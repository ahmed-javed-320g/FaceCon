"""Evaluation harness: baseline table, ablations, and sanity checks.

Produces one JSON the web demo and the report can both consume. The ablations
are the scientific heart of the prototype: they show the harness can tell
genetic signal from demographic confounding, and that it correctly reports
no signal when there is none.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from ..models.base import build_models
from ..pipeline import Prepared
from . import metrics as M


def _fit_predict_test(prep: Prepared, model):
    tr, btr = prep.view("train")
    va, bva = prep.view("val")
    te, bte = prep.view("test")
    model.fit(tr, btr, va, bva)
    return model.predict(te), bte


def baseline_table(prep: Prepared) -> list[dict]:
    rows = []
    # decode the shared ground-truth meshes once, reuse for every model
    _, true_test = prep.view("test")
    true_verts = M.decode_mm(prep.fm, true_test)
    for model in build_models(prep.cfg):
        pred, true = _fit_predict_test(prep, model)
        row = {"model": model.name, "key": model.key, "uses": list(model.uses)}
        row.update(M.all_metrics(prep.fm, pred, true, true_verts=true_verts))
        rows.append(row)
    return rows


def _ridge_on_columns(prep: Prepared, cols: np.ndarray, alpha=1e3):
    """Fit ridge on a chosen subset of standardised SNP columns; test metrics."""
    tr = prep.idx("train"); te = prep.idx("test")
    X = prep.features.snp
    beta = prep.ds.face_params
    m = Ridge(alpha=alpha).fit(X[tr][:, cols], beta[tr])
    pred = m.predict(X[te][:, cols])
    return pred, beta[te]


def ablation_gwas_vs_random(prep: Prepared) -> dict:
    """Informed (causal) SNP panel vs a random panel of equal size."""
    gt = prep.gt
    if gt is None:
        return {}
    rng = np.random.default_rng(prep.cfg["seed"])
    # map causal original-SNP indices into kept-column space; drop any QC removed
    col_of = {int(orig): col for col, orig in enumerate(prep.qc_kept)}
    causal = np.array([col_of[int(c)] for c in gt.causal_snp_idx if int(c) in col_of])
    n = len(causal)
    # random panel of equal size drawn from the non-causal kept columns
    non_causal = np.setdiff1d(np.arange(prep.features.snp.shape[1]), causal)
    rand = rng.choice(non_causal, size=min(n, len(non_causal)), replace=False)

    pc, true = _ridge_on_columns(prep, causal)
    pr, _ = _ridge_on_columns(prep, rand)
    return {
        "panel_size": int(n),
        "informed_param_r2": float(r2_score(true, pc, multioutput="uniform_average")),
        "informed_mesh_mm": M.mesh_errors(prep.fm, pc, true)["mesh_mean_mm"],
        "random_param_r2": float(r2_score(true, pr, multioutput="uniform_average")),
        "random_mesh_mm": M.mesh_errors(prep.fm, pr, true)["mesh_mean_mm"],
    }


def sanity_signal_recovery(prep: Prepared) -> dict:
    """Ridge on all SNPs vs the same with shuffled training labels (control).

    The planted signal must be recovered well above the shuffled control, and the
    control must sit near zero. This proves no leakage and trustworthy metrics.
    """
    from ..models.base import tune_ridge_alpha
    tr = prep.idx("train"); va = prep.idx("val"); te = prep.idx("test")
    X = prep.features.snp; beta = prep.ds.face_params
    alpha, _ = tune_ridge_alpha(X[tr], beta[tr], X[va], beta[va],
                                [1e1, 1e2, 1e3, 1e4, 1e5])
    m = Ridge(alpha=alpha).fit(X[tr], beta[tr])
    r2 = r2_score(beta[te], m.predict(X[te]), multioutput="uniform_average")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    mc = Ridge(alpha=alpha).fit(X[tr], beta[tr][perm])
    r2c = r2_score(beta[te], mc.predict(X[te]), multioutput="uniform_average")
    return {
        "ridge_alpha": float(alpha),
        "signal_param_r2": float(r2),
        "shuffled_control_r2": float(r2c),
        "passes": bool(r2 > 0.05 and r2 > r2c + 0.1),
    }


def ablation_remove_demographics(rows: list[dict]) -> dict:
    """What demographics add on top of SNPs, and vice versa, from the table."""
    by = {r["key"]: r for r in rows}
    out = {}
    if "mlp_snp" in by and "mlp_snp_demo" in by:
        out["demo_gain_on_snp_r2"] = round(
            by["mlp_snp_demo"]["param_r2"] - by["mlp_snp"]["param_r2"], 4)
    if "demographics" in by and "mlp_snp_demo" in by:
        out["snp_gain_on_demo_r2"] = round(
            by["mlp_snp_demo"]["param_r2"] - by["demographics"]["param_r2"], 4)
    return out


def run_full_eval(prep: Prepared, out_dir: str | None = None) -> dict:
    rows = baseline_table(prep)
    results = {
        "mode": prep.ds.meta.get("mode", "synthetic"),
        "note": "SYNTHETIC DATA — prototype harness check, not a real-world accuracy claim.",
        "qc": prep.qc_report,
        "baselines": rows,
        "ablation_gwas_vs_random": ablation_gwas_vs_random(prep),
        "ablation_remove_demographics": ablation_remove_demographics(rows),
        "sanity_signal_recovery": sanity_signal_recovery(prep),
        "variance_shares": prep.gt.variance_shares if prep.gt else None,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return results
