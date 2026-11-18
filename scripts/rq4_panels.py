"""RQ4: do the published Shaffer loci beat frequency-matched random panels?

This runs on 1000 Genomes genotypes with effects planted at the published Shaffer et al.
(2016) lead SNPs and their linkage neighbours. It tests the panel machinery -- that a
panel carrying planted effects is recoverable above panels that do not -- and says
nothing about whether those loci predict real faces. The faces are simulated and the
effects are ours.

Protocol, frozen in PREREGISTRATION.md section 7:

  measure    cov+PC gain of a ridge on the published-loci panel
  reference  100 random panels matched on panel size and allele-frequency bin
  report     the published panel's percentile in that null
  control    the same comparison on a cohort whose effects are planted at RANDOM
             loci, where the published panel must sit near the median

The control is the part that could fail. Without it, a published panel beating random
panels would be explained just as well by the panel being bigger, commoner, or better
imputed than the random draws -- all of which the frequency matching is meant to remove
but none of which it can be assumed to remove.

    python scripts/rq4_panels.py --data-dir outputs/kg --out gpu/rq4_panels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gpu.controls import _ridge_fit_predict, r2  # noqa: E402


def _norm_chrom(x):
    """'chr20' / '20' / 20 -> '20', so the two files' conventions cannot silently differ."""
    t = str(x).strip()
    return t[3:] if t.lower().startswith("chr") else t


def load(data_dir, cfg_path):
    """Dataset plus the chrom/pos arrays, which live in the genotype bundle.

    dataset.npz stores the genotype matrix but not the variant coordinates; those stay
    in the 1000 Genomes bundle named by the config's data.genotype_source. Lead SNPs are
    matched by chromosome and position rather than by identifier, because the bundle
    carries PLINK's own variant ids, which need not be rsIDs. The column alignment
    between the two files is checked rather than assumed: a silent misalignment would
    put the published panel on the wrong columns and produce a quiet null.
    """
    from src.common import load_config
    d = np.load(os.path.join(data_dir, "dataset.npz"), allow_pickle=True)
    src = load_config(cfg_path)["data"].get("genotype_source")
    if not src:
        raise SystemExit(f"{cfg_path} has no data.genotype_source, so there are no coordinates to match")
    if not os.path.isabs(src):
        src = os.path.join(ROOT, src)
    B = np.load(src, allow_pickle=True)
    n = d["genotypes"].shape[1]
    if len(B["pos"]) != n:
        raise SystemExit(f"{len(B['pos'])} variants in the bundle against {n} SNP columns in "
                         f"{data_dir}/dataset.npz: the two are not column-aligned")
    return d, np.array([_norm_chrom(c) for c in B["chrom"]]), np.asarray(B["pos"], float)


def splits_of(d):
    tr, te = np.asarray(d["split_train"]), np.asarray(d["split_test"])
    if np.intersect1d(tr, te).size:
        raise SystemExit("train and test splits overlap")
    return tr, te


def freq_bins(G, n_bins=10):
    """Allele-frequency decile per SNP, from the cohort's own frequencies."""
    f = np.nanmean(G, 0) / 2.0
    f = np.minimum(f, 1 - f)                                      # fold to MAF
    edges = np.quantile(f[np.isfinite(f)], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    return np.clip(np.digitize(f, edges[1:-1]), 0, n_bins - 1), f


def matched_panel(bins, panel_idx, rng, forbid):
    """A random panel with the same per-bin composition as `panel_idx`, avoiding `forbid`."""
    out = []
    for b, want in zip(*np.unique(bins[panel_idx], return_counts=True)):
        pool = np.setdiff1d(np.flatnonzero(bins == b), forbid, assume_unique=False)
        if len(pool) < want:                                      # bin too small: take all
            out.append(pool)
        else:
            out.append(rng.choice(pool, size=want, replace=False))
    return np.concatenate(out) if out else np.array([], int)


def panel_gain(G, X_cov, y, panel, tr, te, alpha, base_r2=None):
    """cov+PC-adjusted R2 gain of a ridge on covariates plus this panel.

    The covariate-only baseline does not depend on the panel, so it is computed once
    and passed in: every panel is scored against the same reference, as the gain
    definition in EVALUATION.md requires.
    """
    if base_r2 is None:
        (pb,) = _ridge_fit_predict(X_cov[tr], y[tr], [X_cov[te]], alpha)
        base_r2 = r2(y[te], pb)
    X = np.hstack([X_cov, G[:, panel]])
    (pf,) = _ridge_fit_predict(X[tr], y[tr], [X[te]], alpha)
    return r2(y[te], pf) - base_r2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="outputs/kg")
    ap.add_argument("--control-dir", default=None,
                    help="cohort with effects planted at RANDOM loci; required for the control")
    ap.add_argument("--config", default="config/kg.yaml",
                    help="config of --data-dir; its data.genotype_source supplies the rsids")
    ap.add_argument("--control-config", default=None,
                    help="config of --control-dir, if different from --config")
    ap.add_argument("--build-report", default="gpu/build_report_kg.json")
    ap.add_argument("--n-random-panels", type=int, default=100)
    ap.add_argument("--ld-window", type=int, default=250_000,
                    help="bp either side of a lead SNP counted as a linkage neighbour")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", default="gpu/rq4_panels.json")
    a = ap.parse_args()

    B = json.load(open(os.path.join(ROOT, a.build_report)))
    leads = [s for s in B["shaffer_snps"] if s.get("found")]
    print(f"{len(leads)} published lead SNPs located in the cohort")

    def published_panel(data_dir, cfg_path):
        d, chrom, pos = load(data_dir, cfg_path)
        sel, n_lead = set(), 0
        for rec in leads:
            c = _norm_chrom(rec["chrom"])
            hit = np.flatnonzero((chrom == c) & (pos == float(rec["pos"])))
            if hit.size:
                sel.update(hit.tolist()); n_lead += 1
            near = np.flatnonzero((chrom == c) & (np.abs(pos - float(rec["pos"])) <= a.ld_window))
            sel.update(near.tolist())
        print(f"  {data_dir}: {n_lead} of {len(leads)} lead SNPs matched by position, "
              f"{len(sel)} SNPs in the panel once linkage neighbours are added")
        if n_lead == 0:
            raise SystemExit(
                "no lead SNP matched by chromosome and position. The panel would be empty, so the "
                "comparison cannot run. Bundle chromosomes look like "
                f"{sorted(set(chrom))[:8]}; the build report expects "
                f"{sorted({_norm_chrom(r['chrom']) for r in leads})[:8]}.")
        return d, np.array(sorted(sel), int)

    rng = np.random.default_rng(a.seed)
    out = {"n_leads_found": len(leads), "ld_window_bp": a.ld_window,
           "n_random_panels": a.n_random_panels, "arms": {}}

    arms = {"published_effects": (a.data_dir, a.config)}
    if a.control_dir:
        arms["random_effects_control"] = (a.control_dir, a.control_config or a.config)

    for name, (ddir, cfgp) in arms.items():
        d, panel = published_panel(ddir, cfgp)
        G = np.nan_to_num(d["genotypes"].astype(np.float64),
                          nan=np.nanmean(d["genotypes"].astype(np.float64)))
        y = d["face_params"].astype(np.float64)
        tr, te = splits_of(d)
        X_cov = d["demographics"].astype(np.float64)
        bins, maf = freq_bins(G)

        (pb,) = _ridge_fit_predict(X_cov[tr], y[tr], [X_cov[te]], a.alpha)
        base_r2 = r2(y[te], pb)
        obs = panel_gain(G, X_cov, y, panel, tr, te, a.alpha, base_r2)
        null = []
        for i in range(a.n_random_panels):
            rp = matched_panel(bins, panel, rng, forbid=panel)
            null.append(panel_gain(G, X_cov, y, rp, tr, te, a.alpha, base_r2))
            if (i + 1) % 25 == 0:
                print(f"  {name}: {i+1}/{a.n_random_panels} random panels")
        null = np.array(null)
        pct = float((null < obs).mean() * 100)
        out["arms"][name] = {
            "panel_size": int(len(panel)), "gain": float(obs),
            "random_panel_gain_mean": float(null.mean()), "random_panel_gain_sd": float(null.std(ddof=1)),
            "random_panel_p05": float(np.percentile(null, 5)),
            "random_panel_p95": float(np.percentile(null, 95)),
            "percentile": pct,
            "mean_maf_published": float(maf[panel].mean())}
        print(f"{name}: panel {len(panel)} SNPs, gain {obs:+.4f}, random "
              f"{null.mean():+.4f}+-{null.std(ddof=1):.4f}, percentile {pct:.1f}")

    acc = {}
    if "published_effects" in out["arms"]:
        acc["published panel above the 95th percentile of matched-random panels"] = \
            bool(out["arms"]["published_effects"]["percentile"] >= 95)
    if "random_effects_control" in out["arms"]:
        p = out["arms"]["random_effects_control"]["percentile"]
        acc["control: published panel near the median (25-75) when effects are at random loci"] = \
            bool(25 <= p <= 75)
    else:
        acc["control NOT RUN: pass --control-dir, the result is uninterpretable without it"] = False
    out["acceptance"] = acc

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in acc.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("wrote " + a.out)


if __name__ == "__main__":
    main()
