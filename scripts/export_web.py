"""Export a compact, web-ready bundle for the Three.js prototype.

Produces app/data/web_data.json containing:
  * a decimated face mesh (mean + triangles) light enough for the browser,
  * the identity basis remapped to the decimated mesh, so the page can decode
    ANY shape vector beta -> mesh (enables the morph/reveal animation),
  * a set of demo subjects with their true beta and each model's predicted beta,
  * the evaluation metrics table.

Arrays are packed as base64 float32/int32 to keep the file small and fast.
"""
from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.pipeline import prepare
from src.models.base import build_models
from src.eval import metrics as M

TARGET_FACES = 9000
N_DEMO = 10


def b64(arr, dtype):
    a = np.ascontiguousarray(arr, dtype=dtype)
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"),
            "shape": list(a.shape), "dtype": dtype}


def decimate(fm):
    mesh = trimesh.Trimesh(vertices=fm.mean, faces=fm.faces, process=False)
    simp = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
    newv = np.asarray(simp.vertices)
    newf = np.asarray(simp.faces)
    # map each decimated vertex to its nearest original vertex (for basis + landmarks)
    tree = cKDTree(fm.mean)
    _, nn = tree.query(newv)
    V = fm.mean.shape[0]
    basis_dec = fm.basis.reshape(fm.basis.shape[0], V, 3)[:, nn, :]      # (K, Vd, 3)
    # landmarks: nearest decimated vertex to each original landmark
    dtree = cKDTree(newv)
    _, lm_dec = dtree.query(fm.mean[fm.landmark_idx])
    # per-decimated-vertex region id (-1 = none), mapped from full-mesh masks
    region_names = list(fm.region_masks.keys())
    region_id = np.full(newv.shape[0], -1, dtype=np.int32)
    for ri, rn in enumerate(region_names):
        mask_full = fm.region_masks[rn]
        region_id[mask_full[nn]] = ri
    return newv, newf, basis_dec, lm_dec, region_id, region_names


# shape-model key -> appearance-model key, so the model selector drives both
SHAPE_TO_APP = {
    "mean_face": "majority", "demographics": "ancestry",
    "ridge_snp": "snp", "mlp_snp": "snp", "mlp_snp_demo": "snp_demo",
}


def _appearance_export(prep, demo_sel):
    ds = prep.ds
    if ds.appearance_snps is None:
        return None, None
    from sklearn.preprocessing import StandardScaler
    from src.appearance.models import build_appearance_models
    from src.appearance.traits import TRAITS

    traits = ds.appearance_traits
    tr = prep.idx("train")
    sc = StandardScaler().fit(ds.appearance_snps[tr])
    snp_tr = sc.transform(ds.appearance_snps[tr])
    snp_sel = sc.transform(ds.appearance_snps[demo_sel])
    demo_tr = ds.demographics[tr]
    demo_sel_feat = ds.demographics[demo_sel]
    y_tr = ds.appearance_labels[tr]

    models = build_appearance_models(traits)
    preds = {}   # app_key -> (preds dict, probs dict)
    for m in models:
        m.fit(snp_tr, demo_tr, y_tr)
        preds[m.key] = m.predict(snp_sel, demo_sel_feat)

    # per-subject: for each appearance model, class + confidence per trait
    by_subject = []
    for r in range(len(demo_sel)):
        entry = {"true": {}, "pred": {}}
        for ti, tk in enumerate(traits):
            entry["true"][tk] = int(ds.appearance_labels[demo_sel[r], ti])
        for app_key, (pr, pb) in preds.items():
            entry["pred"][app_key] = {}
            for tk in traits:
                cls = int(pr[tk][r])
                conf = 1.0
                if pb[tk] is not None:
                    proba, classes = pb[tk]
                    conf = float(proba[r].max())
                entry["pred"][app_key][tk] = [cls, round(conf, 3)]
        by_subject.append(entry)

    meta = {
        "traits": [{"key": t, "name": TRAITS[t].name, "classes": TRAITS[t].classes,
                    "palette": TRAITS[t].palette, "render": TRAITS[t].render}
                   for t in traits],
        "shape_to_app": SHAPE_TO_APP,
        "app_model_names": {m.key: m.name for m in models},
    }
    return meta, by_subject


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml")
    prep = prepare(cfg)
    fm = prep.fm

    newv, newf, basis_dec, lm_dec, region_id, region_names = decimate(fm)
    Vd = newv.shape[0]
    print(f"decimated mesh: {Vd} verts, {len(newf)} faces (from {fm.n_vertices})")

    # fit models, collect predicted betas for the demo subjects
    tr, btr = prep.view("train"); va, bva = prep.view("val")
    test_idx = prep.idx("test")
    # demo subjects: spread across ancestry groups
    anc = prep.ds.ancestry[test_idx]
    demo_sel = []
    for g in np.unique(anc):
        demo_sel += list(test_idx[anc == g][:4])
    demo_sel = sorted(demo_sel)[:N_DEMO]
    demo_feats = prep.features.subset(np.array(demo_sel))

    models = build_models(cfg)
    preds = {}
    for m in models:
        m.fit(tr, btr, va, bva)
        preds[m.key] = m.predict(demo_feats)

    # ---- appearance: fit models, predict for the demo subjects ----
    app_block, app_pred_by_subject = _appearance_export(prep, demo_sel)

    subjects = []
    for r, idx in enumerate(demo_sel):
        subjects.append({
            "id": str(prep.ds.subject_ids[idx]),
            "ancestry": int(prep.ds.ancestry[idx]),
            "demographics": {n: float(v) for n, v in
                             zip(prep.ds.demographic_names, prep.ds.demographics[idx])},
            "beta_true": b64(prep.ds.face_params[idx], "float32"),
            "beta_pred": {k: b64(preds[k][r], "float32") for k in preds},
            "appearance": app_pred_by_subject[r] if app_pred_by_subject else None,
        })

    # reuse the metrics already computed by run_eval.py (avoids a slow re-run)
    eval_path = os.path.join(cfg["data"]["out_dir"], "eval_results.json")
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(
            f"{eval_path} not found. Run `python scripts/run_eval.py` first.")
    results = json.load(open(eval_path))

    out = {
        "note": "SYNTHETIC DATA — prototype demonstration, not a real accuracy claim.",
        "mm_per_unit": float(fm.mm_per_unit),
        "n_shape_dims": int(fm.n_dims),
        "mean": b64(newv, "float32"),
        "faces": b64(newf, "int32"),
        "basis": b64(basis_dec, "float32"),
        "landmarks": b64(lm_dec, "int32"),
        "region_id": b64(region_id, "int32"),
        "region_names": region_names,
        "appearance_meta": app_block,
        "models": [{"key": m.key, "name": m.name, "uses": list(m.uses)} for m in models],
        "subjects": subjects,
        "metrics": results["baselines"],
        "appearance": results.get("appearance"),
        "ablations": {
            "gwas_vs_random": results["ablation_gwas_vs_random"],
            "signal_recovery": results["sanity_signal_recovery"],
            "remove_demographics": results["ablation_remove_demographics"],
        },
        # Shape Error Heatmap display range — see scripts/compute_heatmap_scale.py.
        # Fixed 0-8 nominal mm = rounded 95th percentile of per-vertex error pooled
        # across all shape models on the TRAIN split (never test); one scale for all.
        "heatmap": {"display_max_mm": 8.0, "units": "nominal mm", "above_max_is_red": True,
                    "method": "95th percentile of per-vertex error pooled across models on the train split, rounded up."},
        "variance_shares": results["variance_shares"],
        "dataset": {
            "n_subjects": int(prep.ds.n_subjects),
            "n_snps": int(prep.ds.n_snps),
            "n_test": int(len(test_idx)),
        },
    }
    os.makedirs("app/data", exist_ok=True)
    path = "app/data/web_data.json"
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {path}  ({os.path.getsize(path)/1024/1024:.2f} MB)")
    print(f"demo subjects: {[s['id'] for s in subjects]}")


if __name__ == "__main__":
    main()
