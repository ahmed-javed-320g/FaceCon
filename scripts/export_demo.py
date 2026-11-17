"""Export the data bundle for the demonstrator (app/index.html).

    python scripts/export_demo.py [--subjects 36] [--out app/data/demo.js]

Everything comes from the CORRECTED pipeline (Chapter 7 of the report):
  * cohort    outputs/real_match (simulated genotypes, simulated faces)
  * features  screened SNPs, train-only (gpu/data.py, features="screen")
  * models    mean face, demographics (OLS), ridge on demographics + SNPs
              (two-stage, alpha on validation), MLP (best-validation checkpoint)
  * mesh      decoded with the generator's basis AND identity scale, so the
              millimetres match the report
  * colour    HIrisPlex-S on the 41 marker genotypes (src/appearance/hirisplex.py);
              the legacy colour classifiers trained on synthetic labels are NOT used
  * results   recorded runs: gpu/controls.json, gpu/report_back.json (simulated)
              and gpu/results_kg.json (1000 Genomes genotypes, simulated faces)

Nothing in the bundle is a statement about real faces; the page says so.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression, Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from gpu.data import FaceConData                      # noqa: E402
from gpu.train import train_torch, _param_r2          # noqa: E402
from src.appearance.hirisplex import HirisplexPredictor, LABELS   # noqa: E402

TARGET_FACES = 11000


def pack(arr, dtype):
    a = np.ascontiguousarray(arr, dtype=dtype)
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape), "dtype": dtype}


def pack_i16(arr):
    """int16 with one global scale: half the bytes of float32, error < 1e-4 of the range."""
    a = np.asarray(arr, np.float64)
    scale = float(np.abs(a).max() / 32767.0) or 1.0
    q = np.round(a / scale).astype(np.int16)
    d = pack(q, "int16"); d["scale"] = scale
    return d


def head_mesh(fm_mean, fm_faces):
    """Skin of head and neck plus the two eyeballs, decimated.

    The ICT mesh has 56 parts (skin, eyeballs, teeth, lashes ...). The skin is
    the largest part; the eyeballs are the two ~800-vertex spheres. Without them
    the sockets render as holes.
    """
    m = trimesh.Trimesh(fm_mean, fm_faces, process=False)
    comps = sorted(m.split(only_watertight=False), key=lambda c: -len(c.vertices))
    head = comps[0]
    eyes = [c for c in comps[1:6] if len(c.vertices) == 800]
    parts = [head.simplify_quadric_decimation(face_count=TARGET_FACES)] + \
            [e.simplify_quadric_decimation(face_count=500) for e in eyes]
    v, f, part, off = [], [], [], 0
    for j, p in enumerate(parts):                     # part: 0 skin, 1 and 2 eyeballs
        v.append(np.asarray(p.vertices)); f.append(np.asarray(p.faces) + off); off += len(p.vertices)
        part.append(np.full(len(p.vertices), j, np.uint8))
    v, f, part = np.vstack(v), np.vstack(f), np.concatenate(part)
    _, nn = cKDTree(fm_mean).query(v)                 # decimated vertex -> original vertex
    return v, f, nn, part


def recorded_results():
    C = json.load(open(os.path.join(ROOT, "gpu/controls.json")))
    R = json.load(open(os.path.join(ROOT, "gpu/report_back.json")))
    K = json.load(open(os.path.join(ROOT, "gpu/results_kg.json")))
    ms = lambda xs: [round(float(np.mean(xs)), 3), round(float(np.std(xs)), 3)]
    sim_models = {}
    for feat in ("pca", "screen"):
        row = R["results"][f"A matched N=3118 | {feat}"]
        sim_models[feat] = {"demographics": round(row["demographics"][0], 3)}
        for m in ("mlp", "transformer", "contrastive", "diffusion"):
            sim_models[feat][m] = ms(row[m]); sim_models[feat][m + "_mesh"] = ms(row[m + "_mesh"])
    perm = R["results"]["C NEGATIVE CONTROL permuted | screen"]
    scaling = []
    for n in (2182, 4000, 7000):
        row = R["results"][f"B large, train_n={n} | screen"]
        scaling.append({"n": n, "demographics": round(row["demographics"][0], 3),
                        **{m: round(float(np.mean(row[m])), 3) for m in ("mlp", "transformer", "contrastive", "diffusion")}})
    lm, ls = C["ladder_mean"], C["ladder_sd"]
    return {
        "simulated": {
            "label": "Simulated genotypes, simulated faces",
            "cohort": {"n": 3118, "n_snps": 5000, "train": 2182, "val": 467, "test": 469,
                       "n_causal": 150, "genetic_share": C["genetic_share"]},
            "ladder": {k: [round(lm[k], 3), round(ls[k], 3)] for k in lm},
            "delta": {"pca256": round(float(np.mean(C["delta_pca256"])), 3),
                      "screen": round(float(np.mean([d[0] if isinstance(d, list) else d for d in C["delta_screen"]])), 3),
                      "oracle": round(float(np.mean(C["delta_oracle"])), 3),
                      "permuted": round(float(np.mean([d[0] if isinstance(d, list) else d for d in C["delta_permuted"]])), 3),
                      "null_share": round(float(np.mean(C["delta_null_share"])), 3),
                      "hidden_causal": -0.001},
            "models": sim_models,
            "permuted": {m: round(float(perm[m][0]), 3) for m in ("demographics", "mlp", "transformer", "contrastive", "diffusion")},
            "scaling": scaling,
            "power_grid": C.get("power_grid"),
            "acceptance": C.get("acceptance"),
            "gwas_recovery": {"injected_detected": 14, "injected_total": 14, "bonferroni_false_hits": 0,
                              "non_injected": 246, "fpr_at_0_05": 0.028},
            "source": "gpu/controls.json (5 cohorts), gpu/report_back.json (3 training runs, RTX A4000)",
        },
        "kg": {**K, "label": "Real 1000 Genomes genotypes, simulated faces",
               "source": "gpu/results_kg.json"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=36)
    ap.add_argument("--out", default=os.path.join(ROOT, "app/data/demo.js"),
                    help="a .js file (works from file:// on demo day) or a .json file")
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config/real_match.yaml")))
    print("loading cohort (screened SNP features, train-only) ...", flush=True)
    D = FaceConData(root=ROOT, features="screen", device="cpu")
    tr, va, te = D.tr, D.va, D.te
    demo, snp, beta = D.demo, D.snp_feat, D.beta

    # ---- models -------------------------------------------------------------
    preds, cohort = {}, {}
    preds["mean_face"] = np.repeat(beta[tr].mean(0, keepdims=True), len(te), 0)
    lr = LinearRegression().fit(demo[tr], beta[tr])
    preds["demographics"] = lr.predict(demo[te])
    r_tr = beta[tr] - lr.predict(demo[tr]); best, best_r2 = None, -np.inf
    for alpha in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        m = Ridge(alpha=alpha).fit(snp[tr], r_tr)
        r2 = _param_r2(beta[va], lr.predict(demo[va]) + m.predict(snp[va]))
        if r2 > best_r2:
            best, best_r2 = m, r2
    preds["ridge_demo_snp"] = lr.predict(demo[te]) + best.predict(snp[te])
    print("training MLP (best-validation checkpoint) ...", flush=True)
    import torch
    torch.manual_seed(0); np.random.seed(0)
    mm = train_torch("mlp", D, a.epochs, 1e-3, 256, True, "cpu", patience=4, quiet=True, return_pred=True)
    preds["mlp"] = mm["pred"]
    for k, p in preds.items():
        ev = D.evaluate(beta[te], p)
        cohort[k] = {"param_r2": round(ev["param_r2"], 3), "mesh_mm": round(ev["mesh_mm"], 3)}
        print(f"  {k:16s} test R2 {ev['param_r2']:+.3f}   mesh {ev['mesh_mm']:.2f} mm", flush=True)

    # ---- demo subjects: stratified by group and sex, fixed seed -------------
    ds = np.load(os.path.join(ROOT, "outputs/real_match/dataset.npz"), allow_pickle=True)
    anc = ds["ancestry"]; sex = ds["demographics"][:, 0]
    from generate_matched import _draw_matched_demographics
    _, _, age = _draw_matched_demographics(cfg, anc)
    rng = np.random.default_rng(20261118)
    cells = {}
    for j, i in enumerate(te):
        cells.setdefault((int(anc[i]), int(sex[i])), []).append(j)
    per = max(1, a.subjects // len(cells)); pick = []
    for key in sorted(cells):
        pick += list(rng.choice(cells[key], size=min(per, len(cells[key])), replace=False))
    pick = sorted(pick)[: a.subjects]

    # ---- per-subject numbers (full-resolution mesh, as in the report) -------
    def mesh_err(bt, bp):
        return float(np.linalg.norm(D.decode_mm(bp[None]) - D.decode_mm(bt[None]), axis=2).mean())
    hps = HirisplexPredictor().predict_proba(np.clip(ds["hps_markers"], 0, 2))
    subjects = []
    for j in pick:
        i = int(te[j])
        subjects.append({
            "id": str(ds["subject_ids"][i]),
            "sex": "male" if sex[i] == 1 else "female",
            "age": round(float(age[i]), 1),
            "height_z": round(float(ds["demographics"][i, 3]), 2),
            "weight_z": round(float(ds["demographics"][i, 4]), 2),
            "group": "ABC"[int(anc[i])],
            "beta_true": [round(float(x), 5) for x in beta[i]],
            "pred": {k: [round(float(x), 5) for x in p[j]] for k, p in preds.items()},
            "mesh_mm": {k: round(mesh_err(beta[i], p[j]), 3) for k, p in preds.items()},
            "appearance": {t: [round(float(x), 4) for x in hps[t][i]] for t in hps},
        })

    # ---- mesh for the browser: decimated head, basis in mm ------------------
    v, f, nn, part = head_mesh(D.mean, np.load(os.path.join(ROOT, "assets/ict/ict_face_model.npz"))["faces"])
    K = D.basis.shape[0]
    basis = D.basis.reshape(K, D.V, 3)[:, nn, :] * D.mm_per_unit       # (K, Vd, 3) in mm per unit beta
    mean = D.mean[nn] * D.mm_per_unit
    bundle = {
        "note": "Simulated faces. Not a statement about any real person or real accuracy.",
        "mode": "synthetic_matched",
        "units": "mm",
        "mesh": {"mean": pack(mean, "float32"), "faces": pack(f, "uint32"), "basis": pack_i16(basis), "part": pack(part, "uint8")},
        "models": [
            {"key": "mean_face", "name": "Mean face", "uses": "nothing", "kind": "baseline"},
            {"key": "demographics", "name": "Demographics", "uses": "sex, age, height, weight, ancestry", "kind": "baseline"},
            {"key": "ridge_demo_snp", "name": "Ridge (demographics + SNPs)", "uses": "covariates + 256 screened SNPs", "kind": "linear"},
            {"key": "mlp", "name": "Neural net (MLP)", "uses": "covariates + 256 screened SNPs", "kind": "neural"},
        ],
        "cohort_metrics": cohort,
        "cohort_metrics_note": "This export's own models, one CPU training run on the simulated cohort's test set "
                               f"(n={len(te)}). Headline numbers in the Results tab are the recorded multi-run figures.",
        "appearance_labels": LABELS,
        "subjects": subjects,
        "results": recorded_results(),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    text = json.dumps(bundle, separators=(",", ":"))
    with open(a.out, "w") as fh:          # a script, not JSON: fetch() is blocked on file:// pages
        fh.write(("window.FACECON_DEMO=" + text + ";\n") if a.out.endswith(".js") else text)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB, {len(subjects)} subjects, "
          f"{len(v)} vertices, {len(f)} triangles)")


if __name__ == "__main__":
    main()
