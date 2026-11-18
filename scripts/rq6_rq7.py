"""RQ6 (where on the face) and RQ7 (for whom) for the genetic gain.

    python scripts/rq6_rq7.py                                   # simulated cohort, linear models
    python scripts/rq6_rq7.py --with-mlp                        # + the MLP (CPU is fine, ~1 min)
    python scripts/rq6_rq7.py --data-dir outputs/kg --group-names AFR,AMR,EAS,EUR,SAS \
        --out gpu/rq6_rq7_kg.json                               # real genotypes (GPU machine)

Descriptive analyses, pre-registered in PREREGISTRATION.md §5. Every face is
simulated; nothing here describes real people.

RQ6: per region, per landmark and per shape dimension, how much nearer does each
     model get than the demographic baseline? Paired over test subjects, with a
     bootstrap 95% CI.
RQ7: the same gain within each ancestry group. Reported in millimetres (comparable
     across groups) and in R2 (whose denominator is the group's own variance, so it
     is not comparable across groups and is shown for completeness only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gpu.data import FaceConData            # noqa: E402
from gpu.train import _param_r2, train_torch  # noqa: E402

N_BOOT = 2000


def boot_mean_ci(x, seed=0):
    """Mean and 95% percentile-bootstrap CI of a per-subject quantity."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def r2_gain_ci(bt, pa, pb, seed=0):
    """R2(pa) - R2(pb) on the same subjects, with a bootstrap CI over subjects."""
    r2 = lambda t, p: float((1 - ((t - p) ** 2).sum(0) / (((t - t.mean(0)) ** 2).sum(0) + 1e-9)).mean())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(bt), len(bt))
        bs.append(r2(bt[i], pa[i]) - r2(bt[i], pb[i]))
    return [r2(bt, pa) - r2(bt, pb), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def predictions(D, with_mlp):
    tr, va, te = D.tr, D.va, D.te
    demo, snp, beta = D.demo, D.snp_feat, D.beta
    P = {}
    lr = LinearRegression().fit(demo[tr], beta[tr])
    P["demographics"] = lr.predict(demo[te])
    r_tr = beta[tr] - lr.predict(demo[tr]); best, best_r2 = None, -np.inf
    for alpha in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):          # two-stage, as in gpu/train.baselines
        m = Ridge(alpha=alpha).fit(snp[tr], r_tr)
        v = _param_r2(beta[va], lr.predict(demo[va]) + m.predict(snp[va]))
        if v > best_r2:
            best, best_r2 = m, v
    P["ridge_demo_snp"] = lr.predict(demo[te]) + best.predict(snp[te])
    if with_mlp:
        import torch
        torch.manual_seed(0); np.random.seed(0)
        P["mlp"] = train_torch("mlp", D, 150, 1e-3, 256, True, "cpu", patience=4, quiet=True, return_pred=True)["pred"]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="outputs/real_match")
    ap.add_argument("--features", default="screen")
    ap.add_argument("--with-mlp", action="store_true")
    ap.add_argument("--group-names", default=None, help="comma list, in ancestry-code order")
    ap.add_argument("--out", default="gpu/rq6_rq7.json")
    a = ap.parse_args()

    D = FaceConData(root=ROOT, data_dir=a.data_dir, features=a.features, device="cpu")
    P = predictions(D, a.with_mlp)
    bt = D.beta[D.te]
    E = {k: D.evaluate_detail(bt, p) for k, p in P.items()}
    ref = E["demographics"]
    out = {"data_dir": a.data_dir, "features": a.features, "n_test": int(len(D.te)),
           "note": "Simulated faces. Gains are demographics-only error minus model error (mm, "
                   "positive = model nearer), paired over test subjects, bootstrap 95% CI.",
           "rq6": {}, "rq7": {}}

    # ---------------- RQ6 ----------------
    for k, e in E.items():
        if k == "demographics":
            continue
        reg = {g: boot_mean_ci(ref["region_subject_mm"][g] - e["region_subject_mm"][g])
               for g in e["region_subject_mm"]}
        lmk = (ref["landmark_subject_mm"] - e["landmark_subject_mm"]).mean(0)
        out["rq6"][k] = {
            "whole_head_gain_mm": boot_mean_ci(ref["subject_mm"] - e["subject_mm"]),
            "region_gain_mm": reg,
            "region_error_mm": {g: float(v.mean()) for g, v in e["region_subject_mm"].items()},
            "landmark_gain_mm": [float(x) for x in lmk],
            "landmark_gain_summary": {"min": float(lmk.min()), "median": float(np.median(lmk)),
                                      "max": float(lmk.max()), "n_positive": int((lmk > 0).sum()),
                                      "n": int(len(lmk))},
            "per_dim_r2": [float(x) for x in e["per_dim_r2"]],
            "per_dim_r2_gain": [float(x) for x in e["per_dim_r2"] - ref["per_dim_r2"]],
        }
    out["rq6"]["demographics_region_error_mm"] = {g: float(v.mean()) for g, v in ref["region_subject_mm"].items()}

    # ---------------- RQ7 ----------------
    if D.ancestry is not None:
        anc = D.ancestry[D.te]
        names = a.group_names.split(",") if a.group_names else [f"group_{g}" for g in sorted(set(anc.tolist()))]
        for k, e in E.items():
            if k == "demographics":
                continue
            rows = {}
            for g in sorted(set(anc.tolist())):
                m = anc == g
                rows[names[g] if g < len(names) else f"group_{g}"] = {
                    "n_test": int(m.sum()),
                    "gain_mm": boot_mean_ci(ref["subject_mm"][m] - e["subject_mm"][m], seed=g),
                    "r2_gain_within_group": r2_gain_ci(bt[m], P[k][m], P["demographics"][m], seed=g),
                }
            out["rq7"][k] = rows

    json.dump(out, open(a.out, "w"), indent=1)
    # ---------------- print ----------------
    print(f"RQ6/RQ7 on {a.data_dir} ({len(D.te)} test subjects, features={a.features})  SIMULATED FACES")
    for k in out["rq6"]:
        if k == "demographics_region_error_mm":
            continue
        r = out["rq6"][k]; w = r["whole_head_gain_mm"]
        print(f"\n{k}: whole head {w[0]:+.3f} mm [{w[1]:+.3f},{w[2]:+.3f}] nearer than demographics")
        for g, (m_, lo, hi) in sorted(r["region_gain_mm"].items(), key=lambda kv: -kv[1][0]):
            print(f"  {g:9s} {m_:+.3f} mm [{lo:+.3f},{hi:+.3f}]   (error {r['region_error_mm'][g]:.2f} mm)")
        s = r["landmark_gain_summary"]
        print(f"  landmarks: gain min {s['min']:+.3f}, median {s['median']:+.3f}, max {s['max']:+.3f} mm; "
              f"{s['n_positive']}/{s['n']} improve")
        pd = np.array(r["per_dim_r2_gain"])
        print(f"  shape dims: R2 gain range {pd.min():+.3f} to {pd.max():+.3f}; {int((pd > 0).sum())}/50 improve")
        if k in out["rq7"]:
            print("  by ancestry group (gain in mm, 95% CI):")
            for g, v in out["rq7"][k].items():
                gm = v["gain_mm"]
                print(f"    {g:8s} n={v['n_test']:4d}  {gm[0]:+.3f} [{gm[1]:+.3f},{gm[2]:+.3f}]")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
