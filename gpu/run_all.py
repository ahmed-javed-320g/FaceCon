"""One command for the whole GPU experiment set. Prints one table to send back.

    python -m gpu.run_all            # full run on the GPU (roughly 30-60 min)
    python -m gpu.run_all --quick    # 2-minute smoke test (MLP only, 1 seed)

EXPERIMENTS (all simulated data; nothing here is a finding about real faces)
  A  Matched cohort (N=3,118): every model x {pca, screen} features x seeds.
     Question: once the signal is visible (screen), does architecture matter,
     and does any model beat the demographic baseline?
  B  Learning curve on the N=10,000 scaling dataset: train on nested subsets
     (2,182 = the matched train size, 4,000, 7,000); val/test fixed.
     Question: does more data help, and which models benefit?
  C  NEGATIVE CONTROL: genotype rows permuted (pairing with faces broken).
     Acceptance: no SNP model beats the demographic baseline by more than 0.01.
Acceptance for B: mean R2 at 7,000 >= mean R2 at 2,182 for the screened MLP.

REPLICATE MODE (Track A; only when --data-seeds or --data-dirs is given)
  Each replicate re-draws the DATA as well as the initialisation, so the spread is
  over splits (--data-seeds: split_seed, PCA seed and permutation per replicate) or
  over independently generated cohorts (--data-dirs, each with its own stored split),
  not over network initialisation alone. Baselines are refitted per replicate and the
  deep-vs-linear gap is paired within each replicate.

    python -m gpu.run_all --data-dirs outputs/na_mix,outputs/na_mix_s1,outputs/na_mix_s2 \
                          --feature-sets screen,screen_na --models mlp,transformer
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
from .data import FaceConData
from .train import baselines, train_torch, MODELS


def run(cfg, models, seeds, device, epochs, patience):
    out = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        data = FaceConData(root=".", device=device, **cfg)
        if seed == seeds[0]:
            for k, v in baselines(data).items():
                out[k] = [v["param_r2"]]
            out["_n_train"] = len(data.tr); out["_mesh_demo"] = baselines(data)["demographics"]["mesh_mm"]
        for m in models:
            r = train_torch(m, data, epochs, 1e-3, 256, True, device, 10, patience, quiet=True)
            out.setdefault(m, []).append(r["param_r2"])
            out.setdefault(m + "_mesh", []).append(r["mesh_mm"])
        del data                       # free the genotype matrix before the next seed
    return out


def run_replicates(reps, feat, models, device, epochs, patience, extra=None):
    """One training run per replicate; replicate i also uses initialisation seed i."""
    out = {}
    for i, (dd, kw) in enumerate(reps):
        torch.manual_seed(i); np.random.seed(i)
        data = FaceConData(root=".", device=device, features=feat, data_dir=dd, **kw, **(extra or {}))
        for k, v in baselines(data).items():
            out.setdefault(k, []).append(v["param_r2"])
        out.setdefault("_n_train", []).append(len(data.tr))
        for m in models:
            r = train_torch(m, data, epochs, 1e-3, 256, True, device, 10, patience, quiet=True)
            out.setdefault(m, []).append(r["param_r2"])
            out.setdefault(m + "_mesh", []).append(r["mesh_mm"])
        del data
    return out


def replicate_main(a, device, models, epochs, patience):
    if a.data_seeds and a.data_dirs:
        raise SystemExit("give --data-seeds OR --data-dirs, not both")
    if a.data_dirs:
        dirs = [d.strip().rstrip("/\\").replace("\\", "/") for d in a.data_dirs.split(",")]
        reps, over = [(d, {}) for d in dirs], "independently generated cohorts (each with its own split) and initialisations"
    else:
        dd = a.data_dir.rstrip("/\\").replace("\\", "/")
        ss = [int(x) for x in a.data_seeds.split(",")]
        reps, over = [(dd, {"seed": s, "split_seed": s}) for s in ss], "data splits, PCA seeds and initialisations"
    feats = a.feature_sets.split(",")
    t0 = time.time(); R = {}
    print(f"device={device} models={models} replicates={len(reps)} over {over}", flush=True)
    for feat in feats:
        R[f"A | {feat}"] = run_replicates(reps, feat, models, device, epochs, patience)
        print(f"  done A/{feat} ({time.time()-t0:.0f}s)", flush=True)
        R[f"C NEGATIVE CONTROL permuted | {feat}"] = run_replicates(reps[:1], feat, models, device, epochs,
                                                                    patience, extra={"permute_snps": True})
        print(f"  done C/{feat} ({time.time()-t0:.0f}s)", flush=True)
    cols = ["demographics", "ridge_demo_snp"] + models
    print("\n==================== COPY FROM HERE ====================")
    print(f"test param-R2, mean±sd over {len(reps)} replicates ({over}) | device={device}")
    print(f"{'experiment':36s} " + " ".join(f"{c:>15s}" for c in cols))
    for k, v in R.items():
        print(f"{k:36s} " + " ".join(f"{fmt(v[c]):>15s}" for c in cols))
    checks, gaps = {}, {}
    for feat in feats:
        c = R[f"C NEGATIVE CONTROL permuted | {feat}"]
        checks[f"C/{feat}: permuted SNP models gain <= 0.01 over demographics"] = bool(
            max(np.mean(c[m]) for m in models) - c["demographics"][0] <= 0.01)
        checks[f"C/{feat}: permuted ridge(demographics+SNP) gains <= 0.01 over demographics"] = bool(
            c["ridge_demo_snp"][0] - c["demographics"][0] <= 0.01)
        v = R[f"A | {feat}"]
        best = max(models, key=lambda m: np.mean(v[m]))
        g = np.array(v[best]) - np.array(v["ridge_demo_snp"])          # paired within replicate
        half = 4.303 * g.std(ddof=1) / np.sqrt(len(g)) if len(g) == 3 else (
            1.96 * g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float("nan"))
        gaps[feat] = {"best_model": best, "gap_per_replicate": g.tolist(), "gap_mean": float(g.mean()),
                      "ci95": [float(g.mean() - half), float(g.mean() + half)],
                      "ci_note": "t interval over replicates (t=4.303 for 3)" if len(g) == 3 else "normal approx."}
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("\nDEEP vs LINEAR, paired within replicate (best deep model - ridge on demographics+SNP):")
    for feat, gg in gaps.items():
        print(f"  {feat:10s} best={gg['best_model']:12s} gap {gg['gap_mean']:+.3f}  95% CI "
              f"[{gg['ci95'][0]:+.3f},{gg['ci95'][1]:+.3f}]  per replicate " + " ".join(f"{x:+.3f}" for x in gg["gap_per_replicate"]))
    print(f"total time {time.time()-t0:.0f}s")
    print("==================== TO HERE ====================")
    json.dump({"device": device, "replicates": [[d, kw] for d, kw in reps], "over": over,
               "results": R, "checks": checks, "deep_vs_linear": gaps}, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")
    return all(checks.values())


def fmt(v):
    v = np.asarray(v, float)
    return f"{v.mean():+.3f}" + (f"±{v.std():.3f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="gpu/report_back.json")
    ap.add_argument("--data-dir", default="outputs/real_match",
                    help="dataset to use, e.g. outputs/kg for the real-genotype dataset")
    ap.add_argument("--data-seeds", default=None,
                    help="replicate mode: comma list; each replicate re-draws split, PCA seed and permutation")
    ap.add_argument("--data-dirs", default=None,
                    help="replicate mode: comma list of independently generated cohorts")
    ap.add_argument("--feature-sets", default="screen", help="replicate mode: e.g. screen,screen_na")
    ap.add_argument("--models", default=None, help="comma list; default all four")
    a = ap.parse_args()
    if a.data_seeds or a.data_dirs:      # the default path below is left exactly as it was
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = a.models.split(",") if a.models else (["mlp"] if a.quick else MODELS)
        epochs, patience = (30, 2) if a.quick else (150, 4)
        sys.exit(0 if replicate_main(a, device, models, epochs, patience) else 1)
    dd = a.data_dir.rstrip("/\\").replace("\\", "/")
    is_default = dd == "outputs/real_match"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = ["mlp"] if a.quick else MODELS
    seeds = [0] if a.quick else list(range(a.seeds))
    epochs, patience = (30, 2) if a.quick else (150, 4)
    have_large = is_default and os.path.isfile("outputs/large/dataset.npz")
    import numpy as _np
    n_all = len(_np.load(os.path.join(dd, "dataset.npz"), allow_pickle=True)["face_params"])
    label = "matched N=3118" if is_default else f"{os.path.basename(dd)} N={n_all}"
    t0 = time.time(); R = {}
    print(f"device={device} models={models} seeds={seeds} large_dataset={have_large}", flush=True)

    for feat in ("pca", "screen"):                                    # ---- A
        R[f"A {label} | {feat}"] = run(dict(features=feat, data_dir=dd), models, seeds, device, epochs, patience)
        print(f"  done A/{feat} ({time.time()-t0:.0f}s)", flush=True)
    if have_large:                                                    # ---- B
        for n in ((2182, 7000) if a.quick else (2182, 4000, 7000)):
            R[f"B large, train_n={n} | screen"] = run(
                dict(features="screen", data_dir="outputs/large", train_n=n),
                models, seeds, device, epochs, patience)
            print(f"  done B/train_n={n} ({time.time()-t0:.0f}s)", flush=True)
    R["C NEGATIVE CONTROL permuted | screen"] = run(                  # ---- C
        dict(features="screen", permute_snps=True, data_dir=dd), models, seeds[:1], device, epochs, patience)

    cols = ["demographics", "ridge_snp", "ridge_demo_snp"] + models
    print("\n==================== COPY FROM HERE ====================")
    print(f"test param-R2 (mean±sd over {len(seeds)} seeds) | device={device} | data={dd} | "
          + ("SIMULATED DATA" if is_default else "REAL GENOTYPES, SIMULATED FACES"))
    print(f"{'experiment':40s} {'n_train':>7s} " + " ".join(f"{c:>13s}" for c in cols))
    for k, v in R.items():
        print(f"{k:40s} {v['_n_train']:>7d} " + " ".join(f"{fmt(v[c]):>13s}" for c in cols))
    checks = {}
    c = R["C NEGATIVE CONTROL permuted | screen"]
    checks["C: permuted SNP models gain <= 0.01 over demographics"] = bool(
        max(np.mean(c[m]) for m in models) - c["demographics"][0] <= 0.01)
    checks["C: permuted ridge(demographics+SNP) gains <= 0.01 over demographics"] = bool(
        np.mean(c["ridge_demo_snp"]) - c["demographics"][0] <= 0.01)
    if have_large:
        checks["B: screened MLP at train_n=7000 >= at 2182"] = bool(
            np.mean(R["B large, train_n=7000 | screen"]["mlp"]) >= np.mean(R["B large, train_n=2182 | screen"]["mlp"]))
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("\nDEEP vs LINEAR, same split and same inputs (best deep model - ridge on demographics+SNP):")
    for k, v in R.items():
        if k.startswith("C "):
            continue
        bd = max((np.mean(v[m]), m) for m in models)
        print(f"  {k:40s} best={bd[1]:12s} {bd[0]:+.3f}  ridge(demo+SNP) {np.mean(v['ridge_demo_snp']):+.3f}"
              f"  gap {bd[0] - np.mean(v['ridge_demo_snp']):+.3f}")
    print(f"total time {time.time()-t0:.0f}s")
    print("==================== TO HERE ====================")
    json.dump({"device": device, "seeds": seeds, "data_dir": dd, "results": R, "checks": checks},
              open(a.out, "w"), indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
