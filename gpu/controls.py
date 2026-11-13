"""Controls, baseline ladder and power grid for FaceCon (CPU only, ~minutes).

QUESTION
    Does genotype add predictive information about face shape BEYOND covariates
    and ancestry PCs -- and can this pipeline tell when it does and when it
    does not?

LADDER (all fits on TRAIN; hyper-parameters chosen on VAL; TEST touched once)
    mean            predict the training mean face
    cov             sex, age, age^2, height, weight          (no ancestry info)
    cov+PC          + top genotype PCs   <- the realistic baseline: real data
                    has ancestry PCs, never true ancestry labels
    cov+labels      + TRUE ancestry labels (what earlier results used; an
                    advantage real data will not give)
    +SNP-PCA256     cov+PC, then ridge on 256 SNP PCs (what the deep models saw)
    +SNP-screen     cov+PC, then ridge on SNPs screened by train-only association
    ORACLE          cov+PC + the TRUE causal SNPs (ground truth; an upper bound,
                    not a model anyone could run on real data)

NEGATIVE CONTROLS
    permuted        +SNP-screen with genotype rows shuffled  -> delta must be ~0
    null-share      regenerate with genetic share = 0        -> delta must be ~0

ACCEPTANCE TESTS (printed PASS/FAIL; exit code 1 on any FAIL)
    A1  permuted-genotype delta: no seed's 95% CI lies above 0, |mean| <= 0.005
        (a slightly NEGATIVE delta is expected: fitting noise costs accuracy)
    A2  null-share delta: |mean| <= 0.01
    A3  positive control at the configured share: ORACLE delta >= 0.10 and
        +SNP-screen delta has a 95% CI above 0 in every seed
    A4  train/val/test subject sets are disjoint

Everything here is SIMULATED. It tests the pipeline, not human biology.

    python -m gpu.controls                      # ladder + controls, 5 seeds
    python -m gpu.controls --power              # also the N x genetic-share grid
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.common import load_config, rng_for                      # noqa: E402
from src.synth.generate import _draw_genotypes, _compose_faces   # noqa: E402
from generate_matched import _draw_matched_demographics          # noqa: E402

from sklearn.decomposition import PCA

N_COV = 5            # sex, age, age^2, height, weight
N_PC = 10
ALPHAS = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
KS = [50, 150, 500, 1500]


def make_data(cfg, seed, n=None, share=None):
    """Draw genotypes, demographics and shape targets in memory (no files)."""
    c = copy.deepcopy(cfg)
    c["seed"] = int(seed)
    if n is not None:
        c["data"]["n_subjects"] = int(n)
    if share is not None:
        d = c["face_variance"]["demographic"]
        c["face_variance"] = {"genetic": float(share), "demographic": d,
                              "noise": 1.0 - d - float(share)}
    G, anc, _, _ = _draw_genotypes(c)
    demo, names, _ = _draw_matched_demographics(c, anc)
    k = int(c["face_model"]["n_shape_dims"])
    if c["face_variance"]["genetic"] <= 0:           # compose cannot scale a zero share
        c2 = copy.deepcopy(c); c2["face_variance"]["genetic"] = 1e-12
        c2["face_variance"]["noise"] = 1.0 - c2["face_variance"]["demographic"] - 1e-12
        B, gt = _compose_faces(c2, G, demo, names, k)
    else:
        B, gt = _compose_faces(c, G, demo, names, k)
    rng = rng_for(c["seed"], "splits")
    idx = rng.permutation(len(B)); s = c["split"]
    ntr, nva = int(s["train"] * len(B)), int(s["val"] * len(B))
    return dict(G=G.astype(np.float32), demo=demo, B=B, causal=gt.causal_snp_idx,
                tr=np.sort(idx[:ntr]), va=np.sort(idx[ntr:ntr + nva]),
                te=np.sort(idx[ntr + nva:]))


def r2(bt, bp):
    return float((1 - ((bt - bp) ** 2).sum(0) / (((bt - bt.mean(0)) ** 2).sum(0) + 1e-9)).mean())


def boot_ci(bt, pa, pb=None, n_boot=500, seed=0):
    """95% bootstrap CI (subjects resampled) for R2(pa), or R2(pa)-R2(pb)."""
    rng = np.random.default_rng(seed); n = len(bt); out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = r2(bt[i], pa[i])
        out.append(v - r2(bt[i], pb[i]) if pb is not None else v)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _ols(Ctr, Ytr):
    b, *_ = np.linalg.lstsq(Ctr, Ytr, rcond=None)
    return b


def _ridge_fit_predict(Xtr, Ytr, Xs, alpha):
    """Dual-form ridge (n << p friendly). Returns predictions for each X in Xs."""
    K = Xtr @ Xtr.T
    A = np.linalg.solve(K + alpha * np.eye(len(K)), Ytr)
    W = Xtr.T @ A
    return [X @ W for X in Xs]


def ladder(D, do_pca256=True, permute_seed=None):
    """Run the ladder on one dataset. Returns {row: test prediction} + truth."""
    tr, va, te = D["tr"], D["va"], D["te"]
    G = D["G"]; B = D["B"]
    mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6
    Z = (G - mu) / sd                                         # train-only scaling
    pcs = PCA(N_PC, random_state=0).fit(Z[tr]).transform(Z)   # train-only PCs
    pcs = pcs / (pcs[tr].std(0) + 1e-9)
    one = np.ones((len(B), 1))
    cov = D["demo"][:, :N_COV]
    C_cov = np.c_[one, cov]
    C_pc = np.c_[one, cov, pcs]
    C_lab = np.c_[one, D["demo"]]
    P = {"mean": np.repeat(B[tr].mean(0, keepdims=True), len(te), 0)}
    for name, C in (("cov", C_cov), ("cov+PC", C_pc), ("cov+labels", C_lab)):
        P[name] = C[te] @ _ols(C[tr], B[tr])

    # residual stage: what do SNPs add on top of cov+PC?
    b = _ols(C_pc[tr], B[tr])
    base = C_pc @ b
    Rb = B - base                                             # shape residuals
    def resid_snps(M):                                        # SNPs residualised on cov+PC (train fit)
        return M - C_pc @ _ols(C_pc[tr], M[tr])

    def tuned(Xfull, grid_k=None, score=None):
        best = (-1e9, None)
        for k in (grid_k or [None]):
            cols = score[:k] if k else slice(None)
            X = Xfull[:, cols]
            for a in ALPHAS:
                pv, = _ridge_fit_predict(X[tr], Rb[tr], [X[va]], a)
                v = r2(B[va], base[va] + pv)
                if v > best[0]:
                    best = (v, (k, a))
        k, a = best[1]
        cols = score[:k] if k else slice(None)
        pt, = _ridge_fit_predict(Xfull[tr][:, cols], Rb[tr], [Xfull[te][:, cols]], a)
        return base[te] + pt, {"k": k, "alpha": a}

    info = {}
    Zr = resid_snps(Z)
    if permute_seed is not None:                              # NEGATIVE CONTROL
        Zr = Zr[np.random.default_rng(permute_seed).permutation(len(Zr))]
    Yn = (Rb[tr] - Rb[tr].mean(0)) / (Rb[tr].std(0) + 1e-9)
    Xn = (Zr[tr] - Zr[tr].mean(0)) / (Zr[tr].std(0) + 1e-9)
    score = np.argsort(-((Xn.T @ Yn) ** 2).sum(1))            # train-only screen
    P["+SNP-screen"], info["screen"] = tuned(Zr, KS, score)
    if permute_seed is None:
        if do_pca256:
            pc256 = PCA(256, random_state=0).fit(Z[tr]).transform(Z)[:, N_PC:]
            P["+SNP-PCA256"], info["pca256"] = tuned(pc256)
        ci = D["causal"]
        P["ORACLE"], info["oracle"] = tuned(Zr[:, ci])
    return P, B[te], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/real_match.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--out", default="gpu/controls.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    share = float(cfg["face_variance"]["genetic"])
    seeds = [int(cfg["seed"]) + i for i in range(a.seeds)]
    rows = ["mean", "cov", "cov+PC", "cov+labels", "+SNP-PCA256", "+SNP-screen", "ORACLE"]
    res = {r: [] for r in rows}
    d_screen, d_oracle, d_perm, d_pca = [], [], [], []
    disjoint = True
    t0 = time.time()
    for s in seeds:
        D = make_data(cfg, s)
        disjoint &= not (set(D["tr"]) & set(D["va"]) or set(D["tr"]) & set(D["te"]) or set(D["va"]) & set(D["te"]))
        P, bt, info = ladder(D)
        for r in rows:
            res[r].append(r2(bt, P[r]))
        lo, hi = boot_ci(bt, P["+SNP-screen"], P["cov+PC"], seed=s)
        d_screen.append((res["+SNP-screen"][-1] - res["cov+PC"][-1], lo, hi))
        d_oracle.append(res["ORACLE"][-1] - res["cov+PC"][-1])
        d_pca.append(res["+SNP-PCA256"][-1] - res["cov+PC"][-1])
        Pp, _, _ = ladder(D, do_pca256=False, permute_seed=s)
        lo, hi = boot_ci(bt, Pp["+SNP-screen"], Pp["cov+PC"], seed=s)
        d_perm.append((r2(bt, Pp["+SNP-screen"]) - r2(bt, Pp["cov+PC"]), lo, hi))
        print(f"seed {s}: done ({time.time()-t0:.0f}s)  screen={info['screen']}", flush=True)

    # null-share negative control
    d_null = []
    for s in seeds[:3]:
        D = make_data(cfg, s, share=0.0)
        P, bt, _ = ladder(D, do_pca256=False)
        d_null.append(r2(bt, P["+SNP-screen"]) - r2(bt, P["cov+PC"]))

    print(f"\nLADDER  test param-R2, mean +/- sd over {len(seeds)} seeds  (planted genetic share {share})")
    for r in rows:
        v = np.array(res[r]); print(f"  {r:13s} {v.mean():+.3f} +/- {v.std():.3f}")
    ds = np.array(d_screen); dp = np.array(d_perm)
    print("\nWHAT SNPs ADD beyond cov+PC (delta R2)")
    print(f"  +SNP-PCA256 (what deep models saw) {np.mean(d_pca):+.3f}")
    print(f"  +SNP-screen                         {ds[:,0].mean():+.3f}   per-seed 95% CIs: "
          + " ".join(f"[{l:+.3f},{h:+.3f}]" for _, l, h in ds))
    print(f"  ORACLE (true causal SNPs)           {np.mean(d_oracle):+.3f}")
    print(f"  NEG permuted genotypes              {dp[:,0].mean():+.3f}   per-seed 95% CIs: "
          + " ".join(f"[{l:+.3f},{h:+.3f}]" for _, l, h in dp))
    print(f"  NEG genetic share = 0               {np.mean(d_null):+.3f}")

    tests = {
        "A1 permuted genotypes give no gain: no CI above 0 and |mean delta| <= 0.005":
            bool(np.all(dp[:, 1] <= 0) and abs(dp[:, 0].mean()) <= 0.005),
        "A2 null-share |delta| <= 0.01": bool(abs(np.mean(d_null)) <= 0.01),
        "A3 positive control: ORACLE delta >= 0.10 and screen CI > 0 (all seeds)":
            bool(np.mean(d_oracle) >= 0.10 and np.all(ds[:, 1] > 0)),
        "A4 splits disjoint": bool(disjoint),
    }
    print("\nACCEPTANCE")
    for k, v in tests.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    out = {"config": a.config, "seeds": seeds, "genetic_share": share,
           "ladder_mean": {r: float(np.mean(res[r])) for r in rows},
           "ladder_sd": {r: float(np.std(res[r])) for r in rows},
           "delta_screen": ds.tolist(), "delta_oracle": d_oracle, "delta_pca256": d_pca,
           "delta_permuted": dp.tolist(), "delta_null_share": d_null, "acceptance": tests}

    if a.power:
        print("\nPOWER GRID  delta R2 of +SNP-screen over cov+PC (mean of 2 seeds)."
              "\n  OPTIMISTIC: 5,000 independent SNPs, 150 causal. Real arrays have ~10^6 correlated SNPs.")
        Ns, Ss = [1000, 2000, 3118, 6000], [0.0, 0.1, 0.2]
        grid = {}
        print("      N " + "".join(f"  share={x:<5}" for x in Ss))
        for n in Ns:
            line = f"  {n:5d} "
            for sh in Ss:
                v = []
                for s in seeds[:2]:
                    D = make_data(cfg, s, n=n, share=sh)
                    P, bt, _ = ladder(D, do_pca256=False)
                    v.append(r2(bt, P["+SNP-screen"]) - r2(bt, P["cov+PC"]))
                grid[f"N={n},share={sh}"] = float(np.mean(v)); line += f"  {np.mean(v):+.3f}     "
            print(line, flush=True)
        out["power_grid"] = grid

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}")
    sys.exit(0 if all(tests.values()) else 1)


if __name__ == "__main__":
    main()
