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
KS_NA = [100, 300, 1000, 3000]   # COLUMNS for the 2-df screen: each SNP enters as (additive, het)
NA_TOP = 1500                    # SNPs kept by the 2-df screen (two columns each)
NA_MIN_HET = 10                  # train heterozygotes a SNP needs before it may be selected


_GENO_CACHE = {}


def _load_real(path):
    """Real genotypes (config data.genotype_source), loaded once, mean-imputed."""
    if path not in _GENO_CACHE:
        p = path if os.path.isabs(path) else os.path.join(ROOT, path)
        K = np.load(p, allow_pickle=True)
        g = K["genotypes"].astype(np.float32)
        miss = g < 0
        if miss.any():
            g[miss] = np.nan
            cm = np.nanmean(g, axis=0)
            g = np.where(miss, cm, g).astype(np.float32)
        _GENO_CACHE[path] = (g, K["superpop_code"].astype(np.int64), K["sex"].astype(np.float64))
    return _GENO_CACHE[path]


def n_available(cfg):
    src = cfg["data"].get("genotype_source")
    return len(_load_real(src)[0]) if src else None


def make_data(cfg, seed, n=None, share=None):
    """Build one cohort in memory: genotypes, demographics and shape targets.

    Simulated genotypes are redrawn per seed. With real genotypes
    (data.genotype_source) the people are fixed and each seed draws a new set of
    causal SNPs, effects, noise, demographics and splits; `n` then subsamples people.
    """
    c = copy.deepcopy(cfg)
    c["seed"] = int(seed)
    if n is not None:
        c["data"]["n_subjects"] = int(n)
    if share is not None:
        d = c["face_variance"]["demographic"]
        c["face_variance"] = {"genetic": float(share), "demographic": d,
                              "noise": 1.0 - d - float(share)}
    src = c["data"].get("genotype_source")
    if src:
        G, anc, sex = _load_real(src)
        if n is not None and n < len(G):
            keep = np.sort(np.random.default_rng(seed).choice(len(G), int(n), replace=False))
            G, anc, sex = G[keep], anc[keep], sex[keep]
        demo, names, _ = _draw_matched_demographics(c, anc, sex=sex)
    else:
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
                te=np.sort(idx[ntr + nva:]),
                pairs=gt.interaction_pairs, arch=gt.genetic_architecture or {}, cfg=c, k=k)


def arch_active(cfg):
    a = cfg["data"].get("genetic_architecture") or {}
    return float(a.get("dominance_frac", 0) or 0) > 0 or float(a.get("epistasis_frac", 0) or 0) > 0


def _het(M):
    """Heterozygote indicator. A band, not ==1: real-genotype runs pass mean-imputed dosages."""
    return (np.abs(np.asarray(M, np.float64) - 1.0) < 0.5).astype(np.float64)


def _reference_pairs(D):
    """The interaction pairs the non-additive stream draws at this seed.

    Used when no interaction was planted (additive and DOM arms), so ORACLE-NA has the
    same columns in every arm and its variance penalty is identical across arms: the
    ADD-cal arm then measures exactly the penalty that interprets the others.
    """
    if D.get("pairs") is not None:
        return D["pairs"]
    from src.synth.generate import _nonadditive_terms
    c = copy.deepcopy(D["cfg"])
    c["data"]["genetic_architecture"] = {**(c["data"].get("genetic_architecture") or {}),
                                         "dominance_frac": 0.34, "epistasis_frac": 0.33}
    Xraw = D["G"][:, D["causal"]].astype(np.float64)
    return _nonadditive_terms(c, Xraw, Xraw - Xraw.mean(0), D["k"])[2]


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


def ladder(D, do_pca256=True, permute_seed=None, hide_causal=False, na_rows=False):
    """Run the ladder on one dataset. Returns {row: test prediction} + truth.

    hide_causal: remove the causal SNPs from the features the models may use, so
    signal can only be reached through correlated neighbours (as on a real array,
    where causal variants are usually not genotyped). The ORACLE still uses them.
    """
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
        # by column chunks: each column's result is unchanged, memory stays bounded
        R = np.empty(M.shape, dtype=np.result_type(M.dtype, C_pc.dtype))
        for s in range(0, M.shape[1], 10000):
            blk = M[:, s:s + 10000]
            R[:, s:s + 10000] = blk - C_pc @ _ols(C_pc[tr], blk[tr])
        return R

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
    keep_Z = do_pca256 and permute_seed is None
    if not keep_Z:
        del Z
    perm = None
    if permute_seed is not None:                              # NEGATIVE CONTROL
        perm = np.random.default_rng(permute_seed).permutation(len(Zr))
        for s in range(0, Zr.shape[1], 10000):                # same row shuffle, done by chunks
            Zr[:, s:s + 10000] = Zr[perm, s:s + 10000]
    Yn = (Rb[tr] - Rb[tr].mean(0)) / (Rb[tr].std(0) + 1e-9)
    raw = np.empty(Zr.shape[1])
    for s in range(0, Zr.shape[1], 10000):                    # train-only screen, by chunks
        blk = Zr[tr, s:s + 10000]
        Xn = (blk - blk.mean(0)) / (blk.std(0) + 1e-9)
        raw[s:s + 10000] = ((Xn.T @ Yn) ** 2).sum(1)
    score = np.argsort(-raw)
    if hide_causal:                                           # causal SNPs may never be chosen
        score = score[~np.isin(score, D["causal"])]
    P["+SNP-screen"], info["screen"] = tuned(Zr, KS, score)

    if na_rows:
        # ---- 2-df genotypic screen: additive AND heterozygote code per SNP ----
        # Train-only, residualised on cov+PC like the 1-df screen; closed-form 2-df
        # regression sum of squares, vectorised, in 10k-column chunks (no second
        # full-size matrix is ever held).
        def het_resid(cols):
            Hm = _het(G[:, cols])
            Hm = (Hm - Hm[tr].mean(0)) / (Hm[tr].std(0) + 1e-6)
            Hr = resid_snps(Hm)
            return Hr[perm] if perm is not None else Hr
        M, ntr = G.shape[1], len(tr)
        rows_tr = perm[tr] if perm is not None else tr       # genotype rows paired with train faces
        raw_na = np.full(M, -np.inf)
        for s in range(0, M, 10000):
            e = min(M, s + 10000)
            a_ = Zr[tr, s:e]; a_ = (a_ - a_.mean(0)) / (a_.std(0) + 1e-9)
            h_ = het_resid(np.arange(s, e))[tr]; h_ = (h_ - h_.mean(0)) / (h_.std(0) + 1e-9)
            Sah = np.clip((a_ * h_).mean(0), -0.99, 0.99)     # low-MAF / monomorphic-het guard
            r_a, r_h = a_.T @ Yn / ntr, h_.T @ Yn / ntr
            sc = ((r_a ** 2 - 2 * Sah[:, None] * r_a * r_h + r_h ** 2) / (1 - Sah ** 2 + 1e-9)[:, None]).sum(1)
            sc[_het(G[rows_tr, s:e]).sum(0) < NA_MIN_HET] = -np.inf
            raw_na[s:e] = sc
        order = np.argsort(-raw_na)
        order = order[np.isfinite(raw_na[order])]
        if hide_causal:
            order = order[~np.isin(order, D["causal"])]
        top = order[:NA_TOP]
        assert (_het(G[rows_tr][:, top]).sum(0) >= NA_MIN_HET).all()
        Xna = np.empty((len(B), 2 * len(top)))
        Xna[:, 0::2] = Zr[:, top]; Xna[:, 1::2] = het_resid(top)      # [a1, h1, a2, h2, ...]
        P["+SNP-screen-NA"], info["screen_na"] = tuned(Xna, [k for k in KS_NA if k <= Xna.shape[1]],
                                                       np.arange(Xna.shape[1]))
        del Xna

    if permute_seed is None:
        if do_pca256:
            if hide_causal:
                Z[:, D["causal"]] = 0.0                       # a zero column is a dropped column for PCA
            pc256 = PCA(256, random_state=0).fit(Z[tr]).transform(Z)[:, N_PC:]
            del Z
            P["+SNP-PCA256"], info["pca256"] = tuned(pc256)
        ci = D["causal"]
        P["ORACLE"], info["oracle"] = tuned(Zr[:, ci])
        if na_rows:
            # ORACLE-NA: the true causal SNPs coded correctly for any architecture:
            # additive dosage + heterozygote indicator + centred pair products. Kept beside
            # ORACLE (dosage only): the gap is what coding alone costs a model that is right
            # about which SNPs matter.
            pairs = _reference_pairs(D)
            Gc = G[:, ci].astype(np.float64)
            Xc = Gc - Gc[tr].mean(0)
            T = Xc[:, pairs[:, 0]] * Xc[:, pairs[:, 1]]
            T = resid_snps((T - T[tr].mean(0)) / (T[tr].std(0) + 1e-9))
            Xo = np.hstack([Zr[:, ci], het_resid(ci), T])
            P["ORACLE-NA"], info["oracle_na"] = tuned(Xo)
            info["oracle_na_cols"] = int(Xo.shape[1])
    return P, B[te], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/real_match.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--out", default="gpu/controls.json")
    ap.add_argument("--hide-causal", action="store_true",
                    help="models may not see the causal SNPs, only their neighbours")
    ap.add_argument("--na-rows", action="store_true",
                    help="add +SNP-screen-NA and ORACLE-NA even on an additive config "
                         "(the ADD-cal calibration arm); automatic when the config is non-additive")
    a = ap.parse_args()
    hc = a.hide_causal
    cfg = load_config(a.config)
    nonadd = arch_active(cfg)
    na = bool(a.na_rows or nonadd)
    share = float(cfg["face_variance"]["genetic"])
    seeds = [int(cfg["seed"]) + i for i in range(a.seeds)]
    rows = ["mean", "cov", "cov+PC", "cov+labels", "+SNP-PCA256", "+SNP-screen", "ORACLE"]
    if na:                                   # gated, so a default run keeps its exact 7-row ladder
        rows += ["+SNP-screen-NA", "ORACLE-NA"]
    res = {r: [] for r in rows}
    d_screen, d_oracle, d_perm, d_pca = [], [], [], []
    d_screen_na, d_oracle_na, d_perm_na = [], [], []
    disjoint = True
    t0 = time.time()
    for s in seeds:
        D = make_data(cfg, s)
        disjoint &= not (set(D["tr"]) & set(D["va"]) or set(D["tr"]) & set(D["te"]) or set(D["va"]) & set(D["te"]))
        P, bt, info = ladder(D, hide_causal=hc, na_rows=na)
        for r in rows:
            res[r].append(r2(bt, P[r]))
        lo, hi = boot_ci(bt, P["+SNP-screen"], P["cov+PC"], seed=s)
        d_screen.append((res["+SNP-screen"][-1] - res["cov+PC"][-1], lo, hi))
        d_oracle.append(res["ORACLE"][-1] - res["cov+PC"][-1])
        d_pca.append(res["+SNP-PCA256"][-1] - res["cov+PC"][-1])
        if na:
            lo, hi = boot_ci(bt, P["+SNP-screen-NA"], P["cov+PC"], seed=s)
            d_screen_na.append((res["+SNP-screen-NA"][-1] - res["cov+PC"][-1], lo, hi))
            d_oracle_na.append(res["ORACLE-NA"][-1] - res["cov+PC"][-1])
        Pp, _, _ = ladder(D, do_pca256=False, permute_seed=s, hide_causal=hc, na_rows=na)
        lo, hi = boot_ci(bt, Pp["+SNP-screen"], Pp["cov+PC"], seed=s)
        d_perm.append((r2(bt, Pp["+SNP-screen"]) - r2(bt, Pp["cov+PC"]), lo, hi))
        if na:
            lo, hi = boot_ci(bt, Pp["+SNP-screen-NA"], Pp["cov+PC"], seed=s)
            d_perm_na.append((r2(bt, Pp["+SNP-screen-NA"]) - r2(bt, Pp["cov+PC"]), lo, hi))
        print(f"seed {s}: done ({time.time()-t0:.0f}s)  screen={info['screen']}"
              + (f"  screen_na={info['screen_na']}" if na else ""), flush=True)

    # null-share negative control
    d_null, d_null_na = [], []
    for s in seeds[:3]:
        D = make_data(cfg, s, share=0.0)     # a non-additive config keeps its fracs: this is NULL-NA
        P, bt, _ = ladder(D, do_pca256=False, hide_causal=hc, na_rows=na)
        d_null.append(r2(bt, P["+SNP-screen"]) - r2(bt, P["cov+PC"]))
        if na:
            d_null_na.append({"+SNP-screen-NA": r2(bt, P["+SNP-screen-NA"]) - r2(bt, P["cov+PC"]),
                              "ORACLE-NA": r2(bt, P["ORACLE-NA"]) - r2(bt, P["cov+PC"])})

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
    if na:
        dsn, dpn = np.array(d_screen_na), np.array(d_perm_na)
        print("  -- non-additive-aware rows --")
        print(f"  +SNP-screen-NA (2-df)               {dsn[:,0].mean():+.3f}   per-seed 95% CIs: "
              + " ".join(f"[{l:+.3f},{h:+.3f}]" for _, l, h in dsn))
        print(f"  ORACLE-NA (correct coding)          {np.mean(d_oracle_na):+.3f}")
        print(f"  NEG permuted, +SNP-screen-NA        {dpn[:,0].mean():+.3f}")
        for row in ("+SNP-screen-NA", "ORACLE-NA"):
            print(f"  NEG genetic share = 0, {row:13s}{np.mean([d[row] for d in d_null_na]):+.3f}")

    tests = {
        "A1 permuted genotypes give no gain: no CI above 0 and |mean delta| <= 0.005":
            bool(np.all(dp[:, 1] <= 0) and abs(dp[:, 0].mean()) <= 0.005),
        "A2 null-share |delta| <= 0.01": bool(abs(np.mean(d_null)) <= 0.01),
        "A3 positive control: ORACLE delta >= 0.10 and screen CI > 0 (all seeds)":
            bool(np.mean(d_oracle) >= 0.10 and np.all(ds[:, 1] > 0)),
        "A4 splits disjoint": bool(disjoint),
    }
    info_only = set()
    if na:
        dsn, dpn = np.array(d_screen_na), np.array(d_perm_na)
        tests["A1-NA permuted genotypes give no gain on +SNP-screen-NA: no CI above 0 and |mean delta| <= 0.005"] = \
            bool(np.all(dpn[:, 1] <= 0) and abs(dpn[:, 0].mean()) <= 0.005)
        tests["A2-NA null-share |delta| <= 0.01 on +SNP-screen-NA and ORACLE-NA"] = bool(all(
            abs(np.mean([d[row] for d in d_null_na])) <= 0.01 for row in ("+SNP-screen-NA", "ORACLE-NA")))
    if nonadd:
        # The dosage-only ORACLE is misspecified for a non-additive target: it can fail
        # while the planted signal is fully present. It stays, as information; the
        # required positive control is the correctly coded one. Same 0.10 literal.
        tests["A3-NA positive control: ORACLE-NA delta >= 0.10 and max(+SNP-screen, +SNP-screen-NA) CI > 0 (all seeds)"] = \
            bool(np.mean(d_oracle_na) >= 0.10 and np.all(np.maximum(ds[:, 1], dsn[:, 1]) > 0))
        info_only |= {k for k in tests if k.startswith("A3 ")}
    if hc:                                   # every A3 variant: neighbours carrying signal is the question
        info_only |= {k for k in tests if k.startswith("A3")}
    required = {k: v for k, v in tests.items() if k not in info_only}
    print("\nACCEPTANCE" + ("  (causal SNPs hidden: A3 is reported, not required -- "
                             "whether neighbours carry the signal is the question)" if hc else "")
          + ("  (non-additive: the dosage-only A3 is reported, A3-NA is required)" if nonadd and not hc else ""))
    for k, v in tests.items():
        tag = ("PASS" if v else "FAIL") if k in required else ("info: yes" if v else "info: no")
        print(f"  {tag:9s} {k}")

    out = {"config": a.config, "seeds": seeds, "genetic_share": share,
           "genotype_source": cfg["data"].get("genotype_source"), "hide_causal": hc,
           "causal_dims": cfg["data"].get("causal_dims"),
           "ladder_mean": {r: float(np.mean(res[r])) for r in rows},
           "ladder_sd": {r: float(np.std(res[r])) for r in rows},
           "delta_screen": ds.tolist(), "delta_oracle": d_oracle, "delta_pca256": d_pca,
           "delta_permuted": dp.tolist(), "delta_null_share": d_null, "acceptance": tests}
    if na:                                   # only then, so a default run's JSON keeps its shape
        out.update({"na_rows": True, "genetic_architecture": cfg["data"].get("genetic_architecture"),
                    "delta_screen_na": np.array(d_screen_na).tolist(), "delta_oracle_na": d_oracle_na,
                    "delta_permuted_na": np.array(d_perm_na).tolist(), "delta_null_share_na": d_null_na,
                    "acceptance_info_only": sorted(info_only)})

    if a.power:
        nav = n_available(cfg)
        if nav:
            Ns = [x for x in (1000, 2000) if x < nav] + [nav]
            note = (f"  REAL genotypes ({nav} people available), "
                    f"{cfg['data']['n_causal_snps']} causal SNPs, sparse={cfg['data'].get('causal_dims')}.")
        else:
            Ns = [1000, 2000, 3118, 6000]
            note = "  OPTIMISTIC: 5,000 independent SNPs, 150 causal. Real arrays have ~10^6 correlated SNPs."
        print("\nPOWER GRID  delta R2 of +SNP-screen over cov+PC (mean of 2 seeds).\n" + note)
        Ss = [0.0, 0.1, 0.2]
        grid = {}
        print("      N " + "".join(f"  share={x:<5}" for x in Ss))
        for n in Ns:
            line = f"  {n:5d} "
            for sh in Ss:
                v = []
                for s in seeds[:2]:
                    D = make_data(cfg, s, n=n, share=sh)
                    P, bt, _ = ladder(D, do_pca256=False, hide_causal=hc)
                    v.append(r2(bt, P["+SNP-screen"]) - r2(bt, P["cov+PC"]))
                grid[f"N={n},share={sh}"] = float(np.mean(v)); line += f"  {np.mean(v):+.3f}     "
            print(line, flush=True)
        out["power_grid"] = grid

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}")
    sys.exit(0 if all(required.values()) else 1)


if __name__ == "__main__":
    main()
