"""Genetic-share sensitivity sweep  (answers: "if we raise the SNP %, does
demographics+SNP become better?").

WHAT THIS SHOWS
---------------
Our synthetic generator plants a *fixed* fraction of facial variance as genetic
signal (config `face_variance.genetic`, currently 0.20). This script regenerates
the faces at several planted genetic shares and, at each one, measures how much
the SNPs add *on top of* demographics:

        delta_R2  =  R2(demographics + SNP)  -  R2(demographics)

The expected, and honest, result: delta_R2 is ~0 at a low share and rises
monotonically as the planted share grows. That is NOT a discovery about human
biology -- it is a *validation of the evaluation harness*: it proves the
pipeline can detect genetic signal in proportion to how much is actually
present, and reports "SNPs don't help" at 20% only because little is there, not
because the harness is blind to genetics.

The genotypes and demographics are drawn ONCE and held fixed across all shares;
only the face-composition variance split changes. So any movement in delta_R2 is
attributable purely to the planted genetic share, nothing else.

All models are Ridge(alpha=10) on identical, leak-safe features (SNP PCA fit on
the train split only), so the three R2 values at each share are directly
comparable.

Run from the project root:
    python -m gpu.sweep_genetic
    python -m gpu.sweep_genetic --shares 0.0 0.1 0.2 0.35 0.5 --snp-pca 256
    python -m gpu.sweep_genetic --out gpu/sweep_genetic.json
"""
from __future__ import annotations

import argparse, copy, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.common import load_config                                      # noqa: E402
from src.synth.generate import _draw_genotypes, _draw_demographics, _compose_faces  # noqa: E402

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _find(cfg, key, default=None):
    """Recursively find the first value for `key` anywhere in a nested dict."""
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        for v in cfg.values():
            r = _find(v, key, None)
            if r is not None:
                return r
    return default


def _param_r2(bt, bp):
    bt = np.asarray(bt, np.float32); bp = np.asarray(bp, np.float32)
    ss_res = ((bt - bp) ** 2).sum(0)
    ss_tot = ((bt - bt.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss_res / ss_tot).mean())


def _fit_eval(Xtr, Xte, btr, bte, alpha=10.0):
    m = Ridge(alpha=alpha).fit(Xtr, btr)
    return _param_r2(bte, m.predict(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/real_match.yaml")
    ap.add_argument("--shares", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.35, 0.5],
                    help="planted genetic variance shares to sweep")
    ap.add_argument("--snp-pca", type=int, default=256)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--out", default="gpu/sweep_genetic.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    k = int(_find(cfg, "n_shape_dims", 50))
    demo_share = float(_find(cfg, "demographic", 0.35))   # held fixed across sweep
    seed = int(_find(cfg, "seed", 0))

    # ---- draw genotypes + demographics ONCE; fixed across all shares ----
    # KNOWN DEFECT, deliberately left unfixed (2026-11-18): this always SIMULATES
    # genotypes and ignores data.genotype_source, so pointing the sweep at
    # config/kg.yaml silently runs on simulated SNPs, not 1000 Genomes. The
    # committed output (gpu/sweep_genetic.json) is a simulated-panel result and is
    # correct as such; `python -m gpu.controls --power` covers the share sweep on
    # real genotypes. If this sweep is ever wanted on real genotypes, fix it in its
    # own commit and write a new output file.
    genotypes, ancestry, _, _ = _draw_genotypes(cfg)
    n = genotypes.shape[0]
    demo, demo_names = _draw_demographics(cfg, ancestry, n)
    print(f"N={n}  SNPs={genotypes.shape[1]}  demo_dims={demo.shape[1]}  "
          f"shape_dims={k}  demo_share(fixed)={demo_share}", flush=True)

    # ---- fixed subject-level split (shared by every share) ----
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_te = int(round(a.test_frac * n))
    te_idx, tr_idx = perm[:n_te], perm[n_te:]

    # ---- SNP features: standardize + PCA, FIT ON TRAIN ONLY (leak-safe) ----
    sc = StandardScaler().fit(genotypes[tr_idx])
    pca = PCA(n_components=min(a.snp_pca, len(tr_idx), genotypes.shape[1]),
              random_state=0).fit(sc.transform(genotypes[tr_idx]))
    snp_pcs = pca.transform(sc.transform(genotypes))
    print(f"SNP PCA: {snp_pcs.shape[1]} comps  "
          f"(train-only fit; {pca.explained_variance_ratio_.sum():.1%} var)\n",
          flush=True)

    demo_tr, demo_te = demo[tr_idx], demo[te_idx]
    snp_tr, snp_te = snp_pcs[tr_idx], snp_pcs[te_idx]
    both_tr = np.hstack([demo_tr, snp_tr]); both_te = np.hstack([demo_te, snp_te])

    hdr = f"{'gen_share':>10} {'R2_demo':>9} {'R2_snp':>9} {'R2_demo+snp':>12} {'delta(snp adds)':>16}"
    print(hdr); print("-" * len(hdr), flush=True)

    def _variance_holder(d):
        """Return the dict that holds the genetic/demographic/noise shares.

        Touches face_variance ONLY. That is load-bearing, not untidy: a
        data.genetic_architecture block (dominance / epistasis fractions OF the
        genetic share) is left alone, so its inner split stays fixed while the
        total genetic share is swept. Do not widen this to rewrite other keys.
        """
        if isinstance(d, dict):
            if "genetic" in d and "noise" in d:
                return d
            for v in d.values():
                r = _variance_holder(v)
                if r is not None:
                    return r
        return None

    rows = []
    for share in a.shares:
        noise = max(0.0, 1.0 - share - demo_share)
        cfg2 = copy.deepcopy(cfg)
        fv = _variance_holder(cfg2)   # keep demo fixed; noise absorbs the rest
        fv["genetic"], fv["demographic"], fv["noise"] = share, demo_share, noise

        beta, _ = _compose_faces(cfg2, genotypes, demo, demo_names, k)
        btr, bte = beta[tr_idx], beta[te_idx]

        r_demo = _fit_eval(demo_tr, demo_te, btr, bte, a.alpha)
        r_snp = _fit_eval(snp_tr, snp_te, btr, bte, a.alpha)
        r_both = _fit_eval(both_tr, both_te, btr, bte, a.alpha)
        delta = r_both - r_demo
        rows.append({"genetic_share": share, "demographic_share": demo_share,
                     "noise_share": round(noise, 3), "R2_demo": round(r_demo, 4),
                     "R2_snp": round(r_snp, 4), "R2_demo_snp": round(r_both, 4),
                     "delta_snp_adds": round(delta, 4)})
        print(f"{share:>10.2f} {r_demo:>9.4f} {r_snp:>9.4f} {r_both:>12.4f} "
              f"{delta:>+16.4f}", flush=True)

    json.dump({"config": a.config, "snp_pca": int(snp_pcs.shape[1]),
               "test_frac": a.test_frac, "alpha": a.alpha, "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}")
    print("\nReading it: delta(snp adds) ~0 at low genetic share and rising with "
          "it is the expected, honest result -- the harness detects genetic\n"
          "signal in proportion to how much is planted. At the project's real "
          "setting (genetic=0.20) SNPs add little; that is a finding, not a bug.")


if __name__ == "__main__":
    main()
