"""RQ4: do the published Shaffer loci beat frequency-matched random panels?

This runs on 1000 Genomes genotypes with effects planted at the published Shaffer et al.
(2016) lead SNPs. It tests the panel machinery -- that a panel carrying planted effects
is recoverable above panels that do not -- and says nothing about whether those loci
predict real faces. The effects are ours.

THE TARGET IS THE ANTHROPOMETRIC MEASURES, NOT THE SHAPE VECTOR. The generator injects
the published-locus effects onto the 20 measurements only, and says so explicitly:
"the shape target beta, and so every shape model, is unaffected by them"
(scripts/generate_matched.py, _inject_gwas_effects). Scoring these panels against
face_params would therefore return a null whatever the panels contained, and that null
would look exactly like a real negative result. An earlier version of this script made
that mistake.

The lead SNPs are also not columns of `genotypes`. They are extracted separately from
the named-variant file into `gwas_markers`, while `genotypes` holds the 100k panel. So
a published panel is assembled from both files, and two versions of it are scored:

  leads + neighbours   the lead SNPs themselves plus every panel SNP within the window.
                       The effects were planted ON the leads, so this arm is close to
                       tautological: it confirms the planting and the scoring work, and
                       little else. It is reported for completeness and should not be
                       read as evidence that published loci are predictive.
  neighbours only      panel SNPs near the leads, with the leads removed. This is the
                       informative arm, and the realistic one: a genotyping chip rarely
                       carries the lead SNP itself, so what a real study has is linkage.
                       A gain here means linkage carried the signal.

Protocol, frozen in PREREGISTRATION.md section 7:

  measure    cov gain of a ridge on the published-loci panel
  reference  100 random panels matched on panel size and allele-frequency bin
  report     the published panel's percentile in that null
  control    a cohort where the published loci carry nothing, in which the published
             panel must sit near the median

The control is the part that could fail. Without it, a published panel beating random
panels would be explained just as well by the panel being bigger, commoner, or better
imputed than the random draws.

    python scripts/rq4_panels.py --data-dir outputs/kg --control-dir outputs/kg_noeffects \
        --config config/kg.yaml --out gpu/rq4_panels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gpu.controls import ALPHAS, _ridge_fit_predict, r2  # noqa: E402


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
    """Allele-frequency decile per SNP, plus the edges, from the cohort's own frequencies."""
    f = np.nanmean(G, 0) / 2.0
    f = np.minimum(f, 1 - f)                                      # fold to MAF
    edges = np.quantile(f[np.isfinite(f)], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    return np.clip(np.digitize(f, edges[1:-1]), 0, n_bins - 1), f, edges


def bin_of(M, edges, n_bins=10):
    """Bin a separate marker matrix onto the panel's frequency edges."""
    f = np.nanmean(M, 0) / 2.0
    f = np.minimum(f, 1 - f)
    return np.clip(np.digitize(f, edges[1:-1]), 0, n_bins - 1)


def matched_panel(bins, target_bins, rng, forbid):
    """A random panel from the panel SNPs with the same per-bin composition as `target_bins`.

    `target_bins` is the bin of every SNP in the panel being matched, which may come from
    a different file (the lead markers do), so it is passed as bin values rather than as
    column indices.
    """
    out = []
    for b, want in zip(*np.unique(target_bins, return_counts=True)):
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

    def panels_for(data_dir, cfg_path):
        d, chrom, pos = load(data_dir, cfg_path)
        near = set()
        n_lead = 0
        for rec in leads:
            c = _norm_chrom(rec["chrom"])
            hits = np.flatnonzero((chrom == c) & (np.abs(pos - float(rec["pos"])) <= a.ld_window))
            if hits.size:
                n_lead += 1
                near.update(hits.tolist())
        near = np.array(sorted(near), int)
        print(f"  {data_dir}: {n_lead} of {len(leads)} leads have panel SNPs within "
              f"{a.ld_window//1000} kb; {len(near)} neighbour SNPs")
        if len(near) == 0:
            raise SystemExit("no panel SNP lies near any lead; the comparison cannot run")
        return d, near

    rng = np.random.default_rng(a.seed)
    out = {"n_leads_found": len(leads), "ld_window_bp": a.ld_window,
           "n_random_panels": a.n_random_panels, "arms": {}}
    arms = {"published_effects": (a.data_dir, a.config)}
    if a.control_dir:
        arms["control"] = (a.control_dir, a.control_config or a.config)

    for name, (ddir, cfgp) in arms.items():
        d, near = panels_for(ddir, cfgp)
        G = d["genotypes"].astype(np.float64)
        G = np.where(G < 0, np.nan, G)
        G = np.where(np.isnan(G), np.nanmean(G, 0, keepdims=True), G)
        L = d["gwas_markers"].astype(np.float64)                  # the leads themselves
        L = np.where(L < 0, np.nan, L)
        L = np.where(np.isnan(L), np.nanmean(L, 0, keepdims=True), L)
        y = d["anthro_measures"].astype(np.float64)               # <-- where the effects live
        y = (y - y.mean(0)) / (y.std(0) + 1e-9)
        X_cov = d["demographics"].astype(np.float64)
        tr, te = splits_of(d)
        bins, maf, edges = freq_bins(G)
        lead_bins = bin_of(L, edges)

        va = np.asarray(d["split_val"])

        def tuned_r2(X):
            """Ridge with alpha chosen on the validation split, as everywhere else here.

            A fixed alpha is not usable for this comparison: the panels add a few hundred
            columns to 1,801 training subjects, and an alpha suited to the covariates
            alone overfits them so badly that every panel, published or random, scores
            far below the covariate baseline and the comparison measures nothing. The
            grid is gpu.controls.ALPHAS, and it is selected per panel so the published
            panel gets no tuning advantage over the random ones.
            """
            best = (-1e9, None)
            for al in ALPHAS:
                (pv,) = _ridge_fit_predict(X[tr], y[tr], [X[va]], al)
                v = r2(y[va], pv)
                if v > best[0]:
                    best = (v, al)
            (pt,) = _ridge_fit_predict(X[tr], y[tr], [X[te]], best[1])
            return r2(y[te], pt)

        base_r2 = tuned_r2(X_cov)

        def gain(extra):
            return tuned_r2(np.hstack([X_cov, extra])) - base_r2

        out["arms"][name] = {"baseline_r2": float(base_r2), "n_neighbours": int(len(near)),
                             "n_leads": int(L.shape[1]), "versions": {}}
        for ver, extra, tb in (("leads only", L, lead_bins),
                               ("leads + neighbours", np.hstack([L, G[:, near]]),
                                np.concatenate([lead_bins, bins[near]])),
                               ("neighbours only", G[:, near], bins[near])):
            size = len(tb)
            obs = gain(extra)
            null = []
            for i in range(a.n_random_panels):
                rp = matched_panel(bins, tb, rng, forbid=near)
                if size > len(rp):                                # pad to equal width
                    pool = np.setdiff1d(np.arange(G.shape[1]), np.union1d(rp, near))
                    rp = np.concatenate([rp, rng.choice(pool, size - len(rp), replace=False)])
                null.append(gain(G[:, rp[:size]]))
                if (i + 1) % 25 == 0:
                    print(f"    {name} / {ver}: {i+1}/{a.n_random_panels}", flush=True)
            null = np.array(null)
            pct = float((null < obs).mean() * 100)
            out["arms"][name]["versions"][ver] = {
                "panel_size": int(size), "gain": float(obs),
                "random_mean": float(null.mean()), "random_sd": float(null.std(ddof=1)),
                "random_p05": float(np.percentile(null, 5)), "random_p95": float(np.percentile(null, 95)),
                "percentile": pct}
            print(f"  {name:26s} {ver:20s} gain {obs:+.4f}  random {null.mean():+.4f}"
                  f"+-{null.std(ddof=1):.4f}  percentile {pct:.1f}")

    out["target"] = ("anthro_measures: the published-locus effects are injected onto the "
                     "measurements only, never onto face_params")
    acc = {}
    if "published_effects" in out["arms"]:
        v = out["arms"]["published_effects"]["versions"]
        acc["leads only above the 95th percentile (POSITIVE CONTROL ON THE TEST: the effects "
            "were planted on these very SNPs, so a failure here means the test cannot detect "
            "signal that is certainly present, and the other two lines are uninterpretable)"] = \
            bool(v["leads only"]["percentile"] >= 95)
        acc["leads + neighbours above the 95th percentile (near-tautological: confirms planting)"] = \
            bool(v["leads + neighbours"]["percentile"] >= 95)
        acc["neighbours only above the 95th percentile (the informative arm: linkage carries it)"] = \
            bool(v["neighbours only"]["percentile"] >= 95)
    if "control" in out["arms"]:
        for ver, vv in out["arms"]["control"]["versions"].items():
            acc[f"control, {ver}: near the median (25-75) when the loci carry nothing"] = \
                bool(25 <= vv["percentile"] <= 75)
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
