"""Generate a synthetic paired genotype / face dataset.

Design goals that make the dataset a fair test of the FaceCon pipeline:

  * Genotypes are allele dosages in {0,1,2}, drawn from per-ancestry allele
    frequencies so population structure is real and detectable.
  * Ancestry drives BOTH the genotype (via structured SNPs) AND the face (via a
    demographic weight). That shared pathway is the confounding the project
    exists to measure: a genotype-only model can look predictive by proxying
    ancestry, even when the truly causal SNPs explain little. The baseline
    ladder in Phase 3 is what separates the two.
  * The face is composed from genetic, demographic and noise sources whose
    variance shares are set in config and verified after generation.

Nothing here claims genetics predicts faces. We plant a known, modest mapping
so the pipeline and its evaluation can be exercised and checked.
"""
from __future__ import annotations

import numpy as np

from ..common import Dataset, GroundTruth, rng_for
from ..faces.morphable import FaceModel, load_face_model


def generate(cfg: dict) -> tuple[Dataset, GroundTruth, FaceModel]:
    face_model = load_face_model(cfg)
    k = face_model.n_dims

    genotypes, ancestry, allele_freq, structured_mask = _draw_genotypes(cfg)
    demo, demo_names = _draw_demographics(cfg, ancestry, genotypes.shape[0])

    beta, gt = _compose_faces(cfg, genotypes, demo, demo_names, k)

    # Appearance is NOT simulated. Colour comes from the real HIrisPlex-S model
    # (src/appearance/hirisplex.py) applied to genotypes; there is nothing to plant
    # and nothing to fit. The simulated-appearance ladder that used to live here
    # was removed: its labels were generated from the same panel its "snp" model
    # used as features, so recovering them measured the generator, not prediction.

    subject_ids = np.array([f"SUBJ-{i:05d}" for i in range(genotypes.shape[0])])
    geno_missing = _inject_missingness(cfg, genotypes)

    ds = Dataset(
        subject_ids=subject_ids,
        genotypes=geno_missing,
        demographics=demo,
        demographic_names=demo_names,
        face_params=beta,
        ancestry=ancestry,
        splits=_make_splits(cfg, genotypes.shape[0]),
        meta={
            "mode": "synthetic",
            "allele_freq": allele_freq,
            "structured_snp_mask": structured_mask,
            "n_shape_dims": k,
        },
    )
    ds.check()
    return ds, gt, face_model


# --------------------------------------------------------------------------- #
def _draw_genotypes(cfg: dict):
    d = cfg["data"]
    rng = rng_for(cfg["seed"], "genotypes")
    n, m = int(d["n_subjects"]), int(d["n_snps"])
    n_groups = int(d["ancestry"]["n_groups"])
    structured_frac = float(d["ancestry"]["structured_fraction"])

    # subjects split roughly evenly across ancestry groups
    ancestry = rng.integers(0, n_groups, size=n)

    # base minor-allele frequency per SNP
    base_maf = rng.uniform(d["maf_min"], d["maf_max"], size=m)

    # a subset of SNPs diverge between groups -> detectable structure
    structured_mask = rng.random(m) < structured_frac
    # per-group allele frequency matrix (n_groups, m)
    freq = np.tile(base_maf, (n_groups, 1))
    for g in range(n_groups):
        shift = rng.uniform(-0.25, 0.25, size=m) * structured_mask
        freq[g] = np.clip(base_maf + shift, 0.01, 0.99)

    # draw dosages ~ Binomial(2, p) with each subject's group frequencies
    p = freq[ancestry]                      # (n, m)
    genotypes = rng.binomial(2, p).astype(np.float64)
    return genotypes, ancestry, freq, structured_mask


def _draw_demographics(cfg: dict, ancestry: np.ndarray, n: int):
    d = cfg["data"]["demographics"]
    rng = rng_for(cfg["seed"], "demographics")
    n_groups = int(cfg["data"]["ancestry"]["n_groups"])

    age = rng.uniform(d["age_min"], d["age_max"], size=n)
    age_z = (age - age.mean()) / age.std()
    gender = rng.integers(0, 2, size=n).astype(np.float64)   # {0,1}

    # ancestry one-hot (full, so each group has its own face offset)
    anc_oh = np.zeros((n, n_groups))
    anc_oh[np.arange(n), ancestry] = 1.0

    feats = [age_z, gender] + [anc_oh[:, g] for g in range(n_groups)]
    names = ["age_z", "gender"] + [f"ancestry_{g}" for g in range(n_groups)]
    return np.stack(feats, axis=1), names


def _architecture(cfg):
    """(block, dominance_frac, epistasis_frac); fracs are shares OF the genetic share."""
    arch = cfg["data"].get("genetic_architecture") or {}
    return arch, float(arch.get("dominance_frac", 0.0) or 0.0), float(arch.get("epistasis_frac", 0.0) or 0.0)


def _dim_mask(rng, rows, k, dims):
    mask = np.zeros((rows, k), dtype=bool)
    for i in range(rows):
        mask[i, rng.choice(k, size=int(dims), replace=False)] = True
    return mask


def _nonadditive_terms(cfg, Xraw, Xc, k):
    """Dominance and pairwise-interaction designs and raw weights for the causal SNPs.

    Drawn from their own RNG stream ("faces_nonadditive"), in a fixed order whatever
    the fractions are, so arms that share a seed share every draw: the additive part,
    demographics and noise come from "faces_compose" and are untouched, and the DOM,
    EPI and MIX arms share the same dominance weights, pairs and interaction weights.

    Xraw  causal dosages BEFORE centring. Callers may pass mean-imputed, non-integer
          values (gpu/controls.py:_load_real, scripts/generate_matched.py), so the
          heterozygote indicator is a band |x - 1| < 0.5, not x == 1: equality would
          make it all-zero at imputed cells and plant a dominance effect of nothing.
    Xc    the centred dosages (additive coding). Products of centred codes are close
          to orthogonal to every marginal signal, so a marginal screen is genuinely
          blind to the interaction term; that is the point of the epistasis arm.
    """
    arch, _, _ = _architecture(cfg)
    n_causal = Xraw.shape[1]
    rng = rng_for(cfg["seed"], "faces_nonadditive")
    cdims = cfg["data"].get("causal_dims")

    Hraw = (np.abs(Xraw - 1.0) < 0.5).astype(np.float64)
    Hc = Hraw - Hraw.mean(axis=0, keepdims=True)
    dom_w = rng.normal(size=(n_causal, k))
    ddims = arch.get("dominance_dims") or cdims
    if ddims:
        dom_w = dom_w * _dim_mask(rng, n_causal, k, ddims)

    n_pairs = int(arch.get("n_interaction_pairs") or n_causal // 2)
    iu, ju = np.triu_indices(n_causal, 1)
    sel = np.sort(rng.choice(len(iu), size=n_pairs, replace=False))
    pairs = np.stack([iu[sel], ju[sel]], axis=1)       # positions within the causal set
    T = Xc[:, pairs[:, 0]] * Xc[:, pairs[:, 1]]
    Tc = T - T.mean(axis=0, keepdims=True)
    epi_w = rng.normal(size=(n_pairs, k))
    edims = arch.get("epistasis_dims") or cdims
    if edims:
        epi_w = epi_w * _dim_mask(rng, n_pairs, k, edims)
    return Hc, dom_w, pairs, Tc, epi_w


def _compose_faces(cfg, genotypes, demo, demo_names, k):
    rng = rng_for(cfg["seed"], "faces_compose")
    shares = cfg["face_variance"]
    n = genotypes.shape[0]

    # ---- genetic component from a causal SNP subset ----
    n_causal = int(cfg["data"]["n_causal_snps"])
    causal_idx = rng.choice(genotypes.shape[1], size=n_causal, replace=False)
    causal_idx.sort()
    Xraw = genotypes[:, causal_idx]
    Xc = Xraw - Xraw.mean(axis=0, keepdims=True)      # centre
    gen_w = rng.normal(size=(n_causal, k))
    # Optional sparsity (config data.causal_dims): each causal SNP affects only
    # this many of the k shape dimensions, as real loci affect particular facial
    # features. Unset = every causal SNP affects every dimension (the original
    # behaviour; the random stream is then unchanged, so old results reproduce).
    cdims = cfg["data"].get("causal_dims")
    if cdims:
        mask = np.zeros_like(gen_w, dtype=bool)
        for i in range(n_causal):
            mask[i, rng.choice(k, size=int(cdims), replace=False)] = True
        gen_w = gen_w * mask
    G = Xc @ gen_w

    # ---- demographic component (age, gender, ancestry) ----
    Dc = demo - demo.mean(axis=0, keepdims=True)
    demo_w = rng.normal(size=(demo.shape[1], k))
    # give ancestry a firm effect so it is a real confounder on the face
    for j, name in enumerate(demo_names):
        if name.startswith("ancestry_"):
            demo_w[j] *= 1.8
    D = Dc @ demo_w

    # ---- noise ----
    E = rng.normal(size=(n, k))

    # ---- optional non-additive genetic terms (data.genetic_architecture) ----
    # Placed after the last draw from "faces_compose" and taken from a separate
    # stream, so nothing above can shift; absent or zero fractions never construct it.
    arch, f_d, f_e = _architecture(cfg)
    arch_active = f_d > 0 or f_e > 0
    if arch_active:
        Hc, dom_w, pairs, Tc, epi_w = _nonadditive_terms(cfg, Xraw, Xc, k)
        H0, P0 = Hc @ dom_w, Tc @ epi_w

    # ---- rescale each source to its target variance share ----
    def scale_to(mat, target_var):
        v = mat.var(axis=0).mean()
        return mat * np.sqrt(target_var / v) if v > 0 else mat

    def factor(mat, target_var):
        v = mat.var(axis=0).mean()
        return np.sqrt(target_var / v) if v > 0 else 1.0

    g_tot = shares["genetic"]
    if arch_active:
        # Inner split of the genetic share, then rescale the sum so the genetic TOTAL
        # is exact (the blocks are only nearly uncorrelated). The inner split is a
        # request; what was realised is measured and recorded below.
        f_a = 1.0 - f_d - f_e
        a_f = factor(G, g_tot * f_a) if f_a > 0 else 0.0
        h_f = factor(H0, g_tot * f_d) if f_d > 0 else 0.0
        p_f = factor(P0, g_tot * f_e) if f_e > 0 else 0.0
        S = G * a_f + H0 * h_f + P0 * p_f
        c_f = factor(S, g_tot)
        A_part, H_part, P_part = G * (a_f * c_f), H0 * (h_f * c_f), P0 * (p_f * c_f)
        G = S * c_f
    else:
        G = scale_to(G, g_tot)          # unchanged: byte-identical to the additive pipeline
    D = scale_to(D, shares["demographic"])
    E = scale_to(E, shares["noise"])

    beta = G + D + E
    # normalise so mean per-dim variance == 1 (matches the face-model scale,
    # where beta ~ unit variance yields a ~10% head deformation)
    s = np.sqrt(beta.var(axis=0).mean())
    beta /= s
    # rescale the stored true weights by the same factors, for the manifest
    na = {}
    if arch_active:
        gen_w_scaled = gen_w * (a_f * c_f) / s
        tot = beta.var(axis=0).mean()
        share = lambda m: float((m / s).var(axis=0).mean() / tot)
        na = dict(
            dominance_weights=dom_w * (h_f * c_f) / s,
            interaction_pairs=pairs,
            interaction_weights=epi_w * (p_f * c_f) / s,
            genetic_architecture={
                "additive_frac": f_a, "dominance_frac": f_d, "epistasis_frac": f_e,
                "n_interaction_pairs": int(len(pairs)),
                "dominance_dims": arch.get("dominance_dims") or cfg["data"].get("causal_dims"),
                "epistasis_dims": arch.get("epistasis_dims") or cfg["data"].get("causal_dims"),
                "realised": {"additive": share(A_part), "dominance": share(H_part),
                             "epistasis": share(P_part), "genetic_total": share(G)},
            },
        )
    else:
        gen_w_scaled = gen_w * np.sqrt(shares["genetic"] / (Xc @ gen_w).var(axis=0).mean()) / s
        if cfg["data"].get("genetic_architecture"):
            na = dict(genetic_architecture={"additive_frac": 1.0, "dominance_frac": 0.0,
                                            "epistasis_frac": 0.0, "realised": None})

    gt = GroundTruth(
        causal_snp_idx=causal_idx,
        genetic_weights=gen_w_scaled,
        demo_weights=demo_w,
        variance_shares=dict(shares),
        **na,
    )
    return beta, gt


def _inject_missingness(cfg, genotypes, rate: float = 0.005):
    rng = rng_for(cfg["seed"], "missingness")
    g = genotypes.copy()
    mask = rng.random(g.shape) < rate
    g[mask] = np.nan
    return g


def _make_splits(cfg, n):
    rng = rng_for(cfg["seed"], "splits")
    idx = rng.permutation(n)
    s = cfg["split"]
    n_tr = int(s["train"] * n)
    n_va = int(s["val"] * n)
    return {
        "train": np.sort(idx[:n_tr]),
        "val": np.sort(idx[n_tr:n_tr + n_va]),
        "test": np.sort(idx[n_tr + n_va:]),
    }
