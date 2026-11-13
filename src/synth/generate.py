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

    # ---- appearance module (optional) ----
    app_snps = app_labels = None
    app_traits = []
    if cfg.get("appearance", {}).get("enable", True):
        from ..appearance.generate import generate_appearance
        age_z = demo[:, demo_names.index("age_z")]
        app_snps, app_labels, app_causal, app_traits = generate_appearance(
            cfg, ancestry, age_z)
        gt.appearance_causal = app_causal

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
        appearance_snps=app_snps,
        appearance_labels=app_labels,
        appearance_traits=app_traits,
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


def _compose_faces(cfg, genotypes, demo, demo_names, k):
    rng = rng_for(cfg["seed"], "faces_compose")
    shares = cfg["face_variance"]
    n = genotypes.shape[0]

    # ---- genetic component from a causal SNP subset ----
    n_causal = int(cfg["data"]["n_causal_snps"])
    causal_idx = rng.choice(genotypes.shape[1], size=n_causal, replace=False)
    causal_idx.sort()
    Xc = genotypes[:, causal_idx]
    Xc = Xc - Xc.mean(axis=0, keepdims=True)          # centre
    gen_w = rng.normal(size=(n_causal, k))
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

    # ---- rescale each source to its target variance share ----
    def scale_to(mat, target_var):
        v = mat.var(axis=0).mean()
        return mat * np.sqrt(target_var / v) if v > 0 else mat

    G = scale_to(G, shares["genetic"])
    D = scale_to(D, shares["demographic"])
    E = scale_to(E, shares["noise"])

    beta = G + D + E
    # normalise so mean per-dim variance == 1 (matches the face-model scale,
    # where beta ~ unit variance yields a ~10% head deformation)
    s = np.sqrt(beta.var(axis=0).mean())
    beta /= s
    # rescale the stored true weights by the same factors, for the manifest
    gen_w_scaled = gen_w * np.sqrt(shares["genetic"] / (Xc @ gen_w).var(axis=0).mean()) / s

    gt = GroundTruth(
        causal_snp_idx=causal_idx,
        genetic_weights=gen_w_scaled,
        demo_weights=demo_w,
        variance_shares=dict(shares),
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
