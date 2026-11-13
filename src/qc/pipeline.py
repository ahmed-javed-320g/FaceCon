"""Genotype quality control and feature preparation.

Mirrors the report's QC pipeline at prototype scale: subject- and SNP-level
missingness filtering, minor-allele-frequency filtering, mean imputation of the
remaining missing calls, standardisation, and optional ancestry principal
components as covariates. Every statistic (MAF, means, stds, PCA) is fit on the
training split only, so no test information leaks into preprocessing -- the
report's experimental-integrity requirement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QCResult:
    kept_snp_idx: np.ndarray         # indices (into original SNPs) retained
    report: dict                     # human-readable QC summary


@dataclass
class GenotypeTransformer:
    """Fit on train, apply to any split. Produces standardised SNPs (+ PCs)."""

    kept_snp_idx: np.ndarray
    snp_mean: np.ndarray             # imputation + centring means (per kept SNP)
    snp_std: np.ndarray
    pca_mean: np.ndarray | None
    pca_components: np.ndarray | None   # (n_pcs, n_kept)

    def transform(self, genotypes: np.ndarray) -> np.ndarray:
        g = genotypes[:, self.kept_snp_idx].astype(np.float64)
        # impute missing with the train mean dosage for that SNP
        inds = np.where(~np.isfinite(g))
        if inds[0].size:
            g[inds] = np.take(self.snp_mean, inds[1])
        return (g - self.snp_mean) / self.snp_std

    def pcs(self, genotypes: np.ndarray) -> np.ndarray:
        if self.pca_components is None:
            return np.zeros((genotypes.shape[0], 0))
        z = self.transform(genotypes)
        return (z - self.pca_mean) @ self.pca_components.T


def run_qc(cfg, genotypes, train_idx) -> tuple[QCResult, GenotypeTransformer]:
    """Fit QC on the training subjects; return filters + a fitted transformer."""
    q = cfg["qc"]
    g_tr = genotypes[train_idx]
    n_snp = genotypes.shape[1]

    # SNP-level missingness (computed on train)
    snp_miss = np.mean(~np.isfinite(g_tr), axis=0)
    keep = snp_miss <= q["snp_missingness_max"]

    # MAF filter (train); allele freq p = mean dosage / 2
    with np.errstate(invalid="ignore"):
        mean_dos = np.nanmean(g_tr, axis=0)
    p = mean_dos / 2.0
    maf = np.minimum(p, 1 - p)
    keep &= np.nan_to_num(maf, nan=0.0) >= q["maf_min"]

    kept = np.where(keep)[0]
    if kept.size == 0:
        raise ValueError("QC removed all SNPs; loosen thresholds in config.")

    # fit imputation means + standardisation on kept SNPs (train only)
    g_tr_k = g_tr[:, kept]
    snp_mean = np.nanmean(g_tr_k, axis=0)
    g_imp = np.where(np.isfinite(g_tr_k), g_tr_k, snp_mean)
    snp_std = g_imp.std(axis=0)
    snp_std[snp_std == 0] = 1.0

    # optional ancestry principal components (fit on standardised train)
    pca_mean = pca_comp = None
    n_pcs = int(q.get("n_pcs", 0))
    if n_pcs > 0:
        z = (g_imp - snp_mean) / snp_std
        pca_mean = z.mean(axis=0)
        u, s, vt = np.linalg.svd(z - pca_mean, full_matrices=False)
        pca_comp = vt[:n_pcs]

    # subject-level missingness is reported (synthetic rate is low; no subject cut)
    subj_miss = np.mean(~np.isfinite(genotypes[:, kept]), axis=1)

    report = {
        "n_snps_in": int(n_snp),
        "n_snps_kept": int(kept.size),
        "n_snps_dropped_missing": int(np.sum(snp_miss > q["snp_missingness_max"])),
        "n_snps_dropped_maf": int(np.sum(np.nan_to_num(maf) < q["maf_min"]) ),
        "max_subject_missingness": float(subj_miss.max()),
        "n_pcs": n_pcs,
    }
    transformer = GenotypeTransformer(
        kept_snp_idx=kept, snp_mean=snp_mean, snp_std=snp_std,
        pca_mean=pca_mean, pca_components=pca_comp,
    )
    return QCResult(kept_snp_idx=kept, report=report), transformer
