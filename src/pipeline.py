"""Orchestration: load -> QC -> features -> split views.

Assembles the per-subject FeatureSet once, fit on the training split, so models
and the evaluation harness all share identical, leak-free inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import Dataset, GroundTruth
from .faces.morphable import FaceModel
from .io.loader import load_dataset
from .models.base import FeatureSet
from .qc.pipeline import run_qc


@dataclass
class Prepared:
    cfg: dict
    ds: Dataset
    gt: GroundTruth | None
    fm: FaceModel
    features: FeatureSet          # all subjects, standardised on train stats
    qc_report: dict
    qc_kept: np.ndarray          # original SNP indices retained by QC (kept-column order)

    def view(self, split: str) -> tuple[FeatureSet, np.ndarray]:
        idx = self.ds.splits[split]
        return self.features.subset(idx), self.ds.face_params[idx]

    def idx(self, split: str) -> np.ndarray:
        return self.ds.splits[split]


def prepare(cfg: dict) -> Prepared:
    arch = cfg["data"].get("genetic_architecture") or {}
    if float(arch.get("dominance_frac", 0) or 0) > 0 or float(arch.get("epistasis_frac", 0) or 0) > 0:
        # This harness's positive control (src/eval/harness.py ablation_gwas_vs_random) is
        # a ridge on additive dosages of the causal panel: under a non-additive target it is
        # misspecified and would report a weak or absent effect while the planted signal is
        # fully present. Refuse rather than return a silently wrong control.
        raise ValueError(
            "config has a non-additive data.genetic_architecture; the CPU harness assumes an "
            "additive target. Use `python -m gpu.controls --config <cfg>` (ORACLE-NA, "
            "+SNP-screen-NA) or gpu/run_all.py --features screen_na instead.")
    ds, gt, fm = load_dataset(cfg)
    train_idx = ds.splits["train"]

    qc, transformer = run_qc(cfg, ds.genotypes, train_idx)
    snp = transformer.transform(ds.genotypes)     # (N, kept) standardised
    pcs = transformer.pcs(ds.genotypes)           # (N, n_pcs)
    feats = FeatureSet(snp=snp, demo=ds.demographics, pcs=pcs)

    return Prepared(cfg=cfg, ds=ds, gt=gt, fm=fm, features=feats,
                    qc_report=qc.report, qc_kept=qc.kept_snp_idx)
