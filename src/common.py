"""Shared types, config loading, and reproducibility helpers.

This module is intentionally dependency-light so every other module can import
it without pulling in heavy libraries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced clearly to the user
    raise ImportError(
        "PyYAML is required to load configs. Install with: pip install pyyaml"
    ) from exc


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Fail early and clearly on the invariants the pipeline relies on."""
    shares = cfg.get("face_variance", {})
    total = sum(float(shares.get(k, 0.0)) for k in ("genetic", "demographic", "noise"))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"face_variance shares must sum to 1.0, got {total:.4f}. "
            "Adjust genetic/demographic/noise in the config."
        )
    data = cfg.get("data", {})
    if data.get("n_causal_snps", 0) > data.get("n_snps", 0):
        raise ValueError("n_causal_snps cannot exceed n_snps.")
    split = cfg.get("split", {})
    s = sum(float(split.get(k, 0.0)) for k in ("train", "val", "test"))
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {s:.4f}.")


def rng_for(seed: int, stream: str) -> np.random.Generator:
    """Derive an independent, reproducible RNG for a named stream.

    Using one named sub-stream per concern (genotypes, faces, splits, ...) keeps
    changes in one part of the pipeline from shifting the random draws in another.
    """
    mixed = np.random.SeedSequence([seed, _str_seed(stream)])
    return np.random.default_rng(mixed)


def _str_seed(text: str) -> int:
    """Deterministic non-negative int from a string (stable across runs)."""
    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# Core data containers
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """A paired synthetic dataset, in the canonical in-memory form.

    The real-data loader (phase 2) will produce an object with exactly these
    fields, so nothing downstream needs to change when access lands.
    """

    subject_ids: np.ndarray          # (N,) str
    genotypes: np.ndarray            # (N, M) dosages in {0,1,2}, may contain NaN
    demographics: np.ndarray         # (N, D) encoded covariates
    demographic_names: list[str]
    face_params: np.ndarray          # (N, K) shape vectors (prediction target)
    ancestry: np.ndarray             # (N,) int group label
    splits: dict[str, np.ndarray] = field(default_factory=dict)  # name -> idx
    meta: dict[str, Any] = field(default_factory=dict)
    # ---- appearance (optional; present when the appearance module is enabled) ----
    appearance_snps: np.ndarray | None = None    # (N, P) dosages for the appearance panel
    appearance_labels: np.ndarray | None = None  # (N, T) int class per trait
    appearance_traits: list[str] = field(default_factory=list)   # trait keys, column order

    @property
    def n_subjects(self) -> int:
        return self.genotypes.shape[0]

    @property
    def n_snps(self) -> int:
        return self.genotypes.shape[1]

    @property
    def n_face_dims(self) -> int:
        return self.face_params.shape[1]

    def check(self) -> None:
        """Assert the invariants other modules depend on."""
        n = self.n_subjects
        assert self.subject_ids.shape == (n,), "subject_ids shape mismatch"
        assert self.demographics.shape[0] == n, "demographics row mismatch"
        assert self.face_params.shape[0] == n, "face_params row mismatch"
        assert self.ancestry.shape == (n,), "ancestry shape mismatch"
        finite = self.genotypes[np.isfinite(self.genotypes)]
        assert set(np.unique(finite)).issubset({0.0, 1.0, 2.0}), (
            "genotype dosages must be in {0,1,2}"
        )


@dataclass
class GroundTruth:
    """What we planted, kept apart from the data so evaluation can verify it."""

    causal_snp_idx: np.ndarray       # indices of SNPs that drive the face
    genetic_weights: np.ndarray      # (n_causal, K) true effect sizes
    demo_weights: np.ndarray         # (D, K) true demographic effect sizes
    variance_shares: dict[str, float]
    # appearance: per-trait causal SNP indices (into the appearance panel)
    appearance_causal: dict[str, np.ndarray] = field(default_factory=dict)
