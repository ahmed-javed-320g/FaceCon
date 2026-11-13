"""Model interface, feature container, and registry.

Every model consumes the same FeatureSet and returns FLAME shape vectors, so the
baseline ladder and the (later) transformer / contrastive / diffusion models all
plug into one registry and one evaluation loop. Adding an advanced model is:
write a class with fit/predict, decorate it with @register("key"), list the key
in config -> it appears everywhere automatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureSet:
    """Per-split features a model may draw on. Models pick what they need."""

    snp: np.ndarray          # (N, P) standardised SNP dosages
    demo: np.ndarray         # (N, D) demographic covariates
    pcs: np.ndarray          # (N, C) ancestry principal components (may be width 0)

    def subset(self, idx: np.ndarray) -> "FeatureSet":
        return FeatureSet(self.snp[idx], self.demo[idx], self.pcs[idx])

    @property
    def n(self) -> int:
        return self.snp.shape[0]


class Model(ABC):
    """Predict FLAME shape vectors from genotype and/or demographics."""

    key: str = "base"
    name: str = "Base model"
    #: what the model uses, for honest reporting in the results table
    uses: tuple[str, ...] = ()

    @abstractmethod
    def fit(self, train: FeatureSet, beta: np.ndarray,
            val: FeatureSet | None = None, beta_val: np.ndarray | None = None) -> "Model":
        ...

    @abstractmethod
    def predict(self, feats: FeatureSet) -> np.ndarray:
        ...


_REGISTRY: dict[str, type[Model]] = {}


def register(key: str):
    def deco(cls: type[Model]):
        cls.key = key
        _REGISTRY[key] = cls
        return cls
    return deco


def build_models(cfg) -> list[Model]:
    from . import zoo  # noqa: F401  (import for registration side effects)
    enabled = cfg["models"]["enabled"]
    models = []
    for key in enabled:
        if key not in _REGISTRY:
            raise KeyError(f"Model {key!r} not registered. Known: {sorted(_REGISTRY)}")
        models.append(_REGISTRY[key](cfg))
    return models


def tune_ridge_alpha(Xtr, ytr, Xva, yva, alphas):
    """Pick the ridge alpha with the best validation R2 (uniform over dims)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    best_a, best_s = alphas[0], -np.inf
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        s = r2_score(yva, m.predict(Xva), multioutput="uniform_average")
        if s > best_s:
            best_a, best_s = a, s
    return best_a, best_s
