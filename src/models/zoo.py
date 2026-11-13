"""The core-five baseline ladder.

  mean_face      : training-set mean shape (the floor every model must beat)
  demographics   : ridge on age/gender/ancestry only
  ridge_snp      : ridge on SNP dosages (alpha tuned on validation)
  mlp_snp        : MLP on SNPs (PCA-reduced for stability with P >> N)
  mlp_snp_demo   : MLP on SNPs + demographics

Advanced rungs (transformer, contrastive, diffusion) register the same way.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from .base import FeatureSet, Model, register, tune_ridge_alpha

_ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


@register("mean_face")
class MeanFace(Model):
    name = "Mean face"
    uses = ()

    def __init__(self, cfg):
        self.mean = None

    def fit(self, train, beta, val=None, beta_val=None):
        self.mean = beta.mean(axis=0)
        return self

    def predict(self, feats):
        return np.tile(self.mean, (feats.n, 1))


@register("demographics")
class Demographics(Model):
    name = "Demographics only"
    uses = ("demographics",)

    def __init__(self, cfg):
        self.model = None

    def fit(self, train, beta, val=None, beta_val=None):
        a, _ = tune_ridge_alpha(train.demo, beta, val.demo, beta_val, _ALPHAS) \
            if val is not None else (1.0, None)
        self.model = Ridge(alpha=a).fit(train.demo, beta)
        self.alpha = a
        return self

    def predict(self, feats):
        return self.model.predict(feats.demo)


@register("ridge_snp")
class RidgeSNP(Model):
    name = "Ridge (SNPs)"
    uses = ("snp",)

    def __init__(self, cfg):
        self.model = None

    def fit(self, train, beta, val=None, beta_val=None):
        a, _ = tune_ridge_alpha(train.snp, beta, val.snp, beta_val, _ALPHAS) \
            if val is not None else (float(1e3), None)
        self.model = Ridge(alpha=a).fit(train.snp, beta)
        self.alpha = a
        return self

    def predict(self, feats):
        return self.model.predict(feats.snp)


def _pca_fit(X, n_comp):
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    comp = vt[:n_comp]
    return mean, comp


class _MLPBase(Model):
    """Shared MLP machinery: PCA-reduce SNPs (P >> N) then fit an MLP."""

    use_demo = False

    def __init__(self, cfg):
        m = cfg["models"]["mlp"]
        self.hidden = tuple(m["hidden_layers"])
        self.max_iter = int(m["max_iter"])
        self.alpha = float(m["alpha"])
        self.n_pca = int(m.get("snp_pca", 150))
        self._pca_mean = self._pca_comp = None
        self._feat_mean = self._feat_std = None
        self.net = None

    def _features(self, feats: FeatureSet, fit: bool = False):
        X = feats.snp
        if fit:
            k = min(self.n_pca, X.shape[0], X.shape[1])
            self._pca_mean, self._pca_comp = _pca_fit(X, k)
        Xp = (X - self._pca_mean) @ self._pca_comp.T
        if self.use_demo:
            Xp = np.concatenate([Xp, feats.demo], axis=1)
        # standardise the assembled matrix so SNP-PCs and demographics share a
        # scale and the net can actually use the demographic inputs
        if fit:
            self._feat_mean = Xp.mean(axis=0)
            self._feat_std = Xp.std(axis=0)
            self._feat_std[self._feat_std == 0] = 1.0
        return (Xp - self._feat_mean) / self._feat_std

    def fit(self, train, beta, val=None, beta_val=None):
        Xtr = self._features(train, fit=True)
        self.net = MLPRegressor(
            hidden_layer_sizes=self.hidden, alpha=self.alpha,
            max_iter=self.max_iter, early_stopping=True, n_iter_no_change=15,
            random_state=0,
        ).fit(Xtr, beta)
        return self

    def predict(self, feats):
        return self.net.predict(self._features(feats))


@register("mlp_snp")
class MLPSNP(_MLPBase):
    name = "MLP (SNPs)"
    uses = ("snp",)
    use_demo = False


@register("mlp_snp_demo")
class MLPSNPDemo(_MLPBase):
    name = "MLP (SNPs + demographics)"
    uses = ("snp", "demographics")
    use_demo = True
