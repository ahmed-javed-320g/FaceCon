"""Appearance classifiers, baseline-first, one predictor per feature set.

Mirrors the shape model ladder:
  majority    : most frequent class per trait (the floor)
  ancestry    : logistic regression on demographics (age, gender, ancestry)
  snp         : logistic regression on the standardised appearance panel
  snp_demo    : logistic regression on appearance panel + demographics

Each model fits one independent classifier per trait and returns both the
predicted class and class probabilities (for confidence in the demo).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

APPEARANCE_MODELS = ["majority", "ancestry", "snp", "snp_demo"]
MODEL_NAMES = {
    "majority": "Majority class",
    "ancestry": "Demographics / ancestry",
    "snp": "Appearance SNPs",
    "snp_demo": "Appearance SNPs + demographics",
}


def _features(key, snp_std, demo):
    if key == "ancestry":
        return demo
    if key == "snp":
        return snp_std
    if key == "snp_demo":
        return np.concatenate([snp_std, demo], axis=1)
    return None   # majority uses no features


class AppearancePredictor:
    """Per-trait classifiers for one feature set (one of APPEARANCE_MODELS)."""

    def __init__(self, key, traits):
        self.key = key
        self.name = MODEL_NAMES[key]
        self.traits = traits          # list of trait keys (label column order)
        self.clf = {}                 # trait -> fitted estimator (or majority class)

    def fit(self, snp_std, demo, labels):
        X = _features(self.key, snp_std, demo)
        for ti, tkey in enumerate(self.traits):
            y = labels[:, ti]
            if self.key == "majority" or len(np.unique(y)) < 2:
                vals, counts = np.unique(y, return_counts=True)
                self.clf[tkey] = ("majority", int(vals[np.argmax(counts)]),
                                  {int(v): c / len(y) for v, c in zip(vals, counts)})
            else:
                m = LogisticRegression(max_iter=300, C=1.0)
                m.fit(X, y)
                self.clf[tkey] = ("clf", m, None)
        return self

    def predict(self, snp_std, demo):
        X = _features(self.key, snp_std, demo)
        n = snp_std.shape[0]
        preds, probs = {}, {}
        for tkey in self.traits:
            kind, obj, prior = self.clf[tkey]
            if kind == "majority":
                preds[tkey] = np.full(n, obj, dtype=int)
                probs[tkey] = None
            else:
                preds[tkey] = obj.predict(X)
                probs[tkey] = (obj.predict_proba(X), obj.classes_)
        return preds, probs


def build_appearance_models(traits):
    return [AppearancePredictor(k, traits) for k in APPEARANCE_MODELS]
