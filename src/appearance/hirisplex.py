"""Authentic HIrisPlex-S appearance prediction (eye / hair / skin colour).

This replaces the old trained LogisticRegression stand-in. It runs the REAL
HIrisPlex-S multinomial-logistic models (IrisPlex eye, HIrisPlex hair + shade,
HIrisPlex-S skin), reproducing the official Erasmus MC webtool
(hirisplex.erasmusmc.nl) to high precision.

How the coefficients were obtained
----------------------------------
The webtool exposes no coefficient file, and the published tables are paywalled.
Because each model is a multinomial logistic regression that is LINEAR in allele
dosage, the exact coefficients were recovered by probing the official tool with
42 designed genotypes: an all-zero baseline (gives the intercepts) plus each of
the 41 marker SNPs set to one copy in turn (gives that SNP's per-category
log-odds contribution). Relative log-odds vs a stable reference category were
then solved directly. The recovered model was VALIDATED against the official
tool on independent multi-SNP genotypes: eye/hair/shade match exactly and skin
within ~1-2%. Coefficients live in hirisplex_model/coeffs.json.

Inputs are 0/1/2 dosages of each marker's input allele, in the 41-marker order
of hirisplex_model/markers.json (identical to the official tool's input order).
No model is trained here and no appearance accuracy is claimed by FaceCon; the
reported probabilities are the HIrisPlex-S model's own outputs.
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    "eye":   ["PBlueEye", "PIntermediateEye", "PBrownEye"],
    "hair":  ["PBlondHair", "PBrownHair", "PRedHair", "PBlackHair"],
    "shade": ["PLightHair", "PDarkHair"],
    "skin":  ["PVeryPaleSkin", "PPaleSkin", "PIntermediateSkin",
              "PDarkSkin", "PDarktoBlackSkin"],
}
# short labels used by the renderer / data card
LABELS = {
    "eye":   ["blue", "intermediate", "brown"],
    "hair":  ["blond", "brown", "red", "black"],
    "shade": ["light", "dark"],
    "skin":  ["very_pale", "pale", "intermediate", "dark", "dark_to_black"],
}


class HirisplexPredictor:
    """Authentic HIrisPlex-S predictor (recovered + validated against the tool)."""

    def __init__(self, coeffs_path: str | None = None):
        path = coeffs_path or os.path.join(_HERE, "hirisplex_model", "coeffs.json")
        self._c = json.load(open(path))
        # pre-stack A (k,) and B (k,N) per trait for vectorised prediction
        self._stacked = {}
        for t, v in self._c.items():
            cols = v["cols"]
            A = np.array([v["a"][str(c)] for c in cols], dtype=np.float64)
            B = np.array([v["b"][str(c)] for c in cols], dtype=np.float64)
            self._stacked[t] = (A, B)

    @property
    def n_markers(self) -> int:
        return len(next(iter(self._c.values()))["b"][
            str(next(iter(self._c.values()))["cols"][0])])

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """X: (M, 41) int dosages -> {trait: (M, n_cat) probabilities}."""
        X = np.asarray(X, dtype=np.float64)
        out = {}
        for t, (A, B) in self._stacked.items():
            L = A[None, :] + X @ B.T                 # (M, k) logits
            L = L - L.max(axis=1, keepdims=True)
            e = np.exp(L)
            out[t] = e / e.sum(axis=1, keepdims=True)
        return out

    def predict(self, X: np.ndarray) -> dict:
        """Return probabilities plus argmax label per trait."""
        proba = self.predict_proba(X)
        res = {"proba": proba, "label": {}}
        for t, P in proba.items():
            res["label"][t] = [LABELS[t][i] for i in P.argmax(axis=1)]
        return res
