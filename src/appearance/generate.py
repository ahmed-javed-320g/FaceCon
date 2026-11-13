"""Synthetic generation of the appearance panel and appearance labels.

Mirrors the shape generator's philosophy: a dedicated pigmentation/appearance
SNP panel (with strong ancestry structure, since pigmentation is ancestry-linked)
drives categorical appearance traits, alongside ancestry and — for balding — age.
Each trait's genetic-vs-ancestry-vs-noise mix is set by its calibration in
traits.py so predictability matches the real literature.
"""
from __future__ import annotations

import numpy as np

from ..common import rng_for
from .traits import TRAITS, trait_keys


def _zscore(x):
    s = x.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return (x - x.mean(axis=0, keepdims=True)) / s


def generate_appearance(cfg, ancestry, age_z):
    """Return (panel_snps (N,P), labels (N,T), causal_idx dict, trait_keys list)."""
    ap = cfg.get("appearance", {})
    rng = rng_for(cfg["seed"], "appearance")
    n = ancestry.shape[0]
    n_groups = int(cfg["data"]["ancestry"]["n_groups"])
    P = int(ap.get("n_panel_snps", 300))
    structured_frac = float(ap.get("structured_fraction", 0.45))

    # --- draw the appearance SNP panel (ancestry-structured) ---
    base_maf = rng.uniform(0.05, 0.5, size=P)
    structured = rng.random(P) < structured_frac
    freq = np.tile(base_maf, (n_groups, 1))
    for g in range(n_groups):
        shift = rng.uniform(-0.3, 0.3, size=P) * structured
        freq[g] = np.clip(base_maf + shift, 0.01, 0.99)
    panel = rng.binomial(2, freq[ancestry]).astype(np.float64)

    panel_std = _zscore(panel)
    anc_oh = np.zeros((n, n_groups))
    anc_oh[np.arange(n), ancestry] = 1.0
    anc_std = _zscore(anc_oh)

    keys = trait_keys()
    labels = np.zeros((n, len(keys)), dtype=np.int64)
    causal = {}
    for ti, key in enumerate(keys):
        tr = TRAITS[key]
        rngt = rng_for(cfg["seed"], f"appearance_{key}")
        cidx = rngt.choice(P, size=min(tr.n_causal, P), replace=False)
        cidx.sort()
        causal[key] = cidx

        # latent score = genetic + ancestry + (age for balding) + noise,
        # each component standardised then weighted by the trait calibration
        g_part = _zscore(panel_std[:, cidx] @ rngt.normal(size=len(cidx)))
        a_part = _zscore(anc_std @ rngt.normal(size=n_groups))
        latent = tr.snp_strength * g_part + tr.anc_strength * a_part
        if tr.age_strength > 0:
            latent = latent + tr.age_strength * _zscore(age_z)
        latent = latent + (1.0 - min(0.95, tr.snp_strength + tr.anc_strength
                                     + tr.age_strength)) * rngt.normal(size=n)

        # cut the latent into classes by quantiles (globally balanced,
        # but ancestry-correlated through a_part -> realistic per-group skew)
        c = len(tr.classes)
        qs = np.quantile(latent, np.linspace(0, 1, c + 1)[1:-1])
        labels[:, ti] = np.digitize(latent, qs)

    return panel, labels, causal, keys
