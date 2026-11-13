"""Appearance evaluation: per-trait accuracy against baselines, plus ablation.

Produces, for each trait, the accuracy / balanced accuracy / macro-F1 of each
model, so the honest story is visible: ancestry alone is a strong baseline for
pigmentation, and the SNP panel must beat it to show genuine genetic signal.
Also runs an informed-vs-random appearance-panel ablation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .models import build_appearance_models, MODEL_NAMES
from .traits import TRAITS


def _standardise(train_snp, *others):
    sc = StandardScaler().fit(train_snp)
    return (sc.transform(train_snp), *[sc.transform(o) for o in others])


def evaluate_appearance(prep) -> dict:
    ds = prep.ds
    if ds.appearance_snps is None:
        return {}
    traits = ds.appearance_traits
    tr, va, te = ds.splits["train"], ds.splits["val"], ds.splits["test"]

    snp = ds.appearance_snps
    demo = ds.demographics
    snp_tr, snp_te = _standardise(snp[tr], snp[te])
    demo_tr, demo_te = demo[tr], demo[te]
    y_tr, y_te = ds.appearance_labels[tr], ds.appearance_labels[te]

    # per-model, per-trait metrics
    rows = []
    for model in build_appearance_models(traits):
        model.fit(snp_tr, demo_tr, y_tr)
        preds, _ = model.predict(snp_te, demo_te)
        per_trait = {}
        for ti, tkey in enumerate(traits):
            yp = preds[tkey]; yt = y_te[:, ti]
            per_trait[tkey] = {
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro")),
            }
        rows.append({"key": model.key, "name": model.name, "per_trait": per_trait})

    # informed vs random appearance-panel ablation (per trait, SNP-only logistic)
    ablation = _panel_ablation(prep, snp_tr, snp_te, y_tr, y_te, traits)

    return {
        "traits": [{"key": t, "name": TRAITS[t].name, "classes": TRAITS[t].classes,
                    "render": TRAITS[t].render} for t in traits],
        "models": rows,
        "ablation_informed_vs_random": ablation,
        "note": "SYNTHETIC DATA — per-trait predictability calibrated to the literature.",
    }


def _panel_ablation(prep, snp_tr, snp_te, y_tr, y_te, traits):
    gt = prep.gt
    if gt is None or not gt.appearance_causal:
        return {}
    rng = np.random.default_rng(prep.cfg["seed"])
    P = snp_tr.shape[1]
    out = {}
    for ti, tkey in enumerate(traits):
        causal = gt.appearance_causal.get(tkey, np.array([], dtype=int))
        if len(causal) == 0:
            continue
        rand = rng.choice(np.setdiff1d(np.arange(P), causal), size=len(causal),
                          replace=False)
        acc = {}
        for label, cols in (("informed", causal), ("random", rand)):
            if len(np.unique(y_tr[:, ti])) < 2:
                acc[label] = float("nan"); continue
            m = LogisticRegression(max_iter=300).fit(snp_tr[:, cols], y_tr[:, ti])
            acc[label] = float((m.predict(snp_te[:, cols]) == y_te[:, ti]).mean())
        out[tkey] = acc
    return out
