import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.synth.generate import generate
from src.pipeline import prepare
from src.appearance.evaluate import evaluate_appearance
from src.appearance.traits import trait_keys


def _cfg():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    cfg["data"]["n_subjects"] = 800
    cfg["data"]["n_snps"] = 400
    cfg["data"]["n_causal_snps"] = 30
    cfg["data"]["out_dir"] = "outputs/_test_appearance"
    cfg["appearance"]["n_panel_snps"] = 150
    return cfg


def test_appearance_generated_and_valid():
    ds, gt, _ = generate(_cfg())
    assert ds.appearance_snps is not None
    assert ds.appearance_snps.shape == (ds.n_subjects, 150)
    assert set(np.unique(ds.appearance_snps)).issubset({0.0, 1.0, 2.0})
    assert ds.appearance_labels.shape[1] == len(trait_keys())
    # labels are valid class indices
    for ti, tkey in enumerate(ds.appearance_traits):
        assert ds.appearance_labels[:, ti].min() >= 0


def test_appearance_ladder_and_signal():
    cfg = _cfg()
    prep = prepare(cfg)
    res = evaluate_appearance(prep)
    assert res["models"][0]["key"] == "majority"
    # for pigmentation traits, SNP model must clearly beat majority and the
    # informed panel must beat the random panel (genuine genetic signal)
    by = {m["key"]: m for m in res["models"]}
    for t in ("eye_colour", "hair_colour", "skin_tone"):
        snp = by["snp"]["per_trait"][t]["accuracy"]
        maj = by["majority"]["per_trait"][t]["accuracy"]
        assert snp > maj + 0.1, (t, snp, maj)
        ab = res["ablation_informed_vs_random"][t]
        assert ab["informed"] > ab["random"] + 0.05, (t, ab)
    import shutil
    shutil.rmtree("outputs/_test_appearance", ignore_errors=True)
