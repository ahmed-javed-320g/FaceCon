import os
import sys

import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.pipeline import prepare
from src.models.base import build_models
from src.qc.pipeline import run_qc


def _cfg():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    cfg["data"]["n_subjects"] = 500
    cfg["data"]["n_snps"] = 500
    cfg["data"]["n_causal_snps"] = 40
    cfg["data"]["out_dir"] = "outputs/_test_synthetic"
    cfg["models"]["mlp"]["max_iter"] = 120
    cfg["models"]["mlp"]["snp_pca"] = 80
    return cfg


def test_qc_fits_on_train_only():
    cfg = _cfg()
    from src.synth.generate import generate
    ds, _, _ = generate(cfg)
    _, t = run_qc(cfg, ds.genotypes, ds.splits["train"])
    # imputation means must match train-only means on kept SNPs (no leakage)
    gk = ds.genotypes[ds.splits["train"]][:, t.kept_snp_idx]
    assert np.allclose(t.snp_mean, np.nanmean(gk, axis=0))


def test_feature_widths():
    P = prepare(_cfg())
    assert P.features.snp.shape[0] == P.ds.n_subjects
    assert P.features.demo.shape[1] == len(P.ds.demographic_names)
    assert P.features.pcs.shape[1] == _cfg()["qc"]["n_pcs"]


def test_ladder_ranks_sensibly():
    cfg = _cfg()
    P = prepare(cfg)
    tr, btr = P.view("train"); va, bva = P.view("val"); te, bte = P.view("test")
    res = {}
    for m in build_models(cfg):
        m.fit(tr, btr, va, bva)
        res[m.name] = r2_score(bte, m.predict(te), multioutput="uniform_average")
    assert abs(res["Mean face"]) < 0.04, res            # floor ~ 0
    assert res["Demographics only"] > 0.08, res         # demographics informative
    assert res["Ridge (SNPs)"] > 0.05, res              # SNPs carry signal
    for name in ("Demographics only", "Ridge (SNPs)",
                 "MLP (SNPs)", "MLP (SNPs + demographics)"):
        assert res[name] > res["Mean face"], (name, res)
