import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.synth.generate import generate
from src.io.store import save_bundle, load_bundle


def _small_cfg():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    cfg["data"]["n_subjects"] = 500
    cfg["data"]["n_snps"] = 500
    cfg["data"]["n_causal_snps"] = 40
    return cfg


def test_dosages_and_shapes():
    ds, gt, fm = generate(_small_cfg())
    ds.check()
    finite = ds.genotypes[np.isfinite(ds.genotypes)]
    assert set(np.unique(finite)).issubset({0.0, 1.0, 2.0})
    assert ds.face_params.shape == (ds.n_subjects, fm.n_dims)


def test_splits_disjoint_and_cover():
    ds, _, _ = generate(_small_cfg())
    idx = np.concatenate([ds.splits[k] for k in ("train", "val", "test")])
    assert len(set(idx.tolist())) == ds.n_subjects
    assert len(idx) == ds.n_subjects


def test_variance_shares_match_config():
    cfg = _small_cfg()
    ds, gt, _ = generate(cfg)
    beta = ds.face_params
    X = np.nan_to_num(ds.genotypes)
    Xc = X[:, gt.causal_snp_idx] - X[:, gt.causal_snp_idx].mean(0)
    Dc = ds.demographics - ds.demographics.mean(0)
    Z = np.concatenate([Xc, Dc], 1)
    W, *_ = np.linalg.lstsq(Z, beta, rcond=None)
    gp = Xc @ W[:Xc.shape[1]]
    dp = Dc @ W[Xc.shape[1]:]
    resid = beta - Z @ W
    tot = beta.var(0).mean()
    # realized shares within a reasonable tolerance of the configured targets
    assert abs(gp.var(0).mean() / tot - cfg["face_variance"]["genetic"]) < 0.12
    assert abs(resid.var(0).mean() / tot - cfg["face_variance"]["noise"]) < 0.12


def test_signal_recoverable_above_control():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    ds, gt, _ = generate(_small_cfg())
    X = np.nan_to_num(ds.genotypes)
    tr, te = ds.splits["train"], ds.splits["test"]
    sc = StandardScaler().fit(X[tr])
    beta = ds.face_params
    m = Ridge(alpha=1e3).fit(sc.transform(X[tr]), beta[tr])
    r2 = r2_score(beta[te], m.predict(sc.transform(X[te])), multioutput="uniform_average")
    # shuffled-label control
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    mc = Ridge(alpha=1e3).fit(sc.transform(X[tr]), beta[tr][perm])
    r2c = r2_score(beta[te], mc.predict(sc.transform(X[te])), multioutput="uniform_average")
    assert r2 > 0.05, f"planted signal not recovered (R2={r2:.3f})"
    assert r2 > r2c + 0.1, f"signal not above control (R2={r2:.3f} vs {r2c:.3f})"


def test_roundtrip_save_load(tmp_path):
    ds, gt, fm = generate(_small_cfg())
    out = str(tmp_path / "bundle")
    save_bundle(out, ds, gt, fm)
    ds2, gt2, fm2 = load_bundle(out)
    assert np.allclose(np.nan_to_num(ds.genotypes), np.nan_to_num(ds2.genotypes))
    assert np.allclose(ds.face_params, ds2.face_params)
    assert np.allclose(fm.basis, fm2.basis)
    assert list(fm.region_masks.keys()) == list(fm2.region_masks.keys())
