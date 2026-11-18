"""Non-additive genetic effects (data.genetic_architecture).

The first test is the gate for the whole feature: with the architecture absent or
switched off, generation must be bit-for-bit what it was before the feature existed.
"""
import copy
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.common import load_config                      # noqa: E402
from src.synth.generate import generate, _compose_faces  # noqa: E402

FACE_PARAMS_CHECKSUM = -668.022131456431   # outputs/real_match/dataset.npz, recorded 2026-11-18


def _small(arch=None, n=500):
    cfg = load_config(os.path.join(ROOT, "config", "default.yaml"))
    cfg["data"]["n_subjects"] = n
    cfg["data"]["n_snps"] = 500
    cfg["data"]["n_causal_snps"] = 40
    if arch is not None:
        cfg["data"]["genetic_architecture"] = arch
    return cfg


def test_nonadditive_off_is_bit_identical():
    ds0, gt0, _ = generate(_small())
    ds1, gt1, _ = generate(_small({"dominance_frac": 0.0, "epistasis_frac": 0.0}))
    assert np.array_equal(ds0.face_params, ds1.face_params)
    assert np.array_equal(gt0.genetic_weights, gt1.genetic_weights)
    assert np.array_equal(gt0.demo_weights, gt1.demo_weights)
    assert np.array_equal(gt0.causal_snp_idx, gt1.causal_snp_idx)
    assert gt0.dominance_weights is None and gt1.dominance_weights is None


def test_real_cohort_still_regenerates_exactly():
    """The committed cohort behind every reported number regenerates bit-for-bit."""
    from gpu.controls import make_data
    stored = np.load(os.path.join(ROOT, "outputs", "real_match", "dataset.npz"))["face_params"]
    assert float(stored.sum()) == FACE_PARAMS_CHECKSUM
    cfg = load_config(os.path.join(ROOT, "config", "real_match.yaml"))
    D = make_data(cfg, int(cfg["seed"]))
    assert np.array_equal(D["B"], stored)


# ---------------------------------------------------------------------------
# generator: planted shares are realised, and recorded honestly
# ---------------------------------------------------------------------------
def _compose(arch, n=1500, m=300, causal=40, genetic=0.20, impute_frac=0.0, seed=7):
    cfg = _small(arch, n)
    cfg["data"]["n_causal_snps"] = causal
    cfg["face_variance"] = {"genetic": genetic, "demographic": 0.35, "noise": 0.65 - genetic}
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, rng.uniform(0.1, 0.5, m), size=(n, m)).astype(np.float64)
    if impute_frac:      # what _load_real and generate_matched feed in: column-mean imputed cells
        miss = rng.random(G.shape) < impute_frac
        G[miss] = np.nan
        G = np.where(np.isnan(G), np.nanmean(G, axis=0, keepdims=True), G)
    demo = np.column_stack([rng.normal(size=n), rng.integers(0, 2, n), np.eye(3)[rng.integers(0, 3, n)]])
    names = ["age_z", "gender", "ancestry_0", "ancestry_1", "ancestry_2"]
    beta, gt = _compose_faces(cfg, G, demo, names, 50)
    return cfg, G, demo, beta, gt


def _designs(G, gt):
    Xraw = G[:, gt.causal_snp_idx]
    Xc = Xraw - Xraw.mean(0)
    H = (np.abs(Xraw - 1.0) < 0.5).astype(float)
    Hc = H - H.mean(0)
    T = Xc[:, gt.interaction_pairs[:, 0]] * Xc[:, gt.interaction_pairs[:, 1]]
    return Xc, Hc, T - T.mean(0)


ARMS = [(0.5, 0.0), (0.0, 0.5), (0.34, 0.33)]


@pytest.mark.parametrize("fd,fe", ARMS)
def test_nonadditive_shares_realised(fd, fe):
    # n=6000: an OLS fit with ~105 columns absorbs noise into every block, inflating
    # each measured share by about (columns / n) x noise; at n=1500 that bias alone
    # is ~0.03 on the sum. At 6000 it is ~0.008, below the tolerance being tested.
    cfg, G, demo, beta, gt = _compose({"dominance_frac": fd, "epistasis_frac": fe}, n=6000)
    g = cfg["face_variance"]["genetic"]
    Xc, Hc, Tc = _designs(G, gt)
    tot = beta.var(0).mean()
    # bookkeeping: the recorded realisation equals the planted parts, rebuilt from stored weights
    parts = {"additive": Xc @ gt.genetic_weights, "dominance": Hc @ gt.dominance_weights,
             "epistasis": Tc @ gt.interaction_weights}
    rec = gt.genetic_architecture["realised"]
    for name, part in parts.items():
        assert abs(part.var(0).mean() / tot - rec[name]) < 1e-6, name
    # maths: a correctly specified least-squares fit recovers each share
    Dc = demo - demo.mean(0)
    design = np.hstack([Xc, Hc, Tc, Dc])
    coef, *_ = np.linalg.lstsq(design, beta - beta.mean(0), rcond=None)
    cuts = np.cumsum([0, Xc.shape[1], Hc.shape[1], Tc.shape[1]])
    fitted = {n: design[:, a:b] @ coef[a:b] for n, a, b in zip(parts, cuts[:-1], cuts[1:])}
    share = {n: f.var(0).mean() / tot for n, f in fitted.items()}
    want = {"additive": g * (1 - fd - fe), "dominance": g * fd, "epistasis": g * fe}
    for n in want:
        assert abs(share[n] - want[n]) < 0.05, (n, share[n], want[n])
    assert abs(sum(share.values()) - g) < 0.03


def test_het_coding_handles_imputed_dosages():
    """Mean-imputed (non-integer) dosages must still carry the dominance term."""
    from src.synth.generate import _nonadditive_terms
    cfg, G, demo, beta, gt = _compose({"dominance_frac": 0.5, "epistasis_frac": 0.0}, impute_frac=0.05)
    assert not np.allclose(G, np.round(G)), "fixture must contain imputed, non-integer cells"
    Xraw = G[:, gt.causal_snp_idx]
    Hc, *_ = _nonadditive_terms(cfg, Xraw, Xraw - Xraw.mean(0), 50)
    assert np.abs(Hc).sum() > 0
    assert abs(gt.genetic_architecture["realised"]["dominance"] - 0.10) < 0.03


def test_nonadditive_weights_are_sparse_per_row():
    cfg, G, demo, beta, gt = _compose({"dominance_frac": 0.34, "epistasis_frac": 0.33})
    cfg["data"]["causal_dims"] = 5
    _, _, _, _, gt5 = _compose({"dominance_frac": 0.34, "epistasis_frac": 0.33, "dominance_dims": 5,
                                "epistasis_dims": 5})
    assert ((gt5.dominance_weights != 0).sum(1) == 5).all()
    assert ((gt5.interaction_weights != 0).sum(1) == 5).all()


def test_config_accepts_architecture_block_and_rejects_bad_fracs(tmp_path):
    import yaml
    from src.common import _validate_config
    cfg = _small({"dominance_frac": 0.3, "epistasis_frac": 0.2})
    _validate_config(cfg)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert load_config(str(p))["data"]["genetic_architecture"]["dominance_frac"] == 0.3
    for bad in ({"dominance_frac": 0.7, "epistasis_frac": 0.5}, {"dominance_frac": -0.1},
                {"epistasis_frac": 0.1, "n_interaction_pairs": 10 ** 6}):
        with pytest.raises(ValueError):
            _validate_config(_small(bad))


def test_cpu_harness_refuses_nonadditive_configs():
    from src.pipeline import prepare
    with pytest.raises(ValueError, match="non-additive"):
        prepare(_small({"dominance_frac": 0.5}))


@pytest.mark.parametrize("fd,fe", ARMS)
def test_correctly_coded_signal_above_shuffled_control(fd, fe):
    """Sibling of test_synth.test_signal_recoverable_above_control for non-additive arms.

    Uses the correctly coded design (additive, heterozygote, pair products), keeps the
    shuffled-label clause, and drops the absolute R2 floor: for the interaction arm an
    additive-dosage model genuinely cannot reach the signal, so no floor is asserted.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    ds, gt, _ = generate(_small({"dominance_frac": fd, "epistasis_frac": fe}, n=1500))
    G = np.nan_to_num(ds.genotypes)
    Xc, Hc, Tc = _designs(G, gt)
    X = np.hstack([Xc, Hc, Tc]); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    tr, te = ds.splits["train"], ds.splits["test"]
    beta = ds.face_params
    fit = lambda y: Ridge(alpha=100.0).fit(X[tr], y).predict(X[te])
    r2 = r2_score(beta[te], fit(beta[tr]), multioutput="uniform_average")
    r2c = r2_score(beta[te], fit(beta[tr][np.random.default_rng(0).permutation(len(tr))]),
                   multioutput="uniform_average")
    assert r2 > r2c + 0.1, f"signal not above shuffled control (R2={r2:.3f} vs {r2c:.3f})"


def test_nonadditive_null_share_reports_nothing():
    """Negative control: with no genetic share, the new rows must not manufacture signal."""
    from gpu.controls import make_data, ladder, r2
    cfg = load_config(os.path.join(ROOT, "config", "na_mix.yaml"))
    cfg["data"]["genetic_architecture"].update(dominance_frac=0.5, epistasis_frac=0.5)
    D = make_data(cfg, int(cfg["seed"]), n=1000, share=0.0)
    P, bt, _ = ladder(D, do_pca256=False, na_rows=True)
    base = r2(bt, P["cov+PC"])
    assert abs(r2(bt, P["+SNP-screen-NA"]) - base) <= 0.01
    assert abs(r2(bt, P["ORACLE-NA"]) - base) <= 0.01


def test_oracle_na_column_count_penalty_on_additive_arm():
    """Calibration (PREREGISTRATION.md §4, Addendum A1): on an additive target ORACLE-NA
    sits below ORACLE by the cost of fitting columns with no signal in them.

    The pre-registration expected that penalty to be about 0.02. Measured, it is larger:
    +0.0318 on this single cohort and +0.0301 averaged over the five cohorts of
    `gpu/controls_na_addcal.json`. Per the frozen calibration rule, the threshold is not
    loosened to accommodate it; the measured value is recorded here and reported as a
    finding, and the consequence -- that ORACLE-NA is not a drop-in replacement for
    ORACLE and the two are only compared within an arm, never across arms -- is carried
    into the report.

    What this test still guards is the sign and the scale: the penalty is a cost, not a
    gain (so the additive oracle is not beaten by the larger design), and it is well
    under the gains the NA rows produce on the non-additive arms (~0.055 on DOM, ~0.09
    on MIX), so those gains cannot be an artefact of the extra columns.
    """
    from gpu.controls import make_data, ladder, r2
    cfg = load_config(os.path.join(ROOT, "config", "real_match.yaml"))
    D = make_data(cfg, int(cfg["seed"]))
    P, bt, _ = ladder(D, do_pca256=False, na_rows=True)
    base = r2(bt, P["cov+PC"])
    gap = (r2(bt, P["ORACLE"]) - base) - (r2(bt, P["ORACLE-NA"]) - base)
    assert gap > 0, f"penalty {gap:+.4f} is not a cost: the NA design should not win on an additive target"
    assert abs(gap - 0.0318) <= 0.004, f"recorded penalty moved: {gap:+.4f}, recorded +0.0318"
    assert gap <= 0.05, f"penalty {gap:+.4f} is no longer small against the NA gains it must not explain"


def test_dominance_arm_two_df_screen_beats_additive_screen():
    """On DOM, the 2-df screen gains more than the additive one; both are ~0 when permuted."""
    from gpu.controls import make_data, ladder, r2
    cfg = load_config(os.path.join(ROOT, "config", "na_dom.yaml"))
    D = make_data(cfg, int(cfg["seed"]), n=1000)
    P, bt, _ = ladder(D, do_pca256=False, na_rows=True)
    base = r2(bt, P["cov+PC"])
    assert r2(bt, P["+SNP-screen-NA"]) - base > r2(bt, P["+SNP-screen"]) - base
    Pp, _, _ = ladder(D, do_pca256=False, permute_seed=1, na_rows=True)
    pb = r2(bt, Pp["cov+PC"])
    assert abs(r2(bt, Pp["+SNP-screen"]) - pb) <= 0.01
    assert abs(r2(bt, Pp["+SNP-screen-NA"]) - pb) <= 0.01
