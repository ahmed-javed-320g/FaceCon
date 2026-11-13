"""Regression tests for the methods audit (see EVALUATION.md)."""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DS = os.path.join(ROOT, "outputs/real_match/dataset.npz")


def _ds():
    return np.load(DS, allow_pickle=True)


def test_gwas_markers_one_column_per_snp_and_chrx_hemizygous():
    d = _ds(); ids = list(d["gwas_rsids"])
    assert len(ids) == len(set(ids)) == d["gwas_markers"].shape[1]
    males = d["demographics"][:, 0] == 1
    x = d["gwas_markers"][:, ids.index("rs11093404")]          # chrX lead SNP
    assert not np.any(x[males] == 1), "males cannot be heterozygous on chrX"
    assert np.any(x[~males] == 1)


def test_gwas_recovery_positive_and_negative_control():
    c = json.load(open(os.path.join(ROOT, "outputs/real_match/manifest.json")))["gwas_effects"]["recovery_check"]
    assert c["accept"] and c["null_bonferroni_hits"] == 0
    assert c["injected_detected"] >= 0.8 * c["injected_total"]


def test_gpu_mesh_metric_uses_generator_scale():
    from gpu.data import FaceConData
    from src.common import load_config
    from src.faces.morphable import load_face_model
    D = FaceConData(ROOT, snp_pca=8)
    fm = load_face_model(load_config(os.path.join(ROOT, "config/real_match.yaml")))
    assert np.allclose(D.basis, fm.basis.astype(np.float32), rtol=1e-4)


def test_appearance_predictions_not_degenerate():
    from src.appearance.hirisplex import HirisplexPredictor
    lab = HirisplexPredictor().predict(_ds()["hps_markers"])["label"]
    for trait in ("eye", "hair", "skin"):
        _, c = np.unique(lab[trait], return_counts=True)
        assert c.max() / c.sum() < 0.9, f"{trait}: one class dominates (calibration lost)"


def test_snp_signal_positive_control_and_permuted_negative_control():
    from gpu.controls import make_data, ladder, r2
    from src.common import load_config
    cfg = load_config(os.path.join(ROOT, "config/real_match.yaml"))
    D = make_data(cfg, 7, n=1000)
    P, bt, _ = ladder(D, do_pca256=False)
    assert r2(bt, P["+SNP-screen"]) - r2(bt, P["cov+PC"]) > 0.03      # signal found when planted
    Pp, _, _ = ladder(D, do_pca256=False, permute_seed=1)
    assert r2(bt, Pp["+SNP-screen"]) - r2(bt, Pp["cov+PC"]) < 0.01     # and not when broken
