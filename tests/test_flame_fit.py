"""Tests for gpu/flame_fit.py: scan fitting recovers identity, and a control that could fail.

These use few scans and a reduced iteration budget to stay inside a test run. The full
pre-registered validation (PREREGISTRATION.md section 8, 12 scans per condition,
2500 iterations) is `python -m gpu.flame_fit`, whose output is
gpu/flame_fit_validation.json.
"""
import os

import numpy as np
import pytest
import torch

from gpu.flame_fit import (FaceModelLayer, axis_angle_to_matrix, fit_scan, surface_residual_mm,
                           synthetic_scan)
from src.common import load_config
from src.faces.morphable import load_face_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def model():
    return FaceModelLayer(load_face_model(load_config(os.path.join(ROOT, "config", "real_match.yaml"))))


def _vertex_err_mm(model, b_fit, b_true):
    with torch.no_grad():
        d = (model.decode(torch.tensor(b_fit, dtype=torch.float32))
             - model.decode(torch.tensor(b_true, dtype=torch.float32))).numpy()
    return float(np.linalg.norm(d, axis=1).mean())


def test_axis_angle_round_trip(model):
    """The rotation parameterisation is a rotation: orthogonal, determinant +1."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        R = axis_angle_to_matrix(torch.tensor(rng.normal(size=3) * 0.3)).numpy()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_recovers_identity_from_noise_free_scan(model):
    """A scan of a known face is fitted back to that face, well under the 1.0 mm bound."""
    rng = np.random.default_rng(1)
    errs = []
    for i in range(3):
        b = rng.normal(size=model.K)
        pts, lmk = synthetic_scan(model, b, rng)
        fit = fit_scan(model, pts, lmk, iters=900, seed=i)
        errs.append(_vertex_err_mm(model, fit["beta"], b))
    assert max(errs) <= 1.0, errs


def test_optimisation_improves_on_the_landmark_warm_start(model):
    """The Adam/ICP stage earns its place: it beats the closed-form warm start it begins from."""
    rng = np.random.default_rng(2)
    b = rng.normal(size=model.K)
    pts, lmk = synthetic_scan(model, b, rng)
    fit = fit_scan(model, pts, lmk, iters=900, seed=0)
    warm = _vertex_err_mm(model, fit["beta_warm_start"], b)
    final = _vertex_err_mm(model, fit["beta"], b)
    assert final < warm, (final, warm)


def test_random_mesh_control_residual_is_much_larger(model):
    """Negative control: a mesh of per-vertex noise cannot be explained by any face.

    This is the check that could have contradicted the recovery result. If the fitter
    reported a small residual for anything handed to it, a small residual on a real
    scan would mean nothing.
    """
    rng = np.random.default_rng(3)
    b = rng.normal(size=model.K)
    pts, lmk = synthetic_scan(model, b, rng)
    fit = fit_scan(model, pts, lmk, iters=900, seed=0)
    face_resid = surface_residual_mm(model, fit, pts)

    with torch.no_grad():
        dev = float(np.sqrt(np.mean([((model.decode(torch.tensor(rng.normal(size=model.K), dtype=torch.float32))
                                       - model.mean).numpy() ** 2).sum(1).mean() for _ in range(10)])))
    pts0, lmk0 = synthetic_scan(model, np.zeros(model.K), rng)
    noisy = pts0 + rng.normal(0, dev / np.sqrt(3), pts0.shape)
    fit_c = fit_scan(model, noisy, lmk0 + rng.normal(0, dev / np.sqrt(3), lmk0.shape), iters=900, seed=1)
    assert surface_residual_mm(model, fit_c, noisy) >= 5 * face_resid


def test_fit_is_deterministic(model):
    """Same scan and seed, same answer: the fit is reproducible."""
    rng = np.random.default_rng(4)
    b = rng.normal(size=model.K)
    pts, lmk = synthetic_scan(model, b, rng)
    a = fit_scan(model, pts, lmk, iters=200, seed=7)
    c = fit_scan(model, pts, lmk, iters=200, seed=7)
    assert np.array_equal(a["beta"], c["beta"])


def test_model_layer_is_in_millimetres(model):
    """The layer applies mm_per_unit, so errors are reported in millimetres."""
    fm = load_face_model(load_config(os.path.join(ROOT, "config", "real_match.yaml")))
    with torch.no_grad():
        mean_mm = model.decode(torch.zeros(model.K)).numpy()
    assert np.allclose(mean_mm, fm.mean * fm.mm_per_unit, atol=1e-4)
