"""Fitting the face model to a 3D scan: the step that turns a real scan into a target.

When 3DFN scans arrive, each one is fitted with this module to recover its identity
shape vector beta, which then replaces the simulated beta as the prediction target.

Objective (report Eq. 5.1), in millimetres:

    min_{beta, R, t}  lam_lmk  * mean_k || R v_k(beta) + t - u_k ||^2          landmarks
                    + lam_surf * [ mean_p min_q || p - (R q + t) ||^2          scan -> model
                                 + mean_q min_p || (R q + t) - p ||^2 ]        model -> scan
                    + lam_reg  * || beta ||^2                                  shape prior

The surface term is symmetric: scan->model alone lets the model shrink inside the scan
at no cost. Model->scan uses the face-region vertices only, because a scan covers the
face, not the back of the head. Pose is rigid (rotation R as an axis-angle vector, and
translation t). Expression psi exists only in the real FLAME model; the stand-in has no
expression basis, so none is fitted here.

Optimisation: (1) rigid initialisation from the landmarks (Kabsch), (2) alternating
closed-form steps -- ridge least squares for beta given the pose, Kabsch for the pose
given beta -- which is the warm start (the 68-landmark design has rank 50 and condition
number ~101, so the beta step is well posed; one pass alone is not enough because the
first Kabsch aligns the *mean* landmarks, not the shaped ones), (3) Adam on beta and
pose with a cosine learning-rate schedule, adding the surface term, and stopping when
the loss averaged over `patience` iterations has improved by less than `tol`
(relative). The window is averaged because the surface term subsamples both point sets
each iteration, so the raw loss is noisy and a raw comparison stops far too early.
The previous version ran Adam at a
fixed lr=0.3 from beta=0: its loss stalled at 0.00266 from iteration 100 to 800 and
beta was recovered with error 3.14 against |beta| = 6.51. Its printed explanation, that
68 landmarks under-constrain 50 modes, was false: a direct least-squares solve recovers
beta to 2e-5.

The model comes from src/faces/morphable.load_face_model, the project's single loader
(identity scale applied there). Swapping in FLAME 2020 means changing that loader and
FaceModelLayer below; FLAME needs a licence only the team can accept
(flame.is.tue.mpg.de).

    python -m gpu.flame_fit            # validation on synthetic scans (PREREGISTRATION.md §8)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FaceModelLayer(nn.Module):
    """mean + beta @ basis, in millimetres. The stand-in for the real FLAME layer."""

    def __init__(self, fm):
        super().__init__()
        mm = float(fm.mm_per_unit)
        self.register_buffer("mean", torch.tensor(fm.mean * mm, dtype=torch.float32))
        self.register_buffer("basis", torch.tensor(fm.basis * mm, dtype=torch.float32))
        self.landmark_idx = np.asarray(fm.landmark_idx)
        self.face_idx = np.asarray(fm.face_vert_idx) if fm.face_vert_idx is not None \
            else np.arange(fm.mean.shape[0])
        self.K, self.V = self.basis.shape[0], self.mean.shape[0]

    def decode(self, beta):                                   # (K,) -> (V,3) mm
        return self.mean + (beta @ self.basis).reshape(self.V, 3)


def axis_angle_to_matrix(r):
    theta = torch.sqrt((r ** 2).sum() + 1e-12)
    k = r / theta
    K = torch.zeros(3, 3, dtype=r.dtype, device=r.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=r.dtype, device=r.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def _kabsch(src, dst):
    """Rotation R and translation t minimising ||R src + t - dst|| (rows are points)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cd - R @ cs


def _matrix_to_axis_angle(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return w * theta


def _alternate(mean, basis_l, lmk_mean, scan_lmk, ridge, rounds=25):
    """Alternate closed-form beta (ridge least squares) and pose (Kabsch) on the landmarks."""
    K = basis_l.shape[0]
    A = basis_l.reshape(K, -1).T                                # (3L, K)
    G = np.linalg.inv(A.T @ A + ridge * np.eye(K)) @ A.T
    b = np.zeros(K)
    R, t = _kabsch(lmk_mean, scan_lmk)
    for _ in range(rounds):
        u = (scan_lmk - t) @ R                                  # scan landmarks in model frame
        b_new = G @ (u - lmk_mean).ravel()
        model_l = lmk_mean + (b_new @ A.T).reshape(-1, 3)
        R, t = _kabsch(model_l, scan_lmk)
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new
    return b, R, t


def fit_scan(model: FaceModelLayer, scan_pts, scan_lmk, lam_lmk=1.0, lam_surf=1.0, lmk_sigma_mm=1.5,
             lam_reg=None, iters=800, lr=0.05, tol=1e-5, patience=100, corr_every=25,
             surf_points=2000, seed=0, device="cpu"):
    """Fit beta and rigid pose to one scan. Returns a dict (beta, pose, history, convergence).

    scan_pts : (P,3) scan surface points, mm
    scan_lmk : (L,3) scan landmarks, mm, in the order of model.landmark_idx

    lam_reg defaults to the MAP value for the generator's beta prior N(0, I) against
    landmark noise of `lmk_sigma_mm`: with the landmark term written as a mean over L
    landmarks, that is sigma^2 / L. It is set by this rule, not tuned: the prior is the
    generator's own, and lmk_sigma_mm is the landmark noise of the validation's noisy
    condition. A larger lam_reg shrinks beta toward the mean face, which shows up
    directly as recovery error in the noise-free case.

    The surface term uses ICP-style correspondences over the *full* point sets,
    recomputed every `corr_every` iterations. Resampling both sets each iteration
    instead (an earlier version here) biases the nearest-neighbour distance upward and
    makes its gradient noise: with a weak prior it pushed beta further from the truth
    than the landmark warm start had already reached.
    """
    from scipy.spatial import cKDTree
    if lam_reg is None:
        lam_reg = lmk_sigma_mm ** 2 / len(np.asarray(scan_lmk))
    scan_pts = np.asarray(scan_pts, np.float64); scan_lmk = np.asarray(scan_lmk, np.float64)
    mean = model.mean.cpu().numpy().astype(np.float64)
    basis = model.basis.cpu().numpy().astype(np.float64).reshape(model.K, model.V, 3)
    li = model.landmark_idx
    # (1)+(2) alternating closed-form warm start on the landmarks
    b0, R0, t0 = _alternate(mean, basis[:, li, :], mean[li], scan_lmk,
                            ridge=lam_reg / lam_lmk * len(li))

    dev = torch.device(device)
    beta = torch.tensor(b0, dtype=torch.float32, device=dev, requires_grad=True)
    rvec = torch.tensor(_matrix_to_axis_angle(R0), dtype=torch.float32, device=dev, requires_grad=True)
    tvec = torch.tensor(t0, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([beta, rvec, tvec], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.01)
    S = torch.tensor(scan_pts, dtype=torch.float32, device=dev)
    L = torch.tensor(scan_lmk, dtype=torch.float32, device=dev)
    lmk_i = torch.as_tensor(li, device=dev)
    face_i = torch.as_tensor(model.face_idx, device=dev)
    hist, converged = [], False
    s2m = m2s = None                                            # ICP correspondences
    for it in range(iters):
        opt.zero_grad()
        R = axis_angle_to_matrix(rvec)
        V = model.decode(beta) @ R.T + tvec                     # posed model, mm
        loss_l = ((V[lmk_i] - L) ** 2).sum(1).mean()
        loss_s = torch.zeros((), device=dev)
        if lam_surf > 0:
            if it % corr_every == 0:
                F = V[face_i].detach().cpu().numpy()
                s2m = torch.as_tensor(cKDTree(F).query(scan_pts)[1], device=dev)      # scan -> model
                m2s = torch.as_tensor(cKDTree(scan_pts).query(F)[1], device=dev)      # model -> scan
            Vf = V[face_i]
            loss_s = ((S - Vf[s2m]) ** 2).sum(1).mean() + ((Vf - S[m2s]) ** 2).sum(1).mean()
        loss = lam_lmk * loss_l + lam_surf * loss_s + lam_reg * (beta ** 2).sum()
        loss.backward(); opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if it >= 2 * patience:                                   # compare two averaged windows
            prev = float(np.mean(hist[-2 * patience:-patience])); cur = float(np.mean(hist[-patience:]))
            if (prev - cur) <= tol * max(abs(prev), 1e-12):
                converged = True
                break
    with torch.no_grad():
        R = axis_angle_to_matrix(rvec)
    return {"beta": beta.detach().cpu().numpy().astype(np.float64),
            "R": R.cpu().numpy().astype(np.float64), "t": tvec.detach().cpu().numpy().astype(np.float64),
            "beta_warm_start": b0, "history": hist, "converged": converged, "iterations": len(hist)}


def surface_residual_mm(model, fit, scan_pts):
    """Mean symmetric nearest-neighbour distance (mm) between scan and fitted face region."""
    with torch.no_grad():
        V = (model.decode(torch.tensor(fit["beta"], dtype=torch.float32)).numpy() @ fit["R"].T + fit["t"])
    F = V[model.face_idx]
    from scipy.spatial import cKDTree
    a = cKDTree(F).query(scan_pts)[0].mean()
    b = cKDTree(scan_pts).query(F)[0].mean()
    return float((a + b) / 2)


def synthetic_scan(model, beta, rng, rot_deg=10.0, trans_mm=5.0, pt_noise=0.0, lmk_noise=0.0):
    """A face scan of known identity: face-region surface + landmarks, randomly posed."""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    R = axis_angle_to_matrix(torch.tensor(axis * np.deg2rad(rng.uniform(0, rot_deg)))).numpy()
    t = rng.normal(0, trans_mm, 3)
    with torch.no_grad():
        V = model.decode(torch.tensor(beta, dtype=torch.float32)).numpy().astype(np.float64)
    P = V @ R.T + t
    pts = P[model.face_idx] + rng.normal(0, pt_noise, (len(model.face_idx), 3)) if pt_noise else P[model.face_idx]
    lmk = P[model.landmark_idx] + (rng.normal(0, lmk_noise, (len(model.landmark_idx), 3)) if lmk_noise else 0)
    return pts, lmk


def validate(n_scans=12, seed=0, iters=2500, verbose=True):
    """PREREGISTRATION.md §8: recovery <= 1.0 mm noise-free; random mesh residual >= 5x."""
    from src.common import load_config
    from src.faces.morphable import load_face_model
    fm = load_face_model(load_config(os.path.join(ROOT, "config", "real_match.yaml")))
    model = FaceModelLayer(fm)
    rng = np.random.default_rng(seed)

    def vert_err(b_fit, b_true):
        with torch.no_grad():
            d = model.decode(torch.tensor(b_fit, dtype=torch.float32)) - model.decode(torch.tensor(b_true, dtype=torch.float32))
        d = np.linalg.norm(d.numpy(), axis=1)
        return float(d.mean()), float(d[model.face_idx].mean())

    out = {"n_scans": n_scans, "units": "mm", "mm_per_unit": float(fm.mm_per_unit), "conditions": {}}
    for cond, (pn, ln) in {"noise_free": (0.0, 0.0), "noisy (0.5 mm points, 1.5 mm landmarks)": (0.5, 1.5)}.items():
        rows = []
        for i in range(n_scans):
            b = rng.normal(size=model.K)                        # unit-variance, as the generator's beta
            pts, lmk = synthetic_scan(model, b, rng, pt_noise=pn, lmk_noise=ln)
            fit = fit_scan(model, pts, lmk, iters=iters, seed=i)
            rows.append({"vert_mm": vert_err(fit["beta"], b)[0], "face_mm": vert_err(fit["beta"], b)[1],
                         "warm_start_vert_mm": vert_err(fit["beta_warm_start"], b)[0],
                         "beta_err": float(np.linalg.norm(fit["beta"] - b)), "beta_norm": float(np.linalg.norm(b)),
                         "residual_mm": surface_residual_mm(model, fit, pts), "converged": fit["converged"],
                         "iterations": fit["iterations"]})
        out["conditions"][cond] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out["conditions"][cond]["vert_mm_max"] = float(max(r["vert_mm"] for r in rows))
    # negative control: a random mesh the same size as a face, posed like a scan
    with torch.no_grad():
        dev_rms = float(np.sqrt(np.mean([((model.decode(torch.tensor(rng.normal(size=model.K), dtype=torch.float32))
                                           - model.mean).numpy() ** 2).sum(1).mean() for _ in range(20)])))
    res_c = []
    for i in range(n_scans):
        pts, lmk = synthetic_scan(model, np.zeros(model.K), rng)
        noise = rng.normal(0, dev_rms / np.sqrt(3), pts.shape)  # iid per point: no face can explain it
        lmk_noise = rng.normal(0, dev_rms / np.sqrt(3), lmk.shape)
        fit = fit_scan(model, pts + noise, lmk + lmk_noise, iters=iters, seed=100 + i)
        res_c.append(surface_residual_mm(model, fit, pts + noise))
    out["random_mesh_control"] = {"displacement_rms_mm": dev_rms, "residual_mm": float(np.mean(res_c))}
    nf = out["conditions"]["noise_free"]
    out["acceptance"] = {
        "recovery: mean vertex error <= 1.0 mm (noise-free)": bool(nf["vert_mm"] <= 1.0),
        "control: random-mesh residual >= 5x face residual (noise-free)":
            bool(out["random_mesh_control"]["residual_mm"] >= 5 * nf["residual_mm"]),
    }
    if verbose:
        print(f"FLAME-fit validation, {n_scans} synthetic scans per condition, millimetres")
        for cond, r in out["conditions"].items():
            print(f"  {cond:42s} vertex error {r['vert_mm']:.3f} mm (max {r['vert_mm_max']:.3f}), face region "
                  f"{r['face_mm']:.3f}, warm start alone {r['warm_start_vert_mm']:.3f}; beta err {r['beta_err']:.3f} "
                  f"of |beta| {r['beta_norm']:.2f}; residual {r['residual_mm']:.3f} mm; converged "
                  f"{r['converged']*100:.0f}% in {r['iterations']:.0f} it")
        c = out["random_mesh_control"]
        print(f"  random-mesh control (rms {c['displacement_rms_mm']:.2f} mm)      residual {c['residual_mm']:.3f} mm")
        for k, v in out["acceptance"].items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return out


if __name__ == "__main__":
    res = validate()
    json.dump(res, open(os.path.join(ROOT, "gpu", "flame_fit_validation.json"), "w"), indent=1)
    print("saved gpu/flame_fit_validation.json")
    sys.exit(0 if all(res["acceptance"].values()) else 1)
