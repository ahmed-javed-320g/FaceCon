"""Evaluation metrics, matching the report's Section 5 evaluation plan.

All geometric errors are reported in nominal millimetres: the model's head is
scaled so its height is MM_PER_UNIT * 2 = 200 mm, a realistic human head height.
This is clearly a nominal scale on synthetic geometry, not a real-world accuracy
claim; it just makes the numbers legible and comparable to the report's mm
examples.
"""
from __future__ import annotations

import numpy as np

MM_PER_UNIT = 100.0   # head height 2.0 units -> 200 mm nominal


def param_errors(pred: np.ndarray, true: np.ndarray) -> dict:
    diff = pred - true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    # per-component correlation, averaged over shape dimensions
    corrs = []
    for j in range(true.shape[1]):
        t, p = true[:, j], pred[:, j]
        if t.std() > 1e-9 and p.std() > 1e-9:
            corrs.append(np.corrcoef(t, p)[0, 1])
    return {
        "param_mse": mse,
        "param_rmse": float(np.sqrt(mse)),
        "param_mae": mae,
        "param_mean_corr": float(np.mean(corrs)) if corrs else 0.0,
    }


def decode_mm(fm, beta):
    """Decode shape vectors to vertices in mm, in float32 for speed/memory."""
    scale = getattr(fm, "mm_per_unit", MM_PER_UNIT)
    mean = fm.mean.astype(np.float32)
    basis = fm.basis.astype(np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    v = mean.shape[0]
    if beta.ndim == 1:
        verts = mean + (beta @ basis).reshape(v, 3)
    else:
        verts = mean[None] + (beta @ basis).reshape(beta.shape[0], v, 3)
    return verts * np.float32(scale)


def vertex_dist(fm, pred_beta, true_beta=None, true_verts=None):
    """Per-vertex Euclidean distance in mm, shape (N, V). Decodes each mesh once.

    Pass `true_verts` (from decode_mm) to avoid re-decoding the shared ground
    truth for every model.
    """
    pv = decode_mm(fm, pred_beta)
    tv = true_verts if true_verts is not None else decode_mm(fm, true_beta)
    return np.linalg.norm(pv - tv, axis=2)


def geometric_metrics(fm, d) -> dict:
    """All vertex-based metrics from one precomputed distance array (N, V)."""
    region = {}
    for name, mask in fm.region_masks.items():
        region[name] = float(d[:, mask].mean()) if mask.any() else float("nan")
    region["whole_face"] = float(d.mean())
    return {
        "mesh_mean_mm": float(d.mean()),
        "mesh_median_mm": float(np.median(d)),
        "landmark_mean_mm": float(d[:, fm.landmark_idx].mean()),
        "region_mm": region,
    }


def mesh_errors(fm, pred_beta, true_beta) -> dict:
    d = vertex_dist(fm, pred_beta, true_beta)
    return {"mesh_mean_mm": float(d.mean()), "mesh_median_mm": float(np.median(d))}


def retrieval_metrics(pred_beta, true_beta) -> dict:
    """Rank the true subject among the test gallery by shape distance."""
    n = pred_beta.shape[0]
    # pairwise distance from each prediction to every true subject
    d = np.linalg.norm(pred_beta[:, None, :] - true_beta[None, :, :], axis=2)
    order = np.argsort(d, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n)])  # 0-based
    return {
        "retrieval_top1": float(np.mean(ranks == 0)),
        "retrieval_top5": float(np.mean(ranks < 5)),
        "retrieval_mean_rank": float(np.mean(ranks) + 1),  # 1-based
        "retrieval_gallery": int(n),
        "retrieval_chance_top1": float(1.0 / n),
    }


def all_metrics(fm, pred_beta, true_beta, true_verts=None) -> dict:
    """Full metric set. `true_verts` (decode_mm of true_beta) is decoded once by
    the caller and reused across models to avoid redundant mesh decoding."""
    d = vertex_dist(fm, pred_beta, true_beta=true_beta, true_verts=true_verts)
    out = {}
    out.update(param_errors(pred_beta, true_beta))
    out.update(geometric_metrics(fm, d))
    out.update(retrieval_metrics(pred_beta, true_beta))
    from sklearn.metrics import r2_score
    out["param_r2"] = float(r2_score(true_beta, pred_beta, multioutput="uniform_average"))
    return out
