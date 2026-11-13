"""Build a realistic FaceCon face model from ICT-FaceKit (MIT licence).

Downloads the generic neutral mesh and the leading identity morph targets,
forms the identity basis as (identity_k - neutral), and packages everything into
assets/ict/ict_face_model.npz in the same layout our FaceModel already uses:
mean (V,3), basis (K,3V), faces (F,3), region masks, landmark indices.

Source: https://github.com/ICT-VGL/ICT-FaceKit  (MIT License)
Only the mean and identity shape directions are used; the data is a synthetic,
licence-clean stand-in for FLAME while 3DFN access is pending.

Usage: python scripts/build_face_model.py [n_modes]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

RAW = "https://raw.githubusercontent.com/ICT-VGL/ICT-FaceKit/master/FaceXModel"
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "ict")


def _curl(url: str, dest: str) -> None:
    subprocess.run(["curl", "-sSL", "-o", dest, url], check=True)


def parse_obj(path: str):
    """Raw OBJ parse: geometry vertices ('v ') and triangulated faces ('f ')."""
    verts, faces = [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for i in range(1, len(idx) - 1):          # fan-triangulate
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def assign_regions(verts, landmark_idx, face_vert_idx):
    """Each face vertex -> region of its nearest 68-landmark; others excluded.

    iBUG-68 groups: jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, mouth 48-67.
    Mapped onto our six report regions.
    """
    lm = verts[landmark_idx]                               # (68,3)
    group = (["jaw"] * 17 + ["forehead"] * 10 + ["nose"] * 9 +
             ["eyes"] * 12 + ["mouth"] * 20)               # 17+10+9+12+20 = 68
    names = ["nose", "jaw", "mouth", "eyes", "cheeks", "forehead"]
    masks = {n: np.zeros(verts.shape[0], dtype=bool) for n in names}
    face_set = np.zeros(verts.shape[0], dtype=bool)
    face_set[face_vert_idx] = True
    for v in face_vert_idx:
        d = np.sum((lm - verts[v]) ** 2, axis=1)
        g = group[int(np.argmin(d))]
        # "cheeks" is not a landmark group; infer from distance to face centre
        if g in masks:
            masks[g][v] = True
    # cheeks: face vertices far laterally but not in the other front groups
    cx = verts[face_vert_idx, 0]
    centre = np.median(verts[face_vert_idx], axis=0)
    for v in face_vert_idx:
        if not any(masks[n][v] for n in ("nose", "mouth", "eyes")):
            if abs(verts[v, 0] - centre[0]) > 0.45 * (cx.max() - cx.min()) / 2:
                masks["cheeks"][v] = True
    return masks


def main():
    n_modes = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    os.makedirs(ASSET_DIR, exist_ok=True)
    tmp = "/tmp/claude-0/_ict_dl"
    os.makedirs(tmp, exist_ok=True)

    # neutral mesh + metadata
    neu = os.path.join(ASSET_DIR, "generic_neutral_mesh.obj")
    if not os.path.isfile(neu):
        _curl(f"{RAW}/generic_neutral_mesh.obj", neu)
    vidx_path = os.path.join(ASSET_DIR, "vertex_indices.json")
    if not os.path.isfile(vidx_path):
        _curl(f"{RAW}/vertex_indices.json", vidx_path)

    mean, faces = parse_obj(neu)
    V = mean.shape[0]
    print(f"neutral: {V} vertices, {len(faces)} triangles")

    meta = json.load(open(vidx_path))
    landmark_idx = np.asarray(meta["idx_to_landmark_verts"], dtype=np.int64)
    face_vert_idx = np.asarray(meta["idx_to_fitting_verts"], dtype=np.int64)

    # identity basis = (identity_k - neutral), leading n_modes
    basis = np.zeros((n_modes, 3 * V), dtype=np.float64)
    for k in range(n_modes):
        dest = os.path.join(tmp, f"identity{k:03d}.obj")
        _curl(f"{RAW}/identity{k:03d}.obj", dest)
        vk, _ = parse_obj(dest)
        if vk.shape[0] != V:
            raise ValueError(f"identity{k:03d} vertex count {vk.shape[0]} != {V}")
        basis[k] = (vk - mean).reshape(-1)
        os.remove(dest)
        if (k + 1) % 10 == 0:
            print(f"  loaded {k + 1}/{n_modes} identity modes")

    # calibrate: a beta ~ N(0, I) sample should deform like one plausible identity.
    # Scale so the summed displacement over face vertices matches the mean
    # single-mode displacement (keeps combined faces realistic, not distorted).
    per_mode_disp = np.mean([
        np.linalg.norm(basis[k].reshape(V, 3)[face_vert_idx], axis=1).mean()
        for k in range(n_modes)
    ])
    basis /= np.sqrt(n_modes)            # independent modes sum in quadrature
    samp = np.random.default_rng(0).normal(size=(200, n_modes)) @ basis
    cur = np.mean(np.linalg.norm(samp.reshape(200, V, 3)[:, face_vert_idx], axis=2))
    basis *= per_mode_disp / cur         # match one-identity displacement scale

    # nominal mm: set so neutral face height (brow->chin landmarks) ~ 120 mm
    face_h = (mean[landmark_idx, 1].max() - mean[landmark_idx, 1].min())
    mm_per_unit = 120.0 / face_h

    regions = assign_regions(mean, landmark_idx, face_vert_idx)
    out = os.path.join(ASSET_DIR, "ict_face_model.npz")
    np.savez_compressed(
        out, mean=mean.astype(np.float32), basis=basis.astype(np.float32),
        faces=faces, landmark_idx=landmark_idx, face_vert_idx=face_vert_idx,
        mm_per_unit=np.float64(mm_per_unit),
        **{f"region_{k}": v for k, v in regions.items()},
    )
    sizes = {k: int(v.sum()) for k, v in regions.items()}
    print(f"wrote {out}")
    print(f"  K={n_modes} modes, mm_per_unit={mm_per_unit:.3f}, regions={sizes}")


if __name__ == "__main__":
    main()
