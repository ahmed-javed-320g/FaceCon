"""Shaded orthographic preview of the mean face and sample subjects.

Uses a hand-rolled orthographic projection with Lambertian shading and a
painter's-algorithm sort, so the facial relief reads clearly (matplotlib's 3D
camera makes frontal faces hard to see). This is only a sanity-check preview;
the real-time look lives in the Phase 5 WebGL renderer.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io.store import load_bundle


def rot_y(v, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return v @ R.T


def shade_mesh(ax, verts, faces, azim, base=(0.93, 0.80, 0.70)):
    V = rot_y(verts, azim)
    tri = V[faces]                        # (F,3,3)
    # face normals
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = np.array([-0.3, 0.45, 0.85]); light /= np.linalg.norm(light)
    lam = np.clip(n @ light, 0, 1)
    shade = 0.22 + 0.78 * lam
    colors = np.clip(np.array(base)[None, :] * shade[:, None], 0, 1)
    # cull back faces (normal pointing away from +z viewer) and depth-sort
    facing = n[:, 2] > -0.05
    order = np.argsort(tri[:, :, 2].mean(axis=1))
    order = [i for i in order if facing[i]]
    polys = [tri[i, :, :2] for i in order]
    pc = PolyCollection(polys, facecolors=colors[order], edgecolors="none")
    ax.add_collection(pc)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/synthetic"
    ds, gt, fm = load_bundle(out_dir)
    te = ds.splits["test"]
    pick = [int(te[0]), int(te[1])]
    # normalise all meshes with one shared transform (from the mean) so faces
    # fill the view and stay comparable to each other
    centre = (fm.mean.max(0) + fm.mean.min(0)) / 2
    scale = np.abs(fm.mean - centre).max()

    def norm(V):
        return (V - centre) / scale

    meshes = {
        "Mean face": norm(fm.mean),
        f"Subject {ds.subject_ids[pick[0]]}": norm(fm.decode(ds.face_params[pick[0]])),
        f"Subject {ds.subject_ids[pick[1]]}": norm(fm.decode(ds.face_params[pick[1]])),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5), facecolor="black")
    for col, (title, V) in enumerate(meshes.items()):
        for row, az in enumerate((0, -35)):
            ax = axes[row, col]; ax.set_facecolor("black")
            shade_mesh(ax, V, fm.faces, az)
            if row == 0:
                ax.set_title(title, color="white", fontsize=12)
            ax.text(0.5, -0.04, "front" if row == 0 else "3/4 view",
                    color="#7fdfe8", ha="center", transform=ax.transAxes, fontsize=9)
    fig.suptitle("FaceCon synthetic morphable model  (SYNTHETIC DATA)",
                 color="white", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out_dir, "preview_faces.png")
    plt.savefig(path, dpi=120, facecolor="black")
    print("wrote", path)


if __name__ == "__main__":
    main()
