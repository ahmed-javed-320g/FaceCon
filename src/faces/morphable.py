"""A synthetic morphable face model that stands in for FLAME.

It provides exactly what the pipeline needs from FLAME:
  * a fixed mesh topology shared by every subject,
  * a shape vector `beta` (the prediction target) that decodes to a 3D mesh,
  * anatomical landmark vertices and named face regions for evaluation.

The mean head and the shape basis are generated deterministically from the
config seed, so the "model" is identical across runs. When real FLAME access
lands, `load_face_model` is the single function to replace; everything
downstream consumes the FaceModel interface below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import rng_for


# --------------------------------------------------------------------------- #
# Mean head mesh
# --------------------------------------------------------------------------- #
def _build_mean_head(res: int):
    """Return (verts (V,3), faces (F,3), uv (V,2)) for a head-like mean mesh.

    A UV sphere is shaped into a head: taller than wide, shallow at the back,
    with a protruding nose, brow, cheeks, mouth and chin so the silhouette reads
    clearly as a face in the wireframe hologram.
    """
    us = np.linspace(0.0, 2.0 * np.pi, res + 1)        # azimuth (wraps)
    vs = np.linspace(0.02 * np.pi, 0.98 * np.pi, res + 1)  # polar (avoid poles)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel()
    vv = vv.ravel()

    # base ellipsoid: x width, y height (up), z depth (front = +z)
    x = 0.72 * np.sin(vv) * np.cos(uu)
    y = 1.00 * np.cos(vv)
    z = 0.62 * np.sin(vv) * np.sin(uu)

    # front hemisphere mask (face side, z>0) and its "frontness" weight
    front = np.clip(z / 0.62, 0.0, 1.0)
    yq = y  # vertical coordinate, +1 top .. -1 bottom

    def bump(cx, cy, cz, sx, sy, amp):
        d2 = ((x - cx) / sx) ** 2 + ((yq - cy) / sy) ** 2
        return amp * np.exp(-d2) * np.clip((z - cz) / 0.62 + 0.3, 0.0, 1.0)

    # nose ridge + tip
    z = z + bump(0.0, 0.05, 0.25, 0.10, 0.28, 0.16)
    z = z + bump(0.0, -0.08, 0.35, 0.07, 0.07, 0.10)
    # brow ridge
    z = z + bump(0.0, 0.33, 0.20, 0.42, 0.06, 0.05)
    # cheeks
    z = z + bump(0.33, -0.05, 0.10, 0.17, 0.17, 0.05)
    z = z + bump(-0.33, -0.05, 0.10, 0.17, 0.17, 0.05)
    # chin
    z = z + bump(0.0, -0.72, 0.10, 0.18, 0.16, 0.07)
    # eye sockets (indent)
    z = z - bump(0.26, 0.14, 0.15, 0.13, 0.09, 0.05)
    z = z - bump(-0.26, 0.14, 0.15, 0.13, 0.09, 0.05)
    # mouth line (slight indent)
    z = z - bump(0.0, -0.42, 0.20, 0.20, 0.05, 0.03)

    # taper jaw: narrow the width toward the bottom front
    jaw = np.clip(-yq, 0.0, 1.0) * front
    x = x * (1.0 - 0.18 * jaw)

    verts = np.stack([x, y, z], axis=1).astype(np.float64)
    uv = np.stack([uu, vv], axis=1)

    faces = _grid_faces(res + 1, res + 1)
    return verts, faces, uv


def _grid_faces(n_rows: int, n_cols: int) -> np.ndarray:
    """Two triangles per grid quad; columns wrap (sphere seam)."""
    tris = []
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            a = r * n_cols + c
            b = r * n_cols + (c + 1)
            d = (r + 1) * n_cols + c
            e = (r + 1) * n_cols + (c + 1)
            tris.append((a, b, d))
            tris.append((b, e, d))
    return np.asarray(tris, dtype=np.int64)


def _assign_regions(verts: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean vertex masks per face region, by position on the front face."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    front = z > 0.15
    reg = {}
    reg["forehead"] = front & (y > 0.30)
    reg["eyes"] = front & (y > 0.05) & (y <= 0.30) & (np.abs(x) > 0.10)
    reg["nose"] = front & (y > -0.25) & (y <= 0.25) & (np.abs(x) <= 0.14)
    reg["cheeks"] = front & (y > -0.30) & (y <= 0.10) & (np.abs(x) > 0.18)
    reg["mouth"] = front & (y > -0.55) & (y <= -0.28) & (np.abs(x) <= 0.22)
    reg["jaw"] = front & (y <= -0.45)
    return {k: v for k, v in reg.items()}


def _pick_landmarks(verts: np.ndarray, count: int) -> np.ndarray:
    """Nearest vertex to each anatomical anchor; pad with spread points."""
    anchors = [
        (0.0, -0.08, 0.80),   # nose tip
        (0.0, 0.05, 0.60),    # nose bridge
        (0.26, 0.14, 0.45),   # right eye
        (-0.26, 0.14, 0.45),  # left eye
        (0.0, 0.40, 0.55),    # brow centre
        (0.22, -0.42, 0.45),  # right mouth corner
        (-0.22, -0.42, 0.45), # left mouth corner
        (0.0, -0.42, 0.50),   # mouth centre
        (0.0, -0.78, 0.45),   # chin
        (0.40, -0.10, 0.30),  # right cheek
        (-0.40, -0.10, 0.30), # left cheek
        (0.0, 0.70, 0.30),    # upper forehead
    ]
    anchors = anchors[:count]
    idx = []
    for a in anchors:
        d = np.sum((verts - np.asarray(a)) ** 2, axis=1)
        idx.append(int(np.argmin(d)))
    return np.asarray(idx, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Shape basis + model
# --------------------------------------------------------------------------- #
def _smooth_basis(verts: np.ndarray, uv: np.ndarray, k: int, seed: int) -> np.ndarray:
    """K smooth deformation fields with a decaying spectrum, shape (K, 3V).

    Each field is a low-frequency harmonic of the surface parameters (u, p), one
    spatial axis at a time. Modes are ordered from low to high frequency and
    their amplitude decays with frequency, as in a real morphable model: the
    leading modes make smooth global changes (a longer face, a wider jaw, a
    bigger nose) and later modes add gentle detail. No high-frequency noise, so
    sampled faces read clearly as faces.
    """
    v = verts.shape[0]
    u, p = uv[:, 0], uv[:, 1]

    # enumerate low-frequency modes ordered by total frequency a+b
    modes = []  # (a, b, trig, axis)
    for total in range(0, 7):
        for a in range(0, 4):
            b = total - a
            if b < 0 or b > 3:
                continue
            for axis in range(3):
                for trig in ("cos", "sin"):
                    if a == 0 and trig == "sin":
                        continue  # sin(0)=0, skip the null field
                    modes.append((a, b, trig, axis))
    modes = modes[:k]
    if len(modes) < k:
        raise ValueError(
            f"Only {len(modes)} smooth modes available for n_shape_dims={k}; "
            "lower n_shape_dims or widen the frequency range."
        )

    # taper that vanishes at the sphere poles (p->0, p->pi), so the collapsed
    # UV grid there cannot produce spikes; smooth and strongest on the face band
    pole_taper = np.sin(p) ** 2

    basis = np.zeros((k, 3 * v), dtype=np.float64)
    for i, (a, b, trig, axis) in enumerate(modes):
        ang = np.cos(a * u) if trig == "cos" else np.sin(a * u)
        field = ang * np.cos(b * p)
        field = field - field.mean()      # drop the constant (pure translation)
        field *= pole_taper               # kill pole artifacts
        rms = np.sqrt(np.mean(field ** 2))
        if rms > 0:
            field /= rms                  # unit RMS first ...
        field *= 1.0 / (1.0 + a + b) ** 2.0   # ... then steep spectral decay
        basis[i, axis * v:(axis + 1) * v] = field

    # Deterministic global scale so a typical subject (beta ~ N(0, I)) deforms
    # the head by a few percent of its height, keeping subjects clearly face-like.
    # For independent beta the per-vertex displacement variance is the column
    # energy of the basis; scale its mean to the target.
    col_energy = (basis ** 2).sum(axis=0).reshape(3, v).sum(axis=0)   # (V,)
    expected_disp = np.mean(np.sqrt(col_energy))
    target = 0.055
    if expected_disp > 0:
        basis *= target / expected_disp
    return basis


@dataclass
class FaceModel:
    """FLAME-stand-in: mean mesh + shape basis + eval metadata."""

    mean: np.ndarray                 # (V, 3)
    basis: np.ndarray                # (K, 3V) shape directions
    faces: np.ndarray                # (F, 3)
    region_masks: dict[str, np.ndarray]
    landmark_idx: np.ndarray         # (L,)
    mm_per_unit: float = 100.0       # nominal model-unit -> mm scale
    face_vert_idx: np.ndarray | None = None   # face-region vertices (ICT model)

    @property
    def n_vertices(self) -> int:
        return self.mean.shape[0]

    @property
    def n_dims(self) -> int:
        return self.basis.shape[0]

    def decode(self, beta: np.ndarray) -> np.ndarray:
        """Shape vector(s) -> mesh vertices. beta (K,) -> (V,3); (N,K) -> (N,V,3)."""
        beta = np.asarray(beta, dtype=np.float64)
        v = self.n_vertices
        if beta.ndim == 1:
            disp = (beta @ self.basis).reshape(v, 3)
            return self.mean + disp
        disp = (beta @ self.basis).reshape(beta.shape[0], v, 3)
        return self.mean[None, :, :] + disp


def load_ict_face_model(cfg: dict) -> FaceModel:
    """Load the realistic ICT-FaceKit-derived model built by build_face_model.py."""
    import os
    path = cfg["face_model"].get(
        "ict_path",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "assets", "ict", "ict_face_model.npz"),
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ICT face model not found at {path}. Build it first: "
            "python scripts/build_face_model.py 50"
        )
    f = np.load(path, allow_pickle=True)
    regions = {k[len("region_"):]: f[k] for k in f.files if k.startswith("region_")}
    identity_scale = float(cfg["face_model"].get("identity_scale", 1.0))
    return FaceModel(
        mean=f["mean"].astype(np.float64),
        basis=f["basis"].astype(np.float64) * identity_scale,
        faces=f["faces"],
        region_masks=regions,
        landmark_idx=f["landmark_idx"],
        mm_per_unit=float(f["mm_per_unit"]),
        face_vert_idx=f["face_vert_idx"],
    )


def render_parts(fm: FaceModel) -> list[np.ndarray]:
    """Triangle arrays to draw for display: the skin, plus the eyeballs if the model has them.

    This is the only model-specific display knowledge in the project. The ICT mesh has
    56 parts (skin, eyeballs, teeth, lashes ...): the skin is the largest connected part
    and the eyeballs are the two 800-vertex spheres. A model without such parts (FLAME,
    or the procedural head) returns its whole mesh, so swapping the model needs no change
    elsewhere.
    """
    import trimesh
    m = trimesh.Trimesh(fm.mean, fm.faces, process=False)
    comps = sorted(m.split(only_watertight=False), key=lambda c: -len(c.vertices))
    if len(comps) == 1:
        return [np.asarray(fm.faces)]
    tree_idx = {tuple(v): i for i, v in enumerate(np.round(fm.mean, 9).tolist())}
    def faces_of(c):
        vid = np.array([tree_idx[tuple(v)] for v in np.round(c.vertices, 9).tolist()])
        return vid[np.asarray(c.faces)]
    eyes = [c for c in comps[1:6] if len(c.vertices) == 800]
    return [faces_of(comps[0])] + [faces_of(e) for e in eyes]


def load_face_model(cfg: dict) -> FaceModel:
    """Build or load the face model from config.

    source: "ict"       -> realistic ICT-FaceKit-derived model (default)
    source: "synthetic" -> the lightweight procedural morphable head (fallback)
    """
    if cfg["face_model"].get("source", "ict") == "ict":
        return load_ict_face_model(cfg)
    fm = cfg["face_model"]
    res = int(fm["mesh_resolution"])
    k = int(fm["n_shape_dims"])
    verts, faces, uv = _build_mean_head(res)
    basis = _smooth_basis(verts, uv, k, int(cfg["seed"]))
    regions = _assign_regions(verts)
    landmarks = _pick_landmarks(verts, int(cfg.get("eval", {}).get("landmark_count", 12)))
    model = FaceModel(mean=verts, basis=basis, faces=faces,
                      region_masks=regions, landmark_idx=landmarks)
    return model
