"""The single data-access swap point.

`load_dataset(cfg)` returns the canonical Dataset regardless of source. Today it
serves synthetic data; when controlled 3DFN access lands, only the `real` branch
needs implementing — it must return a Dataset with the identical fields, and
every downstream stage (QC, models, evaluation, demo) keeps working unchanged.
"""
from __future__ import annotations

import os

from ..common import Dataset, load_config
from .store import load_bundle, save_bundle


def load_dataset(cfg: dict):
    """Return (Dataset, GroundTruth|None, FaceModel) for the configured mode."""
    mode = cfg["data"].get("mode", "synthetic")
    if mode == "synthetic":
        out_dir = cfg["data"]["out_dir"]
        if not os.path.isfile(os.path.join(out_dir, "dataset.npz")):
            # generate on demand so a fresh checkout just works
            from ..synth.generate import generate
            ds, gt, fm = generate(cfg)
            save_bundle(out_dir, ds, gt, fm)
            return ds, gt, fm
        return load_bundle(out_dir)
    if mode == "real":
        raise NotImplementedError(
            "Real 3DFN mode is not wired yet. Implement this branch to read "
            "dbGaP genotypes + FaceBase/FLAME fits and return a Dataset with the "
            "same fields (genotypes in {0,1,2} with NaN for missing, demographics, "
            "face_params = FLAME shape coeffs, ancestry, subject-level splits). "
            "Ground truth is None for real data."
        )
    raise ValueError(f"Unknown data mode: {mode!r}")
