"""Persist and reload the canonical dataset, ground truth, and face model.

Everything the pipeline and the web demo consume is written here, so a run is
fully reproducible from disk and the front-end never needs the Python objects.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..common import Dataset, GroundTruth
from ..faces.morphable import FaceModel


def gt_nonadditive_arrays(gt: GroundTruth) -> dict:
    """Non-additive ground-truth arrays for ground_truth.npz; empty for an additive run.

    Shared by both writers (save_bundle and scripts/generate_matched.py) so they
    cannot drift apart. Empty means old bundles stay byte-identical.
    """
    out = {}
    for name in ("dominance_weights", "interaction_pairs", "interaction_weights"):
        v = getattr(gt, name)
        if v is not None:
            out[name] = v
    return out


def gt_nonadditive_manifest(gt: GroundTruth) -> dict:
    """Manifest entry for the architecture; empty when the config has no such block."""
    return {"genetic_architecture": gt.genetic_architecture} if gt.genetic_architecture else {}


def save_bundle(out_dir: str, ds: Dataset, gt: GroundTruth, fm: FaceModel) -> None:
    os.makedirs(out_dir, exist_ok=True)

    app = {}
    if ds.appearance_snps is not None:
        app["appearance_snps"] = ds.appearance_snps
        app["appearance_labels"] = ds.appearance_labels
        app["appearance_traits"] = np.array(ds.appearance_traits, dtype=object)
    np.savez_compressed(
        os.path.join(out_dir, "dataset.npz"),
        subject_ids=ds.subject_ids,
        genotypes=ds.genotypes,
        demographics=ds.demographics,
        face_params=ds.face_params,
        ancestry=ds.ancestry,
        split_train=ds.splits["train"],
        split_val=ds.splits["val"],
        split_test=ds.splits["test"],
        **app,
    )
    gt_app = {f"appcausal_{k}": v for k, v in gt.appearance_causal.items()}
    np.savez_compressed(
        os.path.join(out_dir, "ground_truth.npz"),
        causal_snp_idx=gt.causal_snp_idx,
        genetic_weights=gt.genetic_weights,
        demo_weights=gt.demo_weights,
        **gt_app,
        **gt_nonadditive_arrays(gt),
    )
    extra = {}
    if fm.face_vert_idx is not None:
        extra["face_vert_idx"] = fm.face_vert_idx
    np.savez_compressed(
        os.path.join(out_dir, "face_model.npz"),
        mean=fm.mean,
        basis=fm.basis,
        faces=fm.faces,
        landmark_idx=fm.landmark_idx,
        mm_per_unit=np.float64(fm.mm_per_unit),
        **extra,
        **{f"region_{k}": v for k, v in fm.region_masks.items()},
    )
    manifest = {
        "mode": ds.meta.get("mode", "synthetic"),
        "n_subjects": int(ds.n_subjects),
        "n_snps": int(ds.n_snps),
        "n_face_dims": int(ds.n_face_dims),
        "demographic_names": ds.demographic_names,
        "variance_shares": gt.variance_shares,
        "regions": list(fm.region_masks.keys()),
        **gt_nonadditive_manifest(gt),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_bundle(out_dir: str) -> tuple[Dataset, GroundTruth, FaceModel]:
    d = np.load(os.path.join(out_dir, "dataset.npz"), allow_pickle=True)
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    ds = Dataset(
        subject_ids=d["subject_ids"],
        genotypes=d["genotypes"],
        demographics=d["demographics"],
        demographic_names=manifest["demographic_names"],
        face_params=d["face_params"],
        ancestry=d["ancestry"],
        splits={"train": d["split_train"], "val": d["split_val"], "test": d["split_test"]},
        meta={"mode": manifest["mode"]},
        appearance_snps=d["appearance_snps"] if "appearance_snps" in d.files else None,
        appearance_labels=d["appearance_labels"] if "appearance_labels" in d.files else None,
        appearance_traits=list(d["appearance_traits"]) if "appearance_traits" in d.files else [],
    )
    g = np.load(os.path.join(out_dir, "ground_truth.npz"))
    gt = GroundTruth(
        causal_snp_idx=g["causal_snp_idx"],
        genetic_weights=g["genetic_weights"],
        demo_weights=g["demo_weights"],
        variance_shares=manifest["variance_shares"],
        appearance_causal={k[len("appcausal_"):]: g[k] for k in g.files
                           if k.startswith("appcausal_")},
        dominance_weights=g["dominance_weights"] if "dominance_weights" in g.files else None,
        interaction_pairs=g["interaction_pairs"] if "interaction_pairs" in g.files else None,
        interaction_weights=g["interaction_weights"] if "interaction_weights" in g.files else None,
        genetic_architecture=manifest.get("genetic_architecture"),
    )
    f = np.load(os.path.join(out_dir, "face_model.npz"))
    regions = {k[len("region_"):]: f[k] for k in f.files if k.startswith("region_")}
    fm = FaceModel(
        mean=f["mean"], basis=f["basis"], faces=f["faces"],
        region_masks=regions, landmark_idx=f["landmark_idx"],
        mm_per_unit=float(f["mm_per_unit"]) if "mm_per_unit" in f.files else 100.0,
        face_vert_idx=f["face_vert_idx"] if "face_vert_idx" in f.files else None,
    )
    return ds, gt, fm
