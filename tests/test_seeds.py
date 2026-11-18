"""Seed plumbing: re-drawing the split must be exact, and defaults must not move."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gpu.data import FaceConData   # noqa: E402


def test_split_seed_reproduces_stored_split():
    """split_seed == the cohort's own seed re-draws exactly the stored split."""
    stored = FaceConData(root=ROOT, features="pca", snp_pca=8)
    redrawn = FaceConData(root=ROOT, features="pca", snp_pca=8, split_seed=20261016)
    for a, b in ((stored.tr, redrawn.tr), (stored.va, redrawn.va), (stored.te, redrawn.te)):
        assert np.array_equal(a, b)


def test_other_split_seed_resamples_but_stays_a_partition():
    other = FaceConData(root=ROOT, features="pca", snp_pca=8, split_seed=20261017)
    stored = FaceConData(root=ROOT, features="pca", snp_pca=8)
    assert not np.array_equal(other.te, stored.te)
    idx = np.concatenate([other.tr, other.va, other.te])
    assert len(idx) == len(set(idx.tolist())) == other.N
    assert (len(other.tr), len(other.va), len(other.te)) == (len(stored.tr), len(stored.va), len(stored.te))


def test_generator_refuses_to_overwrite_tracked_dataset():
    import pytest
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import generate_matched
    before = os.path.getmtime(os.path.join(ROOT, "outputs", "real_match", "dataset.npz"))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        generate_matched.main(os.path.join(ROOT, "config", "real_match.yaml"),
                              out_dir=os.path.join(ROOT, "outputs", "real_match"))
    assert os.path.getmtime(os.path.join(ROOT, "outputs", "real_match", "dataset.npz")) == before
