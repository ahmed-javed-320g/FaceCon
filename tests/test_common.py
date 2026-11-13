import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config, rng_for

def test_config_loads_and_validates():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    assert cfg["data"]["n_subjects"] > 0
    assert abs(sum(cfg["face_variance"].values()) - 1.0) < 1e-6

def test_rng_streams_independent_and_reproducible():
    a1 = rng_for(42, "genotypes").random(5)
    a2 = rng_for(42, "genotypes").random(5)
    b  = rng_for(42, "faces").random(5)
    assert (a1 == a2).all()          # reproducible
    assert (a1 != b).any()           # independent streams
