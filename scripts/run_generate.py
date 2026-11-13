"""Generate the synthetic dataset and write it to disk.

Usage:
    python scripts/run_generate.py [config/default.yaml]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.io.store import save_bundle
from src.synth.generate import generate


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    cfg = load_config(cfg_path)
    ds, gt, fm = generate(cfg)
    out_dir = cfg["data"]["out_dir"]
    save_bundle(out_dir, ds, gt, fm)
    print(f"Generated synthetic dataset -> {out_dir}")
    print(f"  subjects={ds.n_subjects}  snps={ds.n_snps}  "
          f"face_dims={ds.n_face_dims}  vertices={fm.n_vertices}")
    print(f"  variance shares (target): {gt.variance_shares}")


if __name__ == "__main__":
    main()
