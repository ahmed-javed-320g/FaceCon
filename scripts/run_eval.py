"""Run the full evaluation and write eval_results.json.

Usage: python scripts/run_eval.py [config/default.yaml]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.pipeline import prepare
from src.eval.harness import run_full_eval


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml")
    prep = prepare(cfg)
    res = run_full_eval(prep, out_dir=cfg["data"]["out_dir"])

    print("Baseline table (test set):")
    print(f"  {'model':30s} {'paramR2':>8} {'mesh_mm':>8} {'lmk_mm':>7} {'top1':>6}")
    for r in res["baselines"]:
        print(f"  {r['model']:30s} {r['param_r2']:+8.3f} {r['mesh_mean_mm']:8.2f} "
              f"{r['landmark_mean_mm']:7.2f} {r['retrieval_top1']:6.2f}")
    regions = sorted(res["baselines"][0].get("region_mm", {}))
    if regions:                                  # RQ6: these were computed but never printed
        print("\nRegional mesh error (mm, test set):")
        print(f"  {'model':30s}" + "".join(f"{r:>11s}" for r in regions))
        for r in res["baselines"]:
            print(f"  {r['model']:30s}" + "".join(f"{r['region_mm'][g]:11.2f}" for g in regions))
    print(f"\nwrote {os.path.join(cfg['data']['out_dir'], 'eval_results.json')} "
          "(per-region, per-landmark and per-dimension error per model)")
    print("\nGWAS-informed vs random panel:", json.dumps(res["ablation_gwas_vs_random"]))
    print("Signal recovery sanity:", json.dumps(res["sanity_signal_recovery"]))
    print("Remove-demographics ablation:", json.dumps(res["ablation_remove_demographics"]))

    if "appearance" in res:
        ap = res["appearance"]
        traits = [t["key"] for t in ap["traits"]]
        print("\nAppearance — per-trait accuracy by model:")
        print(f"  {'trait':13s}" + "".join(f"{m['key']:>10s}" for m in ap["models"]))
        for t in traits:
            print(f"  {t:13s}" + "".join(
                f"{m['per_trait'][t]['accuracy']:>10.2f}" for m in ap["models"]))


if __name__ == "__main__":
    main()
