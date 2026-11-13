import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import load_config
from src.pipeline import prepare
from src.eval.harness import run_full_eval


def _cfg():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    cfg["data"]["n_subjects"] = 500
    cfg["data"]["n_snps"] = 500
    cfg["data"]["n_causal_snps"] = 40
    cfg["data"]["out_dir"] = "outputs/_test_eval"
    cfg["models"]["mlp"]["max_iter"] = 120
    cfg["models"]["mlp"]["snp_pca"] = 80
    return cfg


def test_full_eval_structure_and_signal():
    prep = prepare(_cfg())
    res = run_full_eval(prep)
    # baseline table has all enabled models with the full metric set
    keys = {r["key"] for r in res["baselines"]}
    assert {"mean_face", "demographics", "ridge_snp", "mlp_snp", "mlp_snp_demo"} <= keys
    for r in res["baselines"]:
        for k in ("param_r2", "mesh_mean_mm", "landmark_mean_mm",
                  "retrieval_top1", "region_mm"):
            assert k in r
    # the scientific checks
    ab = res["ablation_gwas_vs_random"]
    assert ab["informed_param_r2"] > ab["random_param_r2"] + 0.05
    assert res["sanity_signal_recovery"]["passes"] is True
    import shutil
    shutil.rmtree("outputs/_test_eval", ignore_errors=True)
