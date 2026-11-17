"""Build gpu/results_kg.json from the recorded 1000 Genomes runs.

    python scripts/summarise_kg.py

Reads the four files saved on the GPU machine (copied into the repo):
  gpu/controls_kg.json          python -m gpu.controls --config config/kg.yaml
  gpu/controls_kg_hidden.json   ... --hide-causal
  gpu/report_back_kg.json       python -m gpu.run_all --data-dir outputs/kg
  gpu/build_report_kg.json      scripts/build_1kg.py
A few values exist only in the console output of scripts/generate_matched.py
(sex counts, splits, the GWAS recovery check); they are kept in CONSOLE below and
labelled as such in the output.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = lambda n: json.load(open(os.path.join(ROOT, "gpu", n)))

CONSOLE = {   # scripts/generate_matched.py config/kg.yaml, run of 2026-11-17
    "males": 1276, "females": 1297, "splits": {"train": 1801, "val": 385, "test": 387},
    "gwas_recovery": {"injected_detected": 14, "injected_total": 14, "correct_sign": 14,
                      "non_injected": 246, "bonferroni_false_hits": 0, "fpr_at_0_05": 0.061},
}


def r(x, d=6):
    return round(float(x), d)


def main():
    C, H, R, B = G("controls_kg.json"), G("controls_kg_hidden.json"), G("report_back_kg.json"), G("build_report_kg.json")
    first = lambda xs: [x[0] if isinstance(x, list) else x for x in xs]
    ci = lambda xs: [[r(x[1]), r(x[2])] for x in xs]
    models = {}
    for feat in ("pca", "screen"):
        row = R["results"][f"A kg N=2573 | {feat}"]
        models[feat] = {k: r(row[k][0]) for k in ("demographics", "ridge_snp", "ridge_demo_snp")}
        for k in ("mlp", "transformer", "contrastive", "diffusion"):
            models[feat][k] = [r(np.mean(row[k])), r(np.std(row[k]))]
            models[feat][k + "_mesh"] = [r(np.mean(row[k + "_mesh"]), 3), r(np.std(row[k + "_mesh"]), 3)]
        models[feat]["demographics_mesh"] = row["_mesh_demo"]
    perm = R["results"]["C NEGATIVE CONTROL permuted | screen"]
    models["permuted_screen_one_run"] = {k: r(perm[k][0]) for k in
                                         ("demographics", "ridge_snp", "ridge_demo_snp", "mlp", "transformer", "contrastive", "diffusion")}
    models["checks"] = R["checks"]; models["seeds"] = len(R["seeds"])
    hps = B["hirisplex_markers"]
    app = B["appearance_by_superpop_NOT_accuracy"]
    out = {
        "_source": "Built by scripts/summarise_kg.py from gpu/controls_kg.json, gpu/controls_kg_hidden.json, "
                   "gpu/report_back_kg.json and gpu/build_report_kg.json. Fields under _console_only come from the "
                   "console output of scripts/generate_matched.py.",
        "_mode": "real_genotypes_1000G_simulated_faces",
        "cohort": {"n_people": B["n_people"], "n_snps": B["n_snps"], "missing_rate": r(B["missing_rate"], 6),
                   "superpop": B["superpop_counts"], "n_causal": 300, "causal_dims": C["causal_dims"],
                   "genetic_share": C["genetic_share"],
                   "hps_found": sum(m["found"] for m in hps), "hps_total": len(hps),
                   "hps_missing": [m["rsid"] for m in hps if not m["found"]],
                   "gwas_found": sum(m["found"] for m in B["shaffer_snps"]), "gwas_total": len(B["shaffer_snps"]),
                   "gwas_flagged": [m["rsid"] for m in B["shaffer_snps"] if abs(m["eur_freq_1kg"] - m["published_freq"]) > 0.15],
                   **{k: CONSOLE[k] for k in ("males", "females", "splits")}},
        "ladder": {"seeds": len(C["seeds"]), **{k: {"mean": r(C["ladder_mean"][k]), "sd": r(C["ladder_sd"][k])} for k in C["ladder_mean"]}},
        "delta": {"pca256": r(np.mean(C["delta_pca256"])), "screen": r(np.mean(first(C["delta_screen"]))),
                  "oracle": r(np.mean(C["delta_oracle"])), "oracle_per_seed": [r(x) for x in C["delta_oracle"]],
                  "permuted": r(np.mean(first(C["delta_permuted"]))), "null_share": r(np.mean(C["delta_null_share"])),
                  "screen_ci": ci(C["delta_screen"]), "permuted_ci": ci(C["delta_permuted"])},
        "hidden_causal": {"screen": r(np.mean(first(H["delta_screen"]))), "screen_ci": ci(H["delta_screen"]),
                          "pca256": r(np.mean(H["delta_pca256"])), "same_on_unlinked_synthetic_panel": -0.001},
        "acceptance": C["acceptance"], "acceptance_hidden": H["acceptance"],
        "models": models,
        "gwas_recovery": CONSOLE["gwas_recovery"],
        "appearance_by_superpop": {g: {t: list(max(d.items(), key=lambda kv: kv[1])) for t, d in v.items()} for g, v in app.items()},
        "appearance_full": app,
        "_console_only": ["cohort.males", "cohort.females", "cohort.splits", "gwas_recovery"],
    }
    json.dump(out, open(os.path.join(ROOT, "gpu/results_kg.json"), "w"), indent=1)
    print("wrote gpu/results_kg.json")


if __name__ == "__main__":
    main()
