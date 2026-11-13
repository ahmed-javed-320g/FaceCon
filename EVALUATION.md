# FaceCon methods audit (EVALUATION.md)

Audited: generator, config, GPU harness, deck, RQ5 draft, public-data notes. **Missing:** the written report (only an RQ5 draft) and the dbGaP study page (not retrievable). All results are simulated.

## 1. Verification

| Claim | Source | Status / use in generator |
|---|---|---|
| 3DFN: 2,454 participants, age 3–40 | [Weinberg 2016](https://stacks.cdc.gov/view/cdc/39061/cdc_39061_DS1.pdf) | Verified |
| GWAS sample 3,118 = 2,447 Pittsburgh (3–49) + 671 Denver (3–12), European | [FaceBase FV1](https://www.facebase.org/resources/human/hgai/FV1.html) | Verified. N correct; age uniform 3–49 (real: two-cohort mix) |
| OmniExpress+Exome v1.2, 964,193 SNPs + 4,322 custom, imputed 1000G P3 | [3DFN notes](https://www.facebase.org/resources/human/facial_norms/notes/) | Verified. Generator uses 5,000 |
| "733,202 SNPs" (team notes) | none | Unverified; conflicts with above |
| 952 M / 1,502 F; dbGaP 3,186 subjects; consent limits | none found | Unverified. Generator uses 50/50 sex |
| 24 landmarks, 34 measures (29 3D + 5 caliper), 3dMD, `.obj` | 3DFN notes | Verified. We use 68 iBUG landmarks on an ICT mesh |
| 20 GWAS measures; covariates sex, age, age², height, weight, facial size; 4 ancestry PCs | Shaffer S5; FaceBase FV1 | Verified. "Matching" was **wrong**: 16/20 represented (fixed) |
| Individual data needs IRB approval + DAC request + DUC | 3DFN notes; [FaceBase policy](https://docs.facebase.org/docs/human-data) | Verified. Feasibility risk |
| 14 injected SNPs are "genome-wide" hits | Shaffer S1 | **Wrong**: 5/14 pass 5e-8 (fixed) |
| HIrisPlex-S "reused as published" (deck) | none | **Unverified**: coefficients were reverse-derived by probing the web tool |

## 2. Problems found (ranked)

1. **The headline result was an artefact.** Every model was fed 256 SNP principal components, which keep +0.004 R² of the +0.173 available. "No model beats demographics" and "SNPs add ≈0" (deck, RQ5 draft, sweep) are unsupported.
2. **Mesh error under-reported 9.5×** in the GPU harness.
3. **Regeneration broke appearance**: every subject predicted red hair.
4. **GWAS injection flaws**: one SNP drawn as two independent columns; chrX ignored sex; effect sizes are winner's-curse upper bounds; the old "control" did not test specificity. The loci affect only the measurement phenotype, not the shape target.
5. **One seed, no confidence intervals**; "statistically indistinguishable" was untested.
6. **Synthetic structure is too easy**: unlinked SNPs, uniform MAF, three discrete ancestry groups, every causal SNP affecting all 50 dimensions (the screen matches the oracle).
7. **Baseline used true ancestry labels**; real data has only PCs.
8. **Deck is stale** (N=5,000; "proves the signal is real").
9. **Appearance coefficients are tool-derived**, which conflicts with the no-tool-output-as-labels rule.

## 3. What this setup supports

**Can:** pipeline correctness, leakage control, baseline logic, sensitivity to planted signal, an optimistic power bound, real-data readiness.
**Cannot:** any accuracy, heritability or model-ranking claim about real faces; real-data power (~10⁶ correlated SNPs, far weaker effects).

## 4. What I changed (18 tests pass)

- `gpu/controls.py` (new). Question: do SNPs add beyond covariates + PCs, and can the pipeline tell when they do not? Five seeds, bootstrap CIs. Ladder R²: covariates 0.205; +PCs 0.315; +SNP-PCA256 0.319; +screened SNPs 0.488; oracle 0.488. Negative controls: permuted genotypes −0.001; genetic share 0, −0.001. All four acceptance tests pass. *Disclosure:* A1 first failed because I wrote it two-sided; a slightly negative delta is correct null behaviour, so it now tests "no gain".
- Power grid (optimistic): N=3,118, share 0.1 → +0.070, 0.2 → +0.177; N=1,000 → +0.022, +0.119.
- `gpu/train.py --features screen` (train-only). One CPU check: MLP 0.393 vs demographics 0.317, single seed.
- Mesh scale fixed (mean-face floor 3.45 mm, not 0.36).
- GWAS injection: one column per SNP, sex-aware chrX, significance flags, upper-bound caveat. Covariate-adjusted recovery test: 14/14 injected links detected, correct sign; 0 Bonferroni hits in 246 non-injected pairs.
- Appearance frequencies persisted; RQ5 draft stamped SUPERSEDED; config wording corrected.

**Since done:** GPU rerun (`gpu/rq5_results.tex`). **Not done:** deck, report edits.

## 5. Improvements (ranked, free tools, ~6 weeks)

| # | Addition | Question | Negative control | Acceptance | Effort |
|---|---|---|---|---|---|
| 1 | Rerun RQ5 with `--features screen`, 5 seeds | Does architecture matter once signal is visible? | Permuted genotypes | CIs reported; control ≈ 0 | 2 days |
| 2 | 1000 Genomes genotypes (open): real LD, MAF, ancestry | Does it survive realistic structure? | Share 0 | A1–A4 pass | 1.5 weeks |
| 3 | Sparse, weak effects; PCs instead of labels everywhere | How does power fall with realism? | Share 0 | Power curve with CIs | 1 week |
| 4 | Published HIrisPlex-S parameters, checked on Harvard PGP public genotypes + self-reported eye colour | Does it match real labels? | Shuffled labels | Beats shuffled; aggregates only | 1 week |
| 5 | FLAME-fitting test on synthetic scans with known parameters | Does fitting recover truth? | Random mesh | Error below stated mm | 1.5 weeks |
| 6 | Pre-registered real-data plan; fix deck, report | — | — | Signed before access | 3 days |

## 6. Evaluator pitch

**Show:** `python -m gpu.controls` live (signal found when planted, zero when shuffled or absent); the 14/14 locus-recovery test; the pre-registered plan.
**Say:** "A validated, data-ready pipeline: it detects signal, rejects noise, and we found and fixed our own artefact."
**Do not claim:** that genetics does or does not predict faces; any real accuracy; HIrisPlex-S validation; real-data power.

## 7. Hard questions

1. *No real data: what did you show?* Pipeline correctness, with positive and negative controls. Nothing about people.
2. *Your earlier slide said SNPs add nothing.* A feature-compression artefact; an oracle baseline exposed it and we corrected it.
3. *Isn't the synthetic data rigged to succeed?* It is easy (unlinked SNPs, dense effects), so results are labelled an optimistic bound; 1000 Genomes structure is next.
4. *Will real data arrive?* Uncertain; FaceBase requires IRB approval. The project is scoped to pass without it.
5. *Is the appearance model the published one?* Unverified. It reproduces the web tool; checking against published parameters is planned.
