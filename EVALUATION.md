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
6. **Synthetic structure is too easy**: unlinked SNPs, uniform MAF, three discrete ancestry groups, every causal SNP affecting all 50 dimensions (the screen matches the oracle). **Addressed** — see §4b; real 1000G genotypes with sparse effects, and the screen no longer matches the oracle (0.100 vs 0.114).
7. **Baseline used true ancestry labels**; real data has only PCs.
8. **Deck is stale** (N=5,000; "proves the signal is real").
9. **Appearance coefficients are tool-derived**, which conflicts with the no-tool-output-as-labels rule.

## 3. What this setup supports

**Can:** pipeline correctness, leakage control, baseline logic, sensitivity to planted signal, an optimistic power bound, real-data readiness.
**Cannot:** any accuracy, heritability or model-ranking claim about real faces; real-data power (~10⁶ correlated SNPs, far weaker effects).

## 4. What I changed (22 tests pass)

- `gpu/controls.py` (new). Question: do SNPs add beyond covariates + PCs, and can the pipeline tell when they do not? Five seeds, bootstrap CIs. Ladder R²: covariates 0.205; +PCs 0.315; +SNP-PCA256 0.319; +screened SNPs 0.488; oracle 0.488. Negative controls: permuted genotypes −0.001; genetic share 0, −0.001. All four acceptance tests pass. *Disclosure:* A1 first failed because I wrote it two-sided; a slightly negative delta is correct null behaviour, so it now tests "no gain".
- Power grid (optimistic): N=3,118, share 0.1 → +0.070, 0.2 → +0.177; N=1,000 → +0.022, +0.119.
- `gpu/train.py --features screen` (train-only). One CPU check: MLP 0.393 vs demographics 0.317, single seed.
- Mesh scale fixed (mean-face floor 3.45 mm, not 0.36).
- GWAS injection: one column per SNP, sex-aware chrX, significance flags, upper-bound caveat. Covariate-adjusted recovery test: 14/14 injected links detected, correct sign; 0 Bonferroni hits in 246 non-injected pairs.
- Appearance frequencies persisted; RQ5 draft stamped SUPERSEDED; config wording corrected.

**Since done:** GPU rerun (`gpu/rq5_results.tex`); report Deliverable III. **Not done:** deck.

## 4b. Real genotypes, simulated faces (1000 Genomes)

Panel: NYGC 30x high-coverage GRCh38 callset. 2,573 unrelated people (deg-2 relatives removed), 100,009 SNPs (MAF ≥ 0.01, missingness ≤ 0.02, biallelic, autosomes + X, unique rsIDs), 5 super-populations. Faces, age, height, weight still simulated; 300 causal SNPs × 5 of 50 dims; share 0.20. Build and commands: `data_prep/README_1KG.md`.

| Result | Value | Status |
|---|---|---|
| Ladder: cov 0.195; +PCs 0.335; +PCA256 0.335; +screen 0.435; oracle 0.449 | gain +0.100 screen, +0.114 oracle | Verified, 5 seeds, `outputs/controls_kg.json` |
| Same quantity from the independent `run_all` path | +0.105 | Verified — two code paths agree to 0.005 |
| **Causal SNPs hidden, only LD neighbours** | **+0.025** (CIs +0.018…+0.035, none spanning 0) | Verified, 5 seeds, `outputs/controls_kg_hidden.json` |
| Same, on the unlinked synthetic panel | −0.001 | Verified — LD is what carries it |
| Deep vs linear, same split, same inputs | MLP +0.366 vs ridge +0.443 (gap −0.077) | Verified, 3 seeds, `gpu/report_back.json` |
| PCA256 features, linear reference | +0.338 = demographics exactly | Verified — the representation is empty, not the models |
| Negative controls (permuted): ridge and all nets | ≤ demographics; both acceptance checks pass | Verified |
| GWAS recovery on real LD | 14/14 injected, 0 Bonferroni FPs, FPR 0.061 | Verified |
| HIrisPlex-S by super-population | EUR 52% brown eyes / 57% brown hair / 90% intermediate skin vs AFR 99.7 / 94.1 / 99.1 | Plausibility only — 1000G has no colour labels |
| HIrisPlex-S marker coverage | 39/41 | Missing N29insA, Y152OCH (rare MC1R red-hair frameshifts) → red hair under-predicted |
| Shaffer lead SNP alignment | 13/13 found, 2 flagged | rs56203676, rs5786757 are indels whose 1000G representation differs from the published table; neither is genome-wide significant and both carry **our** planted weights, so no result depends on the alignment |

**New defect found and fixed (not previously in the list of 5):** the first `ridge_demo_snp` baseline penalised the covariates alongside the SNPs, scoring 0.207 against a 0.317 demographics baseline under permutation — i.e. it would have flattered the deep models exactly in the negative control. Replaced with a two-stage fit (OLS covariates, ridge on the residual); now returns demographics exactly under permutation, and that is an acceptance check.

**What this still cannot support:** the planted mapping is additive, so ridge is correctly specified and is *expected* to win. Nothing here measures non-additive structure, and nothing here is about real faces.

## 5. Improvements (ranked, free tools, ~6 weeks)

| # | Addition | Question | Negative control | Acceptance | Effort |
|---|---|---|---|---|---|
| 1 | Rerun RQ5 with `--features screen`, 5 seeds | Does architecture matter once signal is visible? | Permuted genotypes | CIs reported; control ≈ 0 | 2 days |
| 2 | ~~1000 Genomes genotypes (open): real LD, MAF, ancestry~~ **DONE (§4b)** | Does it survive realistic structure? | Share 0, permuted | A1–A4 pass | done |
| 3 | ~~Sparse effects; PCs instead of labels everywhere~~ **DONE (§4b)** — sparse (5/50 dims), PCs throughout | How does power fall with realism? | Share 0 | Power curve with CIs | done |
| 4 | Published HIrisPlex-S parameters, checked on Harvard PGP public genotypes + self-reported eye colour | Does it match real labels? | Shuffled labels | Beats shuffled; aggregates only | 1 week |
| 5 | FLAME-fitting test on synthetic scans with known parameters | Does fitting recover truth? | Random mesh | Error below stated mm | 1.5 weeks |
| 6 | Pre-registered real-data plan; fix deck, report | — | — | Signed before access | report done; deck open |
| 7 | **Non-additive effects** (interactions, dominance) | Does the linear/neural ranking survive? | Share 0 | Ranking reported either way | 1 week |

## 6. Evaluator pitch

**Show:** `python -m gpu.controls` live (signal found when planted, zero when shuffled or absent); the 14/14 locus-recovery test; the pre-registered plan.
**Say:** "A validated, data-ready pipeline: it detects signal, rejects noise, and we found and fixed our own artefact."
**Do not claim:** that genetics does or does not predict faces; any real accuracy; HIrisPlex-S validation; real-data power.

## 7. Hard questions

1. *No real data: what did you show?* Pipeline correctness, with positive and negative controls. Nothing about people.
2. *Your earlier slide said SNPs add nothing.* A feature-compression artefact; an oracle baseline exposed it and we corrected it.
3. *Isn't the synthetic data rigged to succeed?* It was (unlinked SNPs, dense effects). We rebuilt it on real 1000G genotypes with sparse effects (§4b); the conclusions held with smaller margins, and hiding the causal SNPs showed LD carries only ~22% of the attainable signal.
6. *Why does ridge beat your deep models — isn't that a weak result?* It is a real result and it is reported, but the planted signal is additive, so ridge is the correctly specified model. The honest claim is that these nets are inefficient on additive signal at this N, not that deep learning cannot help. Testing non-additive effects is item 7 in §5.
4. *Will real data arrive?* Uncertain; FaceBase requires IRB approval. The project is scoped to pass without it.
5. *Is the appearance model the published one?* Unverified. It reproduces the web tool; checking against published parameters is planned.
