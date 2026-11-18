# PREREGISTRATION: FaceCon, FYP-II Track A

**Frozen 2026-11-18, at commit `55a8b5d`, before any result of the analyses below
was read.** This file is not revised after results exist. If an analysis has to
deviate from it, the deviation is recorded in a dated addendum at the end and the
original text stays as written.

Every face in every analysis here is simulated, except the eye-colour check in §6,
which uses real people's genotypes and self-reported colours. No claim about real
faces follows from anything below.

A finding that SNPs add little, or that a method does not work, will be reported as
plainly as a positive one.

---

## 1. Primary measure

Gain in test R² from adding SNP features to covariates and ancestry principal
components: R²(model) − R²(`cov+PC`), where `cov+PC` is the ordinary least-squares
fit on sex, age, age², height, weight and ten genotype principal components, all fit
on the training split only (`gpu/controls.py`, `ladder`). Test R² is the mean over
the 50 shape dimensions. Raw R² is never the headline.

## 2. Baseline ladder

Rows as computed by `gpu/controls.py`: `mean`, `cov`, `cov+PC`, `cov+labels`,
`+SNP-PCA256`, `+SNP-screen`, `ORACLE`. With a non-additive configuration or
`--na-rows`, also `+SNP-screen-NA` (two-degree-of-freedom genotypic screen,
additive and heterozygote codes, top 1,500 SNPs) and `ORACLE-NA` (true causal SNPs,
additive + heterozygote + pair-product codes, the full design in every arm).

## 3. Controls and acceptance thresholds

Thresholds are the ones in `gpu/controls.py` (the `tests` dict at the end of `main`)
and `gpu/run_all.py` (`checks`) at the commit above. They are not changed after
results are seen; in particular the A3 literal 0.10 is not lowered.

| Check | Rule |
|---|---|
| A1 | genotypes permuted: no per-cohort 95% CI above 0, and \|mean delta\| ≤ 0.005 |
| A2 | genetic share 0: \|delta\| ≤ 0.01 |
| A3 | additive configs: `ORACLE` delta ≥ 0.10 and `+SNP-screen` CI > 0 in every cohort |
| A3-NA | non-additive configs: `ORACLE-NA` delta ≥ 0.10 and max(`+SNP-screen`, `+SNP-screen-NA`) CI > 0 in every cohort; the dosage-only A3 is then reported, not required |
| A1-NA, A2-NA | the same as A1 and A2 for the new rows (A2-NA covers `+SNP-screen-NA` and `ORACLE-NA`) |
| A4 | train, validation and test disjoint |
| C (neural) | with genotypes permuted, no network and no linear reference gains more than 0.01 over the demographic baseline |

## 4. Non-additive experiment (Phase 4)

Simulated panel (`config/real_match.yaml` design), five cohorts per arm (seeds
20261016 to 20261020), the same planted genetic share 0.20 in every arm.

| Arm | Config | dominance_frac | epistasis_frac |
|---|---|---|---|
| ADD | `real_match.yaml` | absent | absent |
| ADD-cal | `real_match.yaml --na-rows` | absent | absent |
| DOM | `na_dom.yaml` | 0.5 | 0.0 |
| EPI | `na_epi.yaml` | 0.0 | 0.5 |
| MIX | `na_mix.yaml` | 0.34 | 0.33 |
| NULL-NA | the null-share control inside every non-additive run | (arm's) | genetic share 0 |

Confirmatory, on the GPU machine: MIX on real 1000 Genomes genotypes
(`kg_na_mix.yaml`), linear path, five cohorts.

### Expected outcomes, written before any run

| Arm | `+SNP-screen` | `+SNP-screen-NA` | `ORACLE` | `ORACLE-NA` |
|---|---|---|---|---|
| ADD | baseline | ≈ baseline, small variance penalty | baseline | ≈ baseline − penalty |
| DOM | reduced | **clearly > screen** | reduced | ≈ full |
| EPI | ≈ 0 | ≈ 0 | ≈ 0 | **only this recovers** |
| MIX | reduced | between | reduced | ≈ full |

In the EPI arm, pairwise interaction is not reachable by any marginal screen at this
sample size. The conclusion there would be "non-additive signal is not reachable by
marginal screening", not "ridge is wrong".

### Primary comparison

On the MIX arm, at the same planted genetic share: is max(neural) − `ridge_demo_snp`
still negative when **both** are given non-additive-aware screened features
(`--features screen_na`)? Reported as the paired gap within replicate with a 95%
t interval, replicates being three independently generated MIX cohorts, each with
its own split and its own initialisation (`gpu/run_all.py --data-dirs`). Models: MLP
and Transformer at minimum; any architecture not run is named as not run.

### Secondary comparison

Does (`+SNP-screen-NA` − `+SNP-screen`) on DOM and MIX exceed its value on ADD-cal?

### Calibration rule for A3-NA

The ADD-cal arm measures the pure column-count penalty of `ORACLE-NA` (`ORACLE` −
`ORACLE-NA` on an additive target). The expectation is that it is small (about
0.02). A non-additive arm whose `ORACLE-NA` delta falls below 0.10 is reported as an
A3-NA **failure**, with the ADD-cal penalty printed beside it as the interpretation.
The threshold is not moved. If the measured penalty is larger than 0.02, that is
reported as a finding and the calibration test records the measured value with a
note, instead of being loosened silently.

### Validity conditions

The run counts only if: ADD reproduces `gpu/controls.json` to printed precision;
NULL-NA gives \|delta\| ≤ 0.01 on every new row; permutation returns the
demographic baseline.

## 5. RQ6 and RQ7 (descriptive)

RQ6: per-region, per-landmark and per-shape-dimension error, model against
demographic baseline, on the simulated cohort. RQ7: the same gain within each
ancestry group, with bootstrap 95% CIs. Where a CI spans both a meaningful gain and
none, the report says the group is too small to conclude anything. No subgroup is
singled out after the fact.

## 6. Eye colour against real self-reported labels (Harvard PGP)

The only real-label check in the project. Data: public Harvard Personal Genome
Project participants with a consumer genotype file and a self-reported eye colour.

- **Label mapping, fixed now:** blue or grey → blue; green or hazel →
  intermediate; brown or black → brown. If left and right eyes differ, or the text
  fits no category, the person is excluded and counted.
- **Model:** the HIrisPlex-S eye model in `src/appearance/` as it stands when the
  check runs. Coefficient provenance is reported with the result.
- **Inclusion:** all six eye markers (rs12913832, rs1800407, rs12896399,
  rs16891982, rs1393350, rs12203592) genotyped; one genotype file per person;
  alleles aligned with the same rules as the 1000 Genomes build.
- **Measures:** AUC for blue vs not-blue and brown vs not-brown (the measures the
  IrisPlex literature reports), and accuracy with call rate at the published 0.7
  probability threshold. Bootstrap 95% CIs over people.
- **Negative control:** 1,000 label shuffles; the observed AUC is reported with its
  shuffle percentile.
- **Reporting:** aggregates only. No per-person genotype or prediction is written
  to any committed file.
- Hair colour is exploratory, because consumer chips lack many MC1R markers. Skin
  colour is not tested; there is no survey label for it.

## 7. RQ4: published loci against matched-random panels

On 1000 Genomes genotypes, plant effects at the published Shaffer et al. (2016) lead
SNPs and their linkage neighbours. Measure the `cov+PC` gain of a ridge on the
published-loci panel against 100 random panels matched on panel size and allele
frequency bins, and report its percentile. **Control:** the same comparison on a
cohort whose effects are planted at random loci, where the published panel should
sit near the median. This tests the panel machinery and nothing about real faces.

## 8. FLAME fitting (Phase 6)

On synthetic scans with known shape parameters: mean vertex error in millimetres,
noise-free, at most 1.0 mm. **Control:** fitting a random mesh of the same size
must give an error at least five times larger. Reported in millimetres.

---

## Disclosure: what was seen before freezing

Engineering smoke runs, made to check the code worked, not to estimate anything:

- One cohort of 1,200 people (seed 20261016), `ladder(na_rows=True)`:
  - additive design: `+SNP-screen` +0.109, `+SNP-screen-NA` +0.077, `ORACLE` +0.124, `ORACLE-NA` +0.082;
  - MIX design: `+SNP-screen` +0.015, `+SNP-screen-NA` +0.018, `ORACLE` +0.033, `ORACLE-NA` +0.083;
  - permuted `+SNP-screen-NA`: −0.002 and −0.001.
- Replicate-mode smoke on the existing additive cohort (two split seeds, MLP, 30
  epochs): gap −0.070.
- Generator unit tests at n = 1,500 and 6,000 (planted shares realised).

These were at a sample size different from the experiment, so they predict the
direction but not the size of the Phase 4 numbers. Nothing above was changed because
of them: the expectation table and the thresholds are the ones written in the work
order before any code ran.

---

## Addenda

Each entry records something the protocol above got wrong, or did not anticipate, after
the result was seen. No threshold above has been changed. Entries are append-only.

### A1 — the ORACLE-NA calibration penalty is larger than expected (0.032, not 0.02)

§4 expected ORACLE-NA to sit below ORACLE on an additive target by about 0.02 R², the
cost of fitting dominance and interaction columns that carry no signal. Measured:

| measurement | penalty (R²) |
|---|---|
| single cohort, seed 20261016 (`tests/test_nonadditive.py`) | +0.0318 |
| mean over the 5 cohorts of `gpu/controls_na_addcal.json` | +0.0301 |

Per the calibration rule frozen in §4 ("if the measured penalty is larger than 0.02,
that is reported as a finding and the calibration test records the measured value with a
note, instead of being loosened silently"), the 0.02 figure stands as the record of what
was expected and the test now records +0.0318 with a tolerance of 0.004.

Consequence, carried into the report: ORACLE-NA is **not** a drop-in replacement for
ORACLE. The two are compared only within an arm, never across arms, and a lower
ORACLE-NA than ORACLE on an additive target is the expected direction, not a failure.
The penalty is well below the gains the NA rows produce where the architecture is
actually non-additive (+0.055 on DOM, +0.032 on MIX for the screen; the NA oracle gains
0.148 over cov+PC on MIX against the dosage oracle's 0.058), so those gains are not an
artefact of the extra columns. The penalty's size is why it has to be stated: a reader
comparing ORACLE-NA on MIX against ORACLE on ADD would otherwise read a 0.03 modelling
cost as a 0.03 loss of signal.

### A2 — the 2-df screen loses on the pure-epistasis arm

§4 expected the NA screen to beat the additive screen on every non-additive arm. It does
on DOM (0.4107 → 0.4656) and MIX (0.3669 → 0.3988), but **not** on EPI
(0.3815 → 0.3638, i.e. −0.0177). This was not anticipated and is reported as found.

The reason is a property of the screen, not a bug: the per-SNP 2-df score tests a
genotype's own het deviation, so it has power against dominance and none against a
pairwise A×A product whose marginals are flat. On EPI the extra column per SNP is
therefore noise, and it costs what A1 above measures. The pre-registered expectation
table was wrong to treat "non-additive" as one case. ORACLE-NA, which is given the true
interaction pairs, recovers the signal on EPI (0.4591) just as on the other arms, which
is what isolates the cause to the screen's power rather than to the representation.

No new row or screen was added in response: a pair-aware screen is named as future work
in the report and is not run here, because designing it after seeing this result would
make its result unpublishable under this protocol.

### A3 — "error" for the FLAME random-mesh control means fit residual

§8 asks that the random-mesh control show an error "≥5×" the face fit's. For a mesh of
per-vertex noise there is no true β to compare against, so "error" cannot mean the
vertex error used for the recovery check. The quantity reported for both is the mean
symmetric nearest-neighbour **fit residual** between the scan points and the fitted face
region. Measured on the noise-free condition: 0.264 mm for a real face against 2.280 mm
for the control, a ratio of 8.6×. This is a clarification of an ambiguous term in the
protocol, decided by what the control can be measured on at all, not a change of the
threshold.

### A4 — smoke-run disclosure

The fitter in `gpu/flame_fit.py` was rewritten before its validation was run, and two
intermediate versions were run on 3 scans each while diagnosing why the first version
failed both §8 checks (warm start from the mean-face pose, and a resampled chamfer whose
gradient was noise). Those runs informed the optimiser's design — alternating closed-form
warm start, ICP correspondences, averaged convergence window — and the ridge weight was
then set by the MAP rule σ²/L rather than by hand. No §8 threshold was consulted in
choosing any of them, and the reported numbers come from the single full 12-scan run.
