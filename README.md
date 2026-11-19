# FaceCon — genotype → 3D face shape (Final Year Project, FAST NUCES Lahore)

FYP ID **F26-082** · Supervisor: Ms. Rushda Muneer

A pipeline that predicts **3D facial shape** (a 50-dimensional morphable-model
shape vector) and **pigmentation** (eye / hair / skin colour) from SNP genotype,
evaluated honestly against **demographic baselines**.

> ### Status: validated pipeline, synthetic data only
>
> Controlled access to the real cohort (FaceBase `FB00000491.01` + dbGaP
> `phs000949.v1.p1`) is **pending and may not arrive**. Every number in this
> repository comes from **simulated data** in which we planted a known amount of
> genetic signal.
>
> **These results say nothing about how well DNA predicts real human faces.**
> They establish that the pipeline is correct, leak-free, and sensitive to
> signal in proportion to how much is present — i.e. that it is ready for real
> data the day access lands.
>
> Read [`EVALUATION.md`](EVALUATION.md) first: it is an internal methods audit
> that lists what is verified, what is unverified, and the bugs we found in our
> own earlier results.

## What is actually demonstrated

| Claim | Evidence |
|---|---|
| The pipeline finds genetic signal when it exists | Planted 20% genetic share → SNPs add **+0.173 R²** over covariates + ancestry PCs (5 seeds, 95% CIs in `gpu/controls.json`) |
| …and reports none when it does not | Genotypes permuted across subjects → **−0.001**; planted share set to zero → **−0.001** |
| No leakage | Subject-level splits; all scaling, PCA and feature screening fit on train only; 44 tests in `tests/` |
| Known associations are recoverable | 14/14 injected real GWAS loci detected with correct sign, **0** false hits in 246 non-injected SNP×measure pairs |
| Feature representation dominated architecture | SNP-PCA kept only +0.004 of the available +0.173; a train-only SNP screen recovered it |
| Simple beat complex on a linear signal | Tuned ridge **0.488** vs best deep model **0.396** (demographic baseline 0.317) |

Full write-up with tables: [`gpu/rq5_results.tex`](gpu/rq5_results.tex).

## Quickstart

```bash
pip install -r requirements.txt

# 1. regenerate the synthetic cohort (N=3,118, matched to the real study design)
python scripts/generate_matched.py config/real_match.yaml

# 2. controls, baseline ladder and power grid  (CPU, ~2 min; prints PASS/FAIL)
python -m gpu.controls --power

# 3. full model comparison + learning curve      (GPU, ~50 min)
python scripts/generate_matched.py config/real_match_large.yaml   # optional, for the learning curve
python -m gpu.run_all

# tests
python -m pytest -q tests
```

`gpu/` needs PyTorch with CUDA for step 3; steps 1–2 are CPU-only. On Windows,
install torch with `--index-url https://download.pytorch.org/whl/cu121`.

## Layout

```
config/     dataset configs (real_match.yaml = matched to the real study; _large = scaling experiment)
src/
  synth/      synthetic cohort generator + real GWAS effect table (gwas_effects.json)
  appearance/ HIrisPlex-S pigmentation model (see caveat below)
  faces/      morphable face model (ICT-FaceKit stand-in for FLAME)
scripts/    generate_matched.py — builds the dataset
gpu/        training harness (MLP / Transformer / contrastive / diffusion),
            controls.py (baseline ladder + negative controls + power grid),
            run_all.py (one-command experiment set), rq5_results.tex (report section)
tests/      22 regression tests, incl. leakage and control checks
app/        demonstrator (Explore / Results / Pipeline) and hologram reveal, built from the
            corrected pipeline by scripts/export_demo.py; deploy/ is an identical copy
deck/       presentation slides  ** contain stale pre-audit claims — see below **
outputs/real_match/   the dataset behind the reported numbers (tracked for reproducibility)
assets/ict/ derived ICT-FaceKit geometry (MIT, see assets/ict/ATTRIBUTION.md)
```

## Grounding in public data

Individual-level 3DFN data is controlled-access, so the synthetic cohort is
matched to the **public** study description, and real published effect sizes are
injected rather than invented:

- **Cohort design** — N=3,118 (2,447 Pittsburgh aged 3–49 + 671 Denver aged
  3–12), European ancestry, from the
  [FaceBase FV1 GWAS record](https://www.facebase.org/resources/human/hgai/FV1.html).
- **Shape genetics** — 14 lead SNPs (rsID, effect allele, MAF, β, p) taken
  verbatim from Shaffer et al. 2016 S1 Table, in `src/synth/gwas_effects.json`.
  Injected variance is derived as `(β/se)²/N`, which is a **winner's-curse upper
  bound**, not a published quantity.
- **Pigmentation** — HIrisPlex-S marker set and model structure.

## Known caveats

1. **`deck/` is stale.** It still carries pre-audit claims (N=5,000; "SNPs add
   ≈0"), which the audit showed were an artefact of our own feature pipeline. Do
   not cite it; `EVALUATION.md` and `gpu/rq5_results.tex` supersede it. The
   demonstrator in `app/` was rebuilt on the corrected pipeline and is current.
2. **The colour component is untested on real people, and will stay that way.**
   The HIrisPlex-S coefficients were recovered by probing the official calculator
   and validated against it, and FaceCon fits nothing. The check against real
   self-reported eye colour is written and validated on a stand-in cohort but
   cannot run: Harvard PGP now redirects every URL to itself (see
   `scripts/pgp_probe.py` and PREREGISTRATION.md A8), and openSNP shut down in
   April 2025. No accuracy claim is made for this component.
3. **The simulated panel is easier than reality**: 5,000 unlinked SNPs instead of
   ~10⁶ correlated ones, dense causal effects, three cleanly separated ancestry
   groups, so its power figures are optimistic upper bounds. A second cohort uses
   real 1000 Genomes genotypes with sparse effects (`data_prep/README_1KG.md`);
   the faces remain simulated in both.
4. **Licences.** The project's code is MIT (`LICENSE`). `assets/ict/` is derived
   from ICT-FaceKit (MIT) and keeps its attribution; `app/vendor/` is three.js
   (MIT); `app/instrument-sans.woff2` is under the SIL Open Font License.

## Citation of source studies

- Weinberg et al. (2016), *3D Facial Norms* database — Cleft Palate Craniofac J 53(6):e185–e197.
- Shaffer et al. (2016), *Genome-wide association study reveals multiple loci
  influencing normal human facial morphology* — PLoS Genet 12(8):e1006149.
- Chaitanya et al. (2018), HIrisPlex-S — Forensic Sci Int Genet 35:123–135.
- ICT-FaceKit — https://github.com/ICT-VGL/ICT-FaceKit (MIT).
