# FaceCon: Demo-Day Runbook

**Project:** FaceCon (F26-082), genotype to 3D face shape and appearance
**Team:** Ahmed Javed, Afnan Asif, Saad Riaz Taqi. Supervisor: Ms. Rushda Muneer
**Scope:** every face is simulated. Real 3DFN access is pending. The Results tab also
shows a run on real 1000 Genomes genotypes with simulated faces.

Written so any team member can go from a clean checkout to a working demo, and so the
demo survives a dead network, a broken laptop, or a failed rebuild.

---

## 0. The one thing to remember

> **The live demo needs no rebuild and no server.** Open `app/index.html` straight
> from disk (double-click it). Rebuilding the demo data takes about a minute and is
> never part of the talk.

Have these open before you walk in:

| Tab | What | Link / source |
|-----|------|----------------|
| 1 | Demonstrator (Explore, Results, Pipeline) | https://claude.ai/artifact/17atvFs6SFyYHJr9tB1BPg |
| 2 | Hologram reveal | the **Open the hologram reveal** button on tab 1, or `app/holo.html` |
| 3 | Slide deck | `https://claude.ai/artifact/RHErykFTG4dz5UncuaAUeJ` — **stale, see §6** |

Plus the offline copy (§4) on at least two laptops and one USB stick. The artifact link
is private until its owner shares it from its page on claude.ai.

---

## 1. Prerequisites (the day before)

Nothing to install to **run** the demo: two pages, a data file, a font, and three.js
(r147, MIT) in `vendor/`, all local, no network. To **rebuild** its data you need the
Python environment:

```bash
pip install -r requirements.txt -r gpu/requirements.txt    # the second file adds torch, which the export needs
python -m pytest -q                 # 22 tests, all must pass
```

Committed and required:

```
app/index.html, app/holo.html       the two pages
app/data/demo.js                    2.8 MB: mesh, 36 held-out people, recorded results
app/instrument-sans.woff2           the typeface (SIL OFL, licence beside it)
app/vendor/                         three.js r147 and its orbit controls and studio lighting (MIT)
outputs/real_match/dataset.npz      the simulated cohort the demo is drawn from
```

---

## 2. Rebuilding the demo data (only if the pipeline changed)

```bash
python scripts/export_demo.py       # about 1 minute on CPU, writes app/data/demo.js
cp app/index.html app/holo.html deploy/ && cp app/data/demo.js deploy/data/ && cp -r app/vendor deploy/
```

It trains the demo's MLP with a fixed seed, so a rebuild reproduces these exactly
(test set of 469 people, simulated cohort). If they differ, something changed:

| Model | Test R² | Mesh error |
|-------|--------:|-----------:|
| Mean face | −0.002 | 3.45 mm |
| Demographics only | 0.317 | 3.00 mm |
| Ridge, demographics + screened SNPs | 0.467 | 2.64 mm |
| MLP, demographics + screened SNPs | 0.397 | 2.81 mm |

The Results tab does not use these. It shows the recorded multi-run figures from
`gpu/controls.json`, `gpu/report_back.json` and `gpu/results_kg.json`, which are the
numbers in the report.

---

## 3. Running the demo

**Primary:** tab 1, or `app/index.html` opened from disk. Both work offline.

Explore flow to rehearse (about 90 seconds):
1. Pick a person on the left. Person 00076 is a good opener: ridge gets 1.3 mm closer
   than demographics.
2. The left head is the prediction, coloured by its error; the right is the true face.
   Drag either to turn both.
3. Press **←** twice to step down to **Mean face**, then **→** back up through Ridge to the neural net. The
   prediction morphs and the orange shrinks as the model gets more information. This
   is the moment to point at.
4. Point at **How close each model gets** on the right, then the colour predictions.

Then **Results** (start on Real genotypes): the four findings, top to bottom, and the
limits box at the end. Then **Pipeline** if asked how it works.

Keys: **↑ ↓** person, **← →** model, **E** error colouring, **F** face forward.

Hologram: cycles people automatically (wireframe builds the shape, colour resolves,
hold, dissolve). **Click** or **→** for the next person, **Space** pauses. Pause on a
hold frame for photos.

---

## 4. Offline fallback kit

1. **The app folder on disk** — `app/` or `deploy/` (identical). Opens from
   `file://`; makes no network calls except an optional font request that fails
   silently offline. The hosted copy loads three.js from jsDelivr instead, so it
   needs internet.
2. **The same folder on a USB stick.**
3. **Deck PDF**, exported before the day, on the stick and emailed to all three.
4. **A 60 to 90 second screen recording** of the Explore flow and one hologram cycle,
   for a venue machine with WebGL disabled.
5. **Screenshots** of Explore and Results on a backup slide.

Pre-flight on the actual demo laptop:
- [ ] `app/index.html` opened from disk shows two heads within two seconds
- [ ] Switching model morphs the left head; the right one stays put
- [ ] Results tab: both genotype views render their charts
- [ ] `holo.html` reaches a coloured hold frame with eyes and hair coloured
- [ ] Browser zoom 100 per cent, hardware acceleration on, notifications off

---

## 5. If it breaks

| Symptom | Cause | Fix |
|---------|-------|-----|
| Grey panel saying WebGL is turned off | venue browser or GPU blocks WebGL | enable hardware acceleration and reload, or play the recording |
| Panel saying the 3D view could not load three.js | `vendor/` missing, or the hosted copy is offline | copy the whole `app/` folder; the other tabs still work meanwhile |
| Room too bright to read the dark theme | projector washes out dark pages | press **Light mode** in the header; the choice is remembered |
| Message saying the data file was not found | `data/demo.js` missing beside `index.html` | copy the whole `app/` folder, not just the HTML |
| Different typeface than rehearsed | `instrument-sans.woff2` missing | harmless; copy the whole folder next time |
| Hologram never colours in | tab was in the background | click once to restart the cycle |
| Numbers differ from the report | someone rebuilt from a changed config | do not rebuild live; use the committed `demo.js` |

**Rule during Q&A:** never start a rebuild on stage.

---

## 6. What's real, what's simulated, what's stale (say it if asked)

- **Real:** the pipeline and its tests; the controls; the 1000 Genomes genotypes in the
  Results tab's real-genotype view; the HIrisPlex-S method (our coefficients reproduce
  the public calculator and are not yet checked against the paper).
- **Simulated:** every face, and the genotypes in the Explore tab. The genotype-to-face
  link was planted by us and is additive, which is why ridge is expected to win.
  The 41 pigmentation markers in the Explore cohort are also simulated, each drawn
  independently, so a colour combination can be one a real genome would rarely give
  (very dark skin with blond hair, say). The real-genotype colour check is in Results.
- **Stale:** the slide deck still carries pre-correction claims (that SNPs add nothing,
  N=5,000). Do not present those slides until the deck is updated. The demonstrator and
  the report are current.

---

## 7. Key paths

```
app/index.html                demonstrator: Explore, Results, Pipeline
app/holo.html                 hologram reveal
app/data/demo.js              written by scripts/export_demo.py
deploy/                       identical copy for static hosting (see deploy/README.md)
scripts/export_demo.py        corrected-pipeline export (replaces export_web.py)
gpu/*_kg.json                 1000 Genomes runs copied from the GPU machine;
                              scripts/summarise_kg.py turns them into gpu/results_kg.json
gpu/controls.json, gpu/report_back.json    simulated-cohort results
data_prep/README_1KG.md       building the real-genotype panel
EVALUATION.md                 what is verified and what is not
```

---

# The GPU round (Track A, remaining)

Everything else in Track A is done and committed. These six steps need either a GPU or
network access the build sandbox does not have. Run them from the repository root on the
GPU machine, with the repository up to date (`git pull`).

Nothing here overwrites a committed result. Every step writes a new file. If a step
fails, the later steps do not depend on it except where noted.

## 0. Update and check

```bash
git pull
python -m pytest -q                 # expect 44 passed
```

## 1. The mixed architecture on real genotypes (~30 min)

The architecture arms in Chapter 8 are on the simulated panel. This repeats the mixed
arm on the 1000 Genomes panel, where the SNPs are correlated and the structure is
continuous, which is the harder case.

```bash
python scripts/generate_matched.py config/kg_na_mix.yaml --out-dir outputs/kg_na_mix
python -m gpu.controls --config config/kg_na_mix.yaml --na-rows --out gpu/controls_kg_na_mix.json
```

Read: the gap between ORACLE and ORACLE-NA. On the simulated panel the additive oracle
recovered about a third of what the true coding recovered. If linkage closes that gap,
that is a finding worth reporting; if it does not, that is equally worth reporting.

## 2. Replicate cohorts for the mixed arm (~20 min)

Three independently generated cohorts, so the neural comparison in step 3 gets a paired
interval rather than one split's luck.

```bash
for s in 11 12 13; do
  python scripts/generate_matched.py config/na_mix.yaml --seed $s --out-dir outputs/na_mix_s$s
done
```

## 3. Neural models on non-additive signal (several hours, GPU)

**This is the one result the report explicitly marks as open.** Chapter 8 says the
linear reference wins on additive signal and that whether it still wins on non-additive
signal has not been tested. This tests it, and both outcomes are publishable.

```bash
python -m gpu.run_all \
  --data-dirs outputs/na_mix_s11,outputs/na_mix_s12,outputs/na_mix_s13 \
  --feature-sets screen,screen_na \
  --models mlp,transformer \
  --out gpu/run_all_na_mix.json
```

Read: `max(neural) - ridge_demo_snp` on `screen_na`, as the pre-registration's primary
comparison defines it, with the t interval across the three cohorts. A neural model that
beats ridge here, having lost on additive signal, is the clearest case for the nets in
the project.

## 4. Regions and subgroups on real genotypes (~15 min)

```bash
python scripts/rq6_rq7.py --data-dir outputs/kg \
  --group-names AFR,AMR,EAS,EUR,SAS --with-mlp --out gpu/rq6_rq7_kg.json
```

Read: whether the relative regional gains are still flat once the genotypes are real.
The flat profile on simulated data is what a uniformly planted signal predicts, so it
tests the measurement, not the biology.

## 5. RQ4, published loci against matched-random panels (~45 min)

The control is not optional. Without `--control-dir` the script marks its own output
FAILED, because a published panel beating random panels is explained just as well by the
panel being bigger or commoner unless a cohort with effects at random loci shows the
comparison is fair.

```bash
python scripts/generate_matched.py config/kg.yaml --seed 77 --out-dir outputs/kg_randomloci
python scripts/rq4_panels.py --data-dir outputs/kg --control-dir outputs/kg_randomloci \
  --out gpu/rq4_panels.json
```

Read: the published panel's percentile in the matched-random null, and the control's
percentile, which must land near the median. If the control does not, the main number
means nothing and should not be reported.

## 6. Harvard PGP eye colour (~1 hour, mostly download)

Needs ordinary internet access. PGP data is CC0 with no data-use agreement. It has no
face scans, so it tests the colour component only.

```bash
python scripts/pgp_download.py --out data/pgp --max-participants 400
python scripts/pgp_eye_check.py --dir data/pgp --out gpu/pgp_eye.json
```

If the survey export 404s, the script says so and asks for the trait survey to be saved
by hand as `data/pgp/surveys.csv`; the rest still runs. `data/pgp/` is gitignored —
commit `gpu/pgp_eye.json` only, which holds aggregates and no individual's data.

Read: blue-vs-rest AUC against the pre-registered 0.70, and whether it sits outside the
1000-shuffle null. A low AUC here is a real result about a published model on consumer
chips, not a failure of this project, and should be reported as such.

## 7. Bring the results back

```bash
git add gpu/controls_kg_na_mix.json gpu/run_all_na_mix.json gpu/rq6_rq7_kg.json \
        gpu/rq4_panels.json gpu/pgp_eye.json
git commit -m "GPU round results"
git push
```

Then the report needs Chapter 8 extended with steps 1, 3, 5 and 6, and the RQ status
table updated: RQ4 and RQ8 move off "not yet answered", and RQ5 gains the non-additive
neural comparison. PREREGISTRATION.md gets an addendum for anything that came out
against its stated expectation — as Addenda A1 and A2 already do.
