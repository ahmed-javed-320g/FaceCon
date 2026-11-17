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

Nothing to install to **run** the demo: it is three files and a font, no library, no
network. To **rebuild** its data you need the Python environment:

```bash
pip install -r requirements.txt -r gpu/requirements.txt    # the second file adds torch, which the export needs
python -m pytest -q                 # 22 tests, all must pass
```

Committed and required:

```
app/index.html, app/holo.html       the two pages
app/data/demo.js                    2.8 MB: mesh, 36 held-out people, recorded results
app/instrument-sans.woff2           the typeface (SIL OFL, licence beside it)
outputs/real_match/dataset.npz      the simulated cohort the demo is drawn from
```

---

## 2. Rebuilding the demo data (only if the pipeline changed)

```bash
python scripts/export_demo.py       # about 1 minute on CPU, writes app/data/demo.js
cp app/index.html app/holo.html deploy/ && cp app/data/demo.js deploy/data/
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
   silently offline.
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
gpu/results_kg.json           1000 Genomes results, transcribed; replace with the GPU JSONs
gpu/controls.json, gpu/report_back.json    simulated-cohort results
data_prep/README_1KG.md       building the real-genotype panel
EVALUATION.md                 what is verified and what is not
```
