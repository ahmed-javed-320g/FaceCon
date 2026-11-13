# FaceCon — Demo-Day Runbook

**Project:** FaceCon (F26-082) · Genotype → 3D Face Shape & Appearance
**Team:** Ahmed Javed · Afnan Asif · Saad Riaz Taqi · Supervisor: Ms. Rushda Muneer
**Scope of this run:** mid-term prototype on **synthetic** data (real 3DFN access pending).

This runbook is written so that **any team member can take the project from a clean
checkout to a working live demo**, and so that **the demo survives a dead network,
a broken laptop, or a failed rebuild**. Read the "Golden path" once; keep the
"If it breaks" table open during the talk.

---

## 0. The one thing to remember

> **The live demo does NOT need to rebuild anything.** The deck, the prototype, and
> the hologram are already published as links and also run from local files.
> Rebuilding (`generate → eval → export`) is only for regenerating data from scratch;
> it takes ~4 minutes and is **not** part of the talk.

Have **three tabs open and loaded before you walk in**:

| Tab | What | Link / source |
|-----|------|----------------|
| 1 | Slide deck | `https://claude.ai/artifact/RHErykFTG4dz5UncuaAUeJ` |
| 2 | Interactive prototype | `https://claude.ai/artifact/QboAkr5GL1fjN88GJHaxqN` |
| 3 | Hologram reveal | `https://claude.ai/artifact/X5hPH8A2RZMjsfvh2k3S99` |

Plus the **offline copies** (see §4) on at least two laptops and one USB stick.

---

## 1. Prerequisites (do this the day before)

```bash
# Python 3.10+; from the project root:
pip install -r requirements.txt            # numpy, scipy, scikit-learn, trimesh, pyyaml, matplotlib, fast-simplification, pytest
python -c "import numpy, scipy, sklearn, trimesh, yaml; print('deps OK')"
```

Confirm the data assets are present (they are committed):

```bash
ls assets/ict/ict_face_model.npz           # realistic ICT-FaceKit face model (MIT)
ls app/data/web_data.json                  # ~3.9 MB web bundle the demo reads
ls outputs/synthetic/                       # dataset.npz, eval_results.json, etc.
```

If all three exist, **you can demo without ever running the pipeline.**

---

## 2. Golden path — full rebuild (only if you must regenerate data)

Run from the project root. Timings are from a clean dry run (CPU, 5000×5000).

```bash
# Stage 1 — generate synthetic dataset           (~11 s)
python scripts/run_generate.py config/default.yaml

# Stage 2 — full evaluation + ablations           (~2 min)
python scripts/run_eval.py config/default.yaml

# Stage 3 — export the browser bundle              (~2 min)
python scripts/export_web.py config/default.yaml   # writes app/data/web_data.json
```

**Total ≈ 4 minutes.** The pipeline is deterministic: with the committed seed
(`20261016`) the regenerated `app/data/web_data.json` is **byte-for-byte identical**
to the committed one (verified: md5 match). So a rebuild can never silently change
the numbers on the slides.

Expected eval numbers (sanity check — must match the deck):

| Model | Shape R² | Mesh err |
|-------|---------:|---------:|
| Mean face (floor) | −0.00 | 3.5 mm |
| Demographics only | 0.32 | 2.9 mm |
| Ridge (SNPs) | 0.21 | 3.2 mm |
| MLP (SNPs) | 0.17 | 3.2 mm |
| MLP (SNPs + demog.) | 0.29 | 2.9 mm |

Appearance accuracy (SNP / SNP+demog): eye 0.85/0.88 · skin 0.68/0.71 ·
hair-colour 0.80/0.84 · freckling 0.67/0.70 · hair-type 0.84/0.89 ·
baldness 0.46/0.92. Confounder checks: informed panel 0.31 vs random 0.12;
signal-recovery **PASS** vs shuffled control.

---

## 3. Running the live prototype & hologram

**Primary (online):** open Tabs 2 and 3 above. Done.

**Local (no internet, recommended as the real demo surface):** serve the `app/`
folder over HTTP — the pages `fetch()` `web_data.json`, so `file://` will *not* work.

```bash
cd app
python -m http.server 8000
# then open in a browser:
#   http://localhost:8000/index.html   (prototype)
#   http://localhost:8000/holo.html    (hologram)
```

Prototype flow to rehearse:
1. **Test subjects** tab → click a subject (e.g. `SUBJ-00017`, blue eyes; or `SUBJ-00009`).
2. **Prediction** tab → click **▶ Reveal prediction** → predicted face morphs in beside the held-out ground truth; eye/hair/skin render on the mesh.
3. **Evaluation & results** tab → live metrics, appearance-accuracy table, confounder ablations.

Hologram flow: it auto-cycles subjects (wireframe builds shape → colour materializes as the cyan fades → clean held face → dissolve). **Click** = next subject, **Space** = pause. Pause on a clean "hold" frame for photos.

---

## 4. Offline fallback kit (prepare and carry these)

Because a projector + conference Wi-Fi is the classic point of failure, carry a
**chain of fallbacks**, each independent of the previous:

1. **Local HTTP server** (§3) — the main offline surface. No internet needed.
2. **Full project folder on a USB stick** — includes `app/` with `web_data.json`,
   `three.min.js` vendored locally (no CDN), and all slides.
3. **PDF export of the deck** — File → export the deck to PDF from the deck link
   *before* demo day; put the PDF on the USB stick and email it to all three members.
4. **Screen-recording of the prototype + hologram** — record a 60–90 s MP4 of a
   full reveal and a hologram cycle in advance. If WebGL refuses to run on the venue
   machine, play the video. (Record it with the local server running.)
5. **Still screenshots** — a front-on reveal and a hologram "hold" frame, dropped
   into a backup slide at the end of the deck.

> The prototype vendors `three.min.js` locally and reads only `web_data.json` — it
> makes **no external network calls at runtime**, so §1–§2 of the fallback chain work
> on a fully air-gapped laptop.

**Pre-flight checklist (run on the actual demo laptop):**
- [ ] `python -m http.server` serves `index.html` and `holo.html` with no console errors
- [ ] Reveal animation completes and faces show colour (not grey)
- [ ] Hologram shows eye/hair/skin colour and the eyes render cleanly
- [ ] Deck PDF opens; backup screenshots present on the last slide
- [ ] Backup MP4 plays
- [ ] Laptop set to **do-not-disturb**, notifications off, screen-sleep disabled
- [ ] Browser zoom at 100%, hardware acceleration ON (WebGL)

---

## 5. If it breaks — triage during the talk

| Symptom | Cause | Fix (fast) |
|---------|-------|------------|
| Prototype page blank / "load error" | page can't fetch `web_data.json` (opened as `file://`) | serve via `python -m http.server` and use `http://localhost:8000/` |
| Faces/holo are black or don't render | WebGL disabled/blocked on venue GPU | play the backup MP4; or enable hardware acceleration and reload |
| Reveal stays grey | animation frame throttled (tab was backgrounded) | click **Reveal** again; it finalizes to full colour |
| Hologram too dim to read | paused mid-transition | press **Space** to resume, or **click** to advance to the next clean hold |
| Deck link won't load (no Wi-Fi) | network down | open the **deck PDF** from the USB stick |
| Rebuild fails / wrong numbers | someone edited config/seed | **do not rebuild live** — the committed `web_data.json` already matches the slides; just demo |
| `ModuleNotFoundError` on rebuild | deps missing | `pip install -r requirements.txt` (but you don't need a rebuild to demo) |

**Rule during Q&A:** never start a rebuild on stage. Everything needed is pre-built.

---

## 6. What's real vs. synthetic (say it if asked)

- **Real & reproducible:** the whole pipeline, QC, subject-level splits, the model
  ladder, the evaluation harness, confounder ablations, the ICT-FaceKit face model
  (MIT-licensed), and the appearance module calibrated to forensic DNA literature
  (HIrisPlex-S).
- **Synthetic:** the genotype→face *data*, by design, on the supervisor's advice,
  while controlled-access 3DFN data is pending. **Only the data loader changes** when
  real data lands — the swap point is `data.mode` in `config/default.yaml`.

---

## 7. Key paths

```
config/default.yaml          seed, dataset size, variance shares, model list
scripts/run_generate.py      synthetic dataset
scripts/run_eval.py          metrics + ablations -> outputs/synthetic/eval_results.json
scripts/export_web.py        -> app/data/web_data.json (what the demo reads)
app/index.html               interactive prototype
app/holo.html                hologram reveal
deck/project/                slide deck source (Slides artifact)
outputs/synthetic/DATA_CARD.md   dataset documentation
```
