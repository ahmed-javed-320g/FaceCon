# CLAUDE.md: working on FaceCon

FaceCon predicts 3D face shape (a 50-number shape vector) and eye, hair and skin
colour from SNP genotypes, and its main claim is **honest evaluation**: the measure is
the gain over a demographic baseline, never raw R². The real cohort (3DFN, dbGaP
phs000949) has not been released to the team, so **every face in every result is
simulated**. Read `EVALUATION.md` before changing anything that produces a number.

## Rules

1. Never write a claim about real faces. Every number is a property of the pipeline.
2. Every new result needs a control that could have contradicted it: a positive
   control before any null result, a negative control (shuffled genotypes, zero
   genetic share) for every gain.
3. Never overwrite a committed result or tracked dataset. New runs write new files.
   Never run `scripts/generate_matched.py config/real_match.yaml`: it overwrites
   `outputs/real_match/dataset.npz`, which `gpu/report_back.json` and
   `tests/test_audit.py` depend on.
4. Never change an acceptance threshold after seeing a result
   (`gpu/controls.py`, the `tests` dict near the end of `main`).
5. Never edit `report/FastFyp.cls` or any FAST template file. Use `\label`/`\ref`,
   never hard-coded numbers.
6. No secrets in the repo.

## Invariants with tripwires

- `outputs/real_match/dataset.npz` `face_params` sums to `-668.022131456431`.
- With no `data.genetic_architecture` block, generation is bit-for-bit what it was;
  `tests/test_nonadditive.py` checks this with `np.array_equal`.
- `python -m gpu.controls --config config/real_match.yaml` reproduces
  `gpu/controls.json` to printed precision and prints the 7-row ladder.
- `app/` and `deploy/` are identical (`diff -rq app deploy`, ignoring README).
- The test count in `README.md` and the report's Tables 7.7 and 7.8 match
  `pytest --collect-only -q`.

## Commands

```bash
python -m pytest -q                                   # all tests
python -m gpu.controls --config config/real_match.yaml  # ladder + controls, 5 cohorts
python -m gpu.run_all                                 # neural models (GPU machine)
python scripts/summarise_kg.py                        # gpu/results_kg.json from the kg runs
python scripts/export_demo.py                         # demonstrator data
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Where things are

- `gpu/controls.py` linear ladder, controls, power grid; `gpu/run_all.py` neural set
- `src/synth/generate.py:_compose_faces` is the only place the face target is built
- `gpu/*_kg.json` real-genotype runs copied from the GPU machine
- `report/` LaTeX source of the FYP report (FAST template)
- `PLAN.md` what is being worked on now; `PREREGISTRATION.md` frozen protocol
