# PLAN: FYP-II Track A (everything that does not need the real cohort)

Started 2026-11-18. Updated as each step finishes. The order is chosen so no step
depends on a later one: the pre-registration is frozen before any non-additive
result exists, and seed plumbing lands before any neural run.

| # | Step | Where | Status |
|---|------|-------|--------|
| 0 | Orient and verify: 22 tests, controls reproduce `gpu/controls.json` (max diff 4e-8), report builds (106 pp, 17 figures, 33 tables), face_params checksum `-668.022131456431` | here | done |
| 1 | Stale claims and hygiene: test counts, README, LICENSE, CLAUDE.md, PLAN.md, requirements | here | done |
| 2 | `gpu/sweep_genetic.py`: document the genotype_source defect, do not fix | here | done |
| 3 | Non-additive genetic effects: generator, ground truth, ORACLE-NA, 2-df screen, neural `screen_na` features, tests | here | done |
| 5a | Seed plumbing: `--seed`, `--data-seeds`, `split_seed` | here | done |
| 9 | `PREREGISTRATION.md` frozen and committed | here | done |
| 4 | Linear arms ADD, ADD-cal, DOM, EPI, MIX, NULL-NA on the simulated panel, 5 cohorts | here | done |
| 5b | RQ6 (regions, landmarks, dimensions) and RQ7 (ancestry subgroups) | here | done |
| 6 | FLAME fitting: fix the optimiser, validate in mm with a random-mesh control | here | done |
| 7 | Appearance: published coefficients if openly available; Harvard PGP eye-colour check scripts; retire the legacy colour placeholder | here, then GPU machine | scripts done, run on GPU |
| 8 | RQ4: published Shaffer loci vs matched-random panels on 1000 Genomes | script here, run on GPU machine | script done, run on GPU |
| G | GPU round: kg MIX arm, neural runs with data-seed CIs, RQ4, PGP download and check | GPU machine | |
| 10 | Report, deck, demonstrator; push code, report and LaTeX source | here | report done; deck open |

Out of scope by design (Track B): QC of the 3DFN genotype files, FLAME registration
of real scans, the FLAME 2020 licence, and any claim about real faces.

Notes
- openSNP closed on 30 April 2025 and deleted its data; the real-label colour check
  uses the Harvard Personal Genome Project (public, CC0) instead.
- `gpu/flame_fit.py` mentions pytorch3d only in comments; nothing imports it, so it is
  not added as a dependency.
