# FaceCon — GPU training package

Advanced SNP→face models (report Section 5.4, models 6–9) + the FLAME-fitting
scaffold, in PyTorch. Built to run on your allocated GPU box and produce RQ5
report content (simple vs complex models) against the baselines.

## 0. What you need on the GPU box
- Python 3.10+, a CUDA-capable GPU (CPU works too, just slower)
- These data files from this project (copy them alongside the code, keeping the
  `outputs/real_match/` path, **or** regenerate — see §3):
  - `outputs/real_match/dataset.npz`   (genotypes, β target, demographics, splits)
  - `outputs/real_match/face_model.npz` **or** `assets/ict/ict_face_model.npz`

## 1. Install
```bash
pip install -r gpu/requirements.txt
```

## 2. Train
Run from the **project root** (so `gpu` is importable as a package):
```bash
# all four advanced models + baselines, ~minutes on a GPU
python -m gpu.train --model all --epochs 150 --out gpu/results.json

# or one at a time
python -m gpu.train --model transformer --epochs 150
python -m gpu.train --model diffusion   --epochs 200
```
Prints a comparison table (paramR², mesh-mm, retrieval top-1, train seconds) and
writes `gpu/results.json`. Flags: `--snp-pca N` (SNP PCA dim, default 256),
`--no-demo` (SNP-only, to isolate genetic signal), `--batch`, `--lr`.

**Expected, honest result:** under the 20% planted genetic share, the advanced
models should **not** beat the `demographics` baseline — that's the designed-in
RQ5 finding, not a bug. What you're showing is that a fair, confounder-aware
comparison correctly detects a weak genetic contribution even with big models.

## 3. (Optional) regenerate the matched dataset instead of copying it
```bash
python scripts/generate_matched.py config/real_match.yaml
```

## 4. FLAME-fitting scaffold (real-data pipeline, model A)
```bash
python -m gpu.flame_fit          # self-test: recovers a known β on the stand-in
```
This is the swap point for real data. To make it real:
- replace `FlameStandIn` with the real **FLAME** layer (gated download:
  flame.is.tue.mpg.de; e.g. the `flame-pytorch` module),
- feed **FaceBase** 3D scans + landmarks as the fit targets,
- swap the nearest-neighbour surface term for `pytorch3d.loss.chamfer_distance`.
The recovered identity β per scan becomes the real `face_params` target, after
which the same `gpu.train` models run unchanged on real data.

## Files
- `gpu/data.py` — leak-safe loading (train-only standardisation + SNP-PCA) + metrics
- `gpu/models.py` — MLP (6), Transformer (7), Contrastive/CLIP (8), Diffusion (9)
- `gpu/train.py` — training/eval harness + baselines + comparison table
- `gpu/flame_fit.py` — FLAME-fitting scaffold (A)
