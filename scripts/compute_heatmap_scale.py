"""Choose a FIXED per-vertex shape-error display range for the Shape Error
Heatmap, from the TRAIN split only (never test), so the scale is comparable
across subjects and models. Prints percentile stats and a recommended max.

Error convention matches src/eval/metrics.vertex_dist exactly: per-vertex
Euclidean distance between decode_mm(pred) and decode_mm(true), in nominal mm.
"""
from __future__ import annotations
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_config
from src.pipeline import prepare
from src.models.base import build_models
from src.eval import metrics as M

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml")
prep = prepare(cfg); fm = prep.fm
tr, btr = prep.view("train"); va, bva = prep.view("val")
train_idx = prep.idx("train")
# sample up to 400 train subjects for the distribution
rng = np.random.default_rng(0)
samp = np.sort(rng.choice(train_idx, size=min(400, len(train_idx)), replace=False))
feats = prep.features.subset(samp)
true_verts = M.decode_mm(fm, prep.ds.face_params[samp])   # (N,V,3) nominal mm

models = build_models(cfg)
all_err = []
print(f"{'model':28s} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>7}")
for m in models:
    m.fit(tr, btr, va, bva)
    pb = m.predict(feats)
    d = M.vertex_dist(fm, pb, true_verts=true_verts)      # (N,V) nominal mm
    all_err.append(d.ravel())
    ps = np.percentile(d, [50,90,95,99])
    print(f"{m.key:28s} {ps[0]:7.2f} {ps[1]:7.2f} {ps[2]:7.2f} {ps[3]:7.2f} {d.max():7.2f}")
pooled = np.concatenate(all_err)
for q in (90,95,97.5,99):
    print(f"pooled p{q}: {np.percentile(pooled,q):.2f}")
# recommended: round the pooled 95th percentile up to a clean number
p95 = np.percentile(pooled,95)
rec = float(np.ceil(p95))
print(f"RECOMMENDED_DISPLAY_MAX_MM={rec}")
