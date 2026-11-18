"""Builds the non-additive-architecture figure of Chapter 8 (Section: When the Signal Is Not Additive).

Usage: python make_figures_nonadd.py

Reads the committed arm results directly, so the figure cannot drift from the numbers in
Table 8.x: gpu/controls_na_{add,addcal,dom,epi,mix}.json in the code repository, each the
mean over five cohorts. Point REPO at that repository if it is not alongside this one.

All faces are simulated and the genetic architecture is one we imposed; the figure is a
property of the pipeline, not a statement about human faces.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.environ.get("FACECON_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "facecon_repo"))

INK, MUTED, GRID = "#0b0b0b", "#6b6a63", "#e4e3dd"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

ARMS = [("addcal", "Additive"), ("dom", "Dominance"), ("epi", "Epistasis"), ("mix", "Mixed")]
ROWS = [("+SNP-screen", "Additive screen", S[1]),
        ("+SNP-screen-NA", "Two-d.f. screen", S[0]),
        ("ORACLE", "Oracle, additive codes", MUTED),
        ("ORACLE-NA", "Oracle, true codes and pairs", S[2])]

gains = {}
for key, _ in ARMS:
    lm = json.load(open(os.path.join(REPO, "gpu", f"controls_na_{key}.json")))["ladder_mean"]
    base = lm["cov+PC"]
    gains[key] = {r: lm[r] - base for r, _, _ in ROWS if r in lm}

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)

w = 0.2
x = np.arange(len(ARMS))
for j, (r, lab, col) in enumerate(ROWS):
    v = [gains[k].get(r, np.nan) for k, _ in ARMS]
    bars = ax.bar(x + (j - 1.5) * w, v, w, label=lab, color=col,
                  edgecolor="white", linewidth=0.6)
    for b, val in zip(bars, v):
        if np.isfinite(val):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.004, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.4, color=INK)

# the recoverable total is flat across arms once the representation is right
ax.axhline(np.nanmean([gains[k]["ORACLE-NA"] for k, _ in ARMS]), color=S[2],
           lw=0.8, ls=(0, (4, 3)), zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in ARMS])
ax.set_ylabel("Gain in $R^2$ over covariates + components")
ax.set_xlabel("How the planted genetic share of 0.20 is carried")
ax.set_ylim(0, max(max(g.values()) for g in gains.values()) * 1.22)
ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper center", borderaxespad=0.2)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures", "res_nonadd.pdf")
fig.savefig(out)
print("wrote " + out)
for k, lab in ARMS:
    print(f"  {lab:11s} " + "  ".join(f"{r}={gains[k].get(r, float('nan')):+.4f}" for r, _, _ in ROWS))
