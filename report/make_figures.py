"""Builds the Chapter 9 figures from the recorded result files.
Usage: python make_figures.py <report_back.json> <controls.json>
All data are SIMULATED (see Chapter 9)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1]))["results"]; C = json.load(open(sys.argv[2]))
INK, MUTED, GRID = "#0b0b0b", "#6b6a63", "#e4e3dd"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]      # validated categorical order
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
def base(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- Fig 1: baseline ladder -------------------------------------------------
lm, ls = C["ladder_mean"], C["ladder_sd"]
keys = ["mean", "cov", "cov+PC", "+SNP-PCA256", "+SNP-screen", "ORACLE"]
names = ["Mean face", "Covariates", "Covariates +\nancestry PCs", "+ 256 SNP\nprincipal comp.", "+ screened\nSNPs", "Oracle\n(true causal SNPs)"]
fig, ax = plt.subplots(figsize=(6.3, 3.1)); base(ax)
v = [lm[k] for k in keys]
ax.bar(range(len(keys)), v, yerr=[ls[k] for k in keys], width=0.58, color=S[0], ecolor=INK, capsize=2.5, error_kw={"lw": 0.8})
for i, x in enumerate(v):
    ax.text(i, max(x, 0) + 0.022, f"{x:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(len(keys))); ax.set_xticklabels(names, fontsize=8.5, color=INK)
ax.set_ylabel("Test $R^2$ (simulated data)"); ax.set_ylim(-0.03, 0.58)
fig.tight_layout(); fig.savefig("Figures/res_ladder.pdf"); plt.close(fig)

# ---- Fig 2: model comparison, two feature sets -------------------------------
models = ["mlp", "contrastive", "transformer", "diffusion"]; mn = ["MLP", "Contrastive", "Transformer", "Diffusion"]
A, B = R["A matched N=3118 | pca"], R["A matched N=3118 | screen"]
fig, ax = plt.subplots(figsize=(6.3, 3.3)); base(ax); w = 0.36
for j, (D, lab, c, h) in enumerate([(A, "256 SNP principal components", S[0], ""), (B, "256 screened SNPs", S[1], "////")]):
    m = [np.mean(D[k]) for k in models]; sd = [np.std(D[k]) for k in models]
    xs = np.arange(4) + (j - 0.5) * (w + 0.03)
    ax.bar(xs, m, w, yerr=sd, color=c, hatch=h, edgecolor="white", linewidth=0.6, ecolor=INK, capsize=2, error_kw={"lw": 0.8}, label="Model given " + lab)
    for x, val in zip(xs, m):
        ax.text(x, val + 0.012 if val >= 0 else val - 0.012, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8.5, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)
d = A["demographics"][0]
ax.axhline(d, color=INK, lw=1.0, ls="--", label=f"Demographics-only baseline ({d:.3f})", zorder=2)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(mn, color=INK)
ax.set_ylabel("Test $R^2$ (simulated data)"); ax.set_ylim(-0.44, 0.50)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.tight_layout(); fig.savefig("Figures/res_models.pdf"); plt.close(fig)

# ---- Fig 3: learning curve ---------------------------------------------------
ns = [2182, 4000, 7000]; keysB = [f"B large, train_n={n} | screen" for n in ns]
series = [("demographics", "Demographics only", S[0], "o", "--"), ("mlp", "MLP", S[1], "s", "-"), ("contrastive", "Contrastive", S[2], "^", "-"),
          ("transformer", "Transformer", S[3], "D", "-"), ("diffusion", "Diffusion", S[4], "v", "-")]
fig, ax = plt.subplots(figsize=(6.3, 3.4)); base(ax)
off = {"demographics": -0.022, "mlp": 0.02, "contrastive": -0.017, "transformer": 0.014, "diffusion": 0.0}
for k, lab, c, mk, lsy in series:
    y = [np.mean(R[b][k]) for b in keysB]
    ax.plot(ns, y, color=c, lw=2, ls=lsy, marker=mk, ms=6.5, mec="white", mew=1.2, label=lab)
    ax.text(7150, y[-1] + off[k], f"{lab} {y[-1]:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_xticks(ns); ax.set_xticklabels(["2,182\n(matched cohort size)", "4,000", "7,000"]); ax.set_xlim(1900, 9900)
ax.set_xlabel("Training subjects"); ax.set_ylabel("Test $R^2$ (simulated data)"); ax.set_ylim(-0.02, 0.48)
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
fig.tight_layout(); fig.savefig("Figures/res_scaling.pdf"); plt.close(fig)
print("figures written")
