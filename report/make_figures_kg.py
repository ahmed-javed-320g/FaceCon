"""Builds the real-genotype figure of Chapter 8.
Usage: python make_figures_kg.py

Values are the recorded runs on the GPU machine, now committed to the code repository
as gpu/controls_kg.json, gpu/controls_kg_hidden.json and gpu/report_back_kg.json, and
checked against them on 2026-11-18 (1000 Genomes genotypes, simulated faces; Chapter 8).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#6b6a63", "#e4e3dd"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.2))
for ax in (a1, a2):
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- left: what SNPs add beyond covariates and ancestry components ----------
names = ["Permuted genotypes", "256 SNP principal comp.", "Screened, causal hidden",
         "Screened SNPs", "Oracle (causal SNPs)"]
vals = [-0.001, 0.001, 0.025, 0.100, 0.114]
cols = [MUTED, MUTED, S[3], S[0], S[2]]
a1.barh(range(5), vals, height=0.62, color=cols)
for i, v in enumerate(vals):
    a1.text(v + 0.004 if v >= 0 else 0.004, i, f"{v:+.3f}", va="center",
            ha="left", fontsize=8.5, color=INK)
a1.axvline(0, color=MUTED, lw=0.8)
a1.set_yticks(range(5)); a1.set_yticklabels(names, fontsize=8.5, color=INK)
a1.set_xlabel("Gain in test $R^2$", fontsize=9.5)
a1.set_title("Linear analysis", fontsize=10, color=INK)
a1.set_xlim(-0.025, 0.155)
a1.set_ylim(-0.75, 4.6)

# ---- right: models against the linear reference -----------------------------
mnames = ["Diffusion", "Transformer", "Contrastive", "MLP", "Ridge on demo. + SNPs"]
mvals = [-0.045, 0.333, 0.338, 0.366, 0.443]
mcols = [S[1], S[1], S[1], S[1], S[0]]
a2.barh(range(5), mvals, height=0.62, color=mcols)
for i, v in enumerate(mvals):
    a2.text(v + 0.012 if v >= 0 else 0.012, i, f"{v:+.3f}", va="center",
            ha="left", fontsize=8.5, color=INK)
a2.axvline(0.338, color=INK, lw=0.9, ls="--")
a2.text(0.338, -0.62, "demographic\nbaseline", ha="center", va="center", fontsize=7.5, color=MUTED)
a2.axvline(0, color=MUTED, lw=0.8)
a2.set_yticks(range(5)); a2.set_yticklabels(mnames, fontsize=8.5, color=INK)
a2.set_xlabel("Test $R^2$", fontsize=9.5)
a2.set_title("Screened SNPs, all models", fontsize=10, color=INK)
a2.set_xlim(-0.09, 0.60)
a2.set_ylim(-0.95, 4.6)

fig.tight_layout()
fig.savefig("Figures/res_kg.pdf")
plt.close(fig)
print("wrote Figures/res_kg.pdf")
