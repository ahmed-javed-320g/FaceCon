"""Pick a random, reproducible subset of SNP IDs from a plink2 --write-snplist file.

    python data_prep/select_snps.py kg_common.snplist kg_selected.txt 100000
"""
import sys
import numpy as np

src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = [l.strip() for l in open(src) if l.strip() and not l.startswith("#")]
if len(ids) < k:
    raise SystemExit(f"only {len(ids)} SNPs passed the filters, fewer than the {k} requested")
pick = np.sort(np.random.default_rng(20261016).choice(len(ids), k, replace=False))
open(dst, "w").write("\n".join(ids[i] for i in pick) + "\n")
print(f"{len(ids)} SNPs passed the filters; wrote {k} randomly chosen IDs to {dst}")
