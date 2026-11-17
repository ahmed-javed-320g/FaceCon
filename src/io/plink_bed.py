"""Minimal reader/writer for PLINK 1 binary genotype files (.bed/.bim/.fam).

No external dependency: the format is documented at
https://www.cog-genomics.org/plink/1.9/formats#bed

Dosages are returned as the COUNT OF ALLELE A1 (bim column 5) per subject:
    2 = A1/A1, 1 = A1/A2, 0 = A2/A2, -1 = missing.
Which allele is A1 does not matter for the genome-wide panel (it is
standardised). For named markers, `count_allele` re-expresses the dosage as
the count of a specific allele (e.g. a published effect allele).
"""
from __future__ import annotations

import numpy as np

_MAGIC = bytes([0x6C, 0x1B, 0x01])          # variant-major .bed
# 2-bit code -> count of A1:  00 hom A1, 01 missing, 10 het, 11 hom A2
_CODE_TO_A1 = np.array([2, -1, 1, 0], dtype=np.int8)
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _lookup():
    tab = np.zeros((256, 4), dtype=np.int8)
    for b in range(256):
        for k in range(4):
            tab[b, k] = _CODE_TO_A1[(b >> (2 * k)) & 3]
    return tab


_LUT = _lookup()


def read_fam(prefix):
    iid, sex = [], []
    for line in open(prefix + ".fam"):
        f = line.split()
        if len(f) >= 6:
            iid.append(f[1]); sex.append(int(f[4]))
    return np.array(iid, dtype=object), np.array(sex, dtype=np.int8)


def read_bim(prefix):
    chrom, vid, pos, a1, a2 = [], [], [], [], []
    for line in open(prefix + ".bim"):
        f = line.split()
        if len(f) >= 6:
            chrom.append(f[0]); vid.append(f[1]); pos.append(int(f[3]))
            a1.append(f[4].upper()); a2.append(f[5].upper())
    return dict(chrom=np.array(chrom, dtype=object), id=np.array(vid, dtype=object),
                pos=np.array(pos, dtype=np.int64), a1=np.array(a1, dtype=object),
                a2=np.array(a2, dtype=object))


def read_bed(prefix):
    """Return (dosage int8 (N, M) with -1 = missing, bim dict, iid, sex)."""
    iid, sex = read_fam(prefix)
    bim = read_bim(prefix)
    n, m = len(iid), len(bim["id"])
    nb = (n + 3) // 4
    raw = np.fromfile(prefix + ".bed", dtype=np.uint8)
    if raw.size < 3 or bytes(raw[:3]) != _MAGIC:
        raise ValueError(f"{prefix}.bed is not a variant-major PLINK 1 .bed file")
    body = raw[3:]
    if body.size != nb * m:
        raise ValueError(f"{prefix}.bed size mismatch: expected {nb * m} bytes after the "
                         f"header for {n} samples x {m} variants, found {body.size}")
    out = np.empty((n, m), dtype=np.int8)
    step = max(1, 20_000_000 // max(nb, 1))          # decode in chunks (bounded memory)
    for s in range(0, m, step):
        blk = body[s * nb:(s + min(step, m - s)) * nb].reshape(-1, nb)
        dec = _LUT[blk].reshape(blk.shape[0], nb * 4)[:, :n]
        out[:, s:s + blk.shape[0]] = dec.T
    return out, bim, iid, sex


def write_bed(prefix, dosage, bim, iid, sex):
    """Write a PLINK 1 .bed/.bim/.fam (used by tests to build fixtures)."""
    n, m = dosage.shape
    with open(prefix + ".fam", "w") as f:
        for i in range(n):
            f.write(f"{iid[i]} {iid[i]} 0 0 {int(sex[i])} -9\n")
    with open(prefix + ".bim", "w") as f:
        for j in range(m):
            f.write(f"{bim['chrom'][j]}\t{bim['id'][j]}\t0\t{int(bim['pos'][j])}\t"
                    f"{bim['a1'][j]}\t{bim['a2'][j]}\n")
    code = np.full((n, m), 1, dtype=np.uint8)                 # missing
    code[dosage == 2] = 0; code[dosage == 1] = 2; code[dosage == 0] = 3
    nb = (n + 3) // 4
    pad = np.full((nb * 4 - n, m), 1, dtype=np.uint8)
    c = np.vstack([code, pad]).T.reshape(m, nb, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    with open(prefix + ".bed", "wb") as f:
        f.write(_MAGIC); f.write(packed.tobytes())


def match_allele(a1, a2, target):
    """Which bim allele is `target`? Returns ("a1"|"a2", rule) or (None, reason).

    Order of rules: exact string; strand complement (single-base, non-palindromic
    sites only); for indels, the published notation often gives only the inserted
    base(s), so a target that matches neither allele is taken to mean the
    LONGER allele when the target is a single base or the same length as it.
    """
    t = str(target).upper()
    if t == a1:
        return "a1", "exact"
    if t == a2:
        return "a2", "exact"
    if len(a1) == len(a2) == len(t) == 1:
        if {a1, a2} in ({"A", "T"}, {"C", "G"}):
            return None, "palindromic site, strand cannot be resolved"
        c = _COMP.get(t)
        if c == a1:
            return "a1", "strand-flipped"
        if c == a2:
            return "a2", "strand-flipped"
        return None, f"allele {t} not in {a1}/{a2}"
    if len(a1) != len(a2):
        longer = "a1" if len(a1) > len(a2) else "a2"
        if len(t) == 1 or len(t) == max(len(a1), len(a2)):
            return longer, "indel: longer allele"
        if len(t) == min(len(a1), len(a2)):
            return ("a2" if longer == "a1" else "a1"), "indel: shorter allele"
    return None, f"allele {t} not in {a1}/{a2}"


def count_allele(dosage_a1, which):
    """Re-express an A1-count column as the count of the chosen allele (-1 kept)."""
    d = dosage_a1.astype(np.int8).copy()
    if which == "a2":
        ok = d >= 0
        d[ok] = 2 - d[ok]
    return d
