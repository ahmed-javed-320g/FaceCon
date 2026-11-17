"""Convert the PLINK files made from 1000 Genomes into FaceCon's genotype file.

Input (made by the plink2 commands in data_prep/README_1KG.md):
    <dir>/kg_main.{bed,bim,fam}    genome-wide panel (common SNPs, unrelated people)
    <dir>/kg_named.{bed,bim,fam}   the named SNPs in data_prep/named_snps.txt
    <dir>/all_hg38.psam            sample sheet with sex and super-population

Output:
    outputs/kg_geno/genotypes_1kg.npz   genotypes + named markers, aligned
    outputs/kg_geno/build_report.json   what was found, how alleles were matched,
                                        frequency checks, colour predictions by group

Checks printed (and stored):
  * each named SNP is matched to its published allele; the frequency of that
    allele among Europeans is shown next to the published European frequency,
    so a wrong strand or flipped allele shows up as a mismatch;
  * HIrisPlex-S colour predictions per super-population. These are a coarse
    plausibility check (e.g. blue eyes should be common only in Europeans),
    NOT an accuracy measurement: 1000 Genomes has no colour labels.

    python scripts/build_1kg.py --dir D:\\FaceCon\\1000G
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.io.plink_bed import read_bed, match_allele, count_allele   # noqa: E402


def read_psam(path):
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    head = lines[0].lstrip("#").split()
    low = [h.lower() for h in head]
    try:
        i_iid = low.index("iid")
    except ValueError:
        raise SystemExit(f"{path}: no IID column (columns: {head})")
    i_sp = next((i for i, h in enumerate(low) if "superpop" in h), None)
    if i_sp is None:
        raise SystemExit(f"{path}: no super-population column (columns: {head})")
    i_sex = low.index("sex") if "sex" in low else None
    out = {}
    for l in lines[1:]:
        f = l.split()
        out[f[i_iid]] = (f[i_sp], f[i_sex] if i_sex is not None else "0")
    return out


def align_named(dos, bim, rsids, targets, label):
    """Pick, for each rsID, the row whose alleles contain the target; count it."""
    n = dos.shape[0]
    cols, rows = [], []
    for rs, tgt in zip(rsids, targets):
        hit = np.where(bim["id"] == rs)[0]
        rec = {"rsid": rs, "target_allele": tgt, "found": False}
        for j in hit:
            which, rule = match_allele(bim["a1"][j], bim["a2"][j], tgt)
            if which:
                rec.update(found=True, rule=rule, chrom=str(bim["chrom"][j]), pos=int(bim["pos"][j]),
                           alleles=f'{bim["a1"][j]}/{bim["a2"][j]}')
                cols.append(count_allele(dos[:, j], which))
                break
            rec["why_not"] = rule
        if not rec["found"]:
            rec.setdefault("why_not", "rsID not present in the 1000 Genomes file")
            cols.append(np.full(n, -1, dtype=np.int8))
        rows.append(rec)
    nf = sum(r["found"] for r in rows)
    print(f"  {label}: {nf}/{len(rows)} found")
    for r in rows:
        if not r["found"]:
            print(f"    MISSING {r['rsid']}: {r['why_not']}")
    return np.stack(cols, axis=1).astype(np.int8), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with kg_main.*, kg_named.*, all_hg38.psam")
    ap.add_argument("--out", default="outputs/kg_geno")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = lambda name: os.path.join(a.dir, name)

    print("reading genome-wide panel ...", flush=True)
    G, bim, iid, fam_sex = read_bed(P("kg_main"))
    print(f"  {G.shape[0]} people x {G.shape[1]} SNPs, missing rate {np.mean(G < 0):.5f}")
    print("reading named SNPs ...", flush=True)
    Dn, bimn, iidn, _ = read_bed(P("kg_named"))
    if list(iidn) != list(iid):
        raise SystemExit("kg_main and kg_named list different people or a different order; "
                         "rerun both plink2 commands with the same --remove file")

    ps = read_psam(P("all_hg38.psam"))
    sp_raw = np.array([ps.get(i, ("NA", "0"))[0] for i in iid], dtype=object)
    names = sorted(set(sp_raw))
    sp = np.array([names.index(s) for s in sp_raw], dtype=np.int64)
    sex = np.where(fam_sex == 1, 1, np.where(fam_sex == 2, 0, -1))
    for k, i in enumerate(iid):                       # fall back to the psam if the fam lacks sex
        if sex[k] < 0:
            s = ps.get(i, ("", "0"))[1]
            sex[k] = 1 if s == "1" else 0 if s == "2" else 0
    print(f"  super-populations: {dict(zip(names, np.bincount(sp).tolist()))}")
    print(f"  males: {int((sex == 1).sum())}  females: {int((sex == 0).sum())}")

    mk = json.load(open(os.path.join(ROOT, "src/appearance/hirisplex_model/markers.json")))["markers"]
    hps, hps_rows = align_named(Dn, bimn, [m["rsid"] for m in mk],
                                [m["input_allele"] for m in mk], "HIrisPlex-S markers")
    ge = json.load(open(os.path.join(ROOT, "src/synth/gwas_effects.json")))["effects"]
    uniq, eff_allele, pub_maf = [], [], []
    for e in ge:
        if e["rsid"] not in uniq:
            uniq.append(e["rsid"]); eff_allele.append(e["effect_allele"]); pub_maf.append(e["maf"])
    gw, gw_rows = align_named(Dn, bimn, uniq, eff_allele, "Shaffer 2016 lead SNPs")

    # frequency check: published (European) frequency vs 1000 Genomes Europeans
    eur = sp == names.index("EUR") if "EUR" in names else np.ones(len(sp), bool)
    print("  effect-allele frequency, published (European) vs 1000 Genomes EUR:")
    for r, col, pm in zip(gw_rows, gw.T, pub_maf):
        if r["found"]:
            c = col[eur]; c = c[c >= 0]
            males = (sex[eur][col[eur] >= 0] == 1)
            # chrX: males carry one copy, coded 0/2 -> count as 0/1 copies of 1 chromosome
            denom = 2 * len(c) - (males.sum() if r.get("chrom") in ("X", "23", "chrX") else 0)
            num = c.sum() - (c[males].sum() / 2 if r.get("chrom") in ("X", "23", "chrX") else 0)
            f = float(num / max(denom, 1))
            r["eur_freq_1kg"] = round(f, 3); r["published_freq"] = pm
            flag = "  <-- check" if abs(f - pm) > 0.15 else ""
            print(f"    {r['rsid']:13s} {r['target_allele']:>6s}  published {pm:.3f}  1KG-EUR {f:.3f}{flag}")

    # appearance plausibility by super-population (not an accuracy measure)
    from src.appearance.hirisplex import HirisplexPredictor
    X = hps.astype(np.int16).copy(); X[X < 0] = 0
    lab = HirisplexPredictor().predict(X)["label"]
    app = {}
    print("  HIrisPlex-S predicted categories by super-population (% of people):")
    for g, nm in enumerate(names):
        m = sp == g; app[nm] = {}
        line = f"    {nm}: "
        for trait in ("eye", "hair", "skin"):
            u, c = np.unique(np.asarray(lab[trait])[m], return_counts=True)
            d = {str(x): round(100 * cc / m.sum(), 1) for x, cc in zip(u, c)}
            app[nm][trait] = d
            top = max(d, key=d.get)
            line += f"{trait} {top} {d[top]}%  "
        print(line)

    np.savez_compressed(os.path.join(a.out, "genotypes_1kg.npz"),
                        genotypes=G, snp_id=bim["id"], chrom=bim["chrom"], pos=bim["pos"],
                        sex=sex.astype(np.int8), superpop_code=sp,
                        superpop_names=np.array(names, dtype=object),
                        hps_markers=hps, hps_found=np.array([r["found"] for r in hps_rows]),
                        gwas_rsids=np.array(uniq, dtype=object), gwas_dosage=gw,
                        gwas_found=np.array([r["found"] for r in gw_rows]))
    json.dump({"n_people": int(G.shape[0]), "n_snps": int(G.shape[1]),
               "missing_rate": float(np.mean(G < 0)),
               "superpop_counts": dict(zip(names, np.bincount(sp).tolist())),
               "hirisplex_markers": hps_rows, "shaffer_snps": gw_rows,
               "appearance_by_superpop_NOT_accuracy": app},
              open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
    print(f"wrote {a.out}/genotypes_1kg.npz and build_report.json")


if __name__ == "__main__":
    main()
