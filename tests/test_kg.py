"""Tests for the 1000 Genomes path, run on a small synthetic fixture (no download needed)."""
import json, os, subprocess, sys
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.io.plink_bed import read_bed, write_bed, match_allele, count_allele   # noqa: E402

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def test_bed_roundtrip_with_missing():
    rng = np.random.default_rng(0)
    for n in (1, 5, 8, 13):                       # covers every padding case
        d = rng.integers(-1, 3, size=(n, 7)).astype(np.int8)
        bim = dict(chrom=["1"] * 7, id=[f"s{j}" for j in range(7)], pos=list(range(7)),
                   a1=["A"] * 7, a2=["G"] * 7)
        p = os.path.join("/tmp", f"rt{n}")
        write_bed(p, d, bim, [f"i{i}" for i in range(n)], [1] * n)
        got, b, iid, sex = read_bed(p)
        assert np.array_equal(got, d) and list(b["id"]) == bim["id"] and len(iid) == n


def test_match_allele_rules():
    assert match_allele("T", "C", "T") == ("a1", "exact")
    assert match_allele("C", "T", "T") == ("a2", "exact")
    assert match_allele("C", "T", "A") == ("a2", "strand-flipped")          # A pairs with T
    assert match_allele("A", "T", "C")[0] is None                            # palindromic
    assert match_allele("C", "CA", "A") == ("a2", "indel: longer allele")    # HIrisPlex insertion notation
    assert match_allele("CTATA", "C", "CTATA") == ("a1", "exact")
    assert match_allele("C", "CA", "CAAA")[0] is None                        # wrong indel length
    d = np.array([2, 1, 0, -1], dtype=np.int8)
    assert list(count_allele(d, "a2")) == [0, 1, 2, -1]


def _fixture(tmp, n=320, m=1500, seed=1):
    """kg_main / kg_named / all_hg38.psam shaped like the real files."""
    rng = np.random.default_rng(seed)
    pops = np.array(["AFR", "AMR", "EAS", "EUR", "SAS"])
    sp = rng.integers(0, 5, n); sex = rng.integers(1, 3, n)            # 1 male, 2 female
    iid = [f"HG{i:05d}" for i in range(n)]
    base = rng.uniform(0.02, 0.5, m)
    f = np.clip(base[None, :] + rng.uniform(-0.15, 0.15, (5, m)), 0.01, 0.99)
    G = rng.binomial(2, f[sp]).astype(np.int8)
    G[rng.random(G.shape) < 0.001] = -1
    write_bed(os.path.join(tmp, "kg_main"), G,
              dict(chrom=["1"] * m, id=[f"1:{j}" for j in range(m)], pos=list(range(m)),
                   a1=["A"] * m, a2=["G"] * m), iid, sex)
    with open(os.path.join(tmp, "all_hg38.psam"), "w") as fh:
        fh.write("#IID\tPAT\tMAT\tSEX\tSuperPop\tPopulation\n")
        for i in range(n):
            fh.write(f"{iid[i]}\t0\t0\t{sex[i]}\t{pops[sp[i]]}\tX\n")

    mk = json.load(open(os.path.join(ROOT, "src/appearance/hirisplex_model/markers.json")))["markers"]
    ge = json.load(open(os.path.join(ROOT, "src/synth/gwas_effects.json")))["effects"]
    targets = [(m_["rsid"], m_["input_allele"]) for m_ in mk]
    for e in ge:
        if e["rsid"] not in [t[0] for t in targets]:
            targets.append((e["rsid"], e["effect_allele"]))
    other = {"A": "G", "G": "A", "C": "T", "T": "C"}
    truth, rows, cols = {}, [], []
    for k, (rs, t) in enumerate(targets):
        if rs == "rs1805006":                                   # deliberately absent
            continue
        cnt = rng.binomial(2, rng.uniform(0.05, 0.6), n).astype(np.int8)   # copies of target
        if rs == "rs312262906":                                 # insertion, notation 'A'
            a1, a2, which = "C", "CA", "a2"
        elif len(t) > 1:                                        # Shaffer indels, full string
            a1, a2, which = t, "C", "a1"
        elif rs == "rs885479":                                  # reverse strand in the file
            a1, a2, which = COMP[other[t]], COMP[t], "a2"
        elif k % 2:
            a1, a2, which = t, other[t], "a1"
        else:
            a1, a2, which = other[t], t, "a2"
        chrom = "X" if rs == "rs11093404" else "2"
        if chrom == "X":
            male = sex == 1
            cnt[male] = 2 * rng.binomial(1, 0.3, male.sum())   # haploid males stored 0/2
        a1_count = cnt if which == "a1" else (2 - cnt)
        truth[rs] = cnt
        rows.append((chrom, rs, k, a1, a2)); cols.append(a1_count)
    write_bed(os.path.join(tmp, "kg_named"), np.stack(cols, 1).astype(np.int8),
              dict(chrom=[r[0] for r in rows], id=[r[1] for r in rows], pos=[r[2] for r in rows],
                   a1=[r[3] for r in rows], a2=[r[4] for r in rows]), iid, sex)
    return truth, sp, sex


def test_build_generate_and_load(tmp_path):
    tmp = str(tmp_path)
    truth, sp, sex = _fixture(tmp)
    geno = os.path.join(tmp, "geno")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/build_1kg.py"),
                        "--dir", tmp, "--out", geno], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    K = np.load(os.path.join(geno, "genotypes_1kg.npz"), allow_pickle=True)
    mk = [m_["rsid"] for m_ in json.load(open(os.path.join(ROOT, "src/appearance/hirisplex_model/markers.json")))["markers"]]
    assert K["hps_found"].sum() == 40 and not K["hps_found"][mk.index("rs1805006")]
    for j, rs in enumerate(mk):                                  # every allele rule counted right
        if rs in truth:
            assert np.array_equal(K["hps_markers"][:, j], truth[rs]), rs
    for j, rs in enumerate(K["gwas_rsids"]):
        assert K["gwas_found"][j] and np.array_equal(K["gwas_dosage"][:, j], truth[rs]), rs
    assert list(K["superpop_names"]) == ["AFR", "AMR", "EAS", "EUR", "SAS"]
    assert np.array_equal(K["sex"], np.where(sex == 1, 1, 0))

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config/kg.yaml")))
    cfg["data"]["genotype_source"] = os.path.join(geno, "genotypes_1kg.npz")
    cfg["data"]["out_dir"] = os.path.join(tmp, "out")
    cfg["data"]["n_causal_snps"] = 60
    cp = os.path.join(tmp, "kg_test.yaml"); yaml.safe_dump(cfg, open(cp, "w"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/generate_matched.py"), cp],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    d = np.load(os.path.join(tmp, "out", "dataset.npz"), allow_pickle=True)
    assert d["genotypes"].shape == (320, 1500) and d["demographics"].shape[1] == 10
    assert np.array_equal(d["demographics"][:, 0], np.where(sex == 1, 1, 0))     # real sex used
    man = json.load(open(os.path.join(tmp, "out", "manifest.json")))
    assert man["mode"].startswith("real_genotypes") and man["gwas_effects"]["n_loci"] == 14

    from gpu.data import FaceConData
    D = FaceConData(ROOT, snp_pca=16, data_dir=os.path.join(tmp, "out"), features="screen")
    assert D.N == 320 and D.snp_feat.shape == (320, 16) and np.isfinite(D.snp_feat).all()

    from gpu.controls import make_data, ladder, r2
    from src.common import load_config
    c2 = load_config(cp)
    M = make_data(c2, 3)
    P, bt, _ = ladder(M, do_pca256=False, hide_causal=True)
    assert np.isfinite(r2(bt, P["+SNP-screen"]))
    M2 = make_data(c2, 3, n=200)
    assert len(M2["B"]) == 200


def test_sparse_effects_only_touch_chosen_dims():
    from src.synth.generate import _compose_faces
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config/kg.yaml")))
    cfg["data"]["n_causal_snps"] = 40
    rng = np.random.default_rng(0)
    G = rng.binomial(2, 0.3, (500, 400)).astype(float)
    demo = rng.normal(size=(500, 3))
    _, gt = _compose_faces(cfg, G, demo, ["sex", "a", "b"], 50)
    assert ((gt.genetic_weights != 0).sum(1) == 5).all()
