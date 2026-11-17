"""Generate the matched-spec synthetic dataset (config/real_match.yaml).

Matches the real target (dbGaP phs000949 / 3DFN, Shaffer 2016):
  - N = 3118
  - demographics/covariates = sex, age, age^2, height, weight (+ ancestry kept
    as a documented confounding stress-test; facial_size stored with phenotype #2)
  - phenotype #1: FLAME-stand-in beta (50)
  - phenotype #2: 20 linear anthropometric measurements (from mesh landmarks),
    plus facial_size = geometric mean of the 20 (the real GWAS size covariate)
  - the 41 real HIrisPlex-S marker loci embedded as named rsID genotype features,
    exported to a batch file for the official Erasmus MC webtool.

Reuses the existing genotype/face composition so the shape pipeline is unchanged
in spirit; only the subject count, covariate set, and appearance source differ.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.common import load_config, rng_for                      # noqa: E402
from src.faces.morphable import load_face_model                  # noqa: E402
from src.synth.generate import _draw_genotypes, _compose_faces   # noqa: E402

# anthropometric landmark name -> iBUG-68 index (ICT model uses iBUG-68)
IBUG = {
    "en_r": 39, "en_l": 42, "ex_r": 36, "ex_l": 45,
    "al_r": 31, "al_l": 35, "n": 27, "prn": 30, "sn": 33, "gn": 8,
    "sto": 66, "ch_r": 48, "ch_l": 54,
    "zy_r": 2, "zy_l": 14, "go_r": 4, "go_l": 12, "t_r": 0, "t_l": 16,
}


def _draw_hps_markers(cfg, ancestry, rsids):
    """Draw 0/1/2 dosages at the 41 real HIrisPlex-S marker loci, ancestry-structured."""
    rng = rng_for(cfg["seed"], "hps_markers")
    n = ancestry.shape[0]
    P = len(rsids)
    ng = int(cfg["data"]["ancestry"]["n_groups"])
    sfrac = float(cfg["appearance"].get("marker_structured_fraction", 0.6))
    fpath = os.path.join(ROOT, "src/appearance/hirisplex_model/marker_freqs.json")
    if os.path.isfile(fpath):
        cal = json.load(open(fpath))
        fr = np.asarray(cal["freq_by_group"], dtype=float)
        if cal["rsids"] == list(rsids) and fr.shape == (ng, P):
            return rng.binomial(2, fr[ancestry]).astype(np.int8)
    base = rng.uniform(0.05, 0.6, size=P)
    structured = rng.random(P) < sfrac
    freq = np.tile(base, (ng, 1))
    for g in range(ng):
        shift = rng.uniform(-0.4, 0.4, size=P) * structured
        freq[g] = np.clip(base + shift, 0.01, 0.99)
    return rng.binomial(2, freq[ancestry]).astype(np.int8)


def _draw_matched_demographics(cfg, ancestry, sex=None):
    """sex, age_z, age2_z, height_z, weight_z + ancestry one-hot (confounder kept).

    `sex` may be supplied (real genotypes carry real sex, 1 = male); otherwise it
    is drawn. Age, height and weight are always simulated.
    """
    rng = rng_for(cfg["seed"], "demographics")
    n = ancestry.shape[0]
    ng = int(cfg["data"]["ancestry"]["n_groups"])
    dd = cfg["data"]["demographics"]

    drawn = rng.integers(0, 2, size=n).astype(np.float64)        # 0/1
    sex = drawn if sex is None else np.asarray(sex, dtype=np.float64)
    age = rng.uniform(dd["age_min"], dd["age_max"], size=n)
    # height/weight: plausibly correlated with sex (+ mild ancestry + noise)
    sex_c = (sex - 0.5) * 2.0
    anc_c = (ancestry - ancestry.mean())
    height = 0.80 * sex_c + 0.15 * anc_c + rng.normal(size=n)
    weight = 0.55 * height + 0.45 * sex_c + rng.normal(size=n)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    age_z, age2_z, height_z, weight_z = z(age), z(age**2), z(height), z(weight)
    anc_oh = np.zeros((n, ng))
    anc_oh[np.arange(n), ancestry] = 1.0

    cols = [sex, age_z, age2_z, height_z, weight_z] + [anc_oh[:, g] for g in range(ng)]
    names = ["sex", "age_z", "age2_z", "height_z", "weight_z"] + \
            [f"ancestry_{g}" for g in range(ng)]
    return np.stack(cols, axis=1), names, age


def _compute_measures(cfg, fm, beta):
    """Decode only the landmark vertices and compute the 20 linear measures (mm)."""
    V = fm.mean.shape[0]
    basis3 = fm.basis.reshape(fm.n_dims, V, 3)                   # (K,V,3)
    lmk68 = np.asarray(fm.landmark_idx)                          # iBUG-68 -> mesh vert

    names = [m["name"] for m in cfg["anthropometry"]["measures"]]
    pts = {m["name"]: (m["a"], m["b"]) for m in cfg["anthropometry"]["measures"]}

    # which mesh vertices we actually need
    pt_names = sorted({p for ab in pts.values() for p in ab})
    vert_of = {p: int(lmk68[IBUG[p]]) for p in pt_names}
    need = sorted(set(vert_of.values()))
    nmap = {v: i for i, v in enumerate(need)}

    mean_need = fm.mean[need].astype(np.float64)                 # (n_need,3)
    basis_need = basis3[:, need, :].astype(np.float64)           # (K,n_need,3)
    # per subject: coords = mean + beta @ basis_need  -> (N,n_need,3)
    coords = mean_need[None] + np.einsum("nk,kij->nij", beta, basis_need)
    coords *= float(fm.mm_per_unit)                              # nominal mm

    N = beta.shape[0]
    M = np.zeros((N, len(names)), dtype=np.float64)
    for j, nm in enumerate(names):
        a, b = pts[nm]
        ca = coords[:, nmap[vert_of[a]], :]
        cb = coords[:, nmap[vert_of[b]], :]
        M[:, j] = np.linalg.norm(ca - cb, axis=1)
    return M, names


def _draw_gwas_markers(cfg, ancestry, effects, sex):
    """Draw dosages at the published lead SNPs: ONE column per unique rsID.

    A SNP that is the lead for several traits (e.g. rs12792834) shares a single
    dosage column. chrX SNPs (locus_chr == 23) are hemizygous in males
    (sex == 1), coded 0/2 as in standard GWAS practice. Frequencies get a mild
    synthetic ancestry shift; this is a stress-test, not population data.
    """
    rng = rng_for(cfg["seed"], "gwas_markers")
    ng = int(cfg["data"]["ancestry"]["n_groups"])
    uniq, col_of = [], []
    for e in effects:
        if e["rsid"] not in uniq:
            uniq.append(e["rsid"])
        col_of.append(uniq.index(e["rsid"]))
    first = {r: next(e for e in effects if e["rsid"] == r) for r in uniq}
    base = np.array([float(first[r]["maf"]) for r in uniq])
    is_x = np.array([int(first[r].get("locus_chr", 0)) == 23 for r in uniq])
    structured = rng.random(len(uniq)) < 0.5
    freq = np.tile(base, (ng, 1))
    for g in range(ng):
        shift = rng.uniform(-0.15, 0.15, size=len(uniq)) * structured
        freq[g] = np.clip(base + shift, 0.01, 0.99)
    p = freq[ancestry]
    dos = rng.binomial(2, p).astype(np.int8)
    hemi = (2 * rng.binomial(1, p)).astype(np.int8)
    male = (np.asarray(sex) == 1)[:, None] & is_x[None, :]
    dos[male] = hemi[male]
    return dos, uniq, col_of


def _inject_gwas_effects(measures, names, dosages, effects, col_of, var_per_locus):
    """Add published locus->measure signal onto the mesh-derived measures.

    Variance-preserving per measure: z_new = sqrt(1-V)*z + sum_l sign*sqrt(v_l)*d_l.
    NOTE: after injection a measure is no longer a pure function of the mesh
    (beta). The injected loci therefore exist ONLY in the measurement phenotype;
    the shape target beta, and so every shape model, is unaffected by them.
    """
    from collections import defaultdict
    by_measure = defaultdict(list)
    for i, e in enumerate(effects):
        if e["measure"] in names:
            by_measure[e["measure"]].append(i)
    applied = []
    for nm, idxs in by_measure.items():
        j = names.index(nm)
        col = measures[:, j].astype(np.float64)
        mu, sd = col.mean(), col.std() + 1e-9
        z = (col - mu) / sd
        vs = [float(effects[i].get("var_explained", var_per_locus)) for i in idxs]
        vtot = min(0.9, sum(vs))
        znew = np.sqrt(max(1e-6, 1.0 - vtot)) * z
        for i, v in zip(idxs, vs):
            dd = dosages[:, col_of[i]].astype(np.float64)
            dd = (dd - dd.mean()) / (dd.std() + 1e-9)
            sign = 1.0 if float(effects[i].get("beta_mm", 1.0)) >= 0 else -1.0
            znew = znew + sign * np.sqrt(v) * dd
            applied.append({"measure": nm, "rsid": effects[i].get("rsid"),
                            "chr": effects[i].get("locus_chr"),
                            "trait": effects[i].get("trait"), "sign": int(sign),
                            "genome_wide_significant": bool(effects[i].get("genome_wide_significant", False)),
                            "var_explained": round(v, 4)})
        measures[:, j] = znew * sd + mu
    return measures, applied


def gwas_recovery_check(measures, names, dosages, rsids, applied, covars):
    """Covariate-adjusted association scan over ALL (SNP, measure) pairs.

    Question   : are the injected SNP->measure links recoverable, and ONLY those?
    Positive   : injected pairs (should be Bonferroni-significant, right sign).
    Negative   : every non-injected pair (false-positive rate should be ~alpha).
    Both sides are residualised on the covariates (sex, age, age^2, height,
    weight, ancestry) so ancestry structure cannot manufacture associations.
    """
    from scipy import stats
    n = measures.shape[0]
    C = np.c_[np.ones(n), covars]
    def resid(M):
        b, *_ = np.linalg.lstsq(C, M, rcond=None)
        R = M - C @ b
        return (R - R.mean(0)) / (R.std(0) + 1e-12)
    Y = resid(measures.astype(float)); X = resid(dosages.astype(float))
    r = (X.T @ Y) / n                                   # (snps, measures)
    dof = n - C.shape[1] - 1
    t = r * np.sqrt(dof / np.clip(1 - r ** 2, 1e-12, None))
    pv = 2 * stats.t.sf(np.abs(t), dof)
    inj = {(a["rsid"], a["measure"]): a["sign"] for a in applied}
    mask = np.zeros_like(pv, dtype=bool)
    rows = []
    bonf = 0.05 / pv.size
    for (rs, nm), sg in inj.items():
        i, j = rsids.index(rs), names.index(nm)
        mask[i, j] = True
        rows.append({"rsid": rs, "measure": nm, "p": float(pv[i, j]),
                     "sign_ok": bool(np.sign(r[i, j]) == sg),
                     "detected": bool(pv[i, j] < bonf)})
    null_p = pv[~mask]
    out = {"n_pairs": int(pv.size), "bonferroni_alpha": bonf,
           "injected_detected": int(sum(x["detected"] for x in rows)),
           "injected_total": len(rows),
           "injected_sign_ok": int(sum(x["sign_ok"] for x in rows)),
           "null_pairs": int(null_p.size),
           "null_fpr_at_0.05": float((null_p < 0.05).mean()),
           "null_bonferroni_hits": int((null_p < bonf).sum()),
           "pairs": rows}
    # acceptance: >=80% of injected links detected with the right sign, the
    # negative control is calibrated (FPR near 5%) and has no Bonferroni hits.
    out["accept"] = bool(out["injected_detected"] >= 0.8 * len(rows)
                         and all(x["sign_ok"] for x in rows if x["detected"])
                         and 0.02 <= out["null_fpr_at_0.05"] <= 0.09
                         and out["null_bonferroni_hits"] == 0)
    return out


def _make_splits(cfg, n):
    rng = rng_for(cfg["seed"], "splits")
    idx = rng.permutation(n)
    s = cfg["split"]
    n_tr, n_va = int(s["train"] * n), int(s["val"] * n)
    return {"train": np.sort(idx[:n_tr]), "val": np.sort(idx[n_tr:n_tr + n_va]),
            "test": np.sort(idx[n_tr + n_va:])}


def main(cfg_path):
    cfg = load_config(cfg_path)
    out = cfg["data"]["out_dir"]
    os.makedirs(out, exist_ok=True)
    fm = load_face_model(cfg)
    k = fm.n_dims

    markers = json.load(open(os.path.join(ROOT, cfg["appearance"]["markers_file"])))
    rsids = [m["rsid"] for m in markers["markers"]]
    src = cfg["data"].get("genotype_source")
    K = None
    if src:
        # REAL genotypes (1000 Genomes, scripts/build_1kg.py). Faces, age, height
        # and weight remain simulated; sex and ancestry come from the samples.
        K = np.load(src if os.path.isabs(src) else os.path.join(ROOT, src), allow_pickle=True)
        graw = K["genotypes"]
        g_obs = graw.astype(np.float32); g_obs[graw < 0] = np.nan
        cm = np.nanmean(g_obs, axis=0)
        genotypes = np.where(np.isnan(g_obs), cm, g_obs)        # used only to compose the truth
        ancestry = K["superpop_code"].astype(np.int64)
        if int(cfg["data"]["ancestry"]["n_groups"]) != len(K["superpop_names"]):
            raise SystemExit("config data.ancestry.n_groups must equal the number of "
                             f"super-populations in {src} ({len(K['superpop_names'])})")
        n = genotypes.shape[0]
        hps = K["hps_markers"].astype(np.int8).copy(); hps[hps < 0] = 0
        demo, demo_names, age = _draw_matched_demographics(cfg, ancestry, sex=K["sex"])
    else:
        # genotypes + ancestry (reused logic), 3118 x 5000
        genotypes, ancestry, freq, structured = _draw_genotypes(cfg)
        n = genotypes.shape[0]
        hps = _draw_hps_markers(cfg, ancestry, rsids)               # (N,41) int8
        demo, demo_names, age = _draw_matched_demographics(cfg, ancestry)
    beta, gt = _compose_faces(cfg, genotypes, demo, demo_names, k)

    # phenotype #2: 20 anthropometric measures + facial_size
    measures, measure_names = _compute_measures(cfg, fm, beta)

    # inject REAL published GWAS effects (Shaffer 2016) onto the measures
    gwas_dos = np.zeros((n, 0), dtype=np.int8)
    gwas_rsids = []
    gwas_applied = []
    gwcfg = cfg.get("gwas_effects", {})
    if gwcfg.get("enable"):
        gw = json.load(open(os.path.join(ROOT, gwcfg["effects_file"])))
        geff = gw["effects"]
        vpl = float(gw.get("var_explained_per_locus", 0.02))
        if K is not None:                       # real dosages at the published SNPs
            found = {r: j for j, r in enumerate(K["gwas_rsids"]) if K["gwas_found"][j]}
            geff = [e for e in geff if e["rsid"] in found]
            gwas_rsids = [r for r in K["gwas_rsids"] if r in found]
            col_of = [gwas_rsids.index(e["rsid"]) for e in geff]
            gd = K["gwas_dosage"][:, [found[r] for r in gwas_rsids]].astype(np.float32)
            for j in range(gd.shape[1]):
                bad = gd[:, j] < 0
                gd[bad, j] = gd[~bad, j].mean()
            gwas_dos = gd
        else:
            gwas_dos, gwas_rsids, col_of = _draw_gwas_markers(cfg, ancestry, geff, demo[:, 0])
        measures, gwas_applied = _inject_gwas_effects(
            measures, measure_names, gwas_dos, geff, col_of, vpl)
    gwas_check = (gwas_recovery_check(measures, measure_names, gwas_dos, gwas_rsids,
                                      gwas_applied, demo[:, :-1])
                  if gwas_applied else None)

    facial_size = np.exp(np.log(measures).mean(axis=1))         # geometric mean
    facial_size_z = (facial_size - facial_size.mean()) / (facial_size.std() + 1e-9)

    splits = _make_splits(cfg, n)
    subject_ids = np.array([f"SUBJ-{i:05d}" for i in range(n)])

    if K is not None:
        g = g_obs                                   # real missingness kept as NaN
    else:
        # inject genotype missingness (0.5%), matching the real-data representation
        rng = rng_for(cfg["seed"], "missingness")
        g = genotypes.copy()
        g[rng.random(g.shape) < 0.005] = np.nan

    np.savez_compressed(
        os.path.join(out, "dataset.npz"),
        subject_ids=subject_ids, genotypes=g, demographics=demo,
        face_params=beta, ancestry=ancestry,
        split_train=splits["train"], split_val=splits["val"], split_test=splits["test"],
        hps_markers=hps, hps_rsids=np.array(rsids, dtype=object),
        anthro_measures=measures, anthro_names=np.array(measure_names, dtype=object),
        facial_size=facial_size_z,
        gwas_markers=gwas_dos, gwas_rsids=np.array(gwas_rsids, dtype=object),
    )
    # ground truth (shape) carried through
    gt_app = {f"appcausal_{kk}": v for kk, v in gt.appearance_causal.items()}
    np.savez_compressed(
        os.path.join(out, "ground_truth.npz"),
        causal_snp_idx=gt.causal_snp_idx, genetic_weights=gt.genetic_weights,
        demo_weights=gt.demo_weights, **gt_app,
    )
    manifest = {
        "mode": ("real_genotypes_1000G_simulated_faces" if K is not None else "synthetic_matched"),
        "genotype_source": src or None,
        "causal_dims": cfg["data"].get("causal_dims"),
        "n_subjects": int(n), "n_snps": int(genotypes.shape[1]),
        "n_face_dims": int(k), "demographic_names": demo_names,
        "variance_shares": gt.variance_shares, "regions": list(fm.region_masks.keys()),
        "n_hps_markers": len(rsids), "n_anthro_measures": len(measure_names),
        "anthro_names": measure_names,
        "matched_target": "dbGaP phs000949 / 3DFN (Shaffer 2016)",
        "gwas_effects": {
            "enabled": bool(gwas_applied),
            "n_loci": len(gwas_applied),
            "source": "Shaffer 2016 S1 lead SNPs (p<5e-7; rsID/allele/MAF/beta verbatim, Pittsburgh sample). "
                      "Injected variance = Z^2/N from the DISCOVERY sample: an upper bound (winner's curse). "
                      "Affects the measurement phenotype only, not the shape target beta.",
            "applied": gwas_applied,
            "recovery_check": gwas_check,
        },
    }
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)

    # HIrisPlex-S batch upload file (sampleid + 41 rsID columns, 0/1/2)
    header = "sampleid," + ",".join(rsids)
    lines = [header]
    for i in range(n):
        lines.append(subject_ids[i] + "," + ",".join(str(int(x)) for x in hps[i]))
    open(os.path.join(out, "hps_upload.csv"), "w").write("\n".join(lines) + "\n")

    # ---- verification print ----
    print(f"OK: N={n}  genotypes={g.shape}  demo={demo.shape} {demo_names}")
    print(f"    beta={beta.shape}  anthro={measures.shape}  hps_markers={hps.shape}")
    print(f"    ancestry counts={np.bincount(ancestry).tolist()}")
    print(f"    splits train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print("    measure means (mm):")
    for nm, mu in zip(measure_names, measures.mean(axis=0)):
        print(f"      {nm:24s} {mu:7.2f}  (sd {measures[:, measure_names.index(nm)].std():5.2f})")
    print(f"    facial_size mm: mean={facial_size.mean():.2f} sd={facial_size.std():.2f}")
    print(f"    hps dosage distribution: {np.bincount(hps.ravel()).tolist()} (0,1,2 counts)")
    if gwas_check:
        c = gwas_check
        print(f"    GWAS recovery (covariate-adjusted, {c['n_pairs']} SNP x measure pairs):")
        print(f"      injected links detected (Bonferroni): {c['injected_detected']}/{c['injected_total']}"
              f"  correct sign: {c['injected_sign_ok']}/{c['injected_total']}")
        print(f"      NEGATIVE CONTROL {c['null_pairs']} non-injected pairs: FPR@0.05={c['null_fpr_at_0.05']:.3f}"
              f"  Bonferroni hits={c['null_bonferroni_hits']}")
        print(f"      ACCEPT = {c['accept']}")
    print(f"    wrote {out}/dataset.npz, ground_truth.npz, manifest.json, hps_upload.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/real_match.yaml")
