"""Does HIrisPlex-S predict self-reported eye colour in an open human cohort? (RQ5)

Every face number in this project is simulated, so the eye-colour arm is the one place
FaceCon can test a component against real people. The model under test is the real
HIrisPlex-S (src/appearance/hirisplex.py), which FaceCon did not fit: this measures the
published model on an open cohort, and claims nothing about FaceCon's own accuracy.

Protocol, frozen in PREREGISTRATION.md section 7 before any PGP data was read:

  mapping      blue/grey -> blue; green/hazel -> intermediate; brown/black -> brown
  model        IrisPlex eye, the 6 eye SNPs of the HIrisPlex-S marker set
  measure      one-vs-rest AUC per category, and overall accuracy at argmax
  threshold    blue-vs-rest AUC >= 0.7 counts as the model transferring
  control      1000 label shuffles; the real AUC must sit outside that null
  reporting    aggregates only -- never a participant's genotype or trait

Missing markers are mean-imputed at the cohort allele frequency and the number imputed
per participant is reported, because HIrisPlex-S is sensitive to missing eye markers and
23andMe chips do not all carry the same six.

    python scripts/pgp_eye_check.py --dir data/pgp --out gpu/pgp_eye.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.appearance.hirisplex import HirisplexPredictor, LABELS  # noqa: E402

# Frozen mapping from free-text survey answers to the model's three categories.
EYE_MAP = {
    "blue": "blue", "grey": "blue", "gray": "blue", "blue-grey": "blue", "blue-gray": "blue",
    "green": "intermediate", "hazel": "intermediate", "green-hazel": "intermediate",
    "blue-green": "intermediate", "amber": "intermediate",
    "brown": "brown", "black": "brown", "dark brown": "brown", "light brown": "brown",
}


def map_eye(text):
    t = re.sub(r"[^a-z -]", "", str(text).strip().lower())
    if t in EYE_MAP:
        return EYE_MAP[t]
    for k, v in EYE_MAP.items():                                  # e.g. "mostly blue"
        if k in t:
            return v
    return None


def read_surveys(path):
    """participant id -> mapped eye category. Nothing else is read from the survey."""
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []
        idc = next((c for c in cols if re.search(r"participant|huid|user", c, re.I)), None)
        eyec = next((c for c in cols if re.search(r"eye", c, re.I)), None)
        if not idc or not eyec:
            raise SystemExit(f"surveys.csv has no participant/eye columns; saw {cols[:12]}")
        for row in rdr:
            hu = re.findall(r"\b(hu[0-9A-Fa-f]{6})\b", str(row[idc]))
            cat = map_eye(row[eyec])
            if hu and cat:
                out[hu[0].lower()] = cat
    return out


def read_genotype(path, wanted):
    """rsid -> genotype string, for the wanted rsids only (23andMe / Ancestry raw text)."""
    g = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = re.split(r"[\t,]", line.strip())
            if len(p) >= 4 and p[0] in wanted:
                g[p[0]] = p[-1].upper() if len(p) == 4 else (p[3] + p[4]).upper()
    return g


def dosage(call, effect_allele):
    if not call or call in ("--", "__", "NN", "00"):
        return None
    return float(sum(1 for b in call if b == effect_allele))


def auc(y, s):
    """One-vs-rest AUC by rank, ties averaged."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    means = np.zeros(len(uniq))
    np.add.at(means, inv, r); means /= cnt
    r = means[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/pgp")
    ap.add_argument("--out", default="gpu/pgp_eye.json")
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()

    pred = HirisplexPredictor()
    markers = json.load(open(os.path.join(ROOT, "src/appearance/hirisplex_model/markers.json")))["markers"]
    coeffs = json.load(open(os.path.join(ROOT, "src/appearance/hirisplex_model/coeffs.json")))["eye"]
    # The eye model's markers are the columns it gives a nonzero weight: IrisPlex's 6
    # SNPs out of the 41-marker HIrisPlex-S input. Deriving them from the coefficients
    # rather than hard-coding rsids keeps this honest if the coefficient file changes.
    B = np.array([v for _, v in sorted(coeffs["b"].items(), key=lambda kv: int(kv[0]))], float)
    eye_cols = np.flatnonzero(np.abs(B).sum(0) > 0).tolist()
    eye_markers = [markers[j] for j in eye_cols]
    wanted = {m["rsid"] for m in eye_markers}
    print("eye markers: " + ", ".join(f"{m['gene']}/{m['rsid']}" for m in eye_markers))

    eye = read_surveys(os.path.join(a.dir, "surveys.csv"))
    gdir = os.path.join(a.dir, "genotypes")
    rows, imputed_counts = [], []
    for fn in sorted(os.listdir(gdir)):
        hu = fn.split(".")[0].lower()
        if hu not in eye:
            continue
        g = read_genotype(os.path.join(gdir, fn), wanted)
        d = [dosage(g.get(m["rsid"]), m["input_allele"]) for m in eye_markers]
        rows.append({"hu": hu, "y": eye[hu], "d": d})
    if not rows:
        raise SystemExit("no participants with both a mapped eye colour and a genotype file")

    D = np.array([[np.nan if v is None else v for v in r["d"]] for r in rows], float)
    imputed_counts = np.isnan(D).sum(1).tolist()
    freq = np.nanmean(D, 0)
    D = np.where(np.isnan(D), freq, D)

    full = np.zeros((len(rows), len(markers)))                    # 41-marker input; eye reads 6
    full[:, eye_cols] = D
    P = pred.predict_proba(full)["eye"]                           # (N, 3) blue / intermediate / brown
    y = np.array([r["y"] for r in rows])
    cats = LABELS["eye"]
    acc = float((np.array(cats)[P.argmax(1)] == y).mean())

    rng = np.random.default_rng(a.seed)
    res = {"n_participants": len(rows), "source": "Harvard PGP (CC0)",
           "model": "HIrisPlex-S / IrisPlex eye, not fitted by FaceCon",
           "eye_markers": [f"{m['gene']}/{m['rsid']}" for m in eye_markers],
           "imputed_markers_per_participant": {
               "mean": float(np.mean(imputed_counts)), "max": int(np.max(imputed_counts))},
           "class_counts": {c: int((y == c).sum()) for c in cats},
           "accuracy_argmax": acc, "per_category": {}}
    for i, c in enumerate(cats):
        obs = auc(y == c, P[:, i])
        null = np.array([auc(rng.permutation(y) == c, P[:, i]) for _ in range(a.shuffles)])
        res["per_category"][c] = {
            "auc": obs, "null_mean": float(np.nanmean(null)),
            "null_p025": float(np.nanpercentile(null, 2.5)),
            "null_p975": float(np.nanpercentile(null, 97.5)),
            "outside_null": bool(obs > np.nanpercentile(null, 97.5)),
            "p_one_sided": float((np.nansum(null >= obs) + 1) / (a.shuffles + 1))}
    res["acceptance"] = {
        "blue-vs-rest AUC >= 0.70": bool(res["per_category"]["blue"]["auc"] >= 0.70),
        "blue AUC outside the shuffled null": res["per_category"]["blue"]["outside_null"]}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{res['n_participants']} participants, classes {res['class_counts']}")
    print(f"mean markers imputed per participant: {res['imputed_markers_per_participant']['mean']:.2f}")
    print(f"accuracy at argmax: {acc:.3f}")
    for c, v in res["per_category"].items():
        print(f"  {c:13s} AUC {v['auc']:.3f}  null {v['null_p025']:.3f}-{v['null_p975']:.3f}  p={v['p_one_sided']:.4f}")
    for k, v in res["acceptance"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("wrote " + a.out)


if __name__ == "__main__":
    main()
