"""Download Harvard Personal Genome Project data for the eye-colour check (RQ5).

The PGP (pgp.med.harvard.edu) publishes participant-contributed genotype files and
survey responses under CC0, with no application or data-use agreement. That makes it the
only open source the project can use to test HIrisPlex-S against self-reported human
phenotypes. It is NOT a substitute for 3DFN: it has no 3D face scans, so it says nothing
about the shape half of FaceCon.

This script runs on the GPU machine, not in the build sandbox, which has no route to
pgp.med.harvard.edu. It only downloads; scripts/pgp_eye_check.py does the analysis.

What it fetches
    1. the public profile/survey export listing participants and their trait responses
    2. for participants with both an eye-colour response and a genotype file, the
       genotype file itself (23andMe or Ancestry raw-text format)

Privacy. PGP participants consented to open publication under CC0 and are identified by
a public participant ID (hu...). This script still writes nothing outside
`--out`, never prints a genotype or an individual's traits, and the analysis script
reports aggregates only. Do not commit the downloaded files: `--out` defaults to
`data/pgp/`, which .gitignore excludes.

    python scripts/pgp_download.py --out data/pgp --max-participants 400
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
import urllib.request

PROFILE_INDEX = "https://my.pgp-hms.org/public_genetic_data?utf8=%E2%9C%93&data_type=23andMe"
SURVEY_CSV = "https://my.pgp-hms.org/google_surveys/1/download"   # trait survey export
UA = {"User-Agent": "FaceCon-FYP/1.0 (academic; contact via repository)"}


def _get(url, retries=3, timeout=60):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data
        except Exception as e:                                    # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry {i+1} after {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(3 * (i + 1))


def fetch_index(out_dir, max_participants):
    """The public genetic-data index: participant id -> genotype file URL."""
    html = _get(PROFILE_INDEX).decode("utf-8", "replace")
    open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8").write(html)
    rows = []
    for m in re.finditer(r'href="(?P<url>[^"]*?/download/(?P<file>[^"]+))"', html):
        seg = html[max(0, m.start() - 600):m.start()]
        hu = re.findall(r"\b(hu[0-9A-F]{6})\b", seg)
        if hu:
            rows.append({"participant": hu[-1], "url": m.group("url"), "file": m.group("file")})
    seen, uniq = set(), []
    for r in rows:                                                # one file per participant
        if r["participant"] not in seen:
            seen.add(r["participant"]); uniq.append(r)
    uniq = uniq[:max_participants]
    json.dump(uniq, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
    print(f"index: {len(uniq)} participants with a downloadable genotype file")
    return uniq


def fetch_surveys(out_dir):
    """The trait survey export, which carries the self-reported eye colour."""
    try:
        raw = _get(SURVEY_CSV)
    except Exception as e:                                        # noqa: BLE001
        print(f"survey export unavailable ({type(e).__name__}: {e}).", file=sys.stderr)
        print("Download the trait survey by hand from my.pgp-hms.org and save it as "
              f"{os.path.join(out_dir, 'surveys.csv')}.", file=sys.stderr)
        return None
    path = os.path.join(out_dir, "surveys.csv")
    open(path, "wb").write(raw)
    n = sum(1 for _ in csv.reader(io.StringIO(raw.decode("utf-8", "replace")))) - 1
    print(f"surveys: {n} rows -> {path}")
    return path


def fetch_genotypes(rows, out_dir, limit):
    gdir = os.path.join(out_dir, "genotypes")
    os.makedirs(gdir, exist_ok=True)
    got = 0
    for r in rows:
        dest = os.path.join(gdir, f"{r['participant']}.txt")
        if os.path.exists(dest):
            got += 1
            continue
        if got >= limit:
            break
        try:
            data = _get(r["url"], retries=2)
        except Exception as e:                                    # noqa: BLE001
            print(f"  skip {r['participant']}: {type(e).__name__}", file=sys.stderr)
            continue
        open(dest, "wb").write(data)
        got += 1
        if got % 25 == 0:
            print(f"  {got} genotype files")
        time.sleep(0.5)                                           # be polite to the server
    print(f"genotypes: {got} files in {gdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/pgp")
    ap.add_argument("--max-participants", type=int, default=400)
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = fetch_index(a.out, a.max_participants)
    fetch_surveys(a.out)
    if not a.index_only:
        fetch_genotypes(rows, a.out, a.max_participants)
    print("\nnext: python scripts/pgp_eye_check.py --dir " + a.out)


if __name__ == "__main__":
    main()
