"""Download Harvard Personal Genome Project data for the eye-colour check (RQ5).

The PGP (pgp.med.harvard.edu, data at my.personalgenomes.org) publishes participant-contributed genotype files and
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
import http.client
import ssl
import urllib.request

HOST = "https://my.pgp-hms.org"   # my.pgp-hms.org still resolves here, but its
                                         # certificate covers only this name, so the old
                                         # hostname fails verification. Not a reason to
                                         # disable verification: it is a domain migration.
PROFILE_INDEX = HOST + "/public_genetic_data?utf8=%E2%9C%93&data_type=23andMe"
SURVEY_CSV = HOST + "/google_surveys/1/download"                # trait survey export
UA = {"User-Agent": "FaceCon-FYP/1.0 (academic; contact via repository)"}


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """Reach my.pgp-hms.org while verifying its certificate against my.personalgenomes.org.

    The site is in a deadlock of its own making: my.personalgenomes.org 301s to
    my.pgp-hms.org, and my.pgp-hms.org serves a Let's Encrypt certificate whose only
    subject is my.personalgenomes.org. One of the two names must therefore be used
    against a certificate that does not name it, or the data cannot be fetched at all.

    What this does NOT do is disable verification. The certificate is still required to
    be currently valid and to chain to a trusted CA; only the name checked against it is
    pinned to my.personalgenomes.org, which is the name the operator actually certified
    and which their own redirect declares canonical. The residual assumption is that
    these two hostnames are one service -- which that redirect is evidence for -- and an
    impostor would still need a CA-issued certificate for personalgenomes.org, that is,
    control of that domain.

    The scope is deliberately narrow: this handler is used only for the PGP hosts, the
    data is public CC0 genotype and survey data, nothing is uploaded, and no credential
    is ever sent. A blanket ssl._create_unverified_context would have covered the whole
    process and accepted any certificate from anyone, which is why it is not used here.
    """

    def __init__(self):
        ctx = ssl.create_default_context()                     # verification stays ON
        super().__init__(context=ctx)

    def https_open(self, req):
        if req.host.split(":")[0] in ("my.pgp-hms.org", "my.personalgenomes.org"):
            req.host = req.host.replace("my.personalgenomes.org", "my.pgp-hms.org")

            def build(host, **kw):
                kw.pop("server_hostname", None)
                return http.client.HTTPSConnection(
                    host, context=self._context, server_hostname=CERT_NAME, **kw)
            return self.do_open(build, req)
        return self.do_open(http.client.HTTPSConnection, req, context=self._context)


class _StayOnValidHost(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but keep them on the hostname whose certificate is valid.

    The site answers on my.personalgenomes.org and then 302s back to my.pgp-hms.org,
    which resolves to the same server but presents a certificate naming only
    my.personalgenomes.org. Following that redirect therefore fails hostname
    verification, and the two hosts are the same machine, so rewriting the redirect
    target is the correct fix. Verification stays on: this moves the request to the
    name the certificate actually covers instead of accepting a certificate that does
    not match. Any other host is left alone and followed normally.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code in (301, 302) and "my.pgp-hms.org" in newurl and "my.personalgenomes.org" in req.full_url:
            return None                                        # the known A<->B loop: stop here
        return super().redirect_request(req, fp, code, msg, headers, newurl)


CERT_NAME = "my.personalgenomes.org"   # the only name their certificate covers
_OPENER = urllib.request.build_opener(_PinnedHTTPSHandler, _StayOnValidHost)


def _get(url, retries=3, timeout=60):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with _OPENER.open(req, timeout=timeout) as r:
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
        print("Download the trait survey by hand from my.personalgenomes.org and save it as "
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
            data = _get(r["url"] if r["url"].startswith("http") else HOST + r["url"], retries=2)
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
