"""One request to each PGP host, following nothing: print the status and where it points.

The download kept ending in a 301 loop, and a loop cannot be diagnosed while something
is following it. This uses the pinned TLS handler from scripts/pgp_download.py, so the
certificate is verified against the name it actually carries, and installs a redirect
handler that refuses to follow anything. Each line is therefore one request and one
answer.

    python scripts/pgp_probe.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pgp_download", os.path.join(HERE, "pgp_download.py"))
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


class _NoFollow(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None                                            # report it, never follow it


def probe(url, ua="Mozilla/5.0"):
    op = urllib.request.build_opener(pd._PinnedHTTPSHandler, _NoFollow)
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    try:
        r = op.open(req, timeout=25)
        body = r.read(400)
        return r.status, r.headers.get("Location"), len(body), body[:120]
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location"), 0, b""
    except Exception as e:                                     # noqa: BLE001
        return type(e).__name__, str(e)[:120], 0, b""


PATHS = ["/public_genetic_data?data_type=23andMe", "/public_genetic_data", "/google_surveys", "/users", "/"]

print(f"pinning certificate verification to {pd.CERT_NAME}; following no redirects\n")
for host in ("my.pgp-hms.org", "my.personalgenomes.org"):
    print(host)
    for path in PATHS:
        st, loc, n, head = probe(f"https://{host}{path}")
        if loc:
            print(f"  {path:42s} {st}  -> {loc}")
        elif n:
            print(f"  {path:42s} {st}  {n} bytes  {head[:70]!r}")
        else:
            print(f"  {path:42s} {st}  {loc or ''}")
    print()

st, loc, n, head = probe("https://my.pgp-hms.org/public_genetic_data?data_type=23andMe",
                         ua="FaceCon-FYP/1.0")
print(f"same page with a non-browser User-Agent: {st} -> {loc or f'{n} bytes'}")
print("\nIf every path on both hosts answers 301 pointing at the other, the loop is the")
print("server's and no client change resolves it.")
sys.exit(0)
