#!/usr/bin/env python3
"""rcfg_hash.py <geo>: sha256 of every PE's rcfg words by kind (split_geo_gptoss), for the Rust port's check
(geo::gptoss tests: the same hashes)."""
import hashlib, json, sys
import split_geo_gptoss as S
name = sys.argv[1]
tiny = name == "tiny"
g = S.GEOS[name](G2=2 if tiny else 16, G2W=2 if tiny else 24)
g.build()
VR = 8 if tiny else 272
H = {}
for y in range(g.H):
    for x in range(g.W):
        k = g.kind(x, y)
        if k in ("wa", "wm"):
            X0, Y0 = g.unit_origin(next(u for u in range(len(g.order)) if g.unit_origin(u)[0] <= x < g.unit_origin(u)[0] + g.UW and g.unit_origin(u)[1] <= y < g.unit_origin(u)[1] + g.UH))
            ly = y - Y0; r = ly - (1 if ly > g.YC else 0)
            w = g.rcfg_w(x, y, r, rwc=1 if x - X0 == g.XC + 1 else 0)
        elif k == "cx":
            X0, Y0 = g.unit_origin(0)
            w = g.rcfg_cx(x, y, X0 <= x < X0 + g.UW and Y0 <= y < Y0 + g.UH)
        elif k in ("cya", "cym"):
            w = g.rcfg_cy(x, y)
        elif k == "head":
            w = g.rcfg_head(x, y)
        elif k == "hctl":
            w = g.rcfg_hctl(x, y, VR)
        else:
            continue
        H.setdefault(k, hashlib.sha256())
        H[k].update(json.dumps([x, y] + [int(v) for v in w]).encode())
print(json.dumps({k: v.hexdigest()[:16] for k, v in sorted(H.items())}))
