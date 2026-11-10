#!/usr/bin/env cs_python
"""Gemma 4 12B wafer program host (simulator or CS-3; src/spatial/README.md, Gemma 4 12B): places a prep (prep_gemma12.py), then
runs the wafer token loop: every launch the host writes step 0's input (the embedding row into unit 0's CX rows), the
head control root's force table (prompt / teacher forcing) and launches S steps; the ring holds each step's argmax.
  run_gemma12.py --name <out> [--cmaddr A] --report r.json --prep <dir> [--prefix p] [--ntok N] [--greedy 1]
             [--force 1|2] [--wloop S] [--reps R]"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host

ap = host.parser(__doc__)
ap.add_argument("--wloop", type=int, default=0); ap.add_argument("--fine", type=int, default=0)
ap.add_argument("--placeonly", type=int, default=0, help="(checks) stop after the placement (the mock runtime's copy log)")
a = ap.parse_args()
h = host.Host(a)
M, g, npy, put, get = h.meta, h.g, h.npy, h.put, h.get
PY, XC, YC, NU, K = g.PY, g.XC, g.YC, g.units(), M["K"]
HX0, HXC, HR = g.HX0, g.HXC, g.HR
OFF = np.concatenate([[0], np.cumsum([g.hc(r) for r in range(PY)])]).astype(int)
h.place()
if a.placeonly:
    print(json.dumps({"placed_s": round(h.t_place, 1)}), flush=True)
    sys.exit(0)
h.init()
t_load, t_place = h.t_load, h.t_place


def rowblocks():
    """Runs of CX rows of one dims count, split at the CY row: (first row, end, dims)."""
    r0 = 0
    for r in range(1, PY + 1):
        if r == PY or r == YC or g.hc(r) != g.hc(r0):
            yield r0, r, g.hc(r0)
            r0 = r


def put_step0(tok, pos):
    e = np.zeros(OFF[-1], np.float32)
    e[:M["H"]] = (np.asarray(FE[tok]).astype(np.uint32) << 16).view(np.float32)
    U0 = g.unit_origin(0)
    for (r0, r1, hc) in rowblocks():
        put("x", U0[0] + XC, U0[1] + g.wl(r0, YC), 1, r1 - r0, e[OFF[r0]:OFF[r1]].reshape(r1 - r0, 1, hc))
FE = npy("embed")
CAP = 256
gold = h.golden()
ids = gold["prompt_ids"] + gold["tokens"]
NP = len(gold["prompt_ids"])
ntok = a.ntok or len(gold["tokens"])
WLS = a.wloop or 255
reps = []
xd = []
for rep in range(a.reps):
    got, mis, walls = [], [], []
    t = 0
    while t < NP + ntok - 1:
        tok_in = got[-1] if (a.greedy and t >= NP) else ids[t]
        put_step0(tok_in, t)
        if a.force == 2:
            S_ = max(1, min(WLS, NP + ntok - 1 - t))
            fl = [ids[min(t + k + 1, len(ids) - 1)] for k in range(S_)]
        elif t == 0:
            S_ = max(1, min(WLS, NP + ntok - 1))
            fl = [ids[k + 1] for k in range(NP - 1)]
        else:
            S_ = max(1, min(WLS, NP + ntok - 1 - t)); fl = []
        put("force", HX0 + HXC, HR // 2, 1, 1, np.array(fl + [0] * (CAP - len(fl)), np.uint32))
        put("nforce", HX0 + HXC, HR // 2, 1, 1, np.array([len(fl)], np.uint32))
        tw = time.time()
        h.launch(S_, t)
        ring = get("ring", HX0 + HXC, HR // 2, 1, 1, CAP, np.uint32)[0, 0]
        walls += [(time.time() - tw) / S_] * S_
        if a.xdbg:
            ux = []
            for u in range(NU):
                X0, Y0 = g.unit_origin(u)
                xl = np.zeros((K, OFF[-1]), np.float32)
                for (r0, r1, hc) in rowblocks():
                    v = get("xdbg", X0 + XC, Y0 + g.wl(r0, YC), 1, r1 - r0, K * hc)[:, 0].reshape(r1 - r0, K, hc)
                    xl[:, OFF[r0]:OFF[r1]] = v.transpose(1, 0, 2).reshape(K, -1)
                ux.append(xl[:, :M["H"]].tolist())
            xd.append(dict(t=t, S=S_, layers=ux))
        for k in range(S_):
            i = t + k - (NP - 1)
            if i < 0: continue
            best = int(ring[k])
            got.append(best)
            if best != gold["tokens"][i]:
                mis.append({"i": i, "got": best, "want": gold["tokens"][i], "margin": round(gold["margins"][i], 4) if "margins" in gold else None})
            if i < 4 or i % 64 == 0:
                print(json.dumps({"i": i, "got": best, "want": gold["tokens"][i], "wall_ms": round(walls[-1] * 1e3, 3), "S": S_}), flush=True)
        t += S_
    reps.append({"rep": rep, "tf_mismatches": len(mis), "first_mismatch": mis[0]["i"] if mis else None, "mismatches": mis[:20],
                 "wall_ms_median": round(float(np.median(walls)) * 1e3, 4), "tokens": got[:64]})
    print("REP " + json.dumps({k: v for k, v in reps[-1].items() if k != "tokens"}), flush=True)
times = None
if a.tdbg:
    tl = []
    for u in range(NU):
        X0, Y0 = g.unit_origin(u)
        tb = get("tdbg", X0 + XC, Y0, 1, 1, K + 2, np.uint32)[0, 0].astype(np.int64)
        tl.append([int(v) for v in np.diff(tb) % (1 << 32)])
    rb = get("tdbg", HX0 + HXC, HR // 2, 1, 1, 70, np.uint32)[0, 0].astype(np.int64)
    times = dict(units=tl, loop=[int(v) for v in np.diff(rb[2:2 + 64]) % (1 << 32)])
    # (TSTAMP 1 builds) phase stamps from the pass start: a CY and three weight PEs of units 0 and 3
    ph = {}
    for u in (0, min(3, NU - 1)):
        X0, Y0 = g.unit_origin(u)
        for nm, (px_, py_, n_) in {"cy0": (X0, Y0 + YC, 1 + 6 * K), "cymid": (X0 + XC - 1, Y0 + YC, 1 + 6 * K),
                                   "w_far": (X0, Y0, 1 + 9 * K), "w_near": (X0 + XC - 1, Y0 + YC - 1, 1 + 9 * K),
                                   "w_plain": (X0, Y0 + YC + 1, 1 + 9 * K)}.items():
            b_ = get("tdbg", px_, py_, 1, 1, n_, np.uint32)[0, 0].astype(np.int64)
            ph[f"u{u}_{nm}"] = [int(v) for v in (b_ - int(b_[0])) % (1 << 32)]
    times["phases"] = ph
    print("TIMES " + json.dumps(times), flush=True)
h.stop()
if xd:
    json.dump(xd, open(a.report.replace(".json", "_xd.json"), "w"))
r = reps[-1]
rep = {"greedy": bool(a.greedy), "force": a.force, "wloop": WLS, "ntok": ntok, "reps": reps, "tf_mismatches": r["tf_mismatches"],
       "wall_ms_median": r["wall_ms_median"], "tok_s_wall": round(1000.0 / r["wall_ms_median"], 1), "times": times,
       "load_s": round(t_load, 1), "place_s": round(t_place, 1), "passed": all(x["tf_mismatches"] == 0 for x in reps)}
json.dump(rep, open(a.report, "w"), indent=1)
print("GEMMA12 " + json.dumps({k: v for k, v in rep.items() if k != "reps"}), flush=True)
