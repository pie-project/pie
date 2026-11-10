#!/usr/bin/env cs_python
"""Gemma 4 wafer program host (simulator or CS-3; src/spatial/README.md, "Gemma 4"): places a prep (prep_gemma.py), then
runs the wafer token loop (WL always): every launch the host writes step 0's input (the embedding row into unit 0's
CX, the token's PLE words into every CY on a bus), the head control root's force table (prompt / teacher forcing)
and launches S steps; the ring holds each step's argmax.
  run_gemma.py --name <out> [--cmaddr A] --report r.json --prep <dir> [--prefix p] [--ntok N] [--greedy 1]
               [--force 1|2] [--wloop S] [--reps R] [--probe L]  (DIAG 1 builds: L launches of S forced random ids,
               checks the PLE words every CY got and the KVS words every consumer weight PE got)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host

ap = host.parser(__doc__)
ap.add_argument("--wloop", type=int, default=0); ap.add_argument("--fine", type=int, default=0)
ap.add_argument("--probe", type=int, default=0)
ap.add_argument("--kvseq", type=int, default=0, help="(PLOG full-model probes) check only the KVS sequence words")
ap.add_argument("--placeonly", type=int, default=0, help="(checks) stop after the placement (the mock runtime's copy log)")
a = ap.parse_args()
h = host.Host(a)
M, g, npy, put, get = h.meta, h.g, h.npy, h.put, h.get
PX, PY, XC, YC, NU, Hc, K = g.PX, g.PY, g.XC, g.YC, g.units(), M["Hc"], M["K"]
HX0, HXC, HR = g.HX0, g.HXC, g.HR
h.place()
if a.placeonly:
    print(json.dumps({"placed_s": round(h.t_place, 1)}), flush=True)
    sys.exit(0)
h.init()
t_load, t_place = h.t_load, h.t_place

def ple_words(ci, tok):
    col = M["cols"][ci]
    nt, lv = col["nt"], col["lv"]
    k, j = tok // nt, tok % nt
    for f in col["files"]:
        if f["r0"] <= k < f["r1"]:
            return (np.asarray(PT[f["name"]][k - f["r0"], :, j * lv:(j + 1) * lv]).view(np.float16).astype(np.float32))   # [PX][lv]
def put_step0(tok, pos):
    row = FE[tok] if FE is not None else None
    e = (np.asarray(row).astype(np.uint32) << 16).view(np.float32)
    U0 = g.unit_origin(0)
    for (r0, r1) in ((0, YC), (YC, PY)):
        put("x", U0[0] + XC, U0[1] + g.wl(r0, YC), 1, r1 - r0, e[r0 * Hc:r1 * Hc].reshape(r1 - r0, 1, Hc))
    # step 0's PLE words: the storage's, looked up on the head control rows' launch-start TB word
    put("tin", HX0 + HXC, 0, 1, HR, np.full((HR, 1, 1), tok, np.uint32))
FE = npy("embed")
CAP = 256
if a.probe:
    # DIAG 1 builds: launches of S forced random ids; check every CY's PLE words and every consumer's KVS words
    S_ = a.wloop or 64
    rng = np.random.default_rng(7)
    bad_p, bad_k, nchk = [], [], 0
    pos = 0
    ids = rng.integers(0, M["V"], size=a.probe * S_ + 1).tolist()
    PT = {f["name"]: npy(f["name"]) for col in M["cols"] for f in col["files"]}
    for L in range(a.probe):
        put_step0(ids[pos], pos)
        fl = [ids[pos + k + 1] for k in range(S_)]
        put("force", HX0 + HXC, HR // 2, 1, 1, np.array(fl + [0] * (CAP - len(fl)), np.uint32))
        put("nforce", HX0 + HXC, HR // 2, 1, 1, np.array([S_], np.uint32))
        tl = time.time()
        h.launch(S_, pos)
        dt = time.time() - tl
        ring = get("ring", HX0 + HXC, HR // 2, 1, 1, CAP, np.uint32)[0, 0]
        ns = min(S_, 64)
        for ci, col in enumerate(M["cols"]):
            for ui, u in enumerate(col["units"]):
                X0, Y0 = g.unit_origin(u)
                lg = get("plog", X0, Y0 + YC, g.UW, 1, 128)[0]        # [UW][128]
                for s in range(ns):
                    want = ple_words(ci, ids[pos + s])[:, K * ui]
                    for c in range(PX):
                        x = g.wl(c, XC)
                        got, seq = lg[x, 2 * s], lg[x, 2 * s + 1]
                        nchk += 1
                        if got != want[c] or seq != pos + s + 1:
                            if len(bad_p) < 20: bad_p.append(dict(L=L, s=s, u=u, c=c, got=float(got), want=float(want[c]), seq=float(seq), pos=pos + s))
        for u in (getattr(g, "kvdst", None) or []):
            X0, Y0 = g.unit_origin(u)
            lg = get("kvlog", X0, Y0, g.UW, g.UH, 128)
            for s in range(ns):
                p_ = pos + s
                for ly in range(g.UH):
                    if ly == YC: continue
                    for lx in range(g.UW):
                        if lx == XC: continue
                        sm, sq = lg[ly, lx, 2 * s], lg[ly, lx, 2 * s + 1]
                        nchk += 1
                        if (not a.kvseq and sm != 8 * p_ + 120) or sq != p_ + 1:
                            if len(bad_k) < 20: bad_k.append(dict(L=L, s=s, u=u, lx=lx, ly=ly, sum=float(sm), seq=float(sq), pos=p_))
        print(json.dumps({"launch": L, "pos": pos, "steps": S_, "s": round(dt, 3), "bad_ple": len(bad_p), "bad_kvs": len(bad_k)}), flush=True)
        pos += S_
    h.stop()
    rep = dict(probe=a.probe, steps=S_, checks=nchk, bad_ple=bad_p, bad_kvs=bad_k, passed=not bad_p and not bad_k,
               load_s=round(t_load, 1), place_s=round(t_place, 1))
    json.dump(rep, open(a.report, "w"), indent=1)
    print("PROBE " + json.dumps({k: v for k, v in rep.items()}), flush=True)
    sys.exit(0 if rep["passed"] else 1)

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
                xl = np.zeros((PY, K * Hc), np.float32)
                for (r0, r1) in ((0, YC), (YC, PY)):
                    xl[r0:r1] = get("xdbg", X0 + XC, Y0 + g.wl(r0, YC), 1, r1 - r0, K * Hc)[:, 0]
                ux.append(xl.reshape(PY, K, Hc).transpose(1, 0, 2).reshape(K, -1).tolist())
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
        for nm, (px_, py_, n_) in {"cy0": (X0, Y0 + YC, 1 + 6 * K + (12 * K if a.fine else 0)), "cymid": (X0 + XC - 1, Y0 + YC, 1 + 6 * K),
                                   "w_far": (X0, Y0, 1 + 9 * K + (10 * K if a.fine else 0)), "w_near": (X0 + XC - 1, Y0 + YC - 1, 1 + 9 * K + (10 * K if a.fine else 0))}.items():
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
print("GEMMA " + json.dumps({k: v for k, v in rep.items() if k != "reps"}), flush=True)
