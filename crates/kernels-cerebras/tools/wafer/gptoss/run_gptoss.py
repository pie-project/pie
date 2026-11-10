#!/usr/bin/env cs_python
"""gpt-oss wafer program host (simulator or CS-3): places a prep (prep_gptoss.py) on the program (build_gptoss.py), then
runs the wafer token loop: each launch the host writes step 0's embedding row into unit 0's CX, the control root's
force table (prompt / teacher forcing) and launches S steps; the ring holds each step's argmax.
  run_gptoss.py --name <out> [--cmaddr A] --report r.json --prep <dir> [--prefix p] [--ntok N] [--greedy 1]
                [--force 2|1|0] [--wloop S] [--reps R] [--hostck 1] [--xdbg 1] [--tdbg 1] [--override G2=..]
--force 2: teacher-forced (every step's next token from the golden); 1: the prompt forced, then greedy.
--hostck 1: S 1 launches, every control PE's res read: the host argmax over the columns' maxima vs the ring.
--xdbg 1 (XKEEP 1 builds): every unit's x after each layer, each launch (dump json)."""
import argparse, json, os, sys, time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder, SdkRuntime  # pylint: disable=no-name-in-module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compiler

ap = argparse.ArgumentParser()
ap.add_argument("--name"); ap.add_argument("--cmaddr"); ap.add_argument("--report", default="g.json")
ap.add_argument("--prep"); ap.add_argument("--prefix", default=""); ap.add_argument("--ntok", type=int, default=0)
ap.add_argument("--greedy", type=int, default=0); ap.add_argument("--force", type=int, default=2)
ap.add_argument("--wloop", type=int, default=8); ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--hostck", type=int, default=0); ap.add_argument("--xdbg", type=int, default=0)
ap.add_argument("--tdbg", type=int, default=0); ap.add_argument("--override", default="")
ap.add_argument("--golden", default="golden.json")
ap.add_argument("--steps0", type=int, default=0, help="(bisect) one launch of 0 steps, then a d2h")
ap.add_argument("--prof", default="", help="(TSTAMP 1 builds) units (comma) whose PEs' phase stamps to read (last step); also in the report")
ap.add_argument("--d2h-delay", type=float, default=0.0, help="(bisect) seconds a step to wait after a launch before the ring's read (no pending d2h during the loop)")
ap.add_argument("--noweights", type=int, default=0, help="(probes) no weight / head tiles placed (zeros): routes, roles and tables only")
ap.add_argument("--compiled", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "compiled"), help="wafer-compile's output for this prep (build_gptoss.py writes it)")
a = ap.parse_args()
t00 = time.time()
M = json.load(open(os.path.join(a.prep, a.prefix + "meta.json")))
for kv in filter(None, a.override.split(",")):
    k, v = kv.split("="); M[k] = int(v)
def npy(n):
    return np.load(os.path.join(a.prep, a.prefix + n + ".npy"), mmap_mode="r")
tiny = M["preset"] == "tiny"
g = compiler.load(a.compiled)
PY, AX, MXc, XC, YC, NU, Hc, K = g.PY, g.AX, g.MXc, g.XC, g.YC, g.units(), M["Hc"], M["K"]
HB0, HXC, HVC, VR = g.HB0, g.HXC, g.HVC, M["VR"]
runner = SdkRuntime(a.name, cmaddr=a.cmaddr)
runner.load(); runner.run()
t_load = time.time() - t00
k32 = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
k16 = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
ID = {}
def sym(s):
    if s not in ID: ID[s] = runner.get_id(s)
    return ID[s]
def put(s, x0, y0, w, h, arr, b16=False):
    if w * h == 0: return
    arr = np.array(arr, copy=True)
    if b16: arr = arr.view(np.uint16).astype(np.uint32)
    n = arr.size // (w * h)
    runner.memcpy_h2d(sym(s), arr.reshape(-1), x0, y0, w, h, n, **(k16 if b16 else k32))
def get(s, x0, y0, w, h, n, dt=np.float32):
    b = np.zeros(w * h * n, dt)
    runner.memcpy_d2h(b, sym(s), x0, y0, w, h, n, **k32)
    return b.reshape(h, w, n)
HALF = ((0, YC), (YC, PY))
RB = 16 if not tiny else PY
t0 = time.time()
# every copy as wafer-compile's bindings state them (the Rust family's placement, in its order); --noweights
# leaves the weight and head tiles out
WSYMS = ("w", "wx", "wr", "hw", "hemb")
def put_w(s, *r):
    if not (a.noweights and s in WSYMS): put(s, *r)
compiler.place(put_w, npy, g, lambda n: os.path.exists(os.path.join(a.prep, a.prefix + n + ".npy")))
yc = HB0 + HXC
t_place = time.time() - t0
runner.launch("init", nonblock=False)
_bar = np.zeros(g.W * g.H, np.uint32)
runner.memcpy_d2h(_bar, sym("rcfg"), 0, 0, g.W, g.H, 1, **k32)
print(json.dumps({"load_s": round(t_load, 1), "place_s": round(t_place, 1)}), flush=True)
if a.steps0:
    runner.launch("compute", np.uint32(0), np.uint32(0), nonblock=False)
    runner.memcpy_d2h(_bar, sym("rcfg"), 0, 0, g.W, g.H, 1, **k32)
    print("STEPS0 ok", flush=True)
    runner.stop()
    sys.exit(0)

gold = json.load(open(os.path.join(a.prep, a.prefix + a.golden)))
emb = npy("embrows")                                    # [the golden's ids][H] bf16 bits
ids = gold["prompt_ids"] + gold["tokens"]
ntok = a.ntok or len(gold["tokens"])
NP = len(gold["prompt_ids"])
U0 = g.unit_origin(0)
ROOT = (HVC // 2, yc)
CAP = 256
ring = np.zeros(CAP, np.uint32)
WLS = 1 if a.hostck else a.wloop
reps, dumps, ring_bad, diverged = [], [], [], []
def xread(u):
    X0u, Y0u = g.unit_origin(u)
    xl = np.zeros((PY, K * Hc), np.float32)
    for (r0, r1) in HALF:
        xl[r0:r1] = get("xdbg", X0u + XC, Y0u + g.wl(r0, YC), 1, r1 - r0, K * Hc).reshape(r1 - r0, -1)
    return xl.reshape(PY, K, Hc).transpose(1, 0, 2).reshape(K, -1)
def xredread(u):
    X0u, Y0u = g.unit_origin(u)
    xr_ = np.zeros((PY, 2 * K * Hc), np.float32)
    for (r0, r1) in HALF:
        xr_[r0:r1] = get("xred", X0u + XC, Y0u + g.wl(r0, YC), 1, r1 - r0, 2 * K * Hc).reshape(r1 - r0, -1)
    return xr_.reshape(PY, 2 * K, Hc).transpose(1, 0, 2).reshape(2 * K, -1)
for rep in range(a.reps):
    got, walls, mis = [], [], []
    t = 0
    while t < NP + ntok - 1:
        e = (np.asarray(emb[t]).astype(np.uint32) << 16).view(np.float32)
        for (r0, r1) in HALF:
            put("x", U0[0] + XC, U0[1] + g.wl(r0, YC), 1, r1 - r0, e[r0 * Hc:r1 * Hc].reshape(r1 - r0, 1, Hc))
        S_ = max(1, min(WLS, NP + ntok - 1 - t))
        if a.force == 2:                    # teacher-forced: every step's next token from the golden
            fl = [ids[min(t + k + 1, len(ids) - 1)] for k in range(S_)]
        elif t == 0:                        # the prompt forced, then greedy, in one launch
            S_ = max(1, min(max(WLS, NP), NP + ntok - 1, CAP))
            fl = [ids[k + 1] for k in range(NP - 1)]
        else:
            fl = []
            if got and got[-1] != ids[t]:
                diverged.append(t)          # (only the golden's rows are staged: its row instead)
        if fl:
            put("force", *ROOT, 1, 1, np.array(fl + [0] * (CAP - len(fl)), np.uint32))
        put("nforce", *ROOT, 1, 1, np.array([len(fl)], np.uint32))
        tw = time.time()
        runner.launch("compute", np.uint32(S_), np.uint32(t), nonblock=False)
        if a.d2h_delay > 0:
            time.sleep(a.d2h_delay * S_)
        # (a launch returns before the PEs finish it: the ring's read waits for the root's last step)
        runner.memcpy_d2h(ring, sym("ring"), *ROOT, 1, 1, CAP, **k32)
        walls += [(time.time() - tw) / S_] * S_
        if a.hostck:
            rr = get("res", 0, yc, HVC, 1, 4).reshape(HVC, 4)
            best = int(rr[np.argmax(rr[:, 0]), 1])
            if best != int(ring[0]):
                ring_bad.append(t)
        if a.xdbg:
            dumps.append({"t": t, "x": [xread(u).tolist() for u in range(NU)], "red": [xredread(u).tolist() for u in range(NU)]})
        for k in range(S_):
            i = t + k - (NP - 1)
            if i < 0:
                continue
            best = int(ring[k])
            got.append(best)
            if best != gold["tokens"][i]:
                mis.append({"i": i, "got": best, "want": gold["tokens"][i], "golden_margin": round(gold["margins"][i], 4) if "margins" in gold else None})
            if i < 4 or i % 32 == 0:
                print(json.dumps({"i": i, "got": best, "want": gold["tokens"][i], "wall_ms": round(walls[-1] * 1e3, 3), "S": S_}), flush=True)
        t += S_
    reps.append({"rep": rep, "tf_mismatches": len(mis), "mismatches": mis[:20], "wall_ms_median": round(float(np.median(walls)) * 1e3, 3), "tokens": got})
    print("REP " + json.dumps({k: v for k, v in reps[-1].items() if k != "tokens"}), flush=True)
PROFLINE = [None]
times = None
if a.tdbg or a.prof:
    rb_ = get("tdbg", *ROOT, 1, 1, 70, np.uint32).reshape(-1)
    times = {"loop_cycles": [int(v) for v in np.diff(rb_[2:2 + min(WLS, 64)].astype(np.int64)) % (1 << 32)]}
    cx = []
    for u in range(NU):
        X0u, Y0u = g.unit_origin(u)
        tb = get("tdbg", X0u + XC, Y0u, 1, 1, K + 2 + 4 * K, np.uint32).reshape(-1)[:K + 2]
        cx.append([int(v) for v in np.diff(tb.astype(np.int64)) % (1 << 32)])
    times["cx_layers"] = cx
    print("TIMES " + json.dumps(times), flush=True)
PROF = {}
if a.prof:
    # the last step's stamps (TSC, the same clock on every PE: checked on the CS-3), relative to unit 0's CX row 0 pass
    # start: CX [0 pass start, 1 + l layer ends, K + 1 x out]; attention PE [0, then a layer: x in, qkv sent, mixer done,
    # o_proj sent + x2 dropped]; MoE PE [0, then: att x dropped, routes in, gate|up sent, down sent]; CYs [0, then:
    # reduce in, multicast out, done]
    def pe(x_, y_, n_):
        return get("tdbg", x_, y_, 1, 1, n_, np.uint32).reshape(-1).astype(np.int64)
    raw = {}
    for u in [int(v) for v in a.prof.split(",")]:
        X0p, Y0p = g.unit_origin(u)
        for r in (0, PY // 2, PY - 1):
            raw[f"u{u}_cx_r{r}"] = pe(X0p + XC, Y0p + g.wl(r, YC), K + 2 + 4 * K)
        raw[f"u{u}_wa_r0_c0"] = pe(X0p, Y0p, 1 + 6 * K)
        raw[f"u{u}_wa_r159_c63"] = pe(X0p + AX - 1, Y0p + g.wl(PY - 1, YC), 1 + 6 * K)
        raw[f"u{u}_wm_r0_c0"] = pe(X0p + XC + 1, Y0p, 1 + 4 * K)
        raw[f"u{u}_wm_r159_c287"] = pe(X0p + XC + MXc, Y0p + g.wl(PY - 1, YC), 1 + 4 * K)
        raw[f"u{u}_cya_c0"] = pe(X0p, Y0p + YC, 1 + 4 * K)[:1 + 3 * K]
        raw[f"u{u}_cym_c0"] = pe(X0p + XC + 1, Y0p + YC, 1 + 4 * K)
    base = int(raw["u0_cx_r0"][0]) if "u0_cx_r0" in raw else int(next(iter(raw.values()))[0])
    def rel(v):
        d = (int(v) - base) % (1 << 32)
        return d if d < (1 << 31) else d - (1 << 32)
    PROF = {k: [rel(v) for v in arr] for k, arr in raw.items()}
    # every CX row's own layer intervals (clock-offset free): x in -> layer 0 end, layer 0 -> layer 1 end, by unit
    PROF["cx_rows"] = {}
    for u in [int(v) for v in a.prof.split(",")]:
        X0p, Y0p = g.unit_origin(u)
        rows_ = []
        for (r0, r1) in HALF:
            t_ = get("tdbg", X0p + XC, Y0p + g.wl(r0, YC), 1, r1 - r0, K + 2 + 4 * K, np.uint32).reshape(r1 - r0, K + 2 + 4 * K).astype(np.int64)
            rows_ += [[int(v) for v in np.diff(row[:K + 2]) % (1 << 32)] for row in t_]
        PROF["cx_rows"][f"u{u}"] = rows_
    PROF["redo"] = [[int(v) for v in pe(g.unit_origin(u)[0], g.unit_origin(u)[1] + YC, 1 + 4 * K)[1 + 3 * K:]] for u in range(NU)]
    PROFLINE[0] = "PROF " + json.dumps({k: v for k, v in PROF.items() if k.endswith("_cx_r0") or k.endswith("_cx_r159") or k == "redo"}, separators=(",", ":"))
runner.stop()
if dumps:
    json.dump(dumps, open(a.report.replace(".json", "_dump.json"), "w"))
rep = {"prof": PROF, "diverged": diverged[:4], "force": a.force, "greedy": a.greedy, "wloop": WLS, "ntok": ntok, "reps": reps, "ring_bad": ring_bad[:20],
       "load_s": round(t_load, 1), "place_s": round(t_place, 1), "times": times,
       "wall_ms_median": reps[-1]["wall_ms_median"], "tok_s_wall": round(1e3 / max(reps[-1]["wall_ms_median"], 1e-6), 1)}
json.dump(rep, open(a.report, "w"), indent=1)
lc = (times or {}).get("loop_cycles") or []
print("GPTOSS " + json.dumps({k: v for k, v in rep.items() if k not in ("reps", "times", "prof")}), flush=True)
if lc:
    print("LOOP median %d min %d max %d n %d" % (sorted(lc)[len(lc) // 2], min(lc), max(v for v in lc if v < 1 << 31), len(lc)), flush=True)
if PROFLINE[0]:
    print(PROFLINE[0], flush=True)
