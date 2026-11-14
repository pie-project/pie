"""stream/run.py <compiled dir> <program dir> [diffusers velocity .f32]: one step of a stream reading on the
simulator, as wafer-compile stream laid it (host.json): every array the host writes (weights, tables,
the host's rows), one launch of `run`, the host's rows read back and compared with the step's own values
(engine-cpu's Forward::stream_capture: rel L2, cosine), and the step's cycles off every PE's stamps. With a
diffusers golden (flux2-mini: mk_tiny_flux2.py's velocity), the output is unpatchified and compared to it."""
import json, sys, time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

out_dir, prog = sys.argv[1], sys.argv[2]
host = json.load(open(f"{prog}/host.json"))
W, H = host["w"], host["h"]
kw = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
F32, U16 = MemcpyDataType.MEMCPY_32BIT, MemcpyDataType.MEMCPY_16BIT
t0 = time.time()
r = SdkRuntime(out_dir)
r.load()
r.run()
for a in host["h2d"]:
    d = np.load(f"{prog}/{a['file']}")
    data = d.reshape(-1).astype(np.uint32) if a["u16"] else d.reshape(-1).astype(np.float32)
    r.memcpy_h2d(r.get_id(a["sym"]), data, a["x"], a["y"], a["w"], a["h"], a["len"],
                 data_type=U16 if a["u16"] else F32, **kw)
t1 = time.time()
print(f"loaded {len(host['h2d'])} arrays in {t1 - t0:.0f} s", flush=True)
r.launch("run", nonblock=False)
t2 = time.time()
print(f"step ran in {t2 - t1:.0f} s", flush=True)
got = {}
# the host rows out, a column of roots one memcpy
cols = {}
for o in host["d2h"]:
    cols.setdefault((o["sym"], o["x"], o["len"]), []).append(o)
bufs = {}
for (sym, x, n), os_ in cols.items():
    os_.sort(key=lambda o: o["y"])
    runs = [[os_[0]]]
    for o in os_[1:]:
        if o["y"] == runs[-1][-1]["y"] + 1:
            runs[-1].append(o)
        else:
            runs.append([o])
    for run in runs:
        b = np.zeros(len(run) * n, np.float32)
        r.memcpy_d2h(b, r.get_id(sym), x, run[0]["y"], 1, len(run), n, data_type=F32, **kw)
        for i, o in enumerate(run):
            bufs[(x, o["y"])] = b[i * n:(i + 1) * n]
for o in host["d2h"]:
    buf = bufs[(o["x"], o["y"])]
    width = sum(p[2] for p in o["parts"])
    buf = buf.reshape(len(o["rows"]), width)
    off = 0
    for base, frm, n in o["parts"]:
        got.setdefault(base, {})
        for i, tr in enumerate(o["rows"]):
            got[base].setdefault(tr, {})[frm] = buf[i, off:off + n]
        off += n
ts = np.zeros(W * H * 3, np.uint32)
r.memcpy_d2h(ts, r.get_id("tsc"), 0, 0, W, H, 3, data_type=F32, **kw)
r.stop()
ts = ts.reshape(H, W, 3).astype(np.int64)
np.save(f"{out_dir}/../stamps.npy", ts)
start, end = ts[..., 0], ts[..., 1]
live = (end != 0) | (start != 0)
span = int((end[live].max() - start[live].min()) % (1 << 32))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
cos = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
res = dict(cycles=span, load_s=round(t1 - t0), step_s=round(t2 - t1))
for ref in host["refs"]:
    want = np.load(f"{host['gold']}/{ref['file']}").astype(np.float64)
    rows = sorted(got[ref["value"]])
    mine = np.zeros((len(rows), want.shape[1]))
    for i, tr in enumerate(rows):
        for frm, v in got[ref["value"]][tr].items():
            mine[i, frm:frm + len(v)] = v
    w = want[rows]
    res[f"v{ref['value']}"] = dict(rows=len(rows), rel_l2=rel(mine, w), cosine=cos(mine, w),
                                   max_abs=float(np.abs(mine - w).max()), finite=bool(np.isfinite(mine).all()))
    np.save(f"{out_dir}/../v{ref['value']}.npy", mine)
    if len(sys.argv) > 3:
        # diffusers' velocity [voxels, channels]: the patches unpatchified (p = 1 x 2 x 2 over a square grid)
        dv = np.fromfile(sys.argv[3], np.float32).astype(np.float64)
        n, cw = mine.shape
        g = int(round(np.sqrt(n)))
        c = cw // 4
        best = None
        for order in ("c,h,w", "h,w,c"):
            if order == "c,h,w":
                x = mine.reshape(g, g, c, 2, 2).transpose(0, 3, 1, 4, 2)
            else:
                x = mine.reshape(g, g, 2, 2, c).transpose(0, 2, 1, 3, 4)
            x = x.reshape(-1, c)
            if x.size == dv.size:
                rl = rel(x.reshape(-1), dv)
                if best is None or rl < best[0]:
                    best = (rl, cos(x.reshape(-1), dv), order)
        if best:
            res["diffusers"] = dict(rel_l2=best[0], cosine=best[1], order=best[2])
pe = ts[..., 1] - ts[..., 0]
res["passes"] = [int(x) for x in ts[..., 2][ts[..., 2] > 0][:16]]
print("STREAM " + json.dumps(res))
