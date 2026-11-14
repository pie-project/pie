"""prims/reg/run.py <compiled dir> <case dir> <TB>: a printed root region (the prim_region example's case:
reg.csl, rows / res / tabs / want / trace .npy, meta.json) on the simulator, TB rows a launch. Prints the
error against kernels_ir::interp's results and against the trace's, and the cycles a row (the region's
row() alone, and the whole loop with the harness's copies)."""
import json, sys
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

out_dir, case, TB = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = json.load(open(f"{case}/meta.json"))
R, C, T, REC, NOUT = m["roots"], m["chunk"], m["rows"], m["rec"], m["nout"]
rows = np.load(f"{case}/rows.npy")
res = np.load(f"{case}/res.npy")
tabs = np.load(f"{case}/tabs.npy")
want = np.load(f"{case}/want.npy").astype(np.float64)
trace = np.load(f"{case}/trace.npy").astype(np.float64)
kw = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
F32, U32 = MemcpyDataType.MEMCPY_32BIT, MemcpyDataType.MEMCPY_32BIT
r = SdkRuntime(out_dir)
r.load()
r.run()
# the segment's words: chain east into root R-1, its sum multicast west on colour 4
M, PARK = 4, 12
rc = np.zeros((R, 18), np.uint32)
for x in range(R):
    ch = lambda i: 2 + i % 2
    if x == R - 1:
        rc[x, [4, 5, 13, 15, 16, 17]] = [PARK, ch(x - 1) if x > 0 else 11, ch(x - 1) if x > 0 else 11, M, PARK, 6]
    else:
        rc[x, [4, 5, 13, 15, 16, 17]] = [M, ch(x - 1) if x > 0 else 11, ch(x - 1) if x > 0 else 11, ch(x), M, 0 if x == 0 else 1]
r.memcpy_h2d(r.get_id("rcfg"), rc.reshape(-1), 0, 0, R, 1, 18, data_type=U32, **kw)
r.memcpy_h2d(r.get_id("res"), np.ascontiguousarray(res, np.float32).reshape(-1), 0, 0, R, 1, res.shape[1], data_type=F32, **kw)
r.launch("run", np.uint16(0), np.uint16(0), np.uint16(0), nonblock=False)
NT = tabs.shape[1]
for a in range(0, NT, 1024):
    n = min(1024, NT - a)
    st = np.zeros((R, 1024), np.float32)
    st[:, :n] = tabs[:, a:a + n]
    r.memcpy_h2d(r.get_id("stage"), st.reshape(-1), 0, 0, R, 1, 1024, data_type=F32, **kw)
    r.launch("run", np.uint16(1), np.uint16(a), np.uint16(n), nonblock=False)
got = np.zeros((T, NOUT, R * C))
loop, inner = [], []
for t0 in range(0, T, TB):
    nb = min(TB, T - t0)
    rb = np.zeros((R, TB, REC), np.float32)
    rb[:, :nb] = rows[t0:t0 + nb].transpose(1, 0, 2)
    r.memcpy_h2d(r.get_id("rows"), rb.reshape(-1), 0, 0, R, 1, TB * REC, data_type=F32, **kw)
    r.launch("run", np.uint16(2), np.uint16(0), np.uint16(nb), nonblock=False)
    o = np.zeros(R * TB * NOUT * C, np.float32)
    r.memcpy_d2h(o, r.get_id("outs"), 0, 0, R, 1, TB * NOUT * C, data_type=F32, **kw)
    o = o.reshape(R, TB, NOUT, C)[:, :nb]
    got[t0:t0 + nb] = o.transpose(1, 2, 0, 3).reshape(nb, NOUT, R * C)
    ts = np.zeros(R * 3, np.uint32)
    r.memcpy_d2h(ts, r.get_id("tsc"), 0, 0, R, 1, 3, data_type=U32, **kw)
    ts = ts.reshape(R, 3).astype(np.int64)
    loop.append(((ts[:, 1] - ts[:, 0]) % (1 << 32)).max() / nb)
    inner.append(ts[:, 2].max() / nb)
r.stop()
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
res = dict(nodes=m["nodes"], width=m["width"], chunk=C, roots=R, rows=T, ops=m["ops"], round16=m["round16"],
           all_reduces=m["all_reduces"], rel_vs_interp=rel(got, want), max_abs_vs_interp=float(np.abs(got - want).max()),
           rel_vs_trace=rel(got, trace), rel_interp_vs_trace=rel(want, trace),
           cycles_per_row_region=float(np.median(inner)), cycles_per_row_loop=float(np.median(loop)))
print("PRIM_REG " + json.dumps(res))
