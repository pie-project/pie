"""prims/att/run.py <compiled dir> <golden dir> <node> <NH> <PYA> <TB> <scale>: an attn.ragged node of the
stream step (engine-cpu's flux_2_minis_primitive_goldens: q, k, v = n<node>_in0..2, the step's rows packed,
one group, non-causal; o = n<node>_out0) through P7 v0 on the simulator, a kv head a column: store every
row's k / v (TB rows a launch), then every query. Prints the error against softmax(q k^T scale) v in f64 and
against the trace's own output, and the cycles a stored row and a query (the root's), and the passes."""
import json, sys
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

out_dir, gold, node = sys.argv[1], sys.argv[2], int(sys.argv[3])
NH, PYA, TB, SC = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
q, k, v = (np.load(f"{gold}/n{node}_in{i}.npy").astype(np.float64) for i in range(3))
ot = np.load(f"{gold}/n{node}_out0.npy").astype(np.float64)
T, W = q.shape
Dc = W // NH
kw = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
F32 = MemcpyDataType.MEMCPY_32BIT
r = SdkRuntime(out_dir)
r.load()
r.run()


def stamps():
    ts = np.zeros(NH * (PYA + 1) * 3, np.uint32)
    r.memcpy_d2h(ts, r.get_id("tsc"), 0, 0, NH, PYA + 1, 3, data_type=F32, **kw)
    ts = ts.reshape(PYA + 1, NH, 3).astype(np.int64)
    return ((ts[PYA, :, 1] - ts[PYA, :, 0]) % (1 << 32)).max(), ts[PYA, :, 2]


store, attend, got = [], [], np.zeros((T, W))
for t0 in range(0, T, TB):
    nb = min(TB, T - t0)
    kv = np.zeros((1, NH, TB, 2 * Dc), np.float32)
    for h in range(NH):
        kv[0, h, :nb, :Dc] = k[t0:t0 + nb, h * Dc:(h + 1) * Dc]
        kv[0, h, :nb, Dc:] = v[t0:t0 + nb, h * Dc:(h + 1) * Dc]
    r.memcpy_h2d(r.get_id("kvb"), kv.reshape(-1), 0, PYA, NH, 1, TB * 2 * Dc, data_type=F32, **kw)
    r.launch("run", np.uint16(0), np.uint16(t0), np.uint16(nb), nonblock=False)
    store.append(stamps()[0] / nb)
for t0 in range(0, T, TB):
    nb = min(TB, T - t0)
    qq = np.zeros((1, NH, TB, Dc), np.float32)
    for h in range(NH):
        qq[0, h, :nb] = q[t0:t0 + nb, h * Dc:(h + 1) * Dc]
    r.memcpy_h2d(r.get_id("qb"), qq.reshape(-1), 0, PYA, NH, 1, TB * Dc, data_type=F32, **kw)
    r.launch("run", np.uint16(1), np.uint16(t0), np.uint16(nb), nonblock=False)
    o = np.zeros(NH * TB * Dc, np.float32)
    r.memcpy_d2h(o, r.get_id("ob"), 0, PYA, NH, 1, TB * Dc, data_type=F32, **kw)
    got[t0:t0 + nb] = o.reshape(NH, TB, Dc)[:, :nb].transpose(1, 0, 2).reshape(nb, W)
    cyc, passes = stamps()
    attend.append(cyc / nb)
r.stop()
ref = np.zeros((T, W))
for h in range(NH):
    sl = slice(h * Dc, (h + 1) * Dc)
    s = q[:, sl] @ k[:, sl].T * SC
    p = np.exp(s - s.max(1, keepdims=True))
    ref[:, sl] = (p / p.sum(1, keepdims=True)) @ v[:, sl]
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
cos = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
res = dict(node=node, T=T, heads=NH, Dc=Dc, PYA=PYA, NE=-(-T // PYA), rel_vs_f64=rel(got, ref), cos_vs_f64=cos(got, ref),
           max_abs_vs_f64=float(np.abs(got - ref).max()), rel_vs_trace=rel(got, ot), cos_vs_trace=cos(got, ot),
           rel_trace_vs_f64=rel(ot, ref), cycles_per_store=float(np.median(store)),
           cycles_per_query=float(np.median(attend)), passes=[int(p) for p in passes], queries=T)
print("PRIM_ATT " + json.dumps(res))
