"""prims/mv/run.py <compiled dir> <golden dir> <node> <PX> <PY> <TB> <bf16|ieee>: a matmul node of the
stream step (engine-cpu's flux_2_minis_primitive_goldens: x = n<node>_in0, w = n<node>_in1 [N, K],
y = n<node>_out0) through P1-stream on the simulator, TB rows a launch. Prints the error against the
product of the rows as the PEs hold them (x rounded to the 16-bit type, w's bf16 values in it, f64), and
against the trace's own output, and the cycles a row (a weight PE's matvec loop; the root's arrivals)."""
import json, sys
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

out_dir, gold, node = sys.argv[1], sys.argv[2], int(sys.argv[3])
PX, PY, TB, fmt = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
x = np.load(f"{gold}/n{node}_in0.npy")
w = np.load(f"{gold}/n{node}_in1.npy")
yt = np.load(f"{gold}/n{node}_out0.npy")
live = np.isfinite(x).all(1) & np.isfinite(yt).all(1)
x, yt = x[live].astype(np.float32), yt[live]
T, K = x.shape
N = w.shape[0]
Hc, Nc = K // PY, N // PX
assert Hc * PY == K and Nc * PX == N


def bits16(a):
    a = np.ascontiguousarray(a, np.float32)
    if fmt == "ieee":
        return a.astype(np.float16).view(np.uint16).astype(np.uint32)
    b = a.view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint32)


def held16(a):
    if fmt == "ieee":
        return np.asarray(a, np.float32).astype(np.float16).astype(np.float64)
    return (bits16(a) << 16).astype(np.uint32).view(np.float32).astype(np.float64)


kw = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
r = SdkRuntime(out_dir)
r.load()
r.run()
# the tiles: PE (1 + i, j) holds w[i Nc .. , j Hc ..] transposed, Hc rows of Nc
tiles = np.zeros((PY, PX, Hc * Nc + 1), np.uint32)
for j in range(PY):
    for i in range(PX):
        tiles[j, i, 1:] = bits16(w[i * Nc:(i + 1) * Nc, j * Hc:(j + 1) * Hc].T).reshape(-1)
r.memcpy_h2d(r.get_id("w"), tiles.reshape(-1), 1, 0, PX, PY, Hc * Nc + 1,
             data_type=MemcpyDataType.MEMCPY_16BIT, **kw)
got = np.zeros((T, N), np.float32)
cyc = []
for t0 in range(0, T, TB):
    nb = min(TB, T - t0)
    xs = np.zeros((PY, 1, TB * Hc), np.float32)
    for j in range(PY):
        xs[j, 0, :nb * Hc] = x[t0:t0 + nb, j * Hc:(j + 1) * Hc].reshape(-1)
    r.memcpy_h2d(r.get_id("xs"), xs.reshape(-1), 0, 0, 1, PY, TB * Hc, data_type=MemcpyDataType.MEMCPY_32BIT, **kw)
    r.launch("run", np.uint16(nb), nonblock=False)
    o = np.zeros(PX * TB * Nc, np.float32)
    r.memcpy_d2h(o, r.get_id("out"), 1, PY, PX, 1, TB * Nc, data_type=MemcpyDataType.MEMCPY_32BIT, **kw)
    o = o.reshape(PX, TB, Nc)[:, :nb]
    got[t0:t0 + nb] = o.transpose(1, 0, 2).reshape(nb, N)
    ts = np.zeros((PY + 1) * (PX + 1) * 2, np.uint32)
    r.memcpy_d2h(ts, r.get_id("tsc"), 0, 0, PX + 1, PY + 1, 2, data_type=MemcpyDataType.MEMCPY_32BIT, **kw)
    ts = ts.reshape(PY + 1, PX + 1, 2).astype(np.int64)
    span = (ts[..., 1] - ts[..., 0]) % (1 << 32)
    cyc.append(dict(rows=nb, weight=float(span[:PY, 1:].max()) / nb,
                    root=float(span[PY, 1:].max()) / max(nb - 1, 1)))
r.stop()
ref = held16(x) @ held16(w).T
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
cos = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
res = dict(node=node, K=K, N=N, T=T, PX=PX, PY=PY, Hc=Hc, Nc=Nc, fmt=fmt,
           rel_vs_held_product=rel(got, ref), max_abs_vs_held=float(np.abs(got - ref).max()),
           rel_vs_trace=rel(got, yt.astype(np.float64)), cos_vs_trace=cos(got.astype(np.float64), yt.astype(np.float64)),
           rel_trace_vs_f64=rel(yt.astype(np.float64), x.astype(np.float64) @ w.T.astype(np.float64)),
           cycles_per_row_weight=float(np.median([c["weight"] for c in cyc])),
           cycles_per_row_root=float(np.median([c["root"] for c in cyc if c["rows"] > 1])),
           elems_per_pe=Hc * Nc)
res["cycles_per_elem"] = res["cycles_per_row_root"] / (Hc * Nc)
print("PRIM_MV " + json.dumps(res))
