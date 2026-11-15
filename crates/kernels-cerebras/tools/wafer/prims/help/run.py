"""prims/help/run.py <compiled dir> <case dir> <R> <H> <T>: a column region over helpers (layout.csl) on
the simulator: prim_region's case printed at the helpers' chunk Q (R (H + 1) of its roots), each column's
root holding H + 1 of them; the results against kernels_ir::interp's, and the cycles a row at a root."""
import json, sys
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

out_dir, case, R, H, T = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
m = json.load(open(f"{case}/meta.json"))
Q, NR = m["chunk"], m["roots"]
D = H + 1
C = Q * D
assert NR == R * D and m["nout"] == 1 and len(m["params"]) == 2, m
rows = np.load(f"{case}/rows.npy")[:T]          # [T, NR, 2 Q]
want = np.load(f"{case}/want.npy").astype(np.float64)[:T, 0]   # [T, width]
kw = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
F32, U32 = MemcpyDataType.MEMCPY_32BIT, MemcpyDataType.MEMCPY_32BIT
r = SdkRuntime(out_dir)
r.load()
r.run()
# each root's rows: param 0's whole chunk, then param 1's
rr = np.zeros((R, T, 2 * C), np.float32)
for x in range(R):
    g = rows[:, x * D:(x + 1) * D]                  # [T, D, 2Q]
    rr[x, :, :C] = g[:, :, :Q].reshape(T, C)
    rr[x, :, C:] = g[:, :, Q:].reshape(T, C)
r.memcpy_h2d(r.get_id("rows"), rr.reshape(-1), 0, 0, R, 1, T * 2 * C, data_type=F32, **kw)
# the helpers' config words: their place k (1 .. H) and chain role (0: the leaf, which starts the sum)
pc = np.zeros((H, R, 2), np.uint32)
for y in range(1, H + 1):
    pc[y - 1, :, 0] = y
    pc[y - 1, :, 1] = 0 if y == H else 1
r.memcpy_h2d(r.get_id("pcfg"), pc.reshape(-1), 0, 1, R, H, 2, data_type=U32, **kw)
r.launch("run", nonblock=False)
o = np.zeros(R * T * C, np.float32)
r.memcpy_d2h(o, r.get_id("outs"), 0, 0, R, 1, T * C, data_type=F32, **kw)
got = o.reshape(R, T, C).transpose(1, 0, 2).reshape(T, R * C).astype(np.float64)
ts = np.zeros(R * 3, np.uint32)
r.memcpy_d2h(ts, r.get_id("tsc"), 0, 0, R, 1, 3, data_type=U32, **kw)
ts = ts.reshape(R, 3).astype(np.int64)
r.stop()
rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
res = dict(nodes=m["nodes"], chunk=C, helpers=H, q=Q, roots=R, rows=T, rel_vs_interp=rel,
           max_abs=float(np.abs(got - want).max()), cycles_per_row=float(((ts[:, 1] - ts[:, 0]) % (1 << 32)).max() / T))
print("PRIM_HELP " + json.dumps(res))
