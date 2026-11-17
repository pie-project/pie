"""mamba_emu.py <nem1 slice> <golden.json> [variant...]: nemotron -l1 (one Mamba-2 layer) a token a step
as the unit lowering runs it (stream/mamba.rs, ops/cy_sst.csl MAMBA, ops/sst.csl, rg regions,
ops/smv.csl), in f32 with bf16 at the program's roundings and ops/math.csl's scalar functions,
against the same layer in f64 (exact), per op and per position; then the logits (the final norm
and head as the exit unit's) against HF's golden. Variants: exact (math.csl's functions exact),
trunc (every bf16 rounding a truncation), floor (dt floored at 0.001 in the exact run, as
transformers 5.23's native class does: G3.2)."""
import glob, json, os, sys, math
import numpy as np, torch
from safetensors import safe_open
torch.set_num_threads(8)
src, gold = sys.argv[1], sys.argv[2]
var = set(sys.argv[3:])
g = json.load(open(gold)); toks = g["tokens"]
W = {}
for f in glob.glob(src + "/*.safetensors"):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if "layers.0." in k or "layers" not in k:
                W[k.replace("backbone.", "").replace("layers.0.", "")] = s.get_tensor(k)
cfg = json.load(open(src + "/config.json"))
H, P, G, N, K = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["n_groups"], cfg["ssm_state_size"], cfg["conv_kernel"]
D_ = cfg["hidden_size"]; I = H * P; eps = cfg["layer_norm_epsilon"]
f32, f64 = torch.float32, torch.float64

def bf(x):
    if "trunc" in var:
        b = x.float().contiguous().view(torch.int32) & ~0xffff
        return b.view(torch.float32).to(x.dtype)
    return x.to(torch.bfloat16).to(x.dtype)

# ops/math.csl's scalar functions in f32 (numpy f32 arithmetic, step for step)
F = np.float32
def exp1(x0):
    x = F(min(max(F(x0), F(-87.0)), F(88.0)))
    k = F(x * F(1.4426950408889634) + F(12582912.0)); k = F(k - F(12582912.0))
    r = F(x - F(k * F(0.6931471805599453)))
    p = F(0.0013888889)
    for c in (0.008333334, 0.041666668, 0.16666667, 0.5, 1.0, 1.0):
        p = F(F(p * r) + F(c))
    return F(p * F(2.0) ** int(k))
def rsqrt(v):
    v = F(v); y = np.array([0x5f3759df - (np.array(v).view(np.uint32) >> 1)], dtype=np.uint32).view(np.float32)[0]
    for _ in range(3):
        y = F(y * F(F(1.5) - F(F(F(F(0.5) * v) * y) * y)))
    return y
def recip(v):
    a = -v if v < 0 else v; r = rsqrt(a); return F(-(r * r)) if v < 0 else F(r * r)
def sigm(x): return recip(F(F(1.0) + exp1(-x)))
def silu(x): return F(F(x) * sigm(x))
def ln1(x):
    b = np.array([F(x)]).view(np.uint32)[0]
    e = F(int((b >> 23) & 0xff) - 127)
    m = np.array([(b & 0x007fffff) | 0x3f800000], dtype=np.uint32).view(np.float32)[0]
    z = F(F(m - F(1.0)) * recip(F(m + F(1.0)))); z2 = F(z * z)
    p = F(z * F(F(2.0) + F(z2 * F(F(0.6666667) + F(z2 * F(F(0.4) + F(z2 * F(0.2857143))))))))
    return F(F(e * F(0.6931471805599453)) + p)
def softplus(x): return F(x) if x > 20.0 else ln1(F(F(1.0) + exp1(x)))
if "exact" in var:
    exp1 = lambda x: F(math.exp(min(max(float(x), -87), 88)))
    sigm = lambda x: F(1 / (1 + math.exp(-float(x))))
    silu = lambda x: F(float(x) / (1 + math.exp(-float(x))))
    softplus = lambda x: F(math.log1p(math.exp(float(x))) if x < 20 else x)

w = {k: v.to(f64) for k, v in W.items() if v.numel() < 50_000_000}
A_log, Dd, dtb = w["mixer.A_log"], w["mixer.D"], w["mixer.dt_bias"]
convw, convb = w["mixer.conv1d.weight"][:, 0, :], w["mixer.conv1d.bias"]

def run(exact):
    T = f64 if exact else f32
    S = torch.zeros(H, P, N, dtype=T); hist = torch.zeros(K - 1, I + 2 * G * N, dtype=T)
    out = {k: [] for k in ("norm", "inproj", "conv", "dt", "raw", "gated", "y", "mix", "hid", "logits")}
    for t in toks:
        h = W["embeddings.weight"][t].to(T)
        # rg_0_1: the layer's RMS norm, bf16
        r = (h * h).sum() / D_ + eps
        xn = h * (1 / torch.sqrt(r) if exact else torch.tensor(rsqrt(F(r.item())), dtype=T)) * w["norm.weight"].to(T)
        xn = xn if exact else bf(xn)
        out["norm"].append(xn)
        # in_proj: 16-bit x 16-bit, f32 accumulated; the head root rounds its 385 to bf16
        zx = (W["mixer.in_proj.weight"].to(T) @ xn)
        zx = zx if exact else bf(zx)
        out["inproj"].append(zx)
        z, xbc, dt = zx[:I], zx[I:I + I + 2 * G * N], zx[I + I + 2 * G * N:]
        # the conv (oldest first), its bias, SiLU (cy_sst.csl: scalar a channel)
        acc = convw[:, K - 1].to(T) * xbc + convb.to(T)
        for j in range(K - 1):
            acc = acc + convw[:, j].to(T) * hist[j]
        hist = torch.cat([hist[1:], xbc[None]], 0)
        if exact:
            y = acc * torch.sigmoid(acc)
        else:
            y = torch.tensor([silu(F(a)) for a in acc.numpy()], dtype=T)
        out["conv"].append(y)
        xs, B, C = y[:I].view(H, P), y[I:I + G * N].view(G, N), y[I + G * N:].view(G, N)
        raw = torch.zeros(H, P, dtype=T); dts = torch.zeros(H, dtype=T)
        for hd in range(H):
            gi = hd // (H // G)
            if exact:
                st = torch.nn.functional.softplus(dt[hd] + dtb[hd])
                if "floor" in var: st = st.clamp_min(0.001)
                dec = torch.exp(-torch.exp(A_log[hd]) * st)
            else:
                st = torch.tensor(softplus(F(dt[hd].item() + F(dtb[hd].item()))), dtype=T)
                dec = torch.tensor(exp1(F(-exp1(F(A_log[hd].item())) * st.item())), dtype=T)
            dts[hd] = st
            kq = (B[gi] * C[gi]).sum()
            d = st * xs[hd]
            # o = decay Sᵀq + (k.q) delta + D x; S = decay S + k ⊗ delta
            o = (S[hd] @ (dec * C[gi])) + kq * d + Dd[hd].to(T) * xs[hd]
            S[hd] = dec * S[hd] + d[:, None] * B[gi][None, :]
            raw[hd] = o
        out["dt"].append(dts); raw = raw.reshape(-1); out["raw"].append(raw)
        # rg_0_5: g = r silu(z), g²
        gt = raw * (z * torch.sigmoid(z))
        out["gated"].append(gt)
        ss = (gt * gt).view(G, -1).sum(-1)
        yy = gt * (1 / torch.sqrt(ss / (I // G) + eps)).repeat_interleave(I // G) * w["mixer.norm.weight"].to(T)
        yy = yy if exact else bf(yy)
        out["y"].append(yy)
        mix = W["mixer.out_proj.weight"].to(T) @ yy
        out["mix"].append(mix)
        hid = mix + h
        hid = hid if exact else bf(hid)
        out["hid"].append(hid)
        r = (hid * hid).sum() / D_ + eps
        fn = hid / torch.sqrt(r) * w["norm_f.weight"].to(T)
        fn = fn if exact else bf(fn)
        out["logits"].append((W["lm_head.weight"].to(T) @ fn))
    return {k: torch.stack(v).to(f64) for k, v in out.items()}

ex = run(True); em = run(False)
# HF=<hf.pt> (hf_mamba_ops.py's per-op values): the exact lowering against HF f32 per op
if os.environ.get("HF"):
    hf = torch.load(os.environ["HF"])
    zx = hf["inproj"]
    for k, a, b in (("inproj", ex["inproj"], zx), ("raw", ex["raw"], hf["scan"]), ("y", ex["y"], hf["y"]), ("mix", ex["mix"], hf["mix"]), ("hid", ex["hid"], hf["hid"]), ("logits", ex["logits"], hf["logits"])):
        d = (a - b).norm(dim=-1) / b.norm(dim=-1)
        print(f"exact vs HF f32 {k:7s}: max {d.max():.2e} mean {d.mean():.2e} first {', '.join(f'{x:.1e}' for x in d[:6].tolist())}")
print("variant", sorted(var) or ["program"])
for k in ex:
    d = (em[k] - ex[k]).norm(dim=-1) / ex[k].norm(dim=-1).clamp_min(1e-30)
    print(f"{k:7s} rel_l2 per position: max {d.max():.2e} mean {d.mean():.2e}  first {', '.join(f'{x:.1e}' for x in d[:6].tolist())}")
hidg = torch.tensor(g["hidden"], dtype=f64) if "hidden" in g else None
if hidg is not None:
    for name, v in (("exact", ex["hid"]), ("emu", em["hid"])):
        d = (v - hidg).norm(dim=-1) / hidg.norm(dim=-1)
        print(f"hidden vs golden.hidden ({name}): max {d.max():.2e} mean {d.mean():.2e}")
P0 = len(g["prompt"])
for s in range(P0 - 1, len(toks) - 1):
    le, lx = em["logits"][s], ex["logits"][s]
    want = toks[s + 1]; a = int(le.argmax()); b = int(lx.argmax())
    top = torch.topk(lx, 2).values
    print(f"step {s} want {want} exact {b} (gap {float(top[0]-top[1]):.3f}) emu {a} {'ok' if a == want else 'MISS'}  emu-exact at want {float(le[want]-lx[want]):+.3f}  max|dlogit| {float((le-lx).abs().max()):.3f}")
