#!/usr/bin/env python3
"""prep_gemma.py <model dir> <outdir> <golden json> <plan dir> [prefix]: a Gemma 4 checkpoint (bf16
safetensors, "model.language_model." or "model.") in the wafer program tile format (src/spatial/README.md, "Gemma 4"), cut as
the plan in <plan dir> says (`wafer-compile plan`'s tiling.json).
Units of K layers (S .. S G), unit u = layers K u .. K u + K - 1; units below NO (= first shared layer / K) own their
KV, the rest are shared-KV units. Weight column c (of PX): kv copy c // (PX / 2) (its Qh = nq / 2 q heads), within
the copy p = c % (PX / 2): Dc / 2 rotated pairs (j, j + hd / 2), j = p Dc / 2 + t, side by side; PLE dim c.
Writes w<u> (f16 bits), gam (f32: a layer's input / post-attention / pre-FFN / post-FFN / post-PLE gammas, the layer
scalar; + the embedding scale), qkw, rot, hw, hg, hemb, embed, embrows, pt<col>_<part> (the per-layer inputs, f16,
precomputed for every vocab id: (RMSNorm(Wpm e) + 16 T) / sqrt 2), golden.json, meta.json."""
import glob, json, os, shutil, struct, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compiler

MD, out, GOLD = sys.argv[1:4]
PRE = sys.argv[5] if len(sys.argv) > 5 else ""
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(MD, "config.json")))
tc = cfg.get("text_config", cfg)
T = {}
for f in sorted(glob.glob(MD + "/*.safetensors")):
    b = open(f, "rb"); n = struct.unpack("<Q", b.read(8))[0]; h = json.loads(b.read(n)); base = 8 + n
    for k, v in h.items():
        if k == "__metadata__": continue
        T[k] = (f, base + v["data_offsets"][0], v["dtype"], v["shape"])
P = "model.language_model." if any(k.startswith("model.language_model.") for k in T) else "model."
def get(name):
    f, off, dt, shp = T[name]
    assert dt == "BF16", (name, dt)
    a = np.memmap(f, np.uint16, "r", off, tuple(shp))
    return (a.astype(np.uint32) << 16).view(np.float32)
def bits(name):
    f, off, dt, shp = T[name]
    return np.memmap(f, np.uint16, "r", off, tuple(shp))
def f16(a):
    return np.asarray(a, np.float32).astype(np.float16).view(np.uint16)

g = compiler.tiling(sys.argv[4])
PRESET = g.preset
PX, PY, XC = g.PX, g.PY, g.XC
H, NL, I = tc["hidden_size"], tc["num_hidden_layers"], tc["intermediate_size"]
nq, nkv = tc["num_attention_heads"], tc["num_key_value_heads"]
hdS = tc["head_dim"]
hdG = tc.get("global_head_dim") or next(v["head_dim"] for k, v in (tc.get("per_layer_config") or {}).items() if "head_dim" in v)
PLE = tc["hidden_size_per_layer_input"]
V = tc["vocab_size"]
NU = len(g.origins)
K = NL // NU
assert NL == K * NU and nkv in (1, 2) and PLE == PX   # 1 kv head: replicated over the halves; 2: one a half
kinds = ["G" if t == "full_attention" else "S" for t in tc["layer_types"]]
assert all(kinds[u * K:(u + 1) * K] == ["S"] * (K - 1) + ["G"] for u in range(NU)), "every unit S .. S G"
KM = 1 << (K - 1)
FS = NL - tc.get("num_kv_shared_layers", 0)
assert FS % K == 0
NO = FS // K
Hc = H // PY
assert Hc * PY == H and g.Hc == Hc
cpa = PX // 2
Qh = nq // 2
DcS, DcG = hdS // cpa, hdG // cpa
assert DcS * cpa == hdS and DcG * cpa == hdG and DcS % 2 == 0 and DcG % 2 == 0
IcO = I // PX
IcS = (2 * I if tc.get("use_double_wide_mlp") else I) // PX
rp = tc["rope_parameters"]
c = np.arange(PX)
cp, pa = c // cpa, c % cpa
def pdims(hd, Dc):
    nr = Dc // 2
    return np.array([[d for t in range(nr) for d in (p * nr + t, p * nr + t + hd // 2)] for p in pa])   # [PX][Dc]
pdS, pdG = pdims(hdS, DcS), pdims(hdG, DcG)
assert sorted(set(pdS[:cpa].reshape(-1).tolist())) == list(range(hdS))
assert sorted(set(pdG[:cpa].reshape(-1).tolist())) == list(range(hdG))
qh = (Qh * cp)[:, None] + np.arange(Qh)                          # [PX][Qh]

def tile_in(W, idx):
    N = idx.shape[1]
    S_ = f16(W[idx.reshape(-1)]).reshape(PX, N, PY, Hc)
    return S_.transpose(2, 0, 3, 1).reshape(PY, PX, -1)
def tile_out(W, idx):
    N = idx.shape[1]
    S_ = f16(W[:, idx.reshape(-1)]).reshape(PY, Hc, PX, N)
    return S_.transpose(0, 2, 3, 1).reshape(PY, PX, -1)

GL = 5 * Hc + 1
QKS = Qh * DcG + DcG
gam = np.zeros((NU, PY, K * GL + 1), np.float32)
qkw = np.zeros((NU, PX, K * QKS), np.float32)
gam[:, :, K * GL] = np.float32(np.sqrt(np.float32(H)))
LWS = []
for u in range(NU):
    parts = []
    sh = u >= NO
    Ic = IcS if sh else IcO
    for ll in range(K):
        l = K * u + ll
        L = P + f"layers.{l}."
        A = L + "self_attn."
        hd, Dc, pd = (hdG, DcG, pdG) if kinds[l] == "G" else (hdS, DcS, pdS)
        for i, nm in enumerate(("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                                "post_feedforward_layernorm", "post_per_layer_input_norm")):
            gam[u, :, ll * GL + i * Hc:ll * GL + (i + 1) * Hc] = get(L + nm + ".weight").reshape(PY, Hc)
        gam[u, :, ll * GL + 5 * Hc] = get(L + "layer_scalar")[0]
        qrows = (qh[:, :, None] * hd + pd[:, None, :]).reshape(PX, Qh * Dc)
        Wq = get(A + "q_proj.weight")
        if sh:
            parts.append(tile_in(Wq, qrows))
        else:
            Win = np.concatenate([Wq, get(A + "k_proj.weight"), get(A + "v_proj.weight")], 0)
            kvo = (cp * hd)[:, None] if nkv == 2 else 0           # the column's kv head (2 kv heads: its half's)
            idx = np.concatenate([qrows, nq * hd + kvo + pd, nq * hd + nkv * hd + kvo + pd], 1)
            parts.append(tile_in(Win, idx))
        parts.append(tile_out(get(A + "o_proj.weight"), qrows))
        qn = get(A + "q_norm.weight")
        qkw[u, :, ll * QKS:ll * QKS + Qh * Dc] = np.tile(qn[pd], (1, Qh))
        if not sh:
            qkw[u, :, ll * QKS + Qh * Dc:ll * QKS + Qh * Dc + Dc] = get(A + "k_norm.weight")[pd]
        M = L + "mlp."
        Ii = get(M + "gate_proj.weight").shape[0]
        assert Ii == Ic * PX, (l, Ii, Ic)
        gi = (c * Ic)[:, None] + np.arange(Ic)
        Wgu = np.concatenate([get(M + "gate_proj.weight"), get(M + "up_proj.weight")], 0)
        parts.append(tile_in(Wgu, np.concatenate([gi, Ii + gi], 1)))
        parts.append(tile_out(get(M + "down_proj.weight"), gi))
        parts.append(tile_in(get(L + "per_layer_input_gate.weight"), c[:, None]))
        parts.append(tile_out(get(L + "per_layer_projection.weight"), c[:, None]))
    blob = np.concatenate(parts, 2)
    LWS.append(blob.shape[2])
    np.save(os.path.join(out, PRE + f"w{u}.npy"), blob)
    print("unit", u, "shared" if sh else "owner", blob.shape, flush=True)
np.save(os.path.join(out, PRE + "gam.npy"), gam)
np.save(os.path.join(out, PRE + "qkw.npy"), qkw)
# RoPE tables a column: [kind][4][8][cos, sin][2 pairs]
rot = np.zeros((PX, 2, 4, 8, 2, 2), np.float64)
for kd, (hd, Dc, key) in enumerate(((hdS, DcS, "sliding_attention"), (hdG, DcG, "full_attention"))):
    th = float(rp[key]["rope_theta"])
    ra = int(rp[key].get("partial_rotary_factor", 1.0) * hd // 2) if rp[key].get("rope_type") == "proportional" else hd // 2
    nr = Dc // 2
    for p in range(PX):
        for t in range(2):
            j = (p % cpa) * nr + t
            f = (1.0 / th ** (2 * j / hd)) if (t < nr and j < ra) else 0.0
            f = float(np.float32(f))
            for lv in range(4):
                for d in range(8):
                    ang = d * 8 ** (3 - lv) * f
                    rot[p, kd, lv, d, 0, t] = np.cos(ang); rot[p, kd, lv, d, 1, t] = np.sin(ang)
np.save(os.path.join(out, PRE + "rot.npy"), rot.astype(np.float32).reshape(PX, -1))
# head (tied)
HC, HDc, HR = g.HC, 2 * Hc, g.HR
VR = -(-V // HR)
VR = -(-VR // 8) * 8
assert HC * HDc == H and HR * VR >= V
eb = bits(P + "embed_tokens.weight")
Wh = (np.asarray(eb).astype(np.uint32) << 16).view(np.float32)
hw = np.lib.format.open_memmap(os.path.join(out, PRE + "hw.npy"), "w+", np.uint16, (HR, HC, HDc * VR))
he = np.lib.format.open_memmap(os.path.join(out, PRE + "hemb.npy"), "w+", np.uint16, (HR, HC, VR * HDc))
for R in range(HR):
    lo, hi = R * VR, min((R + 1) * VR, V)
    blk = np.zeros((VR, H), np.float32); bb = np.zeros((VR, H), np.uint16)
    if hi > lo:
        blk[:hi - lo] = Wh[lo:hi]; bb[:hi - lo] = eb[lo:hi]
    if hi - lo < VR:
        # padded vocabulary rows: copies of the last real row, so they tie it and the lower (real) id wins
        # (zero rows scored 0 and won where every real logit was negative: 0841's id 262144)
        blk[max(hi - lo, 0):] = Wh[V - 1]
    hw[R] = f16(blk.reshape(VR, HC, HDc).transpose(1, 2, 0)).reshape(HC, -1)
    he[R] = bb.reshape(VR, HC, HDc).transpose(1, 0, 2).reshape(HC, VR * HDc)
hw.flush(); he.flush()
np.save(os.path.join(out, PRE + "hg.npy"), get(P + "norm.weight").reshape(HC, HDc).astype(np.float32))
e = np.lib.format.open_memmap(os.path.join(out, PRE + "embed.npy"), "w+", np.uint16, tuple(eb.shape))
e[:] = eb; e.flush()
gold = json.load(open(GOLD))
ids = gold["prompt_ids"] + gold["tokens"]
np.save(os.path.join(out, PRE + "embrows.npy"), np.asarray(eb)[ids])
shutil.copy(GOLD, os.path.join(out, PRE + "golden.json"))
# the per-layer inputs, every vocab id, f16, into each storage column's tiles
Wpm = get(P + "per_layer_model_projection.weight")             # [NL PLE][H]
pn = get(P + "per_layer_projection_norm.weight")
TP = bits(P + "embed_tokens_per_layer.weight")                   # [V][NL PLE]
esc = np.float32(np.sqrt(np.float32(H)))
cols = []
MAXB = 1900 * 1024 * 1024
for ci, col in enumerate(g.cols):
    y0, y1 = col["srows"]
    nr, nt, units = y1 - y0, col["nt"], col["units"]
    lv = K * len(units)
    assert nr * nt >= V
    rows_per = max(1, MAXB // (PX * nt * lv * 2))
    files = []
    for r0 in range(0, nr, rows_per):
        r1 = min(nr, r0 + rows_per)
        nm = f"pt{ci}_{len(files)}"
        files.append(dict(name=nm, r0=r0, r1=r1))
        np.lib.format.open_memmap(os.path.join(out, PRE + nm + ".npy"), "w+", np.uint16, (r1 - r0, PX, nt * lv)).flush()
    cols.append(dict(x=col["x"], units=units, srows=[y0, y1], nt=nt, lv=lv, files=files))
mm = {f["name"]: np.load(os.path.join(out, PRE + f["name"] + ".npy"), mmap_mode="r+") for c_ in cols for f in c_["files"]}
CH = 4096
for t0 in range(0, V, CH):
    t1 = min(V, t0 + CH)
    E = (np.asarray(eb[t0:t1]).astype(np.uint32) << 16).view(np.float32) * esc
    pr = (E @ Wpm.T) * np.float32(1.0 / np.sqrt(np.float32(H)))
    pr = pr.reshape(t1 - t0, NL, PLE)
    pr = pr / np.sqrt((pr * pr).mean(-1, keepdims=True) + np.float32(1e-6)) * pn
    tb = (np.asarray(TP[t0:t1]).astype(np.uint32) << 16).view(np.float32).reshape(t1 - t0, NL, PLE) * np.float32(np.sqrt(PLE))
    ple = f16((pr + tb) * np.float32(2.0 ** -0.5))                    # [t][NL][PLE] f16 bits
    for c_ in cols:
        nt, lv = c_["nt"], c_["lv"]
        lay = [K * u + ll for u in c_["units"] for ll in range(K)]     # the bus order: top unit first
        vals = ple[:, lay, :]                                          # [t][lv][PX]
        t = t0
        while t < t1:
            k, j = t // nt, t % nt
            te = min(t1, (k + 1) * nt)
            blk = vals[t - t0:te - t0].transpose(2, 0, 1).reshape(PX, -1)    # [PX][(te - t) lv]
            for f in c_["files"]:
                if f["r0"] <= k < f["r1"]:
                    mm[f["name"]][k - f["r0"], :, j * lv:(j + (te - t)) * lv] = blk
            t = te
    if (t0 // CH) % 16 == 0:
        print("ple", t1, "/", V, flush=True)
for a in mm.values():
    a.flush()
meta = dict(preset=PRESET, PX=PX, PY=PY, Hc=Hc, H=H, V=V, K=K, KM=KM, NO=NO, NU=NU, Qh=Qh, DcS=DcS, DcG=DcG,
            HDS=hdS, HDG=hdG, Ic0=IcO, Ic1=IcS, HC=HC, HR=HR, VR=VR, LW=LWS, cols=cols, EPS=tc.get("rms_norm_eps", 1e-6),
            model=os.path.basename(MD.rstrip("/")),
            LOOP=64)   # Pie's posted-loop length (0846 / 0848: 64 beat 32 and 48)
json.dump(meta, open(os.path.join(out, PRE + "meta.json"), "w"))
print("prepared", out, json.dumps({k: v for k, v in meta.items() if k != "cols"}))
