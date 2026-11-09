#!/usr/bin/env python3
"""prep_g12.py <model dir> <outdir> <golden json> <geo: tiny | g12> NEG [prefix]: a Gemma 4 12B-shaped checkpoint
(bf16 safetensors; no PLE, no shared KV, attention_k_eq_v) in run_g12.py's prep format (spatial/split/README.md, Gemma 4 12B).
Units of K layers (S .. S G), unit u = layers K u .. K u + K - 1.
- Rows: attention rows r < RA hold Hc A hidden dims, plain rows Hc W; row r's dims are [off_r, off_r + Hc_r) of the
  padded hidden (dims >= H: zero weights, zero gammas).
- Columns: segment s = c // SWA, p = c % SWA. Sliding: kv head s, q heads Qh s ..; global: the kv head replicated
  (every segment a copy), q heads Qh s ... Within the segment, Dc / 2 rotated pairs (j, j + hd / 2), j = p Dc / 2 + t,
  side by side; j >= hd / 2: padding (zero weights, identity RoPE, zero norm weights).
- MLP: column c holds intermediate dims c Ic .. (>= I: padding).
- Head: column c = the dims of its row group (geo.groups), HDc slots (padding past nx); tied (embed_tokens), f16.
Writes w<u>a / w<u>w (f16 bits, A / W rows), gama / gamw (f32: a layer's input / post-attention / pre-FFN / post-FFN
gammas, the layer scalar; + the embedding scale), qkw, rot, hw, hg, embed, embrows, golden.json, meta.json, params.txt."""
import glob, json, os, shutil, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split_geo_g12 as S

MD, out, GOLD, GEO = sys.argv[1:5]
NEG = int(sys.argv[5])
PRE = sys.argv[6] if len(sys.argv) > 6 else ""
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

g = S.GEOS[GEO](G2=2, WL=1, G2W=2)
PX, PY, RA, SWA = g.PX, g.PY, g.RA, g.SWA
HcA, HcW = g.HcA, g.HcW
H, NL, I = tc["hidden_size"], tc["num_hidden_layers"], tc["intermediate_size"]
nq, nkv = tc["num_attention_heads"], tc["num_key_value_heads"]
hdS = tc["head_dim"]
hdG = tc.get("global_head_dim") or next(v["head_dim"] for k, v in (tc.get("per_layer_config") or {}).items() if "head_dim" in v)
assert tc.get("hidden_size_per_layer_input", 0) == 0 and tc.get("num_kv_shared_layers", 0) == 0 and tc["attention_k_eq_v"]
V = tc["vocab_size"]
NU = len(g.origins)
K = NL // NU
assert NL == K * NU and K == g.K
kinds = ["G" if t == "full_attention" else "S" for t in tc["layer_types"]]
assert all(kinds[u * K:(u + 1) * K] == ["S"] * (K - 1) + ["G"] for u in range(NU)), "every unit S .. S G"
KM = 1 << (K - 1)
NSEG = PX // SWA
assert NSEG * SWA == PX and nkv == NSEG and nq % NSEG == 0
Qh = nq // NSEG
def dcf(hd):
    nr = -(-(hd // 2) // SWA)
    return 2 * nr
DcS, DcG = dcf(hdS), dcf(hdG)
Ic = -(-I // PX)
off = np.concatenate([[0], np.cumsum([g.hc(r) for r in range(PY)])])
HP = int(off[-1])
assert HP >= H, (HP, H)
print("Qh", Qh, "DcS", DcS, "DcG", DcG, "Ic", Ic, "hidden slots", HP, "of", H, flush=True)
c = np.arange(PX)
seg, pa = c // SWA, c % SWA
def pdims(hd, Dc):
    nr = Dc // 2
    out_ = []
    for p in pa:
        row = []
        for t in range(nr):
            j = p * nr + t
            row += [j, j + hd // 2] if j < hd // 2 else [-1, -1]
        out_.append(row)
    return np.array(out_)                                          # [PX][Dc], -1 padding
pdS, pdG = pdims(hdS, DcS), pdims(hdG, DcG)
assert sorted(d for d in set(pdS[:SWA].reshape(-1).tolist()) if d >= 0) == list(range(hdS))
assert sorted(d for d in set(pdG[:SWA].reshape(-1).tolist()) if d >= 0) == list(range(hdG))
qh = (Qh * seg)[:, None] + np.arange(Qh)                         # [PX][Qh]

def sel(W, idx):
    """rows idx of W (-1: a zero row)"""
    Wz = np.concatenate([W, np.zeros((1,) + W.shape[1:], np.float32)], 0)
    return Wz[np.where(idx < 0, W.shape[0], idx)]
def padh(W, axis):
    """the hidden axis padded to HP"""
    pw = [(0, 0)] * W.ndim; pw[axis] = (0, HP - H)
    return np.pad(W, pw)
def tile_in(W, idx):
    """W [out][H]; idx [PX][N] output rows a column: per (row, column) [Hc_r][N] -> (A [RA][PX][HcA N], W [..][PX][HcW N])"""
    N = idx.shape[1]
    Sx = f16(padh(sel(W, idx.reshape(-1)), 1)).reshape(PX, N, HP)
    a = Sx[:, :, :RA * HcA].reshape(PX, N, RA, HcA).transpose(2, 0, 3, 1).reshape(RA, PX, -1)
    w = Sx[:, :, RA * HcA:].reshape(PX, N, PY - RA, HcW).transpose(2, 0, 3, 1).reshape(PY - RA, PX, -1)
    return a, w
def tile_out(W, idx):
    """W [H][in]; idx [PX][N] input columns a column: per (row, column) [N][Hc_r]"""
    N = idx.shape[1]
    Sx = f16(padh(sel(W.T, idx.reshape(-1)), 1)).reshape(PX, N, HP)
    a = Sx[:, :, :RA * HcA].reshape(PX, N, RA, HcA).transpose(2, 0, 1, 3).reshape(RA, PX, -1)
    w = Sx[:, :, RA * HcA:].reshape(PX, N, PY - RA, HcW).transpose(2, 0, 1, 3).reshape(PY - RA, PX, -1)
    return a, w

GLA, GLW = 4 * HcA + 1, 4 * HcW + 1
QKS = Qh * DcG + DcG
gama = np.zeros((NU, RA, K * GLA + 1), np.float32)
gamw = np.zeros((NU, PY - RA, K * GLW + 1), np.float32)
qkw = np.zeros((NU, PX, K * QKS), np.float32)
esc = np.float32(np.sqrt(np.float32(H)))
gama[:, :, K * GLA] = esc
gamw[:, :, K * GLW] = esc
def zpad(v, d):
    return np.where(d < 0, np.float32(0), np.concatenate([v, [0]])[np.where(d < 0, len(v), d)])
LWS = []
for u in range(NU):
    pa_, pw_ = [], []
    for ll in range(K):
        l = K * u + ll
        L = P + f"layers.{l}."
        A = L + "self_attn."
        glob_ = kinds[l] == "G"
        hd, Dc, pd = (hdG, DcG, pdG) if glob_ else (hdS, DcS, pdS)
        for i, nm in enumerate(("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                                "post_feedforward_layernorm")):
            gv = padh(get(L + nm + ".weight"), 0)
            gama[u, :, ll * GLA + i * HcA:ll * GLA + (i + 1) * HcA] = gv[:RA * HcA].reshape(RA, HcA)
            gamw[u, :, ll * GLW + i * HcW:ll * GLW + (i + 1) * HcW] = gv[RA * HcA:].reshape(PY - RA, HcW)
        gama[u, :, ll * GLA + 4 * HcA] = get(L + "layer_scalar")[0]
        gamw[u, :, ll * GLW + 4 * HcW] = get(L + "layer_scalar")[0]
        qrows = np.where(pd[:, None, :] < 0, -1, qh[:, :, None] * hd + pd[:, None, :]).reshape(PX, Qh * Dc)
        Wq = get(A + "q_proj.weight")
        kvo = 0 if glob_ else (seg * hd)[:, None]
        krows = np.where(pd < 0, -1, kvo + pd)
        if glob_:
            Win = np.concatenate([Wq, get(A + "k_proj.weight")], 0)
            idx = np.concatenate([qrows, np.where(krows < 0, -1, nq * hd + krows)], 1)
        else:
            Win = np.concatenate([Wq, get(A + "k_proj.weight"), get(A + "v_proj.weight")], 0)
            idx = np.concatenate([qrows, np.where(krows < 0, -1, nq * hd + krows), np.where(krows < 0, -1, nq * hd + nkv * hd + krows)], 1)
        for a_, w_ in (tile_in(Win, idx), tile_out(get(A + "o_proj.weight"), qrows)):
            pa_.append(a_); pw_.append(w_)
        qn, kn = get(A + "q_norm.weight"), get(A + "k_norm.weight")
        qkw[u, :, ll * QKS:ll * QKS + Qh * Dc] = np.tile(zpad(qn, pd), (1, Qh))
        qkw[u, :, ll * QKS + Qh * Dc:ll * QKS + Qh * Dc + Dc] = zpad(kn, pd)
        M = L + "mlp."
        gi = (c * Ic)[:, None] + np.arange(Ic)
        gi = np.where(gi < I, gi, -1)
        Wgu = np.concatenate([get(M + "gate_proj.weight"), get(M + "up_proj.weight")], 0)
        for a_, w_ in (tile_in(Wgu, np.concatenate([gi, np.where(gi < 0, -1, I + gi)], 1)), tile_out(get(M + "down_proj.weight"), gi)):
            pa_.append(a_); pw_.append(w_)
    ba, bw = np.concatenate(pa_, 2), np.concatenate(pw_, 2)
    LWS.append([ba.shape[2], bw.shape[2]])
    np.save(os.path.join(out, PRE + f"w{u}a.npy"), ba)
    np.save(os.path.join(out, PRE + f"w{u}w.npy"), bw)
    print("unit", u, ba.shape, bw.shape, flush=True)
np.save(os.path.join(out, PRE + "gama.npy"), gama)
np.save(os.path.join(out, PRE + "gamw.npy"), gamw)
np.save(os.path.join(out, PRE + "qkw.npy"), qkw)
# RoPE tables a column: [kind][4][8][cos, sin][NPM pairs]
NPM = DcG // 2
rp = tc["rope_parameters"]
rot = np.zeros((PX, 2, 4, 8, 2, NPM), np.float64)
rot[:, :, :, :, 0, :] = 1.0
for kd, (hd, Dc, key) in enumerate(((hdS, DcS, "sliding_attention"), (hdG, DcG, "full_attention"))):
    th = float(rp[key]["rope_theta"])
    ra = int(rp[key].get("partial_rotary_factor", 1.0) * hd // 2) if rp[key].get("rope_type") == "proportional" else hd // 2
    nr = Dc // 2
    for p in range(PX):
        for t in range(nr):
            j = (p % SWA) * nr + t
            f = (1.0 / th ** (2 * j / hd)) if (j < hd // 2 and j < ra) else 0.0
            f = float(np.float32(f))
            for lv in range(4):
                for d in range(8):
                    ang = d * 8 ** (3 - lv) * f
                    rot[p, kd, lv, d, 0, t] = np.cos(ang); rot[p, kd, lv, d, 1, t] = np.sin(ang)
np.save(os.path.join(out, PRE + "rot.npy"), rot.astype(np.float32).reshape(PX, -1))
# head (tied): column c = its group's dims
HC, HR = g.HC, g.HR
HDc = max(g.nx(cc) for cc in range(HC))
VR = -(-V // HR)
VR = -(-VR // 8) * 8
assert HR * VR >= V
hdims = np.full((HC, HDc), -1, np.int64)
for cc, grp in enumerate(g.groups):
    ds = [d for r in grp for d in range(int(off[r]), int(off[r + 1]))]
    hdims[cc, :len(ds)] = [d if d < H else -1 for d in ds]
eb = bits(P + "embed_tokens.weight")
hw = np.lib.format.open_memmap(os.path.join(out, PRE + "hw.npy"), "w+", np.uint16, (HR, HC, HDc * VR))
for R in range(HR):
    lo, hi = R * VR, min((R + 1) * VR, V)
    blk = np.zeros((VR, H + 1), np.float32)
    if hi > lo:
        blk[:hi - lo, :H] = (np.asarray(eb[lo:hi]).astype(np.uint32) << 16).view(np.float32)
    if hi - lo < VR:
        # padded vocabulary rows: copies of the last real row (a padded id never wins over a real one)
        blk[max(hi - lo, 0):, :H] = (np.asarray(eb[V - 1]).astype(np.uint32) << 16).view(np.float32)
    t_ = blk[:, np.where(hdims < 0, H, hdims)]                     # [VR][HC][HDc]
    hw[R] = f16(t_.transpose(1, 2, 0)).reshape(HC, -1)
    if R % 128 == 0:
        print("head row", R, flush=True)
hw.flush()
nw = np.concatenate([get(P + "norm.weight"), [0]]).astype(np.float32)
np.save(os.path.join(out, PRE + "hg.npy"), nw[np.where(hdims < 0, H, hdims)].astype(np.float32))
e = np.lib.format.open_memmap(os.path.join(out, PRE + "embed.npy"), "w+", np.uint16, tuple(eb.shape))
e[:] = eb; e.flush()
gold = json.load(open(GOLD))
ids = gold["prompt_ids"] + gold["tokens"]
np.save(os.path.join(out, PRE + "embrows.npy"), np.asarray(eb)[ids])
shutil.copy(GOLD, os.path.join(out, PRE + "golden.json"))
SW = tc["sliding_window"]
NES = -(-SW // RA)
WIN = 0 if NES * RA == SW else SW                 # a longer ring: the kernel masks entries out of the window
meta = dict(geo=GEO, PX=PX, PY=PY, RA=RA, HcA=HcA, HcW=HcW, H=H, HP=HP, V=V, K=K, KM=KM, NU=NU, Qh=Qh, DcS=DcS, DcG=DcG,
            NES=NES, WIN=WIN, NEG=NEG, HDS=hdS, HDG=hdG, Ic=Ic, HC=HC, HR=HR, VR=VR, HDc=HDc, LW=LWS,
            EPS=tc.get("rms_norm_eps", 1e-6), SOFTCAP=tc.get("final_logit_softcapping"), model=os.path.basename(MD.rstrip("/")),
            LOOP=64)
json.dump(meta, open(os.path.join(out, PRE + "meta.json"), "w"))
cp = dict(K=K, KM=KM, HcA=HcA, HcW=HcW, Qh=Qh, DcS=DcS, DcG=DcG, NES=NES, WIN=WIN, RING=1 if WIN == 0 else 0, NEG=NEG, HDS=hdS, HDG=hdG, HR_=H, Ic=Ic,
          VR=VR, HDc=HDc, NPX=1)
open(os.path.join(out, PRE + "params.txt"), "w").write(",".join(f"{k}:{v}" for k, v in cp.items()) + "\n")
print("prepared", out, json.dumps(meta))
