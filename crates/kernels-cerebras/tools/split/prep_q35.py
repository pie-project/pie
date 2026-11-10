#!/usr/bin/env python3
"""prep_q35.py <model dir> <outdir> <golden json> <plan dir> [prefix]
Any Qwen3.5
or Qwen3 (dense: every layer attention, no output gate, plain norm weights, full RoPE)
checkpoint (bf16 or f32 safetensors, the VL layout "model.language_model." or text-only "model.") in
the role-split prep format (the files PIE_CEREBRAS_SPLIT=<outdir>:<prefix> names; kernels-cerebras src/spatial/weights.rs
copies them onto the wafer), cut as the plan in <plan dir> says (`split-geo plan`'s tiling.json). Units of PX x PY weight PEs, K layers each (the model's layer kinds, GDN / attention by
`layer_types`, the same K-layer pattern in every unit), UNX x UNY units, the head PY / 2 columns of
2 Hc dims by HR rows of VR vocabulary rows; lm_head = the embedding when tied. Weights as f16 bits.

Column mapping (per weight column c of PX):
  GDN: k head c // (PX / nk), its DK = hk / (PX / nk) dims; v head c // (PX / nv), DV dims.
  attention: kv head c // (PX / nkv), Dc = hd / (PX / nkv) dims: the first rd / Dc columns hold
  Dc / 2 rotated pairs (i, i + rd / 2) side by side, the others Dc unrotated dims; Qh = nq / nkv
  q heads (with their gates) a column.
When PY does not divide the hidden size, Hc = ceil(H / PY) and the padded dims carry zero weights,
gammas and embedding (x stays 0 there; the norms divide by H).
Writes w<u>.npy, gam, cw, gpar, gnw, qkw, rot, hw, hg, embed, hemb, embrows, golden.json, meta.json
(each with the optional prefix)."""
import glob, json, os, shutil, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splitgeo

MD, out, GOLD = sys.argv[1:4]
g = splitgeo.tiling(sys.argv[4])
PX, PY, UNX, UNY, K, HR, VR, NE, NU = g.PX, g.PY, g.UNX, g.UNY, g.K, g.HR, g.VR, g.NE, g.NU
RLV, RBITS = g.RLV, g.RBITS
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
    if dt == "BF16":
        a = np.memmap(f, np.uint16, "r", off, tuple(shp))
        return (a.astype(np.uint32) << 16).view(np.float32)
    assert dt == "F32", (name, dt)
    return np.array(np.memmap(f, np.float32, "r", off, tuple(shp)))
def f16(a):
    return np.asarray(a, np.float32).astype(np.float16).view(np.uint16)
def bf16bits(name):
    f, off, dt, shp = T[name]
    if dt == "BF16":
        return np.memmap(f, np.uint16, "r", off, tuple(shp))
    a = np.array(np.memmap(f, np.float32, "r", off, tuple(shp)))
    return (a.view(np.uint32) >> 16).astype(np.uint16)        # (f32 checkpoints: truncation; tiny tests store bf16)

H, I, NL = tc["hidden_size"], tc["intermediate_size"], tc["num_hidden_layers"]
Q3 = "linear_num_key_heads" not in tc      # Qwen3 (dense): no GDN, no output gate, norm weights w (not 1 + w)
GATE = 0 if Q3 else 1
nq, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
if Q3:
    nk, nv, hk, hv = nkv, nkv, hd, nq // nkv * hd     # (GDN's fields only shape unused buffers: DV = Qh Dc)
else:
    nk, nv, hk, hv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
nw = (lambda w: w) if Q3 else (lambda w: 1 + w)   # a norm's weight as applied
rp = tc.get("rope_parameters") or {}
theta = float(rp.get("rope_theta", tc.get("rope_theta", 1e7)))
prf = 1.0 if Q3 else float(rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
rd = int(hd * prf)
CONV = tc.get("linear_conv_kernel_dim", 4)
tied = bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)))
kinds = ["A" if t == "full_attention" else "G" for t in tc.get("layer_types", ["full_attention"] * NL)]
assert NL == NU * K, (NL, NU, K)
# one slot short of the grid (an odd count on an even number of unit columns): the last column's row-1 slot
# stays out of the snake, so the last unit still ends on row 0 (the wafer loop's return); its PEs relay
GAP = None
if NU < UNX * UNY:
    assert NU == UNX * UNY - 1 and UNX % 2 == 0 and UNY >= 3, "one gap slot, on an even number of unit columns, below the turn row"
    GAP = (UNX - 1, 1)
pat = kinds[:K]
assert all(kinds[u * K:(u + 1) * K] == pat for u in range(NU)), "every unit the same layer kinds"
KM = sum(1 << l for l, k in enumerate(pat) if k == "A")
NLG = pat.count("G")
assert I % PX == 0
Hc, Ic = -(-H // PY), I // PX
Hp = Hc * PY                       # (padded hidden: zero weights past H)
cpk, cpv, cpa = PX // nk, PX // nv, PX // nkv
assert nk * cpk == PX and nv * cpv == PX and nkv * cpa == PX
DK, DV, Dc, Qh = hk // cpk, hv // cpv, hd // cpa, nq // nkv
assert DK * cpk == hk and DV * cpv == hv and Dc * cpa == hd and Dc % 2 == 0
assert Qh * Dc == DV, "the weight PE's o tile has QO = Qh Dc rows for both kinds (GDN's DV)"
assert Q3 or hk <= PY, "GDN state row k = weight row r < hk"
assert rd % Dc == 0 and rd // Dc <= cpa
assert all((PX // 2) % w == 0 for w in (cpa, cpk, cpv)), "a segment (a head's columns) must not span the control column"
assert PY <= PX, "vertical handoffs turn row r at weight column r"
NROT = Dc // 2
nrc = rd // Dc                     # rotated columns a kv head
HC, HDc = PY // 2, 2 * Hc
assert PY % 4 == 0, "the head's CX row pairs must not straddle the CY row (PY / 2 even)"
KD, VD = nk * hk, nv * hv
V = tc["vocab_size"]
assert HR * VR >= V, (HR, VR, V)
NQG = 2 * DK + 2 * DV + 2
NQA = (1 + GATE) * Qh * Dc + 2 * Dc
c = np.arange(PX)
kh, pk = c // cpk, c % cpk
vh, pv = c // cpv, c % cpv
qidx = (kh * hk + pk * DK)[:, None] + np.arange(DK)
vidx = (vh * hv + pv * DV)[:, None] + np.arange(DV)
kvh, pa = c // cpa, c % cpa
def perm(p):
    if p < nrc:
        h2 = rd // 2
        return [d for j in range(NROT) for d in (p * NROT + j, p * NROT + j + h2)]
    return [rd + (p - nrc) * Dc + j for j in range(Dc)]
pd = np.array([perm(p) for p in pa])                       # [PX][Dc] head dims of each column
assert sorted(set(pd[:cpa].reshape(-1).tolist())) == list(range(hd))
inv = 1.0 / (theta ** (np.arange(0, rd, 2) / rd))         # rd / 2 frequencies (pairs i, i + rd / 2)

def tile_in(W, idx):
    """W [out][H], idx [PX cols][N] -> [r][c][Hc][N] (row i = input dim r Hc + i)."""
    N = idx.shape[1]
    S = f16(W[idx.reshape(-1)])
    if Hp > H:
        S = np.pad(S, ((0, 0), (0, Hp - H)))
    S = S.reshape(PX, N, PY, Hc)
    return S.transpose(2, 0, 3, 1)
def tile_out(W, idx):
    """W [H out][in], idx [PX cols][N] (the column's inputs) -> [r][c][N][Hc]."""
    N = idx.shape[1]
    S = f16(W[:, idx.reshape(-1)])
    if Hp > H:
        S = np.pad(S, ((0, Hp - H), (0, 0)))
    S = S.reshape(PY, Hc, PX, N)
    return S.transpose(0, 2, 3, 1)

NCH = 2 * DK + DV
gam = np.zeros((NU, PY, 2 * K * Hc), np.uint16)
cw = np.zeros((NU, PX, max(NLG, 1) * CONV * NCH), np.float32)
gpar = np.zeros((NU, PX, max(NLG, 1) * 2), np.float32)
gnw = np.zeros((NU, PX, max(NLG, 1) * DV), np.float32)
qkw = np.zeros((NU, PX, max(K - NLG, 1) * (Qh * Dc + Dc)), np.float32)
LWG = Hc * NQG + DV * Hc + Hc * 2 * Ic + Ic * Hc
LWA = Hc * NQA + Qh * Dc * Hc + Hc * 2 * Ic + Ic * Hc
ULIM = int(os.environ.get("PREP_UNITS", NU))      # (checks: only the first units)
for u in range(ULIM):
    parts = []
    g = a = 0
    for l in range(K * u, K * u + K):
        L = P + f"layers.{l}."
        ln1 = np.pad(nw(get(L + "input_layernorm.weight")), (0, Hp - H)); ln2 = np.pad(nw(get(L + "post_attention_layernorm.weight")), (0, Hp - H))
        gam[u, :, (l % K) * 2 * Hc:(l % K) * 2 * Hc + Hc] = f16(ln1.reshape(PY, Hc))
        gam[u, :, (l % K) * 2 * Hc + Hc:(l % K + 1) * 2 * Hc] = f16(ln2.reshape(PY, Hc))
        if kinds[l] == "G":
            A = L + "linear_attn."
            Win = np.concatenate([get(A + "in_proj_qkv.weight"), get(A + "in_proj_z.weight"),
                                  get(A + "in_proj_b.weight"), get(A + "in_proj_a.weight")], 0)
            idx = np.concatenate([qidx, KD + qidx, 2 * KD + vidx, 2 * KD + VD + vidx,
                                  2 * KD + 2 * VD + vh[:, None], 2 * KD + 2 * VD + nv + vh[:, None]], 1)
            parts.append(tile_in(Win, idx).reshape(PY, PX, -1))
            parts.append(tile_out(get(A + "out_proj.weight"), vidx).reshape(PY, PX, -1))
            conv = get(A + "conv1d.weight")[:, 0, :]            # [2 KD + VD][CONV]
            ch = np.concatenate([qidx, KD + qidx, 2 * KD + vidx], 1)
            cw[u, :, g * CONV * NCH:(g + 1) * CONV * NCH] = conv[ch].transpose(0, 2, 1).reshape(PX, -1)
            Al, dt = get(A + "A_log"), get(A + "dt_bias")
            gpar[u, :, 2 * g] = Al[vh]; gpar[u, :, 2 * g + 1] = dt[vh]
            gnw[u, :, g * DV:(g + 1) * DV] = get(A + "norm.weight")[(pv * DV)[:, None] + np.arange(DV)]
            g += 1
        else:
            A = L + "self_attn."
            Wq = get(A + "q_proj.weight").reshape(nq, (1 + GATE) * hd, H)
            qh = (Qh * kvh)[:, None] + np.arange(Qh)             # [PX][Qh] the column's q heads
            qrows = (qh[:, :, None] * (1 + GATE) * hd + pd[:, None, :]).reshape(PX, Qh * Dc)
            grows = (qh[:, :, None] * 2 * hd + hd + pd[:, None, :]).reshape(PX, Qh * Dc)
            krows = kvh[:, None] * hd + pd
            Win = np.concatenate([Wq.reshape(-1, H), get(A + "k_proj.weight"), get(A + "v_proj.weight")], 0)
            QR = nq * (1 + GATE) * hd
            idx = np.concatenate([qrows] + ([grows] if GATE else []) + [QR + krows, QR + nkv * hd + krows], 1)
            parts.append(tile_in(Win, idx).reshape(PY, PX, -1))
            oidx = (qh[:, :, None] * hd + pd[:, None, :]).reshape(PX, Qh * Dc)
            parts.append(tile_out(get(A + "o_proj.weight"), oidx).reshape(PY, PX, -1))
            qn, kn = nw(get(A + "q_norm.weight")), nw(get(A + "k_norm.weight"))
            qkw[u, :, a * (Qh * Dc + Dc):a * (Qh * Dc + Dc) + Qh * Dc] = np.tile(qn[pd] / np.sqrt(hd), (1, Qh))
            qkw[u, :, a * (Qh * Dc + Dc) + Qh * Dc:(a + 1) * (Qh * Dc + Dc)] = kn[pd]
            a += 1
        M = L + "mlp."
        gi = (c * Ic)[:, None] + np.arange(Ic)
        Wgu = np.concatenate([get(M + "gate_proj.weight"), get(M + "up_proj.weight")], 0)
        parts.append(tile_in(Wgu, np.concatenate([gi, I + gi], 1)).reshape(PY, PX, -1))
        parts.append(tile_out(get(M + "down_proj.weight"), gi).reshape(PY, PX, -1))
    blob = np.concatenate(parts, 2)
    assert blob.shape[2] == NLG * LWG + (K - NLG) * LWA, blob.shape
    np.save(os.path.join(out, PRE + f"w{u}.npy"), blob)
    print("unit", u, blob.shape, flush=True)
np.save(os.path.join(out, PRE + "gam.npy"), gam)
for n, arr in (("cw", cw), ("gpar", gpar), ("gnw", gnw), ("qkw", qkw)):
    np.save(os.path.join(out, PRE + n + ".npy"), arr)
# RoPE tables: a position is RLV digits of RBITS bits (the plan's)
RB = 1 << RBITS
rot = np.zeros((PX, RLV, RB, 2, NROT), np.float32)
for p in range(PX):
    for j in range(NROT):
        f = inv[(p % cpa) * NROT + j] if p % cpa < nrc else 0.0
        for lv in range(RLV):
            for d in range(RB):
                ang = d * RB ** (RLV - 1 - lv) * f
                rot[p, lv, d, 0, j] = np.cos(ang); rot[p, lv, d, 1, j] = np.sin(ang)
np.save(os.path.join(out, PRE + "rot.npy"), rot.reshape(PX, -1))
if os.environ.get("PREP_NOHEAD"):
    print("no head (PREP_NOHEAD)"); sys.exit(0)
hname = P + "embed_tokens.weight" if tied or "lm_head.weight" not in T else "lm_head.weight"
Wh = get(hname)
hw = np.lib.format.open_memmap(os.path.join(out, PRE + "hw.npy"), "w+", np.uint16, (HR, HC, HDc * VR))
# rows past the vocabulary are copies of its last row: their logits tie with it and the argmax (the
# first maximum) keeps the real, lower id; zero rows would win whenever every real logit is negative
for R in range(HR):
    lo, hi = R * VR, min((R + 1) * VR, V)
    blk = np.repeat(np.asarray(Wh[V - 1:V], np.float32), VR, axis=0)
    if hi > lo:
        blk[:hi - lo] = Wh[lo:hi]
    if Hp > H:
        blk = np.pad(blk, ((0, 0), (0, Hp - H)))
    hw[R] = f16(blk.reshape(VR, HC, HDc).transpose(1, 2, 0)).reshape(HC, -1)
hw.flush()
np.save(os.path.join(out, PRE + "hg.npy"), np.pad(nw(get(P + "norm.weight")), (0, Hp - H)).reshape(HC, HDc).astype(np.float32))
eb = bf16bits(P + "embed_tokens.weight")
e = np.lib.format.open_memmap(os.path.join(out, PRE + "embed.npy"), "w+", np.uint16, tuple(eb.shape))
e[:] = eb; e.flush()
# the head's embedding tiles (hemb: [HR][HC][VR HDc] bf16 bits, a head PE's [vocab row][dim])
he = np.lib.format.open_memmap(os.path.join(out, PRE + "hemb.npy"), "w+", np.uint16, (HR, HC, VR * HDc))
for R in range(HR):
    blk = np.zeros((VR, Hp), np.uint16)
    rows = np.asarray(eb[R * VR:min((R + 1) * VR, V)])
    blk[:rows.shape[0], :H] = rows
    he[R] = blk.reshape(VR, HC, HDc).transpose(1, 0, 2).reshape(HC, VR * HDc)
he.flush()
gold = json.load(open(GOLD))
ids = gold["prompt_ids"] + gold["tokens"]
np.save(os.path.join(out, PRE + "embrows.npy"), np.asarray(eb)[ids])
shutil.copy(GOLD, os.path.join(out, PRE + "golden.json"))
meta = dict(PX=PX, PY=PY, UNX=UNX, UNY=UNY, HC=HC, HR=HR, VR=VR, Hc=Hc, SWA=cpa, SWG=cpk, SWV=cpv, DKG=DK, DKH=hk,
            has_att=KM != 0, has_gdn=NLG > 0, K=K, KM=KM, Ic=Ic, NQA=NQA, NQG=NQG, Qh=Qh, Dc=Dc, DK=DK, DV=DV, NE=NE, HD=hd,
            NROT=NROT, HK=hk, HV=hv, H=H, GATE=GATE, RLV=RLV, RBITS=RBITS, **({"GAPX": GAP[0], "GAPY": GAP[1]} if GAP else {}), EPS=tc.get("rms_norm_eps", 1e-6), V=V, model=os.path.basename(MD.rstrip("/")))
json.dump(meta, open(os.path.join(out, PRE + "meta.json"), "w"))
print("prepared", out, json.dumps(meta))
