#!/usr/bin/env python3
"""prep_gptoss.py <model dir> <out dir> <plan dir> <golden.json> [prefix] [--units a,b] [--nohead]

gpt-oss checkpoint (as released: MXFP4 experts, bf16 elsewhere) -> the role-split tiles (split-geo's
geometry; unit u holds layers K u .. K u + K - 1, K = layers / units):
  wa{u}  u16 [PY][AX][K (Hc NQA + QO Hc)]   attention: per layer [qkv Hc x NQA][o QO x Hc], f16 bits (= bf16 exactly)
  wx{u}  u16 [PY][MX][K NEX XE]              experts, E5M2 pairs (gate|up rows i < Hc/2 high byte; down i < Ic/2)
  wr{u}  f32 [PY][MX][K Hc]                  router rows (output c mod NEX on MoE column c)
  gam{u} f32 [PY][2K Hc], bo{u} f32 [PY][K Hc], bd{u} f32 [PY][K NEX Hc]           (CX)
  qb{u}  f32 [AX][K NQA], snk{u} f32 [AX][K Qh], rot f32 [AX][NLV 8 2 NROT]       (attention CY)
  rb{u}  f32 [MX][K], gb{u} f32 [MX][K NEX 2Ic], ps{u} f32 [MX][K NEX 2]           (MoE CY: 2^Pg, 2^Pd)
  hw u16 [PY][HVC][HDc VR] (f16 of 2^15 lm_head, [dim][vocab row]), hg f32 [PY][HDc], hemb u16 [PY][HVC][VR HDc]
  embrows u16 [the golden's ids][H] (bf16 bits), meta.json, golden.json
E5M2 checker (fails the prep): every expert weight decoded from its byte (byte << 8 as f16, times 2^P) equals
HF's MXFP4 dequant (FP4 LUT x 2^(scale - 127)) exactly, and is 0 or an f16 normal.
Column mapping: attention column c = kv head c // SWA, j = c % SWA: rotated pairs i = j NROT + p, p < NROT,
dims (i, i + 32) side by side (q: per head h of the kv group, [pairs]; k; v the same dims); MoE column c: the
intermediate dims c Ic .. c Ic + Ic; weight row r: hidden dims r Hc .. r Hc + Hc."""
import json, math, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import splitgeo

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
f16, f32 = np.float16, np.float32


class Ckpt:
    def __init__(self, path):
        import glob
        self.where = {}
        for f in sorted(glob.glob(f"{path}/*.safetensors")):
            with open(f, "rb") as h:
                n = struct.unpack("<Q", h.read(8))[0]
                hdr = json.loads(h.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.where[k] = (f, 8 + n, v)

    def raw(self, k):
        f, base, v = self.where[k]
        s, e = v["data_offsets"]
        return v["dtype"], v["shape"], np.memmap(f, np.uint8, "r", base + s, (e - s,))

    def f32(self, k):
        dt, shape, buf = self.raw(k)
        if dt == "BF16":
            return (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(f32).reshape(shape)
        if dt == "F32":
            return np.frombuffer(buf, f32).reshape(shape).copy()
        raise ValueError((k, dt))

    def bf16bits(self, k):
        dt, shape, buf = self.raw(k)
        assert dt == "BF16", (k, dt)
        return np.frombuffer(buf, np.uint16).reshape(shape)

    def u8(self, k):
        dt, shape, buf = self.raw(k)
        assert dt == "U8", (k, dt)
        return np.frombuffer(buf, np.uint8).reshape(shape)


def mxfp4(blocks, scales):
    """HF's dequant: low nibble first, value FP4[code] x 2^(scale - 127). [.., G, 16] u8, [.., G] -> [.., G 32]."""
    lo = FP4[blocks & 0x0F]
    hi = FP4[blocks >> 4]
    v = np.stack([lo, hi], -1).reshape(*blocks.shape[:-1], 32)
    v = v * np.ldexp(f32(1.0), scales.astype(np.int32) - 127)[..., None]
    return v.reshape(*blocks.shape[:-2], -1).astype(f32)


def e5m2(t, P, what):
    """E5M2 bytes of t 2^-P; the checker: exact and normal (fails the prep)."""
    s = (t * f32(2.0 ** -P)).astype(f32)
    h = s.astype(f16)
    hb = h.view(np.uint16)
    back = (((hb & 0xff00).astype(np.uint16)).view(f16).astype(f32) * f32(2.0 ** P))
    bad = np.count_nonzero(back != t)
    mag = hb & 0x7fff
    sub = np.count_nonzero((mag != 0) & (mag < 0x0400))
    if bad or sub:
        raise SystemExit(f"E5M2 checker: {what}: {bad} weights not exact, {sub} subnormal (P {P})")
    return (hb >> 8).astype(np.uint8)


def yarn(cfg):
    rs = cfg["rope_scaling"]
    base, d = cfg["rope_theta"], cfg["head_dim"]
    factor, orig = rs["factor"], rs["original_max_position_embeddings"]
    bf, bs = rs.get("beta_fast", 32), rs.get("beta_slow", 1)
    pos_freqs = base ** (np.arange(0, d, 2, dtype=np.float64) / d)
    inv_e, inv_i = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    def corr(rot):
        return (d * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))
    lo, hi = math.floor(corr(bf)), math.ceil(corr(bs))
    if not rs.get("truncate", True):
        lo, hi = corr(bf), corr(bs)
    lo, hi = max(lo, 0), min(hi, d - 1)
    if lo == hi:
        hi += 0.001
    ramp = np.clip((np.arange(d // 2, dtype=np.float64) - lo) / (hi - lo), 0, 1)
    ext = 1 - ramp
    inv = inv_i * (1 - ext) + inv_e * ext
    att = 0.1 * math.log(factor) + 1.0
    return inv, att


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    model, out, plan, golden = args[:4]
    pre = args[4] if len(args) > 4 else ""
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(f"{model}/config.json"))
    ck = Ckpt(model)
    g = splitgeo.tiling(plan)
    geo = g.preset
    PY, AX, MXc, Hc = g.PY, g.AX, g.MXc, g.Hc
    H, L, NEX, TK = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_local_experts"], cfg["num_experts_per_tok"]
    NH, NKV, HD, I = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"]
    NU = g.NU
    K = L // NU
    Qh, SWA = NH // NKV, g.SWA
    Dc = HD // SWA
    NROT = Dc // 2
    Ic = I // MXc
    assert H == PY * Hc and AX == NKV * SWA and I == MXc * Ic and K * NU == L and Ic % 2 == 0 and Hc % 2 == 0 and MXc % NEX == 0
    NQA, QO = Qh * Dc + 2 * Dc, Qh * Dc
    types = cfg["layer_types"]
    KM = sum(1 << l for l in range(K) if types[l] == "full_attention")
    for u in range(NU):
        assert all(types[K * u + l] == types[l] for l in range(K)), "every unit the same layer kinds"
    XG, XD = (Hc // 2) * 2 * Ic, (Ic // 2) * Hc
    XE = XG + XD
    # a column's attention dims (within its kv head's 64): pairs side by side
    cdims = []
    for j in range(SWA):
        d = []
        for p in range(NROT):
            i = j * NROT + p
            d += [i, i + HD // 2]
        cdims.append(d)
    units = [int(v) for v in opts["units"].split(",")] if "units" in opts else range(NU)
    stats = {"P": {}}
    inv, att = yarn(cfg)
    for u in units:
        LWA = Hc * NQA + QO * Hc
        wa = np.zeros((PY, AX, K * LWA), np.uint16)
        wx = np.zeros((PY, MXc, K * NEX * XE), np.uint16)
        wr = np.zeros((PY, MXc, K * Hc), f32)
        gam = np.zeros((PY, 2 * K * Hc), f32); bo = np.zeros((PY, K * Hc), f32); bd = np.zeros((PY, K * NEX * Hc), f32)
        qb = np.zeros((AX, K * NQA), f32); snk = np.zeros((AX, K * Qh), f32)
        rb = np.zeros((MXc, K), f32); gb = np.zeros((MXc, K * NEX * 2 * Ic), f32); ps = np.zeros((MXc, K * NEX * 2), f32)
        for l in range(K):
            ly = K * u + l
            p = f"model.layers.{ly}."
            Wq, Wk, Wv, Wo = (ck.f32(p + f"self_attn.{n}_proj.weight") for n in "qkvo")
            bq, bk, bv = (ck.f32(p + f"self_attn.{n}_proj.bias") for n in "qkv")
            for t, nm in ((Wq, "q"), (Wk, "k"), (Wv, "v"), (Wo, "o")):
                assert np.array_equal(t.astype(f16).astype(f32), t), f"layer {ly} {nm}_proj not exact in f16"
            for c in range(AX):
                kv, j = c // SWA, c % SWA
                qrows = [(kv * Qh + h) * HD + d for h in range(Qh) for d in cdims[j]]
                krows = [kv * HD + d for d in cdims[j]]
                outs = np.concatenate([Wq[qrows], Wk[krows], Wv[krows]], 0)          # [NQA, H]
                til = outs.T.reshape(PY, Hc, NQA)                                    # [r][i][n]
                ot = Wo[:, qrows].T.reshape(QO, PY, Hc).transpose(1, 0, 2)           # [r][QO][Hc]
                b0 = l * LWA
                wa[:, c, b0:b0 + Hc * NQA] = til.reshape(PY, -1).astype(f16).view(np.uint16)
                wa[:, c, b0 + Hc * NQA:b0 + LWA] = ot.reshape(PY, -1).astype(f16).view(np.uint16)
                qb[c, l * NQA:(l + 1) * NQA] = np.concatenate([bq[qrows], bk[krows], bv[krows]])
                snk[c, l * Qh:(l + 1) * Qh] = ck.f32(p + "self_attn.sinks")[kv * Qh:(kv + 1) * Qh]
            gam[:, (2 * l) * Hc:(2 * l + 1) * Hc] = ck.f32(p + "input_layernorm.weight").reshape(PY, Hc)
            gam[:, (2 * l + 1) * Hc:(2 * l + 2) * Hc] = ck.f32(p + "post_attention_layernorm.weight").reshape(PY, Hc)
            bo[:, l * Hc:(l + 1) * Hc] = ck.f32(p + "self_attn.o_proj.bias").reshape(PY, Hc)
            dbias = ck.f32(p + "mlp.experts.down_proj_bias")                       # [E, H]
            bd[:, l * NEX * Hc:(l + 1) * NEX * Hc] = dbias.reshape(NEX, PY, Hc).transpose(1, 0, 2).reshape(PY, -1)
            Wr, br = ck.f32(p + "mlp.router.weight"), ck.f32(p + "mlp.router.bias")   # [E, H], [E]
            gub = ck.f32(p + "mlp.experts.gate_up_proj_bias")                       # [E, 2I] interleaved
            for c in range(MXc):
                wr[:, c, l * Hc:(l + 1) * Hc] = Wr[c % NEX].reshape(PY, Hc)
                rb[c, l] = br[c % NEX]
                dims = np.arange(c * Ic, (c + 1) * Ic)
                gb[c, l * NEX * 2 * Ic:(l + 1) * NEX * 2 * Ic] = np.concatenate([gub[:, 2 * dims], gub[:, 2 * dims + 1]], 1).reshape(-1)
            gblk = ck.u8(p + "mlp.experts.gate_up_proj_blocks"); gsc = ck.u8(p + "mlp.experts.gate_up_proj_scales")
            dblk = ck.u8(p + "mlp.experts.down_proj_blocks"); dsc = ck.u8(p + "mlp.experts.down_proj_scales")
            for e in range(NEX):
                GU = mxfp4(np.asarray(gblk[e]), np.asarray(gsc[e]))        # [2I (interleaved gate, up), H]
                DN = mxfp4(np.asarray(dblk[e]), np.asarray(dsc[e]))        # [H, I]
                mg = int(np.where((np.asarray(gblk[e]) & 0x77).any(-1), gsc[e], 0).max())
                md = int(np.where((np.asarray(dblk[e]) & 0x77).any(-1), dsc[e], 0).max())
                Pg, Pd = mg - 127, md - 127 - 2
                stats["P"][f"{ly}.{e}"] = [Pg, Pd]
                ps[:, (l * NEX + e) * 2] = 2.0 ** Pg
                ps[:, (l * NEX + e) * 2 + 1] = 2.0 ** Pd
                G, U_ = GU[0::2], GU[1::2]                                   # [I, H]
                bG = e5m2(G, Pg, f"layer {ly} expert {e} gate"); bU = e5m2(U_, Pg, f"layer {ly} expert {e} up")
                bD = e5m2(DN, Pd, f"layer {ly} expert {e} down")
                # gate|up tile of (row r, column c): [Hc][gate Ic | up Ic]
                til = np.concatenate([bG.T.reshape(PY, Hc, MXc, Ic), bU.T.reshape(PY, Hc, MXc, Ic)], 3)   # [r][i][c][2Ic]
                til = til.transpose(0, 2, 1, 3)                              # [r][c][i][2Ic]
                hi_, lo_ = til[:, :, :Hc // 2].astype(np.uint16), til[:, :, Hc // 2:].astype(np.uint16)
                o0 = (l * NEX + e) * XE
                wx[:, :, o0:o0 + XG] = ((hi_ << 8) | lo_).reshape(PY, MXc, XG)
                dt = bD.reshape(PY, Hc, MXc, Ic).transpose(0, 2, 3, 1)       # [r][c][Ic rows][Hc]
                hi_, lo_ = dt[:, :, :Ic // 2].astype(np.uint16), dt[:, :, Ic // 2:].astype(np.uint16)
                wx[:, :, o0 + XG:o0 + XE] = ((hi_ << 8) | lo_).reshape(PY, MXc, XD)
            print(f"unit {u} layer {ly} ok", flush=True)
        for nm, arr in (("wa", wa), ("wx", wx), ("wr", wr), ("gam", gam), ("bo", bo), ("bd", bd), ("qb", qb), ("snk", snk),
                        ("rb", rb), ("gb", gb), ("ps", ps)):
            np.save(f"{out}/{pre}{nm}{u}.npy", arr)
    # RoPE tables: level lv, digit dd: angle dd 8^(NLV-1-lv) inv_freq; the attention scaling in level 0
    NLV = 5
    rot = np.zeros((AX, NLV, 8, 2, NROT), f32)
    for c in range(AX):
        j = c % SWA
        fr = np.array([inv[j * NROT + p] for p in range(NROT)])
        for lv in range(NLV):
            for dd in range(8):
                a = dd * 8.0 ** (NLV - 1 - lv) * fr
                sc = att if lv == 0 else 1.0
                rot[c, lv, dd, 0] = np.cos(a) * sc
                rot[c, lv, dd, 1] = np.sin(a) * sc
    np.save(f"{out}/{pre}rot.npy", rot.reshape(AX, -1))
    gold = json.load(open(golden))
    ids = gold["prompt_ids"] + gold["tokens"]
    emb = ck.bf16bits("model.embed_tokens.weight")
    np.save(f"{out}/{pre}embrows.npy", np.asarray(emb[ids]))
    V = cfg["vocab_size"]
    HVC = g.HVC
    VR = -(-V // HVC)
    VR = -(-VR // 8) * 8                 # (head_ctl's VARG 2 argmax: VR % 8 == 0)
    if "nohead" not in opts:
        Wh = ck.f32("lm_head.weight")                                     # [V, H]
        pad = VR * HVC - V
        Wp = np.concatenate([Wh, np.repeat(Wh[-1:], pad, 0)], 0) if pad else Wh
        h16 = (Wp * f32(2.0 ** 15)).astype(f16)
        inexact = int(np.count_nonzero(h16.astype(f32) * f32(2.0 ** -15) != Wp[:V * 0 + Wp.shape[0]]))
        hw = h16.view(np.uint16).reshape(HVC, VR, PY, Hc).transpose(2, 0, 3, 1).reshape(PY, HVC, Hc * VR)   # [i][j][d][v]
        np.save(f"{out}/{pre}hw.npy", hw)
        np.save(f"{out}/{pre}hg.npy", ck.f32("model.norm.weight").reshape(PY, Hc))
        Ep = np.asarray(emb)
        Ep = np.concatenate([Ep, np.repeat(Ep[-1:], pad, 0)], 0) if pad else Ep
        np.save(f"{out}/{pre}hemb.npy", Ep.reshape(HVC, VR, PY, Hc).transpose(2, 0, 1, 3).reshape(PY, HVC, VR * Hc))
        stats["head_f16_inexact"] = inexact
    json.dump(gold, open(f"{out}/{pre}golden.json", "w"))
    meta = dict(geo=geo, HD=HD, K=K, KM=KM, Hc=Hc, NQA=NQA, QO=QO, Qh=Qh, Dc=Dc, NROT=NROT, Ic=Ic, NEX=NEX, TK=TK, HR=H, NLV=NLV,
                VR=VR, HVC=HVC, V=V, PY=PY, AX=AX, MX=MXc, NU=NU, WIN=cfg["sliding_window"], HSCALE=2.0 ** 15,
                LOOP=64)   # Pie's posted-loop length (0886: L64 the faster)
    json.dump(meta, open(f"{out}/{pre}meta.json", "w"), indent=1)
    json.dump(stats, open(f"{out}/{pre}stats.json", "w"))
    print("ok", out, json.dumps(meta))


if __name__ == "__main__":
    main()
