#!/usr/bin/env python3
"""tf_gptoss.py <model> <golden.json> <modes,comma> <n> <out.json> [--layers L]

Teacher-forced host emulation of the role-split gpt-oss program's arithmetic against a golden
(hf_gptoss_golden.py's json: prompt_ids + tokens). Each mode runs the whole model one position at a
time and reports greedy mismatches against the golden tokens, the gap (our top-1 minus the golden
token's logit, negative = a flip) at the golden's tightest margins, and the router statistics.

Geometry (as built): hidden rows of Hc 18 (160 rows), attention columns 8 a kv head (Dc 8: dims
4j..4j+3 and 32+4j..), MoE columns of Ic 10 (288), router output c mod 32 on MoE column c.

Modes: `ref` (f32 everywhere, the reference arithmetic), or `+`-joined kernel switches:
  k        the kernel as built: x g 2^-S_r in f16 a row chunk, f16 accumulation inside a PE
           (qkv / o / gate|up / down tiles), f32 across PEs, E5M2 experts (exact values, 2^-P per
           layer / expert / matrix), act' = act w 2^Pd in f16, f16 KV and probabilities, the router
           f32 end to end (f32 weights, f32 x), head f16(x) . f16(2^15 W) f32 accumulation
  kv32     KV cache in f32
  r16      router x in one f16 part (f32 accumulation)
  rxp2     router x in two f16 parts
  att32    attention matvecs (qkv, o) with f32 accumulation
Router report: per (position, layer) the 4th / 5th gap of our logits and our logits' error against
an f64 router on the same x2 (the kernel's router error); counts of gaps below that error."""
import argparse, json, math, sys, time
import numpy as np
import torch
torch.set_num_threads(8)

sys.path.insert(0, "/cra-1900/ing-cloud/pie-bringup-20261027/scratch/gptoss")
import gptoss_ref as R   # checkpoint reader, YaRN, MXFP4 decode (exact in f32)

f16, f32 = np.float16, np.float32
HC, PY, DC, IC, PXM, SWA = 18, 160, 8, 10, 288, 8
GEOS = {"full": (18, 160, 8, 10, 288, 8), "tiny": (8, 4, 4, 4, 8, 2)}


def ffh(a):
    return np.asarray(a, f32).astype(f16)


def acc16_chunks(x, W, size):
    """sum_i x_i W[i, :] over each chunk of `size` rows with an f16 accumulator (fused multiply-add: the
    f16 x f16 product is exact in f32, one rounding a step); returns f32 partials [n_chunks, out]."""
    n = x.shape[0] // size
    xt = torch.from_numpy(np.ascontiguousarray(np.asarray(x, f32).reshape(n, size)))
    Wt = W if isinstance(W, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(W))
    Wt = Wt.reshape(n, size, -1)
    acc = torch.zeros((n, Wt.shape[2]), dtype=torch.float16)
    for i in range(size):
        acc = torch.addcmul(acc.float(), xt[:, i:i + 1], Wt[:, i, :].float()).half()
    return acc.float().numpy()


def acc16(x, W):
    return acc16_chunks(x, W, HC)


class Emu:
    def __init__(self, path, layers=None):
        self.m = R.Model(path, layers=layers, cache=0)
        m = self.m
        self.L = m.L
        self.cache = {}
        # f16 copies of the bf16 tables (exactness of bf16 -> f16 checked below)
        self.Wq = []; self.Wo = []; self.Wr = []
        bad = 0
        for l in range(self.L):
            w = m.W[l]
            qkv = np.concatenate([w["self_attn.q_proj.weight"], w["self_attn.k_proj.weight"], w["self_attn.v_proj.weight"]], 0)
            for t in (qkv, w["self_attn.o_proj.weight"], w["mlp.router.weight"]):
                bad += int(np.count_nonzero(t.astype(f16).astype(f32) != t))
            self.Wq.append(qkv.T.copy())                       # [2880, 5120]
            self.Wo.append(w["self_attn.o_proj.weight"].T.copy())   # [4096, 2880]
            self.Wr.append(w["mlp.router.weight"].T.copy())    # [2880, 32]
        self.f16_inexact = bad
        hb = (m.head * f32(2.0 ** 15)).astype(f16).astype(f32) * f32(2.0 ** -15)   # the head as stored: f16 of 2^15 W
        self.head16 = hb
        # o_proj's inputs a column, in the kernel's order: column j of kv head h holds its q heads' rotated
        # pairs i = j NROT + p as (i, i + HD / 2) side by side
        NKV, HD, Qh = m.KV, m.D, m.H // m.KV
        NROT = DC // 2
        self.ocols = []
        for h in range(NKV):
            for j in range(SWA):
                dims = [d for p in range(NROT) for d in (j * NROT + p, j * NROT + p + HD // 2)]
                self.ocols.append([qh * HD + d for qh in range(Qh * h, Qh * h + Qh) for d in dims])
        self.operm = np.array([i for c in self.ocols for i in c])
        self.wocache = {}

    def expert(self, l, e):
        k = (l, e)
        if k in self.cache:
            return self.cache[k]
        G, U, D = self.m.expert(l, e)        # exact f32 dequant: G, U [I, H], D [H, I]
        ck = self.m.ck
        p = f"model.layers.{l}.mlp.experts."
        sg = ck.u8(p + "gate_up_proj_scales")[e].astype(np.int32); sd = ck.u8(p + "down_proj_scales")[e].astype(np.int32)
        bg = ck.u8(p + "gate_up_proj_blocks")[e]; bd = ck.u8(p + "down_proj_blocks")[e]
        mg = int(np.where((bg & 0x77).any(-1), sg, 0).max()); md = int(np.where((bd & 0x77).any(-1), sd, 0).max())
        Pg = mg - 127 - 0; Pd = md - 127 - 2           # T 0 (gate|up), T 2 (down)
        GU = np.concatenate([G, U], 0).T * f32(2.0 ** -Pg)   # [H, 2I], as stored
        Dn = D.T * f32(2.0 ** -Pd)                           # [I, H]
        for t in (GU, Dn):   # E5M2 storage is exact and normal: the prep's checker
            h = t.astype(f16)
            assert np.array_equal(h.astype(f32), t), "not exact in f16"
            hb = h.view(np.uint16)
            assert np.all(hb & 0x00ff == 0), "not E5M2"
            nz = hb & 0x7fff
            assert np.all((nz == 0) | (nz >= 0x0400)), "subnormal"
        v = (torch.from_numpy(GU.astype(f16)), torch.from_numpy(Dn.astype(f16)), Pg, Pd)
        if len(self.cache) > 768:
            self.cache.pop(next(iter(self.cache)))
        self.cache[k] = v
        return v

    def reset(self):
        self.trace = None
        self.K = [[] for _ in range(self.L)]
        self.V = [[] for _ in range(self.L)]
        self.mref = [np.zeros(self.m.H, f32) for _ in range(self.L)]
        self.pos = 0

    def xprime(self, xg):
        """CX: x g a row chunk times 2^-S_r in f16 (max |x'| in [128, 256))."""
        ch = xg.reshape(PY, HC)
        mx = np.abs(ch).max(1)
        S = np.where(mx > 0, np.floor(np.log2(np.maximum(mx, 1e-30))) - 7, 0).astype(np.int32)
        sc = (2.0 ** S).astype(f32)
        return (ch / sc[:, None]).astype(f16).reshape(-1), sc

    def step(self, tok, o, st):
        m = self.m
        ref = o == {"ref"}
        y = m.embed[tok].astype(f32).copy()
        pos = self.pos
        for l in range(self.L):
            w = m.W[l]
            ssq = f32(np.sum(y.astype(np.float64) ** 2))
            rr = f32(1.0 / math.sqrt(ssq / y.size + 1e-5))
            # ---- attention
            xg = y * w["input_layernorm.weight"]
            if ref:
                qkv = (self.Wq[l].T @ (xg * rr)).astype(f32)
            else:
                xp, sc = self.xprime(xg)
                if "att32" in o:
                    part = (xp.astype(f32).reshape(PY, HC)[:, :, None] * self.Wq[l].reshape(PY, HC, -1)).sum(1)
                else:
                    part = acc16(xp, self.Wq[l])
                qkv = (part * sc[:, None]).sum(0, dtype=f32) * rr
            qkv = qkv + np.concatenate([w["self_attn.q_proj.bias"], w["self_attn.k_proj.bias"], w["self_attn.v_proj.bias"]])
            NH, NKV, HD = m.H, m.KV, m.D
            q = qkv[:NH * HD].reshape(NH, HD); k = qkv[NH * HD:(NH + NKV) * HD].reshape(NKV, HD); v = qkv[(NH + NKV) * HD:].reshape(NKV, HD)
            q = m.rope(q, pos) * f32(1.0 / math.sqrt(HD)); k = m.rope(k, pos)
            if ref or "kv32" in o:
                self.K[l].append(k.astype(f32)); self.V[l].append(v.astype(f32))
            else:
                self.K[l].append(ffh(k).astype(f32)); self.V[l].append(ffh(v).astype(f32))
            lo = max(0, pos + 1 - m.win) if m.types[l] == "sliding_attention" else 0
            Kc = np.stack(self.K[l][lo:], 1); Vc = np.stack(self.V[l][lo:], 1)   # [8, T, 64]
            qg = q.reshape(NKV, NH // NKV, HD)
            if not ref:
                qg = ffh(qg).astype(f32)
            s = np.einsum("kgd,ktd->kgt", qg, Kc).reshape(NH, -1).astype(f32)
            sink = w["self_attn.sinks"].astype(f32)
            if ref:
                mm = np.maximum(s.max(-1), sink)
                e = np.exp(s - mm[:, None]); den = e.sum(-1) + np.exp(sink - mm)
                a = (e / den[:, None]).reshape(NKV, NH // NKV, -1)
                att = np.einsum("kgt,ktd->kgd", a, Vc).reshape(NH * HD).astype(f32)
            else:
                mref = self.mref[l]
                for _ in range(8):
                    p = np.minimum(np.exp(np.clip(s - mref[:, None], -87, 88)), 65504).astype(f16).astype(f32)
                    num = np.einsum("kgt,ktd->kgd", p.reshape(NKV, NH // NKV, -1), Vc).reshape(NH, HD)
                    l_ = p.sum(-1)
                    bad = np.any((l_ < 2 ** -8) | (l_ >= 32768))
                    lt = l_ + np.exp(np.minimum(sink - mref, 88))
                    o_ = num / lt[:, None]
                    mref = mref + np.log(np.maximum(l_, 1e-30)).astype(f32)
                    if not bad:
                        break
                    st["redo"] += 1
                self.mref[l] = mref
                att = o_.reshape(NH * HD).astype(f32)
            if ref:
                ov = self.Wo[l].T @ att
            else:
                a16 = ffh(att)
                # a column's o dims (64) by Hc outputs a PE: f16 accumulation over the column's 64 inputs
                if "att32" in o:
                    ov = (a16[self.operm].astype(f32)[:, None] * self.Wo[l][self.operm]).sum(0, dtype=f32)
                else:
                    if l not in self.wocache:
                        self.wocache[l] = torch.from_numpy(np.ascontiguousarray(self.Wo[l][self.operm]))
                    ov = acc16_chunks(a16[self.operm], self.wocache[l], len(self.ocols[0])).sum(0, dtype=f32)
            y = (y + ov + w["self_attn.o_proj.bias"]).astype(f32)
            # ---- router
            ssq = f32(np.sum(y.astype(np.float64) ** 2))
            rr = f32(1.0 / math.sqrt(ssq / y.size + 1e-5))
            xg = y * w["post_attention_layernorm.weight"]
            exact = (self.Wr[l].T.astype(np.float64) @ (xg.astype(np.float64) * float(rr))) + w["mlp.router.bias"]
            if ref:
                lg = (self.Wr[l].T @ (xg * rr) + w["mlp.router.bias"]).astype(f32)
            else:
                xp, sc = self.xprime(xg)
                if "r16" in o:
                    xr = xp.astype(f32)
                elif "rxp2" in o:
                    hi = xp.astype(f32); lo_ = ((xg.reshape(PY, HC) / sc[:, None]).reshape(-1) - hi).astype(f16).astype(f32)
                    xr = hi + lo_
                else:
                    xr = (xg.reshape(PY, HC) / sc[:, None]).reshape(-1).astype(f32)
                part = (xr.reshape(PY, HC)[:, :, None] * self.Wr[l].reshape(PY, HC, -1)).sum(1, dtype=f32)
                lg = ((part * sc[:, None]).sum(0, dtype=f32) * rr + w["mlp.router.bias"]).astype(f32)
            srt = np.sort(lg)[::-1]
            st["rgap"].append(float(srt[3] - srt[4])); st["rerr"].append(float(np.abs(lg - exact).max()))
            top = np.argsort(-lg, kind="stable")[:4]
            tv = lg[top]; pe = np.exp(tv - tv.max()); pe = (pe / pe.sum()).astype(f32)
            st["route"].append(top.tolist())
            # ---- experts
            gub = w["mlp.experts.gate_up_proj_bias"]; db = w["mlp.experts.down_proj_bias"]
            out = np.zeros(y.size, f32)
            if ref:
                xn = xg * rr
                for e, p_ in zip(top, pe):
                    G, U, D = m.expert(l, int(e))
                    gate = G @ xn + gub[e, 0::2]; up = U @ xn + gub[e, 1::2]
                    gate = np.minimum(gate, 7.0); up = np.clip(up, -7, 7)
                    act = ((up + 1) * gate / (1 + np.exp(-gate * 1.702))).astype(f32)
                    out += p_ * (D @ act + db[e])
            else:
                acts = []; Ds = []
                for e, p_ in zip(top, pe):
                    GU, Dn, Pg, Pd = self.expert(l, int(e))
                    part = acc16(xp, GU)                                   # [160, 5760]
                    gu = (part * sc[:, None]).sum(0, dtype=f32) * f32(2.0 ** Pg) * rr
                    I_ = gu.size // 2
                    gate = gu[:I_] + gub[e, 0::2]; up = gu[I_:] + gub[e, 1::2]
                    gate = np.minimum(gate, 7.0); up = np.clip(up, -7, 7)
                    act = ((up + 1) * gate / (1 + np.exp(-gate * 1.702))).astype(f32)
                    acts.append(ffh(act * p_ * f32(2.0 ** Pd))); Ds.append(Dn)
                    out += p_ * db[e]
                # down: a MoE column holds Ic 10 dims of each of the 4 experts: f16 accumulation over 40
                Hd = y.size
                A = np.stack(acts, 0).reshape(4, PXM, IC).transpose(1, 0, 2).reshape(PXM * 4 * IC).astype(f32)
                Wd = torch.stack([d.reshape(PXM, IC, Hd) for d in Ds], 1).reshape(PXM * 4 * IC, Hd)
                out += acc16_chunks(A, Wd, 4 * IC).sum(0, dtype=f32)
            y = (y + out).astype(f32)
            if self.trace is not None: self.trace.append(y.copy())
        self.pos += 1
        ssq = f32(np.sum(y.astype(np.float64) ** 2))
        xn = y * f32(1.0 / math.sqrt(ssq / y.size + 1e-5)) * m.norm
        if ref:
            return (m.head @ xn).astype(f32)
        return (self.head16 @ ffh(xn).astype(f32)).astype(f32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("golden"); ap.add_argument("modes"); ap.add_argument("n", type=int); ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--geo", default="full")
    ap.add_argument("--trace", default=None, help="write every position's x after each layer (json)")
    a = ap.parse_args()
    global HC, PY, DC, IC, PXM, SWA
    HC, PY, DC, IC, PXM, SWA = GEOS[a.geo]
    g = json.load(open(a.golden))
    em = Emu(a.model, a.layers)
    print("bf16 -> f16 inexact attention/router weights:", em.f16_inexact, flush=True)
    toks = g["tokens"][: a.n]
    seq = g["prompt_ids"] + toks
    gm = np.array(g["margins"][: a.n])
    tight = [int(i) for i in np.argsort(gm)[:6]]
    res = {}
    for mode in a.modes.split(","):
        o = set(mode.split("+"))
        em.reset()
        if a.trace: em.trace = []
        st = {"rgap": [], "rerr": [], "route": [], "redo": 0}
        mis = []; gaps = {}; t0 = time.time(); lgs = []
        npre = len(g["prompt_ids"])
        for i, t in enumerate(seq[:-1]):
            lg = em.step(t, o, st)
            j = i - (npre - 1)
            if j >= 0:
                want = seq[i + 1]
                top = int(np.argmax(lg))
                srt = np.sort(lg)
                gap = float(lg[want] - (srt[-2] if top == want else srt[-1]))
                if top != want: mis.append(j)
                if j in tight: gaps[j] = round(gap, 5)
                lgs.append(gap)
            if i % 16 == 0: print(mode, i, len(mis), f"{time.time() - t0:.0f}s", flush=True)
        rg = np.array(st["rgap"]); re_ = np.array(st["rerr"])
        res[mode] = {"mismatches": mis, "n": len(toks), "gaps_at_tightest": gaps, "golden_tightest": {t: g["margins"][t] for t in tight},
                     "min_gap": float(min(lgs)), "redo_passes": st["redo"],
                     "router_err_max": float(re_.max()), "router_gap_min": float(rg.min()),
                     "router_gaps_below_err": int(np.sum(rg < re_.max())), "router_gaps_below_2err": int(np.sum(rg < 2 * re_.max())),
                     "router_decisions": int(rg.size), "routes": st["route"]}
        print(mode, {k: v for k, v in res[mode].items() if k != "routes"}, flush=True)
        json.dump(res, open(a.out, "w"))
        if a.trace:
            json.dump([t.tolist() for t in em.trace], open(a.trace.replace(".json", f"_{mode}.json"), "w"))


if __name__ == "__main__":
    main()
