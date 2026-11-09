"""Teacher-forced check of the serve8b_region kernel's number formats on Qwen3-8B (bf16 release).

  tf_qwen3_8b.py <model dir> <golden.json> <mode,mode,...> <ntok> <outdir>   (writes <outdir>/tf_<mode>.json)

The numerics gate for serve8b_region's lever bits on a q/k-norm model: a TRM / SLW combination ships
only if it keeps 0 teacher-forced mismatches against the HF f32 golden.

Layout (qt22, lever D U order): layer 0 and layers 31-35 run on grids (PY 144, PX 128: hidden in
chunks of 29 over the rows, o_proj 32 a column, down 96 a column); layers 1-30 on regions (RPY 208,
RPX 256: hidden 20 a row, o_proj 16 a column, down 48 a column).

A mode is a base + suffixes (underscore-joined):
  ref     HF f32 (bf16 weights widened): the golden's arithmetic
  k       the shipped qt22 kernel arithmetic (TRM 16629, IEEE f16 build):
            - norm output f16(x * g) (f32 product), the rms scale applied in f32 to the outputs (TRM 1)
            - every projection: x and W in f16, each PE's partial f16-accumulated over its chunk
              (one rounding an FMA, @fmach), PEs' partials summed in f32 (F16 3 / f16accg)
            - q/k RMSNorm, RoPE, softmax in f32; q and the KV cache in f16 for the scores (@fmachs),
              probabilities in f16 (the scaled copy: relative rounding) times f16 v, f32 sums
            - lm_head: f16(x) . f16(W), f32 accumulate (MODE 2 one pass, TRM 32)
  suffixes (the Llama levers):
  _xg     TRM 512: region norms' output f16(f16(x) * f16(g))
  _xgg    TRM 8192: the same in the grids
  _s16    TRM 1024: the regions' MLP SiLU in f16 (silu16.py silu16, bit level)
  _s16n   TRM 1024 + 2048: as s16, the two scales (s log2e, s ln2) rounded to f16 (s = the token's
          rms scale), not corrected
  _s16w / _s16wn  as s16 / s16n with the wide-tail SiLU (silu16.py silu16w: no f16 subnormal flush
          at g < -10)
  _rs2    TRM 256: every rsqrt (layer norms, q/k norms) as 0x5f3759df + two Newton steps in f32
  _tiny   TRM 2 + 8: sums of squares in sequential f32 per PE chunk; exp and the grids' SiLU
          reciprocal with a deterministic relative error of 2.7e-6 / 1.2e-5
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silu16 import silu16, silu16w
import torch
import transformers.models.qwen3.modeling_qwen3 as Q
from transformers import AutoModelForCausalLM

torch.set_num_threads(8)
P, G, modes, ntok, outdir = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]), sys.argv[5]
gold = json.load(open(G))
ntok = min(ntok, len(gold["tokens"]))
ids = gold["prompt_ids"] + gold["tokens"][:ntok]
NPR = len(gold["prompt_ids"])
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager")
m.eval()
NL = m.config.num_hidden_layers
GRIDS = {0} | set(range(31, NL))
LOG2E, LN2 = 1.4426950408889634, 0.6931471805599453


def operm(ncol, dpc):
    # o_proj input order: column c = kv head c // (ncol / 8), dims (c % per) * dpc .. of its 4 q heads
    per = ncol // 8
    return torch.tensor([(kh * 4 + j) * 128 + db * dpc + e
                         for c in range(ncol) for kh, db in [(c // per, c % per)] for j in range(4) for e in range(dpc)])


CHR = {"q_proj": 20, "k_proj": 20, "v_proj": 20, "gate_proj": 20, "up_proj": 20, "o_proj": 16, "down_proj": 48}
CHG = {"q_proj": 29, "k_proj": 29, "v_proj": 29, "gate_proj": 29, "up_proj": 29, "o_proj": 32, "down_proj": 96}
PERM = {"r": operm(256, 4), "g": operm(128, 8)}
NORMED = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
S = {"mode": "ref", "layer": 0}
WCACHE = {}
stats = {}


def flags():
    p = S["mode"].split("_")
    return p[0], set(p[1:])


def hsh(t):
    # deterministic +-1 pattern over a tensor's elements
    n = t.numel()
    v = torch.arange(n, dtype=torch.float64).mul(0.6180339887).frac().lt(0.5).float() * 2 - 1
    return v.reshape(t.shape)


def rsqrt2(v):
    v = v.float().contiguous()
    i = torch.tensor(0x5f3759df, dtype=torch.int32) - (v.view(torch.int32) >> 1)
    y = i.view(torch.float32)
    for _ in range(2):
        y = y * (1.5 - 0.5 * v * y * y)
    return y


def f16acc(x, w, chunk, perm, name):
    K = x.shape[1]
    key = (id(w), chunk)
    if perm is not None:
        x = x[:, perm]
        if key not in WCACHE:
            w = w[:, perm]
    nch = (K + chunk - 1) // chunk
    pad = nch * chunk - K
    # half tensors: CPU addcmul on f16 computes in f32 and rounds once (= @fmach; checked
    # bit-equal against the f32 + round form)
    if key not in WCACHE:
        wt = w.half().t()
        if pad:
            wt = torch.nn.functional.pad(wt, (0, 0, 0, pad))
        WCACHE[key] = wt.reshape(nch, chunk, -1).contiguous()
    wt = WCACHE[key]
    xh = x.half()
    if pad:
        xh = torch.nn.functional.pad(xh, (0, pad))
    xh = xh.reshape(x.shape[0], nch, chunk)
    outs, mp = [], 0.0
    for t in range(0, x.shape[0], 16):
        xb = xh[t:t + 16]
        acc16 = torch.zeros(xb.shape[0], nch, wt.shape[2], dtype=torch.half)
        for k in range(chunk):
            acc16.addcmul_(xb[:, :, k, None], wt[None, :, k, :])
        acc = acc16.float()
        if not torch.isfinite(acc).all():
            stats["f16_overflow"] = stats.get("f16_overflow", 0) + 1
        mp = max(mp, float(acc[torch.isfinite(acc)].abs().max()))
        outs.append(acc.sum(1))
    stats.setdefault("max_abs_partial", {})[name] = max(stats.get("max_abs_partial", {}).get(name, 0.0), mp)
    return torch.cat(outs, 0)


def patch_lin(lin, name, li):
    w = lin.weight
    def fwd(x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1]).float()
        base, fl = flags()
        if base == "ref":
            y = xf @ w.float().t()
        else:
            g = li in GRIDS
            ch = (CHG if g else CHR)[name]
            y = f16acc(xf, w, ch, PERM["g" if g else "r"] if name == "o_proj" else None, name)
            if name in NORMED:
                y = y * S["rs"][0 if name in ("q_proj", "k_proj", "v_proj") else 1].reshape(-1, 1)
        return y.reshape(*shp[:-1], -1)
    lin.forward = fwd


def patch_norm(nm, li, which):
    orig = nm.forward
    def fwd(x):
        base, fl = flags()
        if base == "ref":
            return orig(x)
        xf = x.float()
        ms = xf.pow(2).mean(-1, keepdim=True)
        if "tiny" in fl:
            # each PE's chunk summed in order (f32), then the chunks' sums (all-reduce)
            ch = 29 if li in GRIDS else 20
            K = xf.shape[-1]; nch = (K + ch - 1) // ch
            xp = torch.nn.functional.pad(xf.reshape(-1, K), (0, nch * ch - K)).reshape(-1, nch, ch)
            acc = torch.zeros(xp.shape[0], nch)
            for k in range(ch):
                acc = acc + xp[:, :, k] * xp[:, :, k]
            ms = (acc.sum(-1) / K).reshape(*xf.shape[:-1], 1)
        v = ms + nm.variance_epsilon
        rs = rsqrt2(v) if "rs2" in fl else torch.rsqrt(v)
        S["rs"][which] = rs.reshape(-1)
        wv = nm.weight.float()
        xg = ("xgg" in fl) if li in GRIDS else ("xg" in fl)
        if xg:
            return (xf.half() * wv.half()).float()     # f16 product, one rounding
        return xf * wv                                  # f32 product; f16acc rounds it once
    nm.forward = fwd


def patch_qknorm(nm):
    orig = nm.forward
    def fwd(x):
        base, fl = flags()
        if base == "ref" or "rs2" not in fl:
            return orig(x)
        xf = x.float()
        rs = rsqrt2(xf.pow(2).mean(-1, keepdim=True) + nm.variance_epsilon)
        return nm.weight.float() * (xf * rs)
    nm.forward = fwd


def patch_mlp(mlp, li):
    orig = mlp.forward
    def fwd(x):
        base, fl = flags()
        if base == "ref":
            return orig(x)
        g = mlp.gate_proj(x).float(); u = mlp.up_proj(x).float()
        if li not in GRIDS and fl & {"s16", "s16n", "s16w", "s16wn"}:
            sf = silu16w if fl & {"s16w", "s16wn"} else silu16
            if fl & {"s16n", "s16wn"}:
                s = S["rs"][1].reshape(*g.shape[:-1], 1)
                kg = (s * LOG2E).half().float() / (s * LOG2E)   # the f16 scale's relative rounding
                ku = (s * LN2).half().float() / (s * LN2)
                # silu16 takes s g and s u (f32) and multiplies by log2e / ln2 before its one fs2h;
                # 2048's form: raw g (f16) times f16(s log2e): emulate as the scale's rounding
                # applied, then silu16 (its own f16 roundings of G, U included)
                a = sf((g / s).half().float() * s * kg, (u / s).half().float() * s * ku)
            else:
                a = sf(g, u)
            return mlp.down_proj(a)
        act = torch.nn.functional.silu(g)
        if li in GRIDS and "tiny" in fl:
            act = act * (1 + 1.2e-5 * hsh(act))
        return mlp.down_proj(act * u)
    mlp.forward = fwd


def eager_kernel(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    base, fl = flags()
    if base == "ref":
        return orig_eager(module, query, key, value, attention_mask, scaling, dropout, **kw)
    key_states = Q.repeat_kv(key, module.num_key_value_groups).half().float()
    value_states = Q.repeat_kv(value, module.num_key_value_groups).half().float()
    q = (query.float() * scaling).half().float()
    sc = torch.matmul(q, key_states.transpose(2, 3))
    if attention_mask is not None:
        sc = sc + attention_mask[:, :, :, : key_states.shape[-2]]
    mx = sc.amax(-1, keepdim=True)
    p = torch.exp(sc - mx)
    if "tiny" in fl:
        p = p * (1 + 2.7e-6 * hsh(p))
    l = p.sum(-1, keepdim=True)
    p16 = (p * 16384.0).half().float() / 16384.0
    o = torch.matmul(p16, value_states) / l
    return o.transpose(1, 2).contiguous(), None


orig_eager = Q.eager_attention_forward
Q.eager_attention_forward = eager_kernel
Q.ALL_ATTENTION_FUNCTIONS = getattr(Q, "ALL_ATTENTION_FUNCTIONS", None)

for li, layer in enumerate(m.model.layers):
    at = layer.self_attn
    for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
        patch_lin(getattr(at, nm), nm, li)
    for nm in ("gate_proj", "up_proj", "down_proj"):
        patch_lin(getattr(layer.mlp, nm), nm, li)
    patch_mlp(layer.mlp, li)
    patch_norm(layer.input_layernorm, li, 0)
    patch_norm(layer.post_attention_layernorm, li, 1)
    patch_qknorm(at.q_norm)
    patch_qknorm(at.k_norm)
S["rs"] = [None, None]
lmw = m.lm_head.weight
m.float()
refpred = None
for mode in modes:
    S["mode"] = mode
    stats = {}
    t0 = time.time()
    with torch.no_grad():
        hs = m.model(torch.tensor([ids]), use_cache=False).last_hidden_state[0].float()
        hs = hs[NPR - 1:NPR - 1 + ntok]
        if mode.split("_")[0] == "ref":
            pred = hs @ lmw.float().t()
        else:
            pred = hs.half().float() @ lmw.half().float().t()
    top = torch.topk(pred, 2, dim=-1)
    got = top.indices[:, 0].tolist()
    want = gold["tokens"][:ntok]
    mis = [{"i": i, "got": g, "want": w_, "golden_margin": round(gold["margins"][i], 4),
            "our_margin": round(float(top.values[i, 0] - top.values[i, 1]), 4),
            "logit_gap": round(float(pred[i, g] - pred[i, w_]), 4)}
           for i, (g, w_) in enumerate(zip(got, want)) if g != w_]
    if mode == "ref":
        refpred = pred.clone()
    # our gap at the golden's tightest margins (golden token's logit minus the best other)
    tight = sorted(range(ntok), key=lambda i: gold["margins"][i])[:8]
    def gap(i):
        row = pred[i].clone(); wv = float(row[want[i]]); row[want[i]] = -1e30
        return round(wv - float(row.max()), 4)
    stats["tight"] = [{"i": i, "golden_margin": round(gold["margins"][i], 4), "gap_here": gap(i)} for i in tight]
    stats["min_gap"] = min(gap(i) for i in range(ntok))
    if refpred is not None:
        d = pred - refpred
        stats["vs_ref"] = {"max_abs_logit_err": float(d.abs().max()), "rms_logit_err": float(d.pow(2).mean().sqrt())}
    rep = {"mode": mode, "golden": G, "ntok": ntok, "n_mismatch": len(mis), "mismatches": mis, "stats": stats,
           "seconds": round(time.time() - t0)}
    json.dump(rep, open(f"{outdir}/tf_{mode}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("stats", "mismatches")}),
          json.dumps(mis[:6]), json.dumps(stats.get("tight")[:4]), stats.get("min_gap"), stats.get("vs_ref"), flush=True)
