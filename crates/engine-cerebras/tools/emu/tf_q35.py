"""Teacher-forced check of the wafer program kernels' number formats on any Qwen3.5 against an HF f32
golden (the numerics gate before a build goes to the chip: a lever ships only if it keeps 0
mismatches). The f16 accumulation chunks are the prep's geometry, from its meta.json.

  tf_q35.py <model dir> <golden json> <prep meta.json> <mode,mode,...> <ntok> <outdir>
  (writes <outdir>/tf_<mode>.json)
  extra suffix _h16: the head as the chip's NPX 1: f16(x (1 + w)) . f16(W), f32 accumulation (the final
  rms scale applied after, a positive scalar)

A mode is a base + optional suffixes (underscore-joined):
  ref     x f32, W bf16 widened, f32 accumulate (the golden's arithmetic)
  bf16x   x rounded to bf16 once (one-pass, NP 1), f32 accumulate
  f16x    x and W IEEE f16, f32 accumulate (@fmachs on an IEEE build)
  f16acc  x and W f16, each weight PE's partial accumulated in f16 over its input chunk (rounded
          after every FMA, as @fmach), partials summed in f32: chunk 20 for the hidden-input
          projections (4096 over 208 rows), 16 for GDN out_proj (16 v dims a column) and attention
          o_proj (the column order: kv head c // 64, dims (c % 64) * 4 .. + 3 of its 4 q heads),
          48 for down_proj (12288 over 256 columns)
  _s16    the MLP's SiLU in f16 (silu16, bit-level emulation of the kernel, silu16.py)
  _xg16   the layer norms' output as f16(f16(x) * f16(1 + w)) with the rms scale applied in f32
          to the projections' outputs (TRM 512's form)
  _kv16   attention's cached k and v rounded to f16
Every other op (GDN conv / gates / state, softmax, norms' rsqrt, lm_head) stays f32."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silu16 import silu16
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as Q
from transformers import AutoModelForCausalLM

torch.set_num_threads(8)
torch.set_grad_enabled(False)
P, G, MJ = sys.argv[1:4]
modes, ntok, outdir = sys.argv[4].split(","), int(sys.argv[5]), sys.argv[6]
MT = json.load(open(MJ))
gold = json.load(open(G))
ids = gold["prompt_ids"] + gold["tokens"][:ntok]
NPR = len(gold["prompt_ids"])
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="eager")
m.eval()
tm = m.model.language_model if hasattr(m.model, "language_model") else m.model
PX_, Hc_, Ic_, Qh_, Dc_, DV_, HD_ = MT["PX"], MT["Hc"], MT["Ic"], MT["Qh"], MT["Dc"], MT["DV"], MT["HD"]
cpa = MT["SWA"]; rd = HD_ // 4; nrc = rd // Dc_; NROT_ = Dc_ // 2
def _perm(p):
    if p < nrc:
        return [d for j in range(NROT_) for d in (p * NROT_ + j, p * NROT_ + j + rd // 2)]
    return [rd + (p - nrc) * Dc_ + j for j in range(Dc_)]
# o_proj's inputs in the kernel's order: column c (kv head c // cpa) sums its Qh heads' Dc dims (pd)
OPERM = torch.tensor([((c // cpa) * Qh_ + j) * HD_ + d for c in range(PX_) for j in range(Qh_) for d in _perm(c % cpa)])
CH = {"in_proj_qkv": Hc_, "in_proj_z": Hc_, "in_proj_b": Hc_, "in_proj_a": Hc_, "q_proj": Hc_, "k_proj": Hc_,
      "v_proj": Hc_, "gate_proj": Hc_, "up_proj": Hc_, "out_proj": DV_, "o_proj": Qh_ * Dc_, "down_proj": Ic_}
NORMED = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
S = {"mode": "ref", "rs": None}
stats = {}


def flags():
    p = S["mode"].split("_")
    return p[0], set(p[1:])


def f16acc(x, w, chunk, perm, name):
    K = x.shape[1]
    if perm is not None:
        x = x[:, perm]; w = w[:, perm]
    nch = (K + chunk - 1) // chunk
    pad = nch * chunk - K
    xh = x.half().float()
    wt = w.half().float().t()
    if pad:
        xh = torch.nn.functional.pad(xh, (0, pad)); wt = torch.nn.functional.pad(wt, (0, 0, 0, pad))
    wt = wt.reshape(nch, chunk, -1)
    xh = xh.reshape(x.shape[0], nch, chunk)
    outs, mp = [], 0.0
    for t in range(0, x.shape[0], 8):
        xb = xh[t:t + 8]
        acc = torch.zeros(xb.shape[0], nch, wt.shape[2])
        for k in range(chunk):
            acc.addcmul_(xb[:, :, k, None], wt[None, :, k, :])
            acc = acc.half().float()
        if not torch.isfinite(acc).all():
            stats["f16_overflow"] = stats.get("f16_overflow", 0) + 1
        mp = max(mp, float(acc[torch.isfinite(acc)].abs().max()))
        outs.append(acc.sum(1))
    stats.setdefault("max_abs_partial", {})[name] = max(stats.get("max_abs_partial", {}).get(name, 0.0), mp)
    return torch.cat(outs, 0)


def patch_lin(lin, name):
    w = lin.weight
    def fwd(x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1]).float()
        base, fl = flags()
        stats.setdefault("max_abs_x", {})[name] = max(stats.get("max_abs_x", {}).get(name, 0.0), float(xf.abs().max()))
        if base == "ref":
            y = xf @ w.float().t()
        elif base == "bf16x":
            y = xf.to(torch.bfloat16).float() @ w.float().t()
        elif base == "f16x":
            y = xf.half().float() @ w.half().float().t()
        elif base == "f16acc":
            y = f16acc(xf, w, CH[name], OPERM if name == "o_proj" else None, name)
        else:
            raise SystemExit("mode?")
        if "xg16" in fl and name in NORMED:
            y = y * S["rs"].reshape(-1, 1)
        return y.reshape(*shp[:-1], -1)
    lin.forward = fwd


def patch_norm(nm):
    orig = nm.forward
    def fwd(x):
        base, fl = flags()
        if "xg16" not in fl:
            return orig(x)
        xf = x.float()
        S["rs"] = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + nm.eps).reshape(-1)
        return (xf.half() * (1.0 + nm.weight.float()).half()).float()   # f16 product (one rounding)
    nm.forward = fwd


def patch_mlp(mlp):
    orig = mlp.forward
    def fwd(x):
        base, fl = flags()
        if "s16" in fl:
            g = mlp.gate_proj(x).float(); u = mlp.up_proj(x).float()
            return mlp.down_proj(silu16(g, u))
        return orig(x)
    mlp.forward = fwd


orig_eager = Q.eager_attention_forward
def eager16(module, query, key, value, *a, **k):
    base, fl = flags()
    if "kv16" in fl:
        key = key.half().float(); value = value.half().float()
    return orig_eager(module, query, key, value, *a, **k)
Q.eager_attention_forward = eager16

for layer in tm.layers:
    mix = layer.linear_attn if hasattr(layer, "linear_attn") and layer.linear_attn is not None else layer.self_attn
    for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
        if hasattr(mix, nm):
            patch_lin(getattr(mix, nm), nm)
    for nm in ("gate_proj", "up_proj", "down_proj"):
        patch_lin(getattr(layer.mlp, nm), nm)
    patch_mlp(layer.mlp)
    patch_norm(layer.input_layernorm)
    patch_norm(layer.post_attention_layernorm)
m.float()
cap = {}
tm.norm.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0].detach()[0].float()))
W16 = (m.lm_head.weight.detach().to(torch.bfloat16).float().half().float()).t().contiguous()
w1 = 1 + tm.norm.weight.detach().float()
def head16(n):
    x = cap["x"][NPR - 1:NPR - 1 + n]
    rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + tm.norm.eps)
    return ((x * w1).half().float() @ W16) * rs
refpred = None
for mode in modes:
    S["mode"] = mode
    stats = {}
    t0 = time.time()
    with torch.no_grad():
        lg = m(torch.tensor([ids]), use_cache=False).logits[0].float()
    pred = lg[NPR - 1:NPR - 1 + ntok]
    if "h16" in flags()[1]:
        pred = head16(ntok)
    torch.save({"pred": pred.clone(), "rs": torch.rsqrt(cap["x"][NPR - 1:NPR - 1 + ntok].pow(2).mean(-1) + tm.norm.eps)}, f"{outdir}/pred_{mode}.pt")
    top = torch.topk(pred, 2, dim=-1)
    got = top.indices[:, 0].tolist()
    want = gold["tokens"][:ntok]
    mis = [{"i": i, "got": g, "want": w_, "golden_margin": round(gold["margins"][i], 4),
            "our_margin": round(float(top.values[i, 0] - top.values[i, 1]), 4),
            "logit_gap": round(float(pred[i, g] - pred[i, w_]), 4)}
           for i, (g, w_) in enumerate(zip(got, want)) if g != w_]
    if mode == "ref":
        refpred = pred.clone()
    if refpred is not None:
        d = pred - refpred
        stats["vs_ref"] = {"max_abs_logit_err": float(d.abs().max()), "rms_logit_err": float(d.pow(2).mean().sqrt()),
                           "argmax_differs_from_ref": int((pred.argmax(-1) != refpred.argmax(-1)).sum())}
        # the golden's tightest ties: our logit gap there (golden's token minus runner-up)
        tight = sorted(range(ntok), key=lambda i: gold["margins"][i])[:6]
        stats["tight"] = [{"i": i, "golden_margin": round(gold["margins"][i], 4),
                           "gap_here": round(float(pred[i, want[i]] - torch.topk(pred[i], 2).values[1 if got[i] == want[i] else 0]), 4)}
                          for i in tight]
    rep = {"mode": mode, "ntok": ntok, "n_mismatch": len(mis), "n_mismatch_first64": sum(1 for x in mis if x["i"] < 64),
           "mismatches": mis, "stats": stats, "seconds": round(time.time() - t0)}
    json.dump(rep, open(f"{outdir}/tf_{mode}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("stats",)}), flush=True)
    print(json.dumps(stats), flush=True)
