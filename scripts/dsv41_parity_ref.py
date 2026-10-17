"""DeepSeek-V4.1-Flash reference logits for a release-layout snapshot (the
`mini-l8-e16-fp8` miniature `shrink_checkpoint.py --keep-fp8` cuts), from
DeepSeek's own `inference/model.py` at the release's revision, so a pie
reading of the same snapshot can be read against it.

DeepSeek's model runs as published: MXFP8 projections (e4m3 activations in
1x32 blocks under ue8m0 scales against the 32x32-block weights), MXFP4
experts against the same activations, `wo_a` read out to bf16 under its
quantized input, the FP8 Engram table read out on lookup, the window KV's
fp8 and the compressed KV's FP4 (E2M1, an E4M3 scale a 16) and the
indexer's FP4 quant-dequant. Only `kernel.py` is replaced: its tilelang
kernels are written out below in torch, each computing what its kernel
does (the same scale rules, e2m1 rounding to nearest even, and the sparse
attention's probabilities rounded to bf16 before they weigh the values).

    python3 scripts/dsv41_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]
        [--sim kv,index,o] [--tag ref]

`--sim` names the quant-dequant simulations to keep (default `kv,index,o`,
all of them): `kv` the window KV's fp8 and the compressed KV's FP4, `index`
the indexer's FP4 keys and queries, `o` the attention output's fp8 before
`wo_a`. `--sim none` reads the KV and `wo_a`'s input at bf16, the reference
for a deployment whose cache is bf16.

Writes, per probe NAME (`OUT/probes.json` when no file is stated):

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.
"""

import argparse
import json
import os
import sys
import time
import types
import urllib.request

import numpy as np
import torch

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
SOURCES = ["model.py", "engram.py", "vision.py", "image_processor.py"]

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
]

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_EDGES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def fetch_sources(into):
    os.makedirs(into, exist_ok=True)
    for name in SOURCES:
        path = os.path.join(into, name)
        if not os.path.exists(path):
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/inference/{name}"
            with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:
                f.write(r.read())
            os.replace(path + ".part", path)
    return into


def pow2_ceil(x):
    """2^ceil(log2 x) for positive finite x, as `fast_round_scale` takes it
    off the float's bits."""
    m, e = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(m == 0.5, e - 1, e))


def to_e2m1(v):
    """`v` (already within [-6, 6]) rounded to the nearest e2m1, ties to
    the even code, as `cvt.rn` does."""
    a = v.abs()
    edges = torch.tensor(E2M1_EDGES, device=v.device)
    idx = torch.searchsorted(edges, a.contiguous(), right=False)
    tie = (idx < len(E2M1_EDGES)) & (a == edges[idx.clamp(max=len(E2M1_EDGES) - 1)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.tensor(E2M1, device=v.device)[idx] * torch.sign(v)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    xf = x.float().unflatten(-1, (-1, block_size))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax / 448.0) if scale_fmt is not None else amax / 448.0
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    xf = x.float().unflatten(-1, (-1, block_size))
    amax = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = pow2_ceil(amax.clamp_min(6 * 2.0**-126) / 6.0)
    q = to_e2m1((xf / s).clamp(-6.0, 6.0))
    assert inplace, "the release only quant-dequants FP4 in place"
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def blocks(s, rows, cols):
    """A [rows/r, cols/c] block-scale plane spread over [rows, cols]."""
    s = s.float()
    s = s.repeat_interleave(-(-rows // s.shape[0]), 0)[:rows]
    return s.repeat_interleave(-(-cols // s.shape[1]), 1)[:, :cols]


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    k = a.size(-1)
    af = a.float().unflatten(-1, (-1, block_size)) * a_s.float().unsqueeze(-1)
    bf = b.float() * blocks(b_s, b.size(0), k)
    return (af.flatten(-2) @ bf.T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    codes = b.view(torch.uint8)
    nib = torch.stack([codes & 0x0F, (codes >> 4) & 0x0F], dim=-1).flatten(-2)
    table = torch.tensor(E2M1 + [-v for v in E2M1], device=b.device)
    bf = table[nib.long()] * blocks(b_s, b.size(0), nib.size(-1))
    af = a.float().unflatten(-1, (-1, act_block_size)) * a_s.float().unsqueeze(-1)
    return (af.flatten(-2) @ bf.T).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    for bi in range(b):
        for lo in range(0, m, 64):
            hi = min(m, lo + 64)
            idx = topk_idxs[bi, lo:hi].long()
            valid = idx >= 0
            g = kv[bi][idx.clamp_min(0)].float() * valid.unsqueeze(-1)
            s = torch.einsum("mhd,mtd->mht", q[bi, lo:hi].float(), g) * softmax_scale
            s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = s.amax(-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(s - mx)
            den = p.sum(-1) + torch.exp(attn_sink.float().view(1, h) - mx.squeeze(-1))
            o = torch.einsum("mht,mtd->mhd", p.to(torch.bfloat16).float(), g)
            out[bi, lo:hi] = (o / den.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def install_kernel(sim):
    k = types.ModuleType("kernel")
    k.act_quant = act_quant
    def fp4_sim(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        # e4m3 scales a 16 are the compressed KV's; ue8m0 a 32 the indexer's.
        which = "kv" if scale_dtype == torch.float8_e4m3fn else "index"
        return fp4_act_quant(x, block_size, inplace, scale_dtype) if which in sim else x

    k.fp4_act_quant = fp4_sim
    k.fp8_gemm = fp8_gemm
    k.fp4_gemm = fp4_gemm
    k.sparse_attn = sparse_attn
    k.hc_split_sinkhorn = hc_split_sinkhorn
    sys.modules["kernel"] = k


def model_args(model_mod, cfg):
    t = cfg.get("text_config", cfg)
    q = cfg.get("quantization_config", t.get("quantization_config", {}))
    fields = dict(
        max_batch_size=1, max_seq_len=4096, temperature=0.0, dtype="fp8",
        expert_dtype=q.get("expert_dtype", "fp4"),
        vocab_size=t["vocab_size"], dim=t["hidden_size"], moe_inter_dim=t["moe_intermediate_size"],
        n_layers=t["num_hidden_layers"], n_mtp_layers=0, n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"], n_shared_experts=t["n_shared_experts"],
        n_activated_experts=t["num_experts_per_tok"], score_func=t["scoring_func"],
        norm_topk_prob=t["norm_topk_prob"], route_scale=t["routed_scaling_factor"],
        swiglu_limit=t["swiglu_limit"], q_lora_rank=t["q_lora_rank"], head_dim=t["head_dim"],
        rope_head_dim=t["qk_rope_head_dim"], norm_eps=t["rms_norm_eps"], o_groups=t["o_groups"],
        o_lora_rank=t["o_lora_rank"], window_size=t["sliding_window"],
        compress_ratios=tuple(t["compress_ratios"]),
        kv_source_layers=tuple(t["kv_source_layer_ids"]),
        index_source_layers=tuple(t["index_source_layer_ids"]),
        compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=t["rope_scaling"]["original_max_position_embeddings"],
        rope_theta=t["rope_theta"], rope_factor=t["rope_scaling"]["factor"],
        beta_fast=t["rope_scaling"]["beta_fast"], beta_slow=t["rope_scaling"]["beta_slow"],
        index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"], index_topk=t["index_topk"],
        candidate_source_layer=t.get("candidate_source_layer_id", -1),
        candidate_topk_blocks=t.get("candidate_topk_blocks", 0),
        candidate_block_size=t.get("candidate_block_size", 0),
        hc_mult=t["hc_mult"], hc_sinkhorn_iters=t["hc_sinkhorn_iters"], hc_eps=t["hc_eps"],
        engram_layer_ids=tuple(t.get("engram_layer_ids", ())),
        engram_num_embeddings=tuple(t.get("engram_num_embeddings", ())),
        engram_max_ngram_size=t.get("engram_max_ngram_size", 1),
        engram_vocab_size=t.get("engram_vocab_size", 0), engram_n_heads=t.get("engram_n_heads", 0),
        engram_head_dim=t.get("engram_head_dim", 0), engram_pad_id=t.get("engram_pad_token_id", 2),
        engram_compressed_vocab_size=t.get("engram_compressed_vocab_size", 0),
        vision_n_layers=0, dspark_block_size=0,
    )
    return model_mod.ModelArgs(**fields)


def load(model, snapshot):
    """The snapshot's tensors into the model as `convert.py` lands them at
    one rank: `wo_a` read out to bf16 under its 32x32 exponents, the fp4
    experts' code pairs viewed as e2m1 pairs, everything else as stored
    (into the parameter's own dtype)."""
    from safetensors import safe_open

    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    params = dict(model.named_parameters())
    seen = set()
    for shard in sorted(set(index.values())):
        with safe_open(os.path.join(snapshot, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in params:
                    continue
                t = f.get_tensor(name)
                if name.endswith("wo_a.weight"):
                    s = f.get_tensor(name[: -len("weight")] + "scale")
                    t = (t.float() * blocks(s, t.size(0), t.size(1))).bfloat16()
                p = params[name]
                if p.dtype == torch.float4_e2m1fn_x2:
                    p.data.copy_(t.view(torch.float4_e2m1fn_x2).to(p.device))
                else:
                    p.data.copy_(t.to(p.device))
                seen.add(name)
    missing = [n for n in params if n not in seen]
    if missing:
        raise SystemExit(f"{len(missing)} parameters not in the snapshot, first {missing[:4]}")


def logits_all(model, mod, ids, start_pos):
    """`Transformer.forward` with every position's logits."""
    engram_hashes = model.engram_hash(ids, start_pos, None) if model.engram_hash is not None else None
    h = model.embed(ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = mod.make_identity_pre_mix(h, model.hc_mult)
    for layer in model.layers:
        if layer.engram is not None:
            h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre_mix = layer(h, start_pos, pre_mix, None)
    h = layer.hc_pre(h, pre_mix)
    return model.head(model.norm(h), full_logits=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--sim", default="kv,index,o")
    ap.add_argument("--act", default="fp8", choices=["fp8", "bf16"],
                    help="bf16: projections read their activations unquantized (the weights still "
                         "read out exactly), the reference for a deployment at bf16 activations")
    ap.add_argument("--inference", default=None, help="DeepSeek's inference/ sources (fetched when absent)")
    ap.add_argument("--layers", type=int, default=None,
                    help="run only the first N layers (then the head), to bisect a disagreement")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sim = set(args.sim.split(",")) - {"none", ""}

    src = fetch_sources(args.inference or os.path.join(args.out, ".inference"))
    install_kernel(sim)
    sys.path.insert(0, src)
    import model as mod
    from transformers import AutoTokenizer

    if "o" not in sim:
        mod_act_quant = mod.act_quant

        def act_quant_but_o(x, *a, **kw):
            # `o` is the only 4-d activation the release quant-dequants in place.
            if kw.get("inplace") or (len(a) >= 4 and a[3]):
                if x.dim() == 4:
                    return x
            return mod_act_quant(x, *a, **kw)

        mod.act_quant = act_quant_but_o
    if "kv" not in sim:
        mod_act_quant2 = mod.act_quant

        def act_quant_but_window(x, *a, **kw):
            # The window KV is the only 3-d activation quant-dequanted in place.
            if (kw.get("inplace") or (len(a) >= 4 and a[3])) and x.dim() == 3:
                return x
            return mod_act_quant2(x, *a, **kw)

        mod.act_quant = act_quant_but_window

    if args.act == "bf16":
        quantized = mod.linear

        def linear_bf16(x, weight, bias=None):
            if weight.dtype == torch.float4_e2m1fn_x2:
                codes = weight.view(torch.uint8)
                nib = torch.stack([codes & 0x0F, (codes >> 4) & 0x0F], dim=-1).flatten(-2)
                table = torch.tensor(E2M1 + [-v for v in E2M1], device=weight.device)
                w = table[nib.long()] * blocks(weight.scale, nib.size(0), nib.size(-1))
            elif weight.dtype == torch.float8_e4m3fn:
                w = weight.float() * blocks(weight.scale, weight.size(0), weight.size(1))
            else:
                return quantized(x, weight, bias)
            return torch.nn.functional.linear(x, w.to(x.dtype))

        mod.linear = linear_bf16

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    cfg = json.load(open(os.path.join(args.snapshot, "config.json")))
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    margs = model_args(mod, cfg)
    if args.layers is not None:
        margs.n_layers = args.layers
    with torch.device("cuda"):
        model = mod.Transformer(margs, tokenizer)
    model.eval()
    load(model, args.snapshot)

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        probes = [{"name": n, "ids": bos + tokenizer.encode(t, add_special_tokens=False)} for n, t in PROMPTS]
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        with torch.inference_mode():
            x = torch.tensor([ids], device="cuda")
            tf = logits_all(model, mod, x, 0)[0].float().cpu().numpy()
            gen_rows = [tf[-1].copy()]
            produced = []
            last = int(tf[-1].argmax())
            pos = len(ids)
            for _ in range(args.steps):
                produced.append(last)
                row = logits_all(model, mod, torch.tensor([[last]], device="cuda"), pos)[0, -1]
                row = row.float().cpu().numpy()
                gen_rows.append(row)
                last = int(row.argmax())
                pos += 1
        vocab = tf.shape[-1]
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens, vocab {vocab}, gen={produced[:12]}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
