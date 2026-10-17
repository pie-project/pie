"""DeepSeek-V4.1 reference logits for a release-layout snapshot (the full
release or a truncation of it to its first layers), from DeepSeek's own
`inference/model.py` run on the CPU, so a pie reading of the same snapshot
can be read against it at the release's own precision.

    python3 scripts/dsv41_release_parity_ref.py <snapshot-dir> <inference-dir> OUT
        [--steps 8] [--probes probes.json] [--no-sims] [--threads 64]

`inference-dir` holds the release's `inference/` (`model.py`, `engram.py`,
`vision.py`, `image_processor.py`); its `kernel.py` is tilelang (GPU), so
this file stands in for it with the same arithmetic in torch:

  act_quant       e4m3 under a 2^ceil(log2(amax / 448)) scale a 1xblock, the
                  amax floored at 1e-4 (`scale_fmt` ue8m0; f32 `amax / 448`
                  without it), quant-dequantized in place or returned
  fp8_gemm        a's codes x b's codes, each block under its two scales,
                  summed in f32 (the products are exact, so one f32 matmul
                  over the dequantized values)
  fp4_gemm        the same over e2m1 nibbles (low first) under e8m0 a 1x32
  fp4_act_quant   e2m1 under an e8m0 (or e4m3) scale a block, in place
  sparse_attn     gathered keys, softmax with the sink, f32
  hc_split_sinkhorn  the hyper-connection split and Sinkhorn normalisation

`--no-sims` turns off the reference's quant-dequant simulations outside the
products (the window KV's and the attention output's fp8 1x32, the
compressed KV's and the indexer's fp4), which pie does not apply, so a run
can attribute what remains to the products.

Writes OUT/probes.json (or reads `--probes`) and per probe NAME:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] last prompt position, then each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`a_family_is_read_against_its_reference.rs`) against these.
"""

import argparse
import json
import os
import re
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the mountains,"),
    ("chat", "Q: What is the boiling point of water at sea level?\nA:"),
]

SIMS = True

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _pow2_ceil(v):
    """2^ceil(log2(v)), by the exponent bits as the kernel takes them."""
    bits = v.float().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return torch.pow(2.0, e.float())


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / 448.0) if scale_fmt is not None else amax * (1.0 / 448.0)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        # In place, the reference simulates a quantization outside its
        # products (the window KV, the attention output before wo_a).
        if SIMS:
            x.copy_((q.float() * s).reshape(shape).to(x.dtype))
        return x
    return q.reshape(shape), s.squeeze(-1).to(scale_dtype)


def _e2m1_round(v):
    """The e2m1 value nearest each of v (|v| <= 6), ties to even."""
    grid = E2M1[:8]
    a = v.abs()
    d = (a.unsqueeze(-1) - grid).abs()
    best = d.min(-1, keepdim=True).values
    # ties go to the even mantissa: indices 0,2,4,6 of the grid
    tie = (d == best)
    even = torch.tensor([True, False, True, False, True, False, True, False])
    idx = torch.where((tie & even).any(-1), (tie & even).float().argmax(-1), tie.float().argmax(-1))
    return grid[idx] * torch.sign(v)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace
    if not SIMS:
        return x
    shape = x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
    amax = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2.0**-126) / 6.0)
    q = _e2m1_round((xf / s).clamp(-6.0, 6.0))
    x.copy_((q * s).reshape(shape).to(x.dtype))
    return x


def _blocked(scale, rows, cols):
    return scale.float().repeat_interleave(rows, 0).repeat_interleave(cols, 1)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    k = a.size(-1)
    av = a.float().reshape(-1, k // block_size, block_size) * a_s.float().reshape(-1, k // block_size, 1)
    n = b.size(0)
    bv = b.float() * _blocked(b_s, block_size, block_size)[:n, :k]
    out = av.reshape(-1, k) @ bv.t()
    return out.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def _fp4_values(b, b_s):
    raw = b.view(torch.uint8)
    lo, hi = (raw & 0xF).long(), (raw >> 4).long()
    vals = torch.stack([E2M1[lo], E2M1[hi]], -1).reshape(raw.size(0), -1)
    return vals * b_s.float().repeat_interleave(32, 1)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    k = a.size(-1)
    av = a.float().reshape(-1, k // act_block_size, act_block_size) * a_s.float().reshape(-1, k // act_block_size, 1)
    out = av.reshape(-1, k) @ _fp4_values(b, b_s).t()
    return out.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    for bi in range(b):
        idx = topk_idxs[bi].long()  # [m, topk]
        valid = idx >= 0
        g = kv[bi].float()[idx.clamp_min(0)]  # [m, topk, d]
        s = torch.einsum("mhd,mtd->mht", q[bi].float(), g) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
        mx = torch.maximum(s.amax(-1, keepdim=True), torch.full_like(s[..., :1], -1e30))
        p = torch.exp(s - mx)
        denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, h, 1) - mx)
        p = p.to(torch.bfloat16).float()
        out[bi] = (torch.einsum("mht,mtd->mhd", p, g) / denom).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def install_kernels():
    k = types.ModuleType("kernel")
    for f in (act_quant, fp4_act_quant, fp4_gemm, fp8_gemm, hc_split_sinkhorn, sparse_attn):
        setattr(k, f.__name__, f)
    sys.modules["kernel"] = k


def model_args(model_mod, snapshot, inference_cfg):
    """The reference's arguments: its own config.json, cut to the snapshot's
    layers as the snapshot's config states them."""
    base = json.load(open(inference_cfg))
    t = json.load(open(os.path.join(snapshot, "config.json")))
    t = t.get("text_config", t)
    L = t["num_hidden_layers"]
    base.update(
        n_layers=L,
        n_mtp_layers=0,
        dspark_block_size=0,
        dspark_target_layer_ids=[],
        n_routed_experts=t["n_routed_experts"],
        compress_ratios=t["compress_ratios"][:L],
        kv_source_layers=t["kv_source_layer_ids"],
        index_source_layers=t["index_source_layer_ids"],
        candidate_source_layer=t["candidate_source_layer_id"],
        engram_layer_ids=t["engram_layer_ids"],
        engram_num_embeddings=t["engram_num_embeddings"],
        engram_vocab_size=t["engram_vocab_size"],
        vision_n_layers=0,
        max_batch_size=1,
        max_seq_len=512,
    )
    return model_mod.ModelArgs(**base)


def load(model, snapshot):
    """The snapshot's tensors under the reference's names (convert.py at one
    rank): wo_a dequantized to bf16 under its block exponents, the routed
    experts' int8 nibble pairs read as float4_e2m1fn_x2."""
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    params = dict(model.named_parameters())
    files = {}
    for name, fn in index.items():
        files.setdefault(fn, []).append(name)
    loaded = set()
    for fn, names in files.items():
        with safe_open(os.path.join(snapshot, fn), "pt") as f:
            for name in names:
                if name.endswith("wo_a.scale"):
                    continue
                t = f.get_tensor(name)
                if name.endswith("wo_a.weight"):
                    s = f.get_tensor(name.replace("weight", "scale"))
                    br, bc = t.size(0) // s.size(0), t.size(1) // s.size(1)
                    t = (t.float() * _blocked(s, br, bc)).bfloat16()
                elif ".experts." in name and "shared" not in name and t.dtype == torch.int8:
                    t = t.view(torch.float4_e2m1fn_x2)
                if name not in params:
                    continue
                p = params[name]
                with torch.no_grad():
                    if p.dtype != t.dtype and p.dtype in (torch.float32,) and t.dtype == torch.bfloat16:
                        p.data = t.float()
                    else:
                        assert p.shape == t.shape, (name, p.shape, t.shape)
                        p.data = t
                loaded.add(name)
    missing = sorted(set(params) - loaded)
    if missing:
        raise SystemExit(f"the snapshot does not hold {len(missing)} of the reference's tensors: {missing[:20]}")
    # Linear modules reach their scale through `.weight.scale`.
    for mod in model.modules():
        if hasattr(mod, "scale") and isinstance(getattr(mod, "scale"), torch.nn.Parameter) and hasattr(mod, "weight"):
            mod.weight.scale = mod.scale


def main():
    global SIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("inference")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--no-sims", action="store_true")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    SIMS = not args.no_sims
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.inference)
    install_kernels()
    import model as M
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    torch.set_default_dtype(torch.bfloat16)
    margs = model_args(M, args.snapshot, os.path.join(args.inference, "config.json"))
    started = time.time()
    model = M.Transformer(margs, tokenizer)
    load(model, args.snapshot)
    model.eval()
    print(f"loaded in {time.time() - started:.0f}s", flush=True)
    full = M.ParallelHead.forward
    M.ParallelHead.forward = lambda self, x, full_logits=True: full(self, x, True)

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = [{"name": n, "ids": tokenizer.encode(t, add_special_tokens=False)} for n, t in PROMPTS]
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        t0 = time.time()
        _, logits, _ = model(torch.tensor([ids]), 0)
        tf = logits[0].float().numpy()
        gen_rows, produced = [tf[-1].copy()], []
        last = int(tf[-1].argmax())
        pos = len(ids)
        for _ in range(args.steps):
            produced.append(last)
            _, logits, _ = model(torch.tensor([[last]]), pos)
            pos += 1
            row = logits[0, -1].float().numpy()
            gen_rows.append(row)
            last = int(row.argmax())
        vocab = tf.shape[-1]
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens gen={produced} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
