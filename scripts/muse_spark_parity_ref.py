"""Muse Spark reference logits for a text-only snapshot (the release's bf16
layout, or AMD's Quark MXFP4 one), from transformers' own `muse_spark`
(`MuseSparkTextModel`, the lm_head, then `output_multiplier` and the tanh
softcap as `MuseSparkForConditionalGeneration` applies them), so a pie
reading of the same snapshot can be read against it.

    python3 scripts/muse_spark_parity_ref.py <snapshot-dir> <transformers-src> OUT
        [--steps 8] [--probes probes.json] [--tag ref] [--threads 64]

`transformers-src` is a checkout of transformers' `src/` that carries
`models/muse_spark` (main, past 5.20). An MXFP4 snapshot's experts (`U8`
nibble pairs, the low one first, beside `weight_scale` e8m0 exponents a
1x32 block, gate / up / down one plane an expert) are written out to bf16,
which holds each value exactly, and stacked as the module holds them; the
activations stay bf16 (pie serves Quark's MXFP4 experts at W4, its
activations as they are).

Writes OUT/probes.json (or reads `--probes`) and per probe NAME:
OUT/NAME.TAG.tf.f32 [len(ids), vocab], OUT/NAME.TAG.gen.f32 [steps + 1,
vocab], OUT/NAME.TAG.json — as `scripts/dsv4_mini_parity_compare.py` reads.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the mountains,"),
    ("chat", "Q: What is the boiling point of water at sea level?\nA:"),
]

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
TRUNK = "model.language_model."


def mxfp4(codes, scales):
    lo, hi = (codes & 0xF).long(), (codes >> 4).long()
    v = torch.stack([E2M1[lo], E2M1[hi]], -1).reshape(codes.size(0), -1)
    e = scales.view(torch.uint8).float() if scales.dtype != torch.uint8 else scales.float()
    return (v * torch.pow(2.0, e - 127.0).repeat_interleave(32, 1)).bfloat16()


def read(snapshot, experts, inter, hidden):
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    raw = {}
    for fn in sorted(set(index.values())):
        with safe_open(os.path.join(snapshot, fn), "pt") as f:
            for k in f.keys():
                if k.startswith(TRUNK) or k == "lm_head.weight":
                    raw[k] = f.get_tensor(k)
    out = {}
    legs = {}
    for k, t in raw.items():
        if k.endswith("weight_scale"):
            continue
        m = re.match(r"(.*\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$", k)
        if m:
            v = mxfp4(t, raw[k + "_scale"]) if t.dtype == torch.uint8 else t
            legs.setdefault(m.group(1), {})[(int(m.group(2)), m.group(3))] = v
            continue
        out[k] = t
    for prefix, parts in legs.items():
        out[prefix + ".gate_up_proj"] = torch.stack([torch.cat([parts[(e, "gate")], parts[(e, "up")]], 0) for e in range(experts)])
        out[prefix + ".down_proj"] = torch.stack([parts[(e, "down")] for e in range(experts)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.src)
    from transformers import AutoTokenizer
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkConfig
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextModel

    cfg = MuseSparkConfig.from_pretrained(args.snapshot)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    started = time.time()
    model = MuseSparkTextModel(tc).to(torch.bfloat16)
    state = read(args.snapshot, tc.num_local_experts, tc.moe_intermediate_size, tc.moe_hidden_size)
    head = state.pop("lm_head.weight")
    trunk = {k[len(TRUNK):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(trunk, strict=False)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    if missing or unexpected:
        raise SystemExit(f"missing {missing}\nunexpected {unexpected}")
    for name, mod in model.named_modules():
        if name.endswith("mlp.gate"):
            mod.float()
            mod.e_score_correction_bias = mod.e_score_correction_bias.float()
    model.eval()
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    def logits(h):
        y = torch.nn.functional.linear(h, head).float() * tc.output_multiplier
        cap = tc.final_logit_softcapping
        return torch.tanh(y / cap) * cap

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = [{"name": n, "ids": tokenizer.encode(t, add_special_tokens=False)} for n, t in PROMPTS]
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    from transformers.cache_utils import DynamicCache

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        t0 = time.time()
        with torch.no_grad():
            cache = DynamicCache(config=tc)
            out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
            tf = logits(out.last_hidden_state[0]).numpy()
            rows, produced = [tf[-1].copy()], []
            last = int(tf[-1].argmax())
            for _ in range(args.steps):
                produced.append(last)
                out = model(input_ids=torch.tensor([[last]]), past_key_values=out.past_key_values, use_cache=True)
                row = logits(out.last_hidden_state[0, -1]).numpy()
                rows.append(row)
                last = int(row.argmax())
        vocab = tf.shape[-1]
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        np.stack(rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens gen={produced} ({time.time() - t0:.0f}s)", flush=True)
        print(f"      {tokenizer.decode(produced)!r}", flush=True)


if __name__ == "__main__":
    main()
