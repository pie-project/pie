"""DeepSeek-V4 reference logits for a release-layout snapshot (V4-Pro's or
V4-Flash's `mini-l5-e16`), from `transformers`' own `deepseek_v4`, so a pie
reading of the same snapshot can be read against it. Activations stay bf16 (or
f32): the release's FP8 activation quantization (`act_quant` before every fp8
or fp4 GEMM, and the KV and indexer quant-dequant) is not applied, so this is
the reference for the deployments that serve dequantized weights at bf16
activations.

    python3 scripts/dsv4_release_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]
        [--dtype float32]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME, the three files every reference writes:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.

`transformers` reads bf16 planes, and the release stores its routed experts
as MXFP4 codes (two e2m1 nibbles a byte, the low one first) beside `.scale`
e8m0 exponents, one a 32-wide block, and V4-Flash's release its dense planes as
fp8 beside `.scale` e8m0 exponents, one a 128-square block. Each is written out
to bf16, which holds every such value exactly, into `OUT/.bf16-view`, the configuration's
`quantization_config` and `expert_dtype` taken off. The reference runs in
float32 by default: its hyper-connection streams are f32 and its bf16 arm
hands them to bf16 projections unconverted.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, DeepseekV4ForCausalLM

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
]

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

def read_out(codes, scale):
    codes = codes.view(torch.uint8)
    pairs = torch.stack([codes & 0x0F, (codes >> 4) & 0x0F], dim=-1).reshape(codes.shape[0], -1)
    values = E2M1[pairs.long()].reshape(codes.shape[0], -1, 32)
    values = values * torch.pow(2.0, scale.view(torch.uint8).float() - 127.0).unsqueeze(-1)
    return values.reshape(codes.shape[0], -1).to(torch.bfloat16)

def read_out_fp8(codes, scale):
    """An fp8 plane under its `.scale` e8m0 exponents, one a square block
    (128 wide in V4): each code times two to its block's exponent, which
    bf16 holds exactly."""
    rows, cols = codes.shape
    group = max(-(-rows // scale.shape[0]), -(-cols // scale.shape[1]))
    factors = torch.pow(2.0, scale.view(torch.uint8).float() - 127.0)
    factors = factors.repeat_interleave(group, 0)[:rows].repeat_interleave(group, 1)[:, :cols]
    return (codes.float() * factors).to(torch.bfloat16)

def bf16_view(snapshot, view):
    if os.path.exists(os.path.join(view, "model.safetensors")):
        return
    os.makedirs(view, exist_ok=True)
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    shards = {}

    def get(name):
        shard = index[name]
        if shard not in shards:
            shards[shard] = safe_open(os.path.join(snapshot, shard), "pt")
        return shards[shard].get_tensor(name)

    state = {}
    for name in index:
        if name.endswith(".scale") and name[: -len(".scale")] + ".weight" in index:
            continue
        t = get(name)
        if ".experts." in name and t.dtype in (torch.int8, torch.uint8):
            t = read_out(t, get(name[: -len(".weight")] + ".scale"))
        elif t.dtype == torch.float8_e4m3fn:
            t = read_out_fp8(t, get(name[: -len(".weight")] + ".scale"))
        state[name] = t.contiguous()
    save_file(state, os.path.join(view, "model.safetensors"), metadata={"format": "pt"})
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config.pop("quantization_config", None)
    config.pop("expert_dtype", None)
    json.dump(config, open(os.path.join(view, "config.json"), "w"), indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    view = os.path.join(args.out, ".bf16-view")
    bf16_view(args.snapshot, view)
    model = DeepseekV4ForCausalLM.from_pretrained(view, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        probes = []
        for name, text in PROMPTS:
            probes.append({"name": name, "ids": bos + tokenizer.encode(text, add_special_tokens=False)})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=args.device), use_cache=True)
            tf = out.logits[0].float().cpu().numpy()
            past = out.past_key_values
            gen_rows = [tf[-1].copy()]
            produced = []
            last = int(tf[-1].argmax())
            for _ in range(args.steps):
                produced.append(last)
                step = model(
                    input_ids=torch.tensor([[last]], device=args.device),
                    past_key_values=past,
                    use_cache=True,
                )
                past = step.past_key_values
                row = step.logits[0, -1].float().cpu().numpy()
                gen_rows.append(row)
                last = int(row.argmax())
        vocab = tf.shape[-1]
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.tf.f32"))
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
            open(os.path.join(args.out, f"{name}.ref.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens, vocab {vocab}, gen={produced[:12]}  ({time.time() - started:.1f}s)")

if __name__ == "__main__":
    main()
