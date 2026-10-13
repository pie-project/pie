"""Qwen3.5-MoE reference logits (Ornith-1.5, Qwen3.8 Max), from
`transformers`' own `qwen3_5_moe`, so a pie reading of the same snapshot can be
read against it.

    python3 scripts/qwen3_5_moe_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME, the three files every reference writes:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`)
against these.

A checkpoint with a tower (`Qwen3_5MoeForConditionalGeneration`, its trunk
under `model.language_model.`) is read by that class, its text positions the
same on every mrope axis; a text-only one (`Qwen3_5MoeForCausalLM`, its trunk
under `model.`) by the causal LM. Qwen3.8 Max's `output_gate_type: swish` is
the silu gate on the GDN's normed output that both already apply.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
    ("chat", "<|im_start|>user\nWhat is 17 times 23?<|im_end|>\n<|im_start|>assistant\n<think>\n"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    stated = json.load(open(os.path.join(args.snapshot, "config.json")))
    cls = Qwen3_5MoeForConditionalGeneration if "text_config" in stated else Qwen3_5MoeForCausalLM
    model = cls.from_pretrained(args.snapshot, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()

    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = []
        for name, text in PROMPTS:
            ids = tokenizer.encode(text, add_special_tokens=False)
            probes.append({"name": name, "ids": ids})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        input_ids = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
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
        print(f"      {tokenizer.decode(produced)!r}")

if __name__ == "__main__":
    main()
