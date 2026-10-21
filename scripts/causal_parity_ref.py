"""Reference logits for a text checkpoint transformers reads as a causal LM
(Llama 3.x, Qwen3 / Qwen3-MoE, GLM-4.5, MiniMax-M2, ...), whole or a
`scripts/bench/shrink_checkpoint.py` miniature, for pie's parity harness.

    /root/venvs/vllm/bin/python scripts/causal_parity_ref.py <snapshot-dir> OUT \
        [--steps 16] [--probes probes.json] [--dtype float32|bfloat16] [--device cuda]

Writes `OUT/probes.json` (a stated file is used instead) and, per probe NAME:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

The chat probe is the tokenizer's own chat template. The pie side is
`crates/engine-cuda/tests/engine_cuda/a_family_is_read_against_its_reference.rs`;
`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` grades it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
]


def logits_of(model, ids, device):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids], device=device)).logits[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--dtype", default="float32", help="float32, or bfloat16 for the bf16 floor")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = [{"name": name, "ids": tokenizer.encode(text)} for name, text in PROMPTS]
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 17 times 23?"}], tokenize=True, add_generation_prompt=True
        )
        chat = chat["input_ids"] if hasattr(chat, "keys") else chat
        probes.append({"name": "chat", "ids": list(chat)})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"))

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        tf = logits_of(model, ids, args.device)
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        # greedy by re-reading the whole sequence: the reference's answer at
        # each step, independent of its own cache
        seq, gen, gen_rows = list(ids), [], [tf[-1]]
        for _ in range(args.steps):
            nxt = int(gen_rows[-1].argmax())
            gen.append(nxt)
            seq.append(nxt)
            gen_rows.append(logits_of(model, seq, args.device)[-1])
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(i) for i in tf.argmax(-1)], "gen": gen, "vocab": int(tf.shape[-1])},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"{name}: {len(ids)} tokens, gen={gen[:12]} -> {tokenizer.decode(gen)!r}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
