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

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_ref  # noqa: E402
import time

import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration

PROMPTS = [
    *parity_ref.PROMPTS,
    ("chat", "<|im_start|>user\nWhat is 17 times 23?<|im_end|>\n<|im_start|>assistant\n<think>\n"),
]

def main():
    ap = parity_ref.parser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    stated = json.load(open(os.path.join(args.snapshot, "config.json")))
    cls = Qwen3_5MoeForConditionalGeneration if "text_config" in stated else Qwen3_5MoeForCausalLM
    model = cls.from_pretrained(args.snapshot, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    for probe in parity_ref.probes(args, encode, PROMPTS):
        started = time.time()
        tf, rows, gen = parity_ref.greedy_cached(model, probe["ids"], args.steps, args.device)
        parity_ref.dump(args, probe["name"], probe["ids"], tf, rows, gen, started, tokenizer.decode)


if __name__ == "__main__":
    main()
