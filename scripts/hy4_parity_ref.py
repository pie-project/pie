"""Hy4-preview reference logits, from transformers' own `hy_v4`, for the
miniature snapshot (`scripts/bench/shrink_checkpoint.py --repo
tencent/Hy4-preview --experts 16`: layers 0-5, sixteen experts, an indexer
that keeps sixteen keys), read in float32 on the CPU so pie's bf16 run is
graded against the model rather than against another bf16 rounding.

transformers 5.17 turns the rope dims of both the attention and the indexer
as halves (`rotate_half`); the checkpoint, vLLM and sglang turn them in
interleaved pairs, and so does pie. The reference is patched to the pairs, and
its indexer's key norm to the eps vLLM and sglang use (1e-6, not
`rms_norm_eps`).

    /root/venvs/vllm/bin/python scripts/hy4_parity_ref.py <snapshot-dir> OUT \
        [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (a stated file is used instead) and, per probe NAME:

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
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import apply_rotary_pos_emb_interleave
from transformers.models.hy_v4 import modeling_hy_v4

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river, there lived an old fisherman who "
              "every morning rowed out before dawn and every evening came home with a story about the fish "
              "that got away."),
    ("chat", "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>"
             "reasoning_effort:no_think<｜hy_end:opensource｜><｜hy_start:opensource｜>user"
             "<｜hy_middle:opensource｜>What is 17 times 23?<｜hy_end:opensource｜>"
             "<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>"
             "</think:opensource>"),
]


def interleaved(q, k, cos, sin, unsqueeze_dim=1):
    return apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


def logits_of(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--dtype", default="float32", help="float32, or bfloat16 for the bf16 floor")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    modeling_hy_v4.apply_rotary_pos_emb = interleaved
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=getattr(torch, args.dtype))
    model.eval()
    for layer in model.model.layers:
        if layer.self_attn.indexer is not None:
            layer.self_attn.indexer.k_norm.eps = 1e-6
    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = [{"name": name, "ids": tokenizer.encode(text, add_special_tokens=False)} for name, text in PROMPTS]
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"))

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        tf = logits_of(model, ids)
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        # greedy by re-reading the whole sequence: the reference's answer at
        # each step, independent of its own cache
        seq, gen, gen_rows = list(ids), [], [tf[-1]]
        for _ in range(args.steps):
            nxt = int(gen_rows[-1].argmax())
            gen.append(nxt)
            seq.append(nxt)
            gen_rows.append(logits_of(model, seq)[-1])
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(i) for i in tf.argmax(-1)], "gen": gen, "vocab": int(tf.shape[-1])},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"{name}: {len(ids)} tokens, gen={gen[:12]} -> {tokenizer.decode(gen)!r}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
