"""MiniMax-M3 reference logits, from transformers' own `minimax_m3_vl`, for a
miniature snapshot (`scripts/bench/shrink_checkpoint.py --repo
MiniMaxAI/MiniMax-M3 --layers 0-5 --experts 8`), read in float32 on the CPU
so pie's bf16 run is graded against the model rather than against another
bf16 rounding.

    /root/venvs/vllm/bin/python scripts/minimax_m3_parity_ref.py \
        <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Beside the short prompts it writes two long ones, past sixteen 128-key
blocks, so the indexer's selection drops blocks rather than keeping every
one. Writes `OUT/probes.json` (a stated file is used instead) and, per probe
NAME:

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
from transformers import AutoModelForImageTextToText, AutoTokenizer
import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl as reference

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
    ("chat", "]~b]user\nWhat is 17 times 23?[e~[\n]~b]ai\n"),
]

# Long probes: the reference's own source, and its docstrings joined into
# prose, cut at a length past sixteen blocks of 128.
LONG = [("long_code", 3000), ("long_prose", 2400)]


def long_texts():
    source = open(reference.__file__).read()
    prose = " ".join(
        line.strip().strip('"').strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("def ", "class ", "import ", "from ", "@", "return", "self."))
    )
    return {"long_code": source, "long_prose": prose}


def logits_of(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--long-steps", type=int, default=8)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--only", default=None, help="comma-separated probe names")
    ap.add_argument("--dtype", default="float32", help="float32, or bfloat16 for the bf16 floor")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(min(64, os.cpu_count() or 1))

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForImageTextToText.from_pretrained(
        args.snapshot, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    )
    model.eval()
    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = [{"name": name, "ids": tokenizer.encode(text)} for name, text in PROMPTS]
        texts = long_texts()
        for name, length in LONG:
            ids = tokenizer.encode(texts[name])
            assert len(ids) >= length, f"{name}: {len(ids)} tokens, {length} wanted"
            probes.append({"name": name, "ids": ids[:length]})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"))
    if args.only:
        keep = set(args.only.split(","))
        probes = [p for p in probes if p["name"] in keep]

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        steps = args.long_steps if len(ids) > 2048 else args.steps
        started = time.time()
        tf = logits_of(model, ids)
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        # greedy by re-reading the whole sequence: the reference's answer at
        # each step, independent of its own cache
        seq, gen, gen_rows = list(ids), [], [tf[-1]]
        for _ in range(steps):
            nxt = int(gen_rows[-1].argmax())
            gen.append(nxt)
            seq.append(nxt)
            gen_rows.append(logits_of(model, seq)[-1])
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": [int(i) for i in tf.argmax(-1)], "gen": gen, "vocab": int(tf.shape[-1])},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"{name}: {len(ids)} tokens, gen={gen[:12]} -> {tokenizer.decode(gen)!r}  ({time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
