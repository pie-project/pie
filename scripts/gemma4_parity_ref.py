"""gemma4 reference logits, from `mlx_lm`'s own gemma4 (the model mlx-community's
conversions are made for), so a pie artifact of the same snapshot can be read
against an external implementation organ by organ.

    python3 scripts/gemma4_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes `OUT/probes.json` (`{"probes": [{"name", "ids"}]}`; a stated file is
used instead) and, per probe NAME:

    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]  teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab] logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"ids", "argmax", "gen", "vocab"}

`scripts/dsv4_mini_parity_compare.py OUT --a pie --b ref` reads the pie side
(`crates/engine-metal/tests/a_family_is_read_against_its_reference.rs`)
against these.
"""

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
    ("chat", "<start_of_turn>user\nWhat is 17 times 23?<end_of_turn>\n<start_of_turn>model\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--layers", type=int, default=None, help="truncate to the first N layers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tokenizer = load(args.snapshot)
    if args.layers is not None:
        text = model.language_model.model
        text.layers = text.layers[: args.layers]
        text.previous_kvs = text.previous_kvs[: args.layers]
    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        probes = []
        for name, text in PROMPTS:
            ids = tokenizer.encode(text)
            if not ids or ids[0] != tokenizer.bos_token_id:
                ids = [tokenizer.bos_token_id] + ids
            probes.append({"name": name, "ids": ids})
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"))

    for probe in probes:
        name, ids = probe["name"], probe["ids"]
        started = time.time()
        # Teacher-forced: one forward over the prompt, every position's logits.
        tf = model(mx.array([ids]))[0].astype(mx.float32)
        mx.eval(tf)
        np.asarray(tf).astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.tf.f32"))
        # Greedy: the prompt through a cache, then one token a step.
        cache = make_prompt_cache(model)
        out = model(mx.array([ids]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out)
        gen_rows = [np.asarray(out)]
        gen = []
        for _ in range(args.steps):
            nxt = int(mx.argmax(out).item())
            gen.append(nxt)
            out = model(mx.array([[nxt]]), cache=cache)[0, -1].astype(mx.float32)
            mx.eval(out)
            gen_rows.append(np.asarray(out))
        np.stack(gen_rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.ref.gen.f32"))
        argmax = [int(i) for i in np.asarray(tf).argmax(-1)]
        json.dump(
            {"ids": ids, "argmax": argmax, "gen": gen, "vocab": int(tf.shape[-1])},
            open(os.path.join(args.out, f"{name}.ref.json"), "w"),
        )
        print(f"{name}: {len(ids)} tokens, gen={gen[:12]} -> {tokenizer.decode(gen)!r}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
