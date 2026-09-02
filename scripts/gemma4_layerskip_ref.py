"""Greedy tokens from mlx_lm's gemma4 truncated to its first N layers, for the
layerskip bisection: pie's `max_layers = N` fire reads the head at layer N,
and the two must agree (up to ties) at every N if the layers agree.

    python3 scripts/gemma4_layerskip_ref.py <snapshot-dir> --layers 1,2,6,12,23,24,30,42 [--steps 16]

Prints one line per (prompt, N): the greedy ids.
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    ("capital", "The capital of France is"),
    ("story", "Once upon a time, in a small village by the river,"),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--layers", default="1,2,6,12,23,24,30,42")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    model, tokenizer = load(args.snapshot)
    text = model.language_model.model
    all_layers = list(text.layers)
    all_prev = list(text.previous_kvs)
    out = {}
    for n in [int(x) for x in args.layers.split(",")]:
        text.layers = all_layers[:n]
        text.previous_kvs = all_prev[:n]
        for name, prompt in PROMPTS:
            ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
            cache = make_prompt_cache(model)
            logits = model(mx.array([ids]), cache=cache)[0, -1]
            gen = []
            for _ in range(args.steps):
                nxt = int(mx.argmax(logits).item())
                gen.append(nxt)
                logits = model(mx.array([[nxt]]), cache=cache)[0, -1]
            out[f"{name}@{n}"] = gen
            print(f"{name} layers={n}: {gen}")
    json.dump(out, open("/tmp/warmstream/e4b-layerskip-ref.json", "w"))


if __name__ == "__main__":
    main()
