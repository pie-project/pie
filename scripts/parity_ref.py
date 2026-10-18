"""What every `scripts/*_parity_ref.py` shares: the probe battery, the greedy
walk and the dump `scripts/dsv4_mini_parity_compare.py` reads.

Per probe NAME a reference writes `OUT/NAME.<tag>.tf.f32` (float32
[len(ids), vocab], the teacher-forced logits), `OUT/NAME.<tag>.gen.f32`
([steps + 1, vocab], the last prompt row then each greedy step's) and
`OUT/NAME.<tag>.json` (`{"ids", "argmax", "gen", "vocab"}`); `OUT/probes.json`
holds the battery (`{"probes": [{"name", "ids"}]}`).
"""

import argparse
import json
import os
import time

import numpy as np

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
]


def parser(doc=None):
    """`snapshot OUT [--steps] [--probes] [--tag] [--device]`."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None, help="a probes.json to read instead of the battery")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--device", default="cuda")
    return ap


def probes(args, encode, prompts=PROMPTS, more=()):
    """The stated battery, or `prompts` encoded (plus `more`, ready
    `{"name", "ids"}` probes), written to `OUT/probes.json`."""
    os.makedirs(args.out, exist_ok=True)
    if args.probes:
        return json.load(open(args.probes))["probes"]
    battery = [{"name": name, "ids": list(encode(text))} for name, text in prompts] + list(more)
    json.dump({"probes": battery}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)
    return battery


def greedy(logits, ids, steps):
    """The teacher-forced rows of `ids` and `steps` greedy steps, each read
    again over the whole sequence (`logits(ids) -> [len, vocab]`), so no
    cache of the reference's own is trusted."""
    tf = np.asarray(logits(list(ids)), dtype=np.float32)
    rows, gen, seq = [tf[-1]], [], list(ids)
    for _ in range(steps):
        gen.append(int(rows[-1].argmax()))
        seq.append(gen[-1])
        rows.append(np.asarray(logits(seq), dtype=np.float32)[-1])
    return tf, rows, gen


def greedy_cached(model, ids, steps, device):
    """As `greedy`, over a transformers model's own KV cache."""
    import torch

    with torch.no_grad():
        out = model(input_ids=torch.tensor([list(ids)], device=device), use_cache=True)
        tf = out.logits[0].float().cpu().numpy()
        past, rows, gen = out.past_key_values, [tf[-1].copy()], []
        for _ in range(steps):
            gen.append(int(rows[-1].argmax()))
            step = model(input_ids=torch.tensor([[gen[-1]]], device=device), past_key_values=past, use_cache=True)
            past = step.past_key_values
            rows.append(step.logits[0, -1].float().cpu().numpy())
    return tf, rows, gen


def dump(args, name, ids, tf, rows, gen, started=None, decode=None, **extra):
    """Writes one probe's three files and says what it generated."""
    at = lambda suffix: os.path.join(args.out, f"{name}.{args.tag}.{suffix}")  # noqa: E731
    tf = np.asarray(tf, dtype=np.float32)
    tf.tofile(at("tf.f32"))
    np.stack(rows).astype(np.float32).tofile(at("gen.f32"))
    summary = {"ids": list(ids), "argmax": [int(a) for a in tf.argmax(-1)], "gen": gen, "vocab": int(tf.shape[-1])}
    json.dump({**summary, **extra}, open(at("json"), "w"))
    took = f"  ({time.time() - started:.1f}s)" if started else ""
    said = f" -> {decode(gen)!r}" if decode else ""
    print(f"  {name}: {len(ids)} tokens, gen={gen[:12]}{said}{took}", flush=True)
