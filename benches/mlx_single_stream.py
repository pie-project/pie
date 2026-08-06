#!/usr/bin/env python3
"""mlx-lm's single-stream prefill and decode, measured the way `llama_bench` measures pie's.

`mlx_bench.py` next door drives mlx-lm's HTTP server, which is the right
comparison for a serving shape and the wrong one for this question: it needs a
`pie model build` artifact beside the checkpoint to compare against, and on this
machine the four models under test are 65 GiB of checkpoint with 65 GiB of
artifact nowhere to put. This runs mlx-lm in-process against the SAME
`.safetensors` the driver loads, so the only thing that differs between the two
numbers is the implementation.

What is held equal, because a throughput number is only worth the shape it was
taken at:

* **The same prompt.** `llama_bench` prefills 128 tokens and decodes 64; so does
  this. The ids are its counting prompt repeated, which is a sentence in Qwen's
  vocabulary and noise in gemma's -- that does not matter for speed, and using
  the same ids means the KV lengths match exactly.
* **The same readback.** `llama_bench`'s headline decode figure is its
  `[token fed back]` one: the host reads each step's argmax and feeds it in, so
  the GPU cannot run ahead. `int(mx.argmax(...).item())` here is the same
  barrier. mlx-lm's own generator does the same thing.
* **The same warm state.** The first fire of anything on Metal pays for
  pipeline specialization and first-touch residency, so the first trial is
  discarded on both sides.

What is NOT held equal, and is not fixable here: mlx-lm evaluates lazily, so a
prefill's wall clock is only real if something forces it. `mx.eval` does. If a
future version fuses across the boundary this number becomes optimistic, in
mlx-lm's favour.

Usage: mlx_single_stream.py <checkpoint-dir> [--decode N] [--prefill N]
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# `llama_bench`'s counting prompt, in Qwen3.6's vocabulary. Repeated to length
# rather than transcribed, for the reason that file gives: a hand-typed list of
# ids is a chance to fix a typo into the measurement.
COUNTING = [16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220,
            20, 11, 220, 21, 11, 220, 22, 11, 220, 23, 11]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--trials", type=int, default=2)
    args = ap.parse_args()

    model, _ = load(args.checkpoint)
    ids = (COUNTING * (args.prefill // len(COUNTING) + 1))[: args.prefill]

    prefill_best = 0.0
    decode_best = 0.0
    for trial in range(args.trials + 1):
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        logits = model(mx.array([ids]), cache=cache)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0

        tok = int(mx.argmax(logits[0, -1]).item())
        t0 = time.perf_counter()
        for _ in range(args.decode):
            logits = model(mx.array([[tok]]), cache=cache)
            tok = int(mx.argmax(logits[0, -1]).item())
        decode_s = time.perf_counter() - t0

        if trial == 0:  # warm-up: pipelines and first-touch residency
            continue
        prefill_best = max(prefill_best, len(ids) / prefill_s)
        decode_best = max(decode_best, args.decode / decode_s)

    print(f"prefill: {len(ids)} tok  =  {prefill_best:.1f} tok/s")
    print(f"decode : {args.decode} tok  =  {decode_best:.1f} tok/s  "
          f"({1000.0 / decode_best:.2f} ms/tok)  [token fed back]")


if __name__ == "__main__":
    main()
