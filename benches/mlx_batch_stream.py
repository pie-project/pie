#!/usr/bin/env python3
"""mlx-lm's concurrent decode, measured the way `llama_bench`'s Tier 3 measures pie's.

`macos-bench.md` §3 leaves the mlx throughput cell empty and says why: mlx-lm's
HTTP server has a five-deep backlog and an accept loop that starves under load,
so 16 requests fired at once complete 10. A number produced by an accept queue
is not a number about the engine. The document's own remedy is to drive mlx-lm
in-process, and this is that.

It does NOT use `mlx_lm.batch_generate`, which §3 suggests, and the reason is
the same one that makes the server unusable -- it would not be the same
measurement. `batch_generate` owns its sampler and its stopping rule, so the
rows would finish at different lengths and the total would be divided by a wall
clock covering a shape `llama_bench` never ran. What is held equal matters more
than which entry point is used, so this mirrors `mlx_single_stream.py` next
door instead: the model driven directly, one batched prompt cache, and the
host reading every step's tokens back before the next is issued.

Held equal to `PIE_BENCH_TPUT=n llama_bench <ckpt> 128 64`:

* **n identical sequences in ONE forward.** `llama_bench`'s fleet is n copies of
  the same prompt, so this batches the same ids n ways. Identical rows also make
  a wrong answer visible: every row must decode the same token, which is the
  check `llama_bench` makes and the one that fails on qwen3.5 (§7, §8).
* **The prefill is untimed.** `llama_bench` prefills the fleet one member at a
  time and outside the clock, because what is being measured is the decode
  fleet; folding a prefill in would report one number for two regimes.
* **The same readback.** Every step's tokens come back to the host before the
  next fire is issued. `tolist()` forces the eval and the copy; without it mlx
  would queue the whole decode and the number would be a submission rate.
* **The same warm state.** First trial discarded, as in `mlx_single_stream.py`.

What is not held equal: mlx-lm has no notion of independent sequences here.
These n rows share one cache tensor and one causal length, which is the shape
mlx-lm is fast at and NOT what a server does -- pie's fleet members carry their
own pages and can be at different lengths. So this is the friendly case for
mlx, and a fair reading treats it as an upper bound on what its batching gives
rather than as a like-for-like serving number.

Usage: mlx_batch_stream.py <checkpoint-dir> [--batch N] [--prefill N] [--decode N]
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# `llama_bench`'s counting prompt, as `mlx_single_stream.py` uses it and for the
# reason that file gives: repeated to length rather than transcribed.
COUNTING = [16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220,
            20, 11, 220, 21, 11, 220, 22, 11, 220, 23, 11]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--trials", type=int, default=2)
    args = ap.parse_args()

    model, _ = load(args.checkpoint)
    ids = (COUNTING * (args.prefill // len(COUNTING) + 1))[: args.prefill]
    n = args.batch

    best = 0.0
    agree = True
    for trial in range(args.trials + 1):
        cache = make_prompt_cache(model)
        # Untimed, as in `llama_bench`: the fleet's prefill is not the figure.
        logits = model(mx.array([ids] * n), cache=cache)
        mx.eval(logits)
        toks = mx.argmax(logits[:, -1], axis=-1)
        row = toks.tolist()

        t0 = time.perf_counter()
        for _ in range(args.decode):
            logits = model(toks[:, None], cache=cache)
            toks = mx.argmax(logits[:, -1], axis=-1)
            row = toks.tolist()  # the readback barrier, and the answer
        decode_s = time.perf_counter() - t0

        # Identical rows must decode identically. `llama_bench` refuses a
        # throughput number when they do not, and so does this.
        if len(set(row)) != 1:
            agree = False

        if trial == 0:  # warm-up: pipelines and first-touch residency
            continue
        best = max(best, n * args.decode / decode_s)

    if not agree:
        print(f"FAIL  {n} identical rows did not decode the same token; "
              f"no throughput number for this shape")
        raise SystemExit(1)
    print(f"tput   : {n} seqs x {args.decode} tok  =  {best:.1f} tok/s  "
          f"({best / n:.1f} tok/s per seq)")


if __name__ == "__main__":
    main()
