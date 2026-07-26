"""Where does a constrained decode step actually spend its time?

A mask fill is easy to dismiss as overlapped with the forward pass, and on the
numbers it looks small. But a step is not only the fill: every accepted token
has to advance the parser, and in a serving batch that is one advance per
sequence per step. This replays a decode loop for both backends and charges each
of them for both halves, so the comparison is of the whole per-step cost rather
than the part that is easiest to measure.

No model runs here. That is deliberate: the forward pass is the same for both
backends, so including it only dilutes the difference being measured.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch", type=int, nargs="+", default=[256])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    import gpugrammar
    import xgrammar as xgr
    from transformers import AutoTokenizer

    from gpu_lr1.device_parser import DeviceGrammar

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")
    vocab_size = len(vocabulary)
    words = (vocab_size + 31) // 32
    schema = json.dumps(SCHEMA)

    # A document the grammar accepts, replayed so both backends walk the same
    # states. Sampling would make the two diverge and the comparison meaningless.
    document = '{"name": "Ada Lovelace", "age": 36, "active": true}'
    script = tokenizer.encode(document, add_special_tokens=False)[: arguments.steps]

    print(f"{len(script)} steps, vocabulary {vocab_size}")
    print(
        f"{'batch':>6} {'xgr fill':>9} {'xgr adv':>9} {'xgr tot':>9} "
        f"{'our fill':>9} {'our adv':>9} {'our tot':>9} {'ratio':>7}"
    )
    everything = []
    for batch_size in arguments.batch:
        everything.append(_one(batch_size, script, tokenizer, vocabulary, schema, xgr, gpugrammar, DeviceGrammar))

    if arguments.output:
        arguments.output.write_text(
            json.dumps(
                {"steps": len(script), "vocab_size": vocab_size, "batches": everything},
                indent=2,
            )
        )
        print(f"wrote {arguments.output}")


def _one(batch, script, tokenizer, vocabulary, schema, xgr, gpugrammar, DeviceGrammar):
    import time

    import torch

    vocab_size = len(vocabulary)
    words = (vocab_size + 31) // 32
    results = {}
    arguments = argparse.Namespace(batch=batch)

    # --- XGrammar -----------------------------------------------------------
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiled = xgr.GrammarCompiler(info, cache_enabled=True).compile_json_schema(schema)
    best = None
    for threads in (1, 2, 4, 8, 16):
        # A fresh set of matchers per sweep, so each thread count replays the
        # same script from the same states.
        matchers = [xgr.GrammarMatcher(compiled) for _ in range(arguments.batch)]
        batched = xgr.BatchGrammarMatcher(max_threads=threads)
        mask = torch.empty(
            xgr.get_bitmask_shape(arguments.batch, vocab_size),
            dtype=xgr.bitmask_dtype,
            pin_memory=True,
        )
        fill_seconds = 0.0
        advance_seconds = 0.0
        for token in script:
            start = time.perf_counter()
            batched.batch_fill_next_token_bitmask(matchers, mask)
            fill_seconds += time.perf_counter() - start
            start = time.perf_counter()
            for matcher in matchers:
                matcher.accept_token(token)
            advance_seconds += time.perf_counter() - start
        total = fill_seconds + advance_seconds
        if best is None or total < best[0]:
            best = (total, threads, fill_seconds, advance_seconds)
    total, threads, fill_seconds, advance_seconds = best
    per_step = total / len(script) * 1e6
    results["xgrammar"] = {
        "threads": threads,
        "fill_us": fill_seconds / len(script) * 1e6,
        "advance_us": advance_seconds / len(script) * 1e6,
        "total_us": per_step,
    }

    # --- gpugrammar ---------------------------------------------------------
    ours = gpugrammar.Compiler(vocabulary).compile_json_schema(schema)
    device = DeviceGrammar(ours)
    matchers = [ours.matcher(0) for _ in range(arguments.batch)]
    rows = device.new_batch(arguments.batch)
    host = torch.empty((arguments.batch, words), dtype=torch.int32, pin_memory=True)
    # Warm up: the first launch of a Triton kernel compiles it, which is
    # hundreds of milliseconds and would otherwise be charged to the first step
    # and dominate the average.
    for _ in range(3):
        rows.set_batch_configurations(
            {index: matcher.configurations() for index, matcher in enumerate(matchers)}
        )
        host.copy_(rows.fill_mask())
    torch.cuda.synchronize()

    fill_seconds = 0.0
    advance_seconds = 0.0
    for token in script:
        start = time.perf_counter()
        rows.set_batch_configurations(
            {index: matcher.configurations() for index, matcher in enumerate(matchers)}
        )
        host.copy_(rows.fill_mask())
        torch.cuda.synchronize()
        fill_seconds += time.perf_counter() - start
        start = time.perf_counter()
        for matcher in matchers:
            matcher.accept_token(token)
        advance_seconds += time.perf_counter() - start
    per_step = (fill_seconds + advance_seconds) / len(script) * 1e6
    results["gpugrammar"] = {
        "fill_us": fill_seconds / len(script) * 1e6,
        "advance_us": advance_seconds / len(script) * 1e6,
        "total_us": per_step,
    }

    ratio = results["xgrammar"]["total_us"] / results["gpugrammar"]["total_us"]
    print(
        f"{batch:>6} {results['xgrammar']['fill_us']:>9.0f} "
        f"{results['xgrammar']['advance_us']:>9.0f} {results['xgrammar']['total_us']:>9.0f} "
        f"{results['gpugrammar']['fill_us']:>9.0f} "
        f"{results['gpugrammar']['advance_us']:>9.0f} "
        f"{results['gpugrammar']['total_us']:>9.0f} {ratio:>6.2f}x"
    )
    return {"batch_size": batch, "backends": results, "ratio": ratio}


if __name__ == "__main__":
    main()
