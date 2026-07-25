"""Where does grammar work stop hiding behind the forward pass?

The case for moving constrained decoding onto the device is not that a CPU
matcher is slow in absolute terms. It is that its cost scales with the batch
while a decode step does not, so past some batch size the grammar stops fitting
in the shadow of the forward pass and starts adding to it. Below that point
XGrammar is free and nothing is worth doing; above it, every microsecond of
grammar work is a microsecond of end-to-end latency.

This measures both sides at the same batch sizes: the model's decode step, and
XGrammar filling one mask per sequence. It reports the ratio, so the crossover
is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _time_cuda(function, warmup: int = 3, iterations: int = 10) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _time_wall(function, warmup: int = 3, iterations: int = 10) -> float:
    for _ in range(warmup):
        function()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - start) * 1e6 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0, help="0 means xgrammar's default")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    import xgrammar as xgr
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    # A serving engine's decode step, not a transformers loop: the Python
    # overhead in the latter is an order of magnitude above the compute and
    # would hide exactly the effect being measured.
    engine = LLM(
        model=arguments.model,
        max_model_len=1024,
        gpu_memory_utilization=0.45,
        disable_log_stats=True,
    )
    vocab_size = engine.llm_engine.vllm_config.model_config.get_vocab_size()

    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = (
        xgr.GrammarCompiler(info, cache_enabled=True, max_threads=arguments.threads)
        if arguments.threads
        else xgr.GrammarCompiler(info, cache_enabled=True)
    )
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["name", "age", "active"],
            "additionalProperties": False,
        }
    )
    compiled = compiler.compile_json_schema(schema)

    print(f"model {arguments.model}, context {arguments.context}, vocab {vocab_size}")
    print(
        f"{'batch':>6} {'decode step':>13} {'xgrammar fill':>15} "
        f"{'threads':>8} {'share of step':>14}"
    )

    results = []
    steps = 64
    for batch in arguments.batches:
        prompts = [f"Write a long story about topic {index}." for index in range(batch)]
        params = SamplingParams(
            temperature=1.0, max_tokens=steps, min_tokens=steps, ignore_eos=True
        )
        engine.generate(prompts, params, use_tqdm=False)
        started = time.perf_counter()
        outputs = engine.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        generated = sum(len(output.outputs[0].token_ids) for output in outputs)
        decode_us = elapsed * 1e6 / steps
        del generated, outputs

        matchers = [xgr.GrammarMatcher(compiled) for _ in range(batch)]
        mask = torch.empty(
            xgr.get_bitmask_shape(batch, vocab_size),
            dtype=xgr.bitmask_dtype,
            pin_memory=True,
        )
        # Give XGrammar its best thread count rather than a default: measured
        # earlier, the wrong one costs it a factor of two to three.
        best = None
        for threads in (1, 2, 4, 8, 16):
            batched = xgr.BatchGrammarMatcher(max_threads=threads)

            def fill(batched=batched) -> None:
                batched.batch_fill_next_token_bitmask(matchers, mask)

            candidate = _time_wall(fill)
            if best is None or candidate < best[0]:
                best = (candidate, threads)
        fill_us, threads_used = best

        share = 100.0 * fill_us / decode_us
        print(
            f"{batch:>6} {decode_us:>11.0f}us {fill_us:>13.0f}us "
            f"{threads_used:>8} {share:>13.1f}%"
        )
        results.append(
            {
                "batch_size": batch,
                "decode_us": decode_us,
                "xgrammar_fill_us": fill_us,
                "xgrammar_threads": threads_used,
                "fill_share_percent": share,
            }
        )

    if arguments.output:
        arguments.output.write_text(
            json.dumps(
                {
                    "model": arguments.model,
                    "context": arguments.context,
                    "vocab_size": vocab_size,
                    "measurements": results,
                },
                indent=2,
            )
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
