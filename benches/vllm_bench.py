#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from common import (
    RequestResult,
    add_mode_subcommands,
    finish,
    hf_chat_prompts_and_counts,
    make_prompts,
    summarize,
)


def run(args: argparse.Namespace):
    from vllm import LLM, SamplingParams

    n = args.requests if args.mode == "latency" else args.num_requests
    prompts, prompt_counts = hf_chat_prompts_and_counts(
        args.model, args.system, make_prompts(args, n + args.warmup)
    )
    max_num_seqs = args.num_requests if args.mode == "tput" else 1
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    if args.warmup:
        llm.generate(prompts[: args.warmup], sampling)

    run_prompts = prompts[args.warmup:]
    run_prompt_counts = prompt_counts[args.warmup:]
    results: list[RequestResult] = []
    start = time.perf_counter()
    if args.mode == "latency":
        for p, prompt_count in zip(run_prompts, run_prompt_counts):
            req_start = time.perf_counter()
            outputs = llm.generate([p], sampling)
            req_wall = time.perf_counter() - req_start
            for out in outputs:
                latency = req_wall
                metrics = getattr(out, "metrics", None)
                if (
                    metrics
                    and metrics.finished_time is not None
                    and metrics.arrival_time is not None
                ):
                    latency = metrics.finished_time - metrics.arrival_time
                results.append(
                    RequestResult(
                        True, float(latency), len(out.outputs[0].token_ids), prompt_count
                    )
                )
    else:
        outputs = llm.generate(run_prompts, sampling)
        for out, prompt_count in zip(outputs, run_prompt_counts):
            results.append(
                RequestResult(True, 0.0, len(out.outputs[0].token_ids), prompt_count)
            )
    wall = time.perf_counter() - start

    summary = summarize(
        mode=args.mode,
        engine="vllm",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "enable_prefix_caching": False,
            "max_num_seqs": max_num_seqs,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    args = parser.parse_args()
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
