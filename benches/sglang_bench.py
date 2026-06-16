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
    maybe_set_cpu_affinity,
    summarize,
    visible_cuda_devices,
)


def run(args: argparse.Namespace):
    import sglang as sgl

    cpu_affinity = maybe_set_cpu_affinity(args, visible_cuda_devices(args.tp_size))
    n = args.requests if args.mode == "latency" else args.num_requests
    prompts, prompt_counts = hf_chat_prompts_and_counts(
        args.model, args.system, make_prompts(args, n + args.warmup)
    )
    if args.mode == "latency":
        max_running_requests = 1
    elif args.concurrency == 0:
        max_running_requests = max(1, args.num_requests)
    else:
        max_running_requests = args.concurrency
    engine_kwargs = {
        "model_path": args.model,
        "mem_fraction_static": args.gpu_mem_util,
        "disable_cuda_graph": args.sglang_disable_cuda_graph,
        "disable_piecewise_cuda_graph": args.sglang_disable_piecewise_cuda_graph,
        "disable_radix_cache": True,
        "max_running_requests": max_running_requests,
        "tp_size": args.tp_size,
        "context_length": args.max_model_len,
    }
    if args.sglang_attention_backend:
        engine_kwargs["attention_backend"] = args.sglang_attention_backend
    if args.sglang_sampling_backend:
        engine_kwargs["sampling_backend"] = args.sglang_sampling_backend
    engine = sgl.Engine(**engine_kwargs)
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
    }
    try:
        if args.warmup:
            warmup_sampling = sampling
            if args.warmup_max_tokens is not None:
                warmup_sampling = {
                    **sampling,
                    "max_new_tokens": args.warmup_max_tokens,
                }
            engine.generate(prompts[: args.warmup], warmup_sampling)
        run_prompts = prompts[args.warmup:]
        run_prompt_counts = prompt_counts[args.warmup:]
        results: list[RequestResult] = []
        start = time.perf_counter()
        if args.mode == "latency":
            for p, prompt_count in zip(run_prompts, run_prompt_counts):
                req_start = time.perf_counter()
                outputs = engine.generate([p], sampling)
                req_wall = time.perf_counter() - req_start
                for o in outputs:
                    results.append(
                        RequestResult(
                            True,
                            req_wall,
                            int(o.get("meta_info", {}).get("completion_tokens", 0)),
                            prompt_count,
                        )
                    )
        else:
            outputs = engine.generate(run_prompts, sampling)
            results = [
                RequestResult(
                    True,
                    0.0,
                    int(o.get("meta_info", {}).get("completion_tokens", 0)),
                    prompt_count,
                )
                for o, prompt_count in zip(outputs, run_prompt_counts)
            ]
        wall = time.perf_counter() - start
    finally:
        engine.shutdown()

    summary = summarize(
        mode=args.mode,
        engine="sglang",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "disable_cuda_graph": args.sglang_disable_cuda_graph,
            "disable_piecewise_cuda_graph": args.sglang_disable_piecewise_cuda_graph,
            "disable_radix_cache": True,
            "attention_backend": args.sglang_attention_backend,
            "sampling_backend": args.sglang_sampling_backend,
            "max_running_requests": max_running_requests,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
            "cpu affinity": cpu_affinity,
            "warmup max tokens": args.warmup_max_tokens,
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="SGLang canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    args = parser.parse_args()
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
