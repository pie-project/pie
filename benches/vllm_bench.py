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
    from vllm import LLM, SamplingParams

    cpu_affinity = maybe_set_cpu_affinity(args, visible_cuda_devices(args.tp_size))
    n = args.requests if args.mode == "latency" else args.num_requests
    prompts, prompt_counts = hf_chat_prompts_and_counts(
        args.model, args.system, make_prompts(args, n + args.warmup)
    )
    # Concurrency 0 means "no batch cap" — match pie's --concurrency 0 path.
    if args.mode == "latency":
        max_num_seqs = 1
    elif args.concurrency == 0:
        max_num_seqs = max(1, args.num_requests)
    else:
        max_num_seqs = args.concurrency
    llm_kwargs = {}
    if args.attention_backend:
        llm_kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        **llm_kwargs,
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
                results.append(
                    RequestResult(
                        True, float(req_wall), len(out.outputs[0].token_ids), prompt_count
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
            "attention_backend": args.attention_backend,
            "enforce_eager": args.enforce_eager,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
            "cpu affinity": cpu_affinity,
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument("--attention-backend", default=None)
        sp.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
