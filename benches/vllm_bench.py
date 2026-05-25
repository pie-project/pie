#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import time

from common import (
    RequestResult,
    add_mode_subcommands,
    cuda_profiler_range,
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
        args.model,
        args.system,
        make_prompts(args, n + args.warmup),
        trust_remote_code=args.trust_remote_code,
    )
    # Concurrency 0 means "no batch cap" — match pie's --concurrency 0 path.
    # Latency mode defaults to one sequence, but keep an explicit override so
    # we can compare vLLM and Pie with the same cudagraph/capture shape.
    if args.max_num_seqs is not None:
        max_num_seqs = args.max_num_seqs
    elif args.mode == "latency":
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
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    if args.text_only_mm:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
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
        warmup_sampling = sampling
        if args.warmup_max_tokens is not None:
            warmup_sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.warmup_max_tokens,
                ignore_eos=args.ignore_eos,
            )
        llm.generate(prompts[: args.warmup], warmup_sampling)

    run_prompts = prompts[args.warmup:]
    run_prompt_counts = prompt_counts[args.warmup:]
    results: list[RequestResult] = []
    first_output_text: str | None = None
    first_output_tokens: list[int] | None = None
    with cuda_profiler_range(args.cuda_profiler_range):
        start = time.perf_counter()
        if args.mode == "latency":
            for p, prompt_count in zip(run_prompts, run_prompt_counts):
                req_start = time.perf_counter()
                outputs = llm.generate([p], sampling)
                req_wall = time.perf_counter() - req_start
                for out in outputs:
                    if first_output_text is None:
                        first = out.outputs[0]
                        first_output_text = first.text
                        first_output_tokens = list(first.token_ids)
                    results.append(
                        RequestResult(
                            True, float(req_wall), len(out.outputs[0].token_ids), prompt_count
                        )
                    )
        else:
            outputs = llm.generate(run_prompts, sampling)
            for out, prompt_count in zip(outputs, run_prompt_counts):
                if first_output_text is None:
                    first = out.outputs[0]
                    first_output_text = first.text
                    first_output_tokens = list(first.token_ids)
                results.append(
                    RequestResult(True, 0.0, len(out.outputs[0].token_ids), prompt_count)
                )
        wall = time.perf_counter() - start

    if args.dump_first_text and first_output_text is not None:
        sha = hashlib.sha256(first_output_text.encode()).hexdigest()[:16]
        print(f"\nFIRST REQUEST OUTPUT (sha256[:16]={sha}):")
        print(first_output_text)
        if first_output_tokens is not None:
            print(f"TOKEN IDS: {first_output_tokens}")
        print(f"END OUTPUT (chars={len(first_output_text)})")

    summary = summarize(
        mode=args.mode,
        engine="vllm",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "enable_prefix_caching": False,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "attention_backend": args.attention_backend,
            "enforce_eager": args.enforce_eager,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
            "text_only_mm": args.text_only_mm,
            "cpu affinity": cpu_affinity,
            "warmup max tokens": args.warmup_max_tokens,
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument("--attention-backend", default=None)
        sp.add_argument("--enforce-eager", action="store_true")
        sp.add_argument("--dump-first-text", action="store_true")
        sp.add_argument("--max-num-seqs", type=int, default=None)
        sp.add_argument("--max-num-batched-tokens", type=int, default=None)
        sp.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
        sp.add_argument(
            "--text-only-mm",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Set vLLM multimodal limits to zero for text-only generation.",
        )
    args = parser.parse_args()
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
