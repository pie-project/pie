#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from common import (
    RequestResult,
    add_mode_subcommands,
    cuda_profiler_start,
    cuda_profiler_stop,
    finish,
    hf_chat_prompts_and_counts,
    make_prompts,
    maybe_set_cpu_affinity,
    summarize,
    visible_cuda_devices,
)


def _vllm_metric_value(llm: Any, name: str) -> int:
    try:
        metrics = llm.get_metrics()
    except Exception:
        return 0
    total = 0
    for metric in metrics:
        if getattr(metric, "name", None) != name:
            continue
        value = getattr(metric, "value", None)
        if value is not None:
            total += int(value)
    return total


def _vllm_metric_vector(llm: Any, name: str) -> list[int]:
    try:
        metrics = llm.get_metrics()
    except Exception:
        return []
    values: list[int] = []
    for metric in metrics:
        if getattr(metric, "name", None) != name:
            continue
        metric_values = getattr(metric, "values", None)
        if metric_values is None:
            continue
        if not values:
            values = [0] * len(metric_values)
        for i, value in enumerate(metric_values):
            values[i] += int(value)
    return values


def _vllm_spec_metrics(llm: Any) -> dict[str, Any]:
    return {
        "drafts": _vllm_metric_value(llm, "vllm:spec_decode_num_drafts"),
        "draft_tokens": _vllm_metric_value(
            llm, "vllm:spec_decode_num_draft_tokens"
        ),
        "accepted_tokens": _vllm_metric_value(
            llm, "vllm:spec_decode_num_accepted_tokens"
        ),
        "accepted_per_position": _vllm_metric_vector(
            llm, "vllm:spec_decode_num_accepted_tokens_per_pos"
        ),
    }


def _vllm_spec_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    pos_after = after.get("accepted_per_position") or []
    pos_before = before.get("accepted_per_position") or []
    pos_len = max(len(pos_after), len(pos_before))
    accepted_per_position = [
        (pos_after[i] if i < len(pos_after) else 0)
        - (pos_before[i] if i < len(pos_before) else 0)
        for i in range(pos_len)
    ]
    drafts = int(after.get("drafts", 0)) - int(before.get("drafts", 0))
    draft_tokens = int(after.get("draft_tokens", 0)) - int(
        before.get("draft_tokens", 0)
    )
    accepted_tokens = int(after.get("accepted_tokens", 0)) - int(
        before.get("accepted_tokens", 0)
    )
    out: dict[str, Any] = {
        "vllm spec drafts": drafts,
        "vllm spec draft tokens": draft_tokens,
        "vllm spec accepted tokens": accepted_tokens,
    }
    if accepted_per_position:
        out["vllm spec accepted per position"] = accepted_per_position
    if draft_tokens > 0:
        out["vllm spec acceptance rate"] = accepted_tokens / draft_tokens
    if drafts > 0:
        out["vllm spec mean acceptance length"] = 1.0 + (
            accepted_tokens / drafts
        )
    return out


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
    speculative_config = None
    if args.speculative_config is not None:
        speculative_config = json.loads(args.speculative_config)
    if args.spec_method is not None or args.spec_tokens is not None:
        speculative_config = dict(speculative_config or {})
        if args.spec_method is not None:
            if "method" in speculative_config:
                raise ValueError("--spec-method conflicts with speculative_config.method")
            speculative_config["method"] = args.spec_method
        if args.spec_tokens is not None:
            if "num_speculative_tokens" in speculative_config:
                raise ValueError(
                    "--spec-tokens conflicts with speculative_config.num_speculative_tokens"
                )
            speculative_config["num_speculative_tokens"] = args.spec_tokens
    if args.mtp_assistant_model is not None:
        speculative_config = dict(speculative_config or {})
        if "model" in speculative_config:
            raise ValueError("--mtp-assistant-model conflicts with speculative_config.model")
        speculative_config["model"] = args.mtp_assistant_model
        if "method" in speculative_config:
            raise ValueError("--mtp-method conflicts with speculative_config.method")
        speculative_config["method"] = args.mtp_method
        if "num_speculative_tokens" in speculative_config:
            raise ValueError(
                "--mtp-num-drafts conflicts with "
                "speculative_config.num_speculative_tokens"
            )
        speculative_config["num_speculative_tokens"] = args.mtp_num_drafts
        if args.mtp_draft_tp_size is not None:
            if "draft_tensor_parallel_size" in speculative_config:
                raise ValueError(
                    "--mtp-draft-tp-size conflicts with "
                    "speculative_config.draft_tensor_parallel_size"
                )
            speculative_config["draft_tensor_parallel_size"] = args.mtp_draft_tp_size
    summary_speculative_config = dict(speculative_config) if speculative_config else None
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = dict(speculative_config)
    if args.print_llm_kwargs:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "gpu_memory_utilization": args.gpu_mem_util,
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "tensor_parallel_size": args.tp_size,
                    "max_model_len": args.max_model_len,
                    "enable_prefix_caching": False,
                    "disable_log_stats": False,
                    **llm_kwargs,
                },
                indent=2,
            )
        )

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        disable_log_stats=False,
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

    spec_metrics_before = _vllm_spec_metrics(llm)
    run_prompts = prompts[args.warmup:]
    run_prompt_counts = prompt_counts[args.warmup:]
    results: list[RequestResult] = []
    cuda_profiler_start(args.cuda_profiler_capture)
    start = time.perf_counter()
    try:
        if args.mode == "latency":
            for p, prompt_count in zip(run_prompts, run_prompt_counts):
                req_start = time.perf_counter()
                outputs = llm.generate([p], sampling)
                req_wall = time.perf_counter() - req_start
                for out in outputs:
                    results.append(
                        RequestResult(
                            True,
                            float(req_wall),
                            len(out.outputs[0].token_ids),
                            prompt_count,
                        )
                    )
        else:
            outputs = llm.generate(run_prompts, sampling)
            for out, prompt_count in zip(outputs, run_prompt_counts):
                results.append(
                    RequestResult(True, 0.0, len(out.outputs[0].token_ids), prompt_count)
                )
    finally:
        wall = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_capture)
    spec_metrics_after = _vllm_spec_metrics(llm)

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
            "speculative_config": summary_speculative_config,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
            "cuda profiler capture": args.cuda_profiler_capture,
            "cpu affinity": cpu_affinity,
            "warmup max tokens": args.warmup_max_tokens,
            **_vllm_spec_delta(spec_metrics_after, spec_metrics_before),
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument("--attention-backend", default=None)
        sp.add_argument("--enforce-eager", action="store_true")
        sp.add_argument("--max-num-batched-tokens", type=int, default=None)
        sp.add_argument(
            "--speculative-config",
            default=None,
            help="JSON object passed through to vLLM's speculative_config.",
        )
        sp.add_argument("--spec-method", default=None)
        sp.add_argument("--spec-tokens", type=int, default=None)
        sp.add_argument(
            "--mtp-assistant-model",
            default=None,
            help="Assistant checkpoint/model for vLLM Gemma4 MTP speculative decoding.",
        )
        sp.add_argument(
            "--mtp-method",
            default="gemma4_mtp",
            help="vLLM speculative method for the assistant. Default: gemma4_mtp.",
        )
        sp.add_argument(
            "--mtp-num-drafts",
            type=int,
            default=3,
            help="Number of vLLM speculative tokens. Match Pie --mtp-num-drafts.",
        )
        sp.add_argument(
            "--mtp-draft-tp-size",
            type=int,
            default=None,
            help="Optional draft_tensor_parallel_size for vLLM speculative_config.",
        )
        sp.add_argument(
            "--print-llm-kwargs",
            action="store_true",
            help="Print the vLLM LLM kwargs used by the benchmark before loading.",
        )
    args = parser.parse_args()
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
