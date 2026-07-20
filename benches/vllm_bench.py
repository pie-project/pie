#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
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
    request_max_tokens,
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
                    "enable_prefix_caching": args.prefix_caching,
                    "disable_log_stats": False,
                    **llm_kwargs,
                },
                indent=2,
            )
        )

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.prefix_caching,
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
            "enable_prefix_caching": args.prefix_caching,
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


def run_streaming(args: argparse.Namespace):
    """tput with per-token client stamps via the AsyncLLM streaming engine.

    Vantage mirrors pie's --report-timing client: a closed loop of
    `concurrency` in-flight requests, TTFT stamped on the first token
    delivery after submit, inter-token gaps stamped per delivery event.
    """
    import asyncio

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    cpu_affinity = maybe_set_cpu_affinity(args, visible_cuda_devices(args.tp_size))
    n = args.num_requests
    prompts, prompt_counts = hf_chat_prompts_and_counts(
        args.model, args.system, make_prompts(args, n + args.warmup)
    )
    if args.concurrency == 0:
        max_num_seqs = max(1, args.num_requests)
    else:
        max_num_seqs = args.concurrency
    engine_kwargs = {}
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_mem_util,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            tensor_parallel_size=args.tp_size,
            max_model_len=args.max_model_len,
            enable_prefix_caching=args.prefix_caching,
            disable_log_stats=False,
            **engine_kwargs,
        )
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )

    def sampling_for(i: int) -> "SamplingParams":
        if not getattr(args, "mixed_phase", False):
            return sampling
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=request_max_tokens(args, i),
            ignore_eos=args.ignore_eos,
        )

    async def stream_one(
        request_id: str,
        prompt,
        prompt_count: int,
        params=None,
        measured_epoch_monotonic_ns: int | None = None,
    ) -> RequestResult:
        start = time.perf_counter()
        send_monotonic_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        client_send_s = (
            (send_monotonic_ns - measured_epoch_monotonic_ns)
            / 1_000_000_000.0
            if measured_epoch_monotonic_ns is not None
            else None
        )
        ttft_s = None
        last_tick = None
        gaps_us: list[int] = []
        token_arrival_s: list[float] = []
        token_arrival_monotonic_ns: list[int] = []
        n_tokens = 0
        try:
            async for out in engine.generate(
                prompt, params or sampling, request_id
            ):
                now = time.perf_counter()
                now_monotonic_ns = time.clock_gettime_ns(
                    time.CLOCK_MONOTONIC
                )
                new_total = len(out.outputs[0].token_ids)
                if new_total > n_tokens:
                    if measured_epoch_monotonic_ns is not None:
                        token_arrival_s.extend(
                            [
                                (
                                    now_monotonic_ns
                                    - measured_epoch_monotonic_ns
                                )
                                / 1_000_000_000.0
                            ]
                            * (new_total - n_tokens)
                        )
                        token_arrival_monotonic_ns.extend(
                            [now_monotonic_ns] * (new_total - n_tokens)
                        )
                    if ttft_s is None:
                        ttft_s = now - start
                    else:
                        gaps_us.append(int((now - last_tick) * 1e6))
                    last_tick = now
                    n_tokens = new_total
            returned = time.perf_counter()
            returned_monotonic_ns = time.clock_gettime_ns(
                time.CLOCK_MONOTONIC
            )
            return RequestResult(
                True,
                returned - start,
                n_tokens,
                prompt_count,
                ttft_s=ttft_s,
                intertoken_us=gaps_us or None,
                client_send_s=client_send_s,
                token_arrival_s=token_arrival_s or None,
                token_arrival_monotonic_ns=(
                    token_arrival_monotonic_ns or None
                ),
                client_return_s=(
                    (
                        returned_monotonic_ns
                        - measured_epoch_monotonic_ns
                    )
                    / 1_000_000_000.0
                    if measured_epoch_monotonic_ns is not None
                    else None
                ),
                process_id=request_id,
            )
        except Exception as e:  # noqa: BLE001
            return RequestResult(
                False,
                time.perf_counter() - start,
                n_tokens,
                prompt_count,
                error=f"{type(e).__name__}: {e}",
            )

    async def drive() -> tuple[list[RequestResult], float, float, int]:
        if args.warmup:
            for j, p in enumerate(prompts[: args.warmup]):
                await stream_one(f"warmup-{j}", p, prompt_counts[j])

        # Submit ALL requests at t=0 — concurrency is enforced engine-side by
        # max_num_seqs, exactly like pie's tput client (all launch_process at
        # once, engine admission = concurrency). A client-side semaphore here
        # would stop the TTFT clock during queueing that pie's clock counts,
        # making the comparison asymmetric.
        epoch_unix_s = time.time()
        epoch_monotonic_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        profiler_task = None
        if (
            args.cuda_profiler_capture
            and args.cuda_profiler_duration_s > 0
        ):
            async def capture_profiler_window() -> None:
                await asyncio.sleep(args.cuda_profiler_delay_s)
                cuda_profiler_start(True)
                try:
                    await asyncio.sleep(args.cuda_profiler_duration_s)
                finally:
                    cuda_profiler_stop(True)

            profiler_task = asyncio.create_task(capture_profiler_window())
        else:
            cuda_profiler_start(args.cuda_profiler_capture)
        start = time.perf_counter()
        try:
            results = list(
                await asyncio.gather(
                    *(
                        stream_one(
                            f"req-{i}",
                            prompts[args.warmup + i],
                            prompt_counts[args.warmup + i],
                            sampling_for(i),
                            epoch_monotonic_ns,
                        )
                        for i in range(n)
                    )
                )
            )
        finally:
            if profiler_task is None:
                cuda_profiler_stop(args.cuda_profiler_capture)
            else:
                if not profiler_task.done():
                    profiler_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await profiler_task
        return (
            results,
            time.perf_counter() - start,
            epoch_unix_s,
            epoch_monotonic_ns,
        )

    try:
        results, wall, epoch_unix_s, epoch_monotonic_ns = asyncio.run(drive())
    finally:
        engine.shutdown()

    summary = summarize(
        mode=args.mode,
        engine="vllm",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "streaming_client": True,
            "client timing epoch unix s": epoch_unix_s,
            "client timing epoch monotonic ns": epoch_monotonic_ns,
            "enable_prefix_caching": args.prefix_caching,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
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
        sp.add_argument(
            "--prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
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
        sp.add_argument(
            "--report-timing",
            action="store_true",
            help="Collect per-request TTFT and inter-token gap distributions. "
                 "Switches tput mode to the AsyncLLM streaming engine with a "
                 "closed-loop client (mirrors pie's client vantage: stamps on "
                 "token delivery, all requests submitted at t=0 when "
                 "num_requests == concurrency).",
        )
        sp.add_argument(
            "--report-arrivals",
            action="store_true",
            help="Collect absolute per-token client arrivals without enabling "
            "additional latency reporting.",
        )
    args = parser.parse_args()
    if (
        getattr(args, "report_timing", False)
        or getattr(args, "report_arrivals", False)
    ) and args.mode == "tput":
        summary, results = run_streaming(args)
    else:
        summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
