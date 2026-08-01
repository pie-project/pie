#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any

from common import (
    ArrivalPacer,
    RequestResult,
    add_mode_subcommands,
    arrival_schedule,
    add_output_dump_args,
    cuda_profiler_start,
    cuda_profiler_stop,
    finish,
    gpu_clock_state,
    hash_output_tokens,
    hf_chat_prompts_and_counts,
    make_prompts,
    maybe_set_cpu_affinity,
    print_first_output,
    request_max_tokens,
    request_max_tokens_varies,
    run_timed_warmup_sync,
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
    if getattr(args, "num_gpu_blocks_override", 0):
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if getattr(args, "block_size", 0):
        llm_kwargs["block_size"] = args.block_size
    if getattr(args, "kv_cache_dtype", "auto") != "auto":
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
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

    def sampling_for(i: int) -> "SamplingParams":
        mt = request_max_tokens(args, i)
        if mt == args.max_tokens:
            return sampling
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=mt,
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
        def warmup_pass() -> None:
            llm.generate(prompts[: args.warmup], warmup_sampling)

        warmup_pass()
        # Optional duration-based extension of the warmup; see common.py.
        run_timed_warmup_sync(
            warmup_pass, args.warmup_seconds, label="vllm")

    spec_metrics_before = _vllm_spec_metrics(llm)
    clocks_at_start = gpu_clock_state()
    run_prompts = prompts[args.warmup:]
    run_prompt_counts = prompt_counts[args.warmup:]
    results: list[RequestResult] = []
    first_output_text: str | None = None

    def record(out: Any, req_wall: float, prompt_count: int) -> None:
        nonlocal first_output_text
        token_ids = [int(t) for t in out.outputs[0].token_ids]
        result = RequestResult(
            True,
            float(req_wall),
            len(token_ids),
            prompt_count,
        )
        result.output_token_sha256 = hash_output_tokens(token_ids)
        if getattr(args, "dump_all_token_ids", False):
            result.output_token_ids = token_ids
        if getattr(args, "dump_all_texts", False):
            result.output_text = out.outputs[0].text
        if first_output_text is None:
            first_output_text = out.outputs[0].text
        results.append(result)

    cuda_profiler_start(args.cuda_profiler_capture)
    start = time.perf_counter()
    try:
        if args.mode == "latency":
            for i, (p, prompt_count) in enumerate(
                zip(run_prompts, run_prompt_counts),
                start=args.warmup,
            ):
                req_start = time.perf_counter()
                outputs = llm.generate([p], sampling_for(i))
                req_wall = time.perf_counter() - req_start
                for out in outputs:
                    record(out, req_wall, prompt_count)
        else:
            measured_sampling = (
                [
                    sampling_for(args.warmup + i)
                    for i in range(len(run_prompts))
                ]
                if request_max_tokens_varies(args)
                else sampling
            )
            outputs = llm.generate(run_prompts, measured_sampling)
            for out, prompt_count in zip(outputs, run_prompt_counts):
                record(out, 0.0, prompt_count)
    finally:
        wall = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_capture)
        clocks_at_end = gpu_clock_state()
    spec_metrics_after = _vllm_spec_metrics(llm)
    print_first_output(args, first_output_text)

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
            "warmup seconds": args.warmup_seconds,
            "gpu clocks at start": clocks_at_start,
            "gpu clocks at end": clocks_at_end,
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
    if getattr(args, "num_gpu_blocks_override", 0):
        engine_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if getattr(args, "block_size", 0):
        engine_kwargs["block_size"] = args.block_size
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
        mt = request_max_tokens(args, i)
        if mt == args.max_tokens:
            return sampling
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=mt,
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

    pacer = ArrivalPacer(
        arrival_schedule(
            n, args.arrival_rate, args.arrival_process, args.arrival_seed
        )
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
        pacer.start()

        async def offer(i: int) -> RequestResult:
            await pacer.wait(i)
            return await stream_one(
                f"req-{i}",
                prompts[args.warmup + i],
                prompt_counts[args.warmup + i],
                sampling_for(args.warmup + i),
                epoch_monotonic_ns,
            )

        try:
            results = list(
                await asyncio.gather(*(offer(i) for i in range(n)))
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
            "arrival_rate": args.arrival_rate,
            "arrival_process": args.arrival_process,
            **pacer.stats(),
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        add_output_dump_args(sp)
        sp.add_argument("--attention-backend", default=None)
        sp.add_argument("--enforce-eager", action="store_true")
        sp.add_argument(
            "--num-gpu-blocks-override",
            type=int,
            default=0,
            help="Exact KV block count. Paired with --block-size 16 this is "
                 "the token-for-token match to pie's `total_pages` driver "
                 "option, so both engines can be sized by the same budget "
                 "instead of by a calibrated memory fraction.",
        )
        sp.add_argument(
            "--block-size",
            type=int,
            default=0,
            help="KV block size in tokens. 0 leaves vLLM's default; set 16 to "
                 "match pie's page size when using --num-gpu-blocks-override.",
        )
        sp.add_argument(
            "--prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        sp.add_argument("--max-num-batched-tokens", type=int, default=None)
        sp.add_argument(
            "--kv-cache-dtype",
            default="auto",
            help="vLLM KV cache dtype. DeepSeek-V4 needs 'fp8' because its "
                 "fp8_ds_mla cache layout refuses anything else.",
        )
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
    # vLLM refuses in-process data parallelism ("not supported for
    # single-process usage and may hang"), so a replica is a process. The
    # parent shards the request set, runs one child per replica pinned to
    # its own tp-size-wide slice of the devices, and merges the results —
    # which is what `data_parallel_offline.py` demonstrates, reduced to
    # what this benchmark needs.
    if args.dp_size > 1 and not os.environ.get("_VLLM_BENCH_REPLICA"):
        summary, results = run_data_parallel(args)
        finish(summary, results, args.json_out)
        return
    if (
        getattr(args, "report_timing", False)
        or getattr(args, "report_arrivals", False)
    ) and args.mode == "tput":
        summary, results = run_streaming(args)
    else:
        summary, results = run(args)
    finish(summary, results, args.json_out)


def run_data_parallel(args):
    """Fan the request set out over `dp_size` single-replica children.

    Wall clock is the max over children (they run concurrently), so
    throughput sums and latency percentiles pool — the same numbers a
    real DP deployment would report."""
    import subprocess
    import tempfile

    total = args.requests if args.mode == "latency" else args.num_requests
    per = [total // args.dp_size] * args.dp_size
    for i in range(total % args.dp_size):
        per[i] += 1

    procs, outs = [], []
    tmpdir = tempfile.mkdtemp(prefix="vllm-dp-")
    for replica, count in enumerate(per):
        if count == 0:
            continue
        devices = ",".join(
            str(replica * args.tp_size + i) for i in range(args.tp_size))
        out = os.path.join(tmpdir, f"replica{replica}.json")
        outs.append(out)
        # Forward the parent's own argv rather than reconstructing it from
        # the namespace: `store_true` flags have no `--no-` form, and the
        # namespace cannot tell them from `BooleanOptionalAction` ones.
        # Only the per-replica options are rewritten.
        rewritten = {"--dp-size", "--json-out", "--requests", "--num-requests"}
        forwarded, skip_next = [], False
        for token in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if token in rewritten:
                skip_next = True
                continue
            if any(token.startswith(f"{flag}=") for flag in rewritten):
                continue
            forwarded.append(token)
        argv = [sys.executable, os.path.abspath(__file__), *forwarded]
        argv += ["--json-out", out]
        argv += (["--requests", str(count)] if args.mode == "latency"
                 else ["--num-requests", str(count)])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices,
               "_VLLM_BENCH_REPLICA": str(replica)}
        procs.append(subprocess.Popen(argv, env=env))
    failures = [p.wait() for p in procs]
    if any(rc != 0 for rc in failures):
        raise RuntimeError(f"vllm data-parallel replica failed: {failures}")

    # Wall clock is the SLOWEST replica's own measured window, not the
    # parent's subprocess lifetime — the children spend minutes loading
    # weights before they start measuring, and counting that would report
    # a 35B model's throughput as a fraction of its real value. The
    # replicas run concurrently, so the max is the wall the merged set
    # would have seen.
    merged: list = []
    wall = 0.0
    for path in outs:
        with open(path) as fh:
            payload = json.load(fh)
        wall = max(wall, float(payload["summary"]["wall_s"]))
        for record in payload["requests"]:
            merged.append(RequestResult(**record))
    summary = summarize(
        mode=args.mode,
        engine="vllm",
        model=args.model,
        results=merged,
        wall_s=wall,
        config={
            "data_parallel_size": args.dp_size,
            "tensor_parallel_size": args.tp_size,
            "dp_replica_requests": per,
            "max_tokens": args.max_tokens,
        },
    )
    return summary, merged


if __name__ == "__main__":
    main()
