#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import socket
import sys
import time
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from common import (
    ROOT,
    RequestResult,
    add_mode_subcommands,
    cuda_profiler_start,
    cuda_profiler_stop,
    finish,
    hf_chat_token_ids_and_counts,
    make_prompts,
    maybe_set_cpu_affinity,
    request_max_tokens,
    resolve_local_model,
    summarize,
    visible_cuda_devices,
)

SERVER_SDK = ROOT / "sdk" / "python-server" / "python"
if str(SERVER_SDK) not in sys.path:
    sys.path.insert(0, str(SERVER_SDK))


EMBEDDED_CLI_DRIVERS: set[str] = {
    "dummy",
    "vllm",
    "sglang",
    "tensorrt_llm",
}


def hash_output_tokens(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()
KV_CACHE_DTYPES = [
    "auto",
    "bf16",
    "bfloat16",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8_per_token_head",
    "fp8_per_token_head",
    "fp4_e2m1",
    "nvfp4",
]


def reconstruct_token_arrivals(
    first_arrival_s: float,
    intertoken_us: list[int],
    output_tokens: int,
) -> list[float]:
    arrivals = [first_arrival_s]
    for gap_us in intertoken_us:
        arrivals.append(arrivals[-1] + gap_us / 1_000_000.0)
    if len(arrivals) != output_tokens:
        raise ValueError(
            f"token timing count is {len(arrivals)}, expected {output_tokens}"
        )
    return arrivals


def bench_inferlet_paths(inferlet_dir: str | None) -> tuple[Path, Path, str]:
    if not inferlet_dir:
        raise FileNotFoundError(
            "text-completion-bench is not part of the curated inferlets; pass "
            "--inferlet-dir or set PIE_BENCH_INFERLET_DIR"
        )
    inferlet_dir = Path(inferlet_dir).expanduser().resolve()
    wasm = (
        inferlet_dir / "target" / "wasm32-wasip2" / "release"
        / "text_completion_bench.wasm"
    )
    manifest = inferlet_dir / "Pie.toml"
    if not wasm.exists():
        raise FileNotFoundError(
            f"missing {wasm}; build with: cd {inferlet_dir} && "
            "cargo build --target wasm32-wasip2 --release"
        )
    pkg = tomllib.loads(manifest.read_text())["package"]
    return wasm, manifest, f"{pkg['name']}@{pkg['version']}"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def embedded_engine_identity() -> dict[str, str]:
    from pie import _engine

    engine_path = Path(_engine.__file__).resolve()
    source_suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".rs"}
    source_roots = (
        ROOT / "driver",
        ROOT / "interface",
        ROOT / "runtime",
        ROOT / "sdk" / "python-server" / "src",
    )
    newest_source = max(
        (
            path
            for root in source_roots
            for path in root.rglob("*")
            if path.is_file() and path.suffix in source_suffixes
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if engine_path.stat().st_mtime_ns < newest_source.stat().st_mtime_ns:
        raise RuntimeError(
            f"embedded engine {engine_path} is older than {newest_source}; "
            "rebuild with PIE_COMPILER_LAUNCHER=env CARGO_BUILD_JOBS=2 "
            "CMAKE_BUILD_PARALLEL_LEVEL=2 uv --project sdk/python-server sync "
            "--reinstall-package pie-server"
        )
    digest = hashlib.sha256()
    with engine_path.open("rb") as engine:
        for chunk in iter(lambda: engine.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "embedded engine": str(engine_path),
        "embedded engine sha256": digest.hexdigest(),
    }


def is_cumulative_status_key(key: str) -> bool:
    suffix = key.rsplit(".", 1)[-1]
    return (
        suffix.endswith("_sum")
        or suffix
        in {
            "batch_size_hist",
            "bypass_hits",
            "chain_drops",
            "chain_submits",
            "cold_hold_fires",
            "cumulative_batch_latency_us",
            "escape_fires",
            "readiness_miss",
            "spec_attempted",
            "spec_budget_skipped",
            "spec_dropped_orphan",
            "spec_hits",
            "spec_misses",
            "spec_need_pages",
            "spec_rule_skipped",
            "submit_ahead_fires",
            "total_batches",
            "total_requests_processed",
            "total_tokens_processed",
            "wave_fires",
        }
    )


def measured_average(
    status: dict[str, Any],
    numerator_key: str,
    denominator: int | float,
    output_key: str,
) -> None:
    numerator = status.get(numerator_key)
    if isinstance(numerator, (int, float)) and denominator > 0:
        status[output_key] = numerator / denominator


def build_config(args: argparse.Namespace):
    from pie.config import (
        AuthConfig,
        Config,
        DriverConfig,
        ModelConfig,
        RuntimeConfig,
        SchedulerConfig,
        ServerConfig,
        TelemetryConfig,
    )

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else [args.device]
    driver_options: dict[str, Any]
    if args.driver == "cuda_native":
        driver_options = {
            "gpu_mem_utilization": args.gpu_mem_util,
            "ready_timeout_s": float(args.server_startup_timeout),
        }
        if args.memory_profile != "auto":
            driver_options["memory_profile"] = args.memory_profile
        if args.kv_cache_dtype != "auto":
            driver_options["kv_cache_dtype"] = args.kv_cache_dtype
        if args.runtime_quant:
            driver_options["runtime_quant"] = args.runtime_quant
        if args.mxfp4_moe:
            driver_options["mxfp4_moe"] = args.mxfp4_moe
        if args.mtp_assistant_snapshot_dir:
            driver_options["mtp_assistant_snapshot_dir"] = (
                args.mtp_assistant_snapshot_dir
            )
        if args.mtp_num_drafts is not None:
            driver_options["mtp_num_drafts"] = args.mtp_num_drafts
        if args.enable_system_speculation:
            driver_options["enable_system_speculation"] = True
    elif args.driver == "vllm":
        driver_options = {
            "gpu_memory_utilization": args.gpu_mem_util,
        }
        if args.vllm_max_num_seqs is not None:
            driver_options["max_num_seqs"] = args.vllm_max_num_seqs
        if args.vllm_max_num_batched_tokens is not None:
            driver_options["max_num_batched_tokens"] = args.vllm_max_num_batched_tokens
        if args.vllm_max_model_len is not None:
            driver_options["max_model_len"] = args.vllm_max_model_len
        if getattr(args, "vllm_spec_ngram", False):
            driver_options["spec_ngram_enabled"] = True
            driver_options["spec_ngram_num_drafts"] = args.vllm_spec_ngram_num_drafts
            driver_options["spec_ngram_min_n"] = args.vllm_spec_ngram_min_n
            driver_options["spec_ngram_max_n"] = args.vllm_spec_ngram_max_n
        if getattr(args, "venv", None):
            driver_options["venv"] = args.venv
        if args.vllm_attention_backend:
            driver_options["attention_backend"] = args.vllm_attention_backend
    elif args.driver == "sglang":
        driver_options = {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": args.sglang_disable_cuda_graph,
            "disable_radix_cache": True,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
        }
        if getattr(args, "venv", None):
            driver_options["venv"] = args.venv
        if args.sglang_attention_backend:
            driver_options["attention_backend"] = args.sglang_attention_backend
    elif args.driver == "tensorrt_llm":
        driver_options = {}
        if getattr(args, "venv", None):
            driver_options["venv"] = args.venv
        if args.trtllm_backend:
            driver_options["backend"] = args.trtllm_backend
        if args.trtllm_attn_backend:
            driver_options["attn_backend"] = args.trtllm_attn_backend
        if args.trtllm_lookahead_tokens is not None:
            driver_options["lookahead_tokens"] = args.trtllm_lookahead_tokens
        if args.trtllm_execution_mode:
            driver_options["execution_mode"] = args.trtllm_execution_mode
        if args.trtllm_pyexecutor_max_tokens is not None:
            driver_options["pyexecutor_max_tokens"] = args.trtllm_pyexecutor_max_tokens
        if args.trtllm_pyexecutor_lookahead:
            driver_options["pyexecutor_lookahead"] = True
        if args.trtllm_pyexecutor_lookahead_min_batch_size is not None:
            driver_options["pyexecutor_lookahead_min_batch_size"] = (
                args.trtllm_pyexecutor_lookahead_min_batch_size
            )
        if args.trtllm_pyexecutor_direct_token_limit is not None:
            driver_options["pyexecutor_direct_token_limit"] = (
                args.trtllm_pyexecutor_direct_token_limit
            )
        if args.trtllm_pyexecutor_speculative_lookahead:
            driver_options["pyexecutor_speculative_lookahead"] = True
        if args.trtllm_max_seq_len is not None:
            driver_options["max_seq_len"] = args.trtllm_max_seq_len
        if args.trtllm_max_batch_size is not None:
            driver_options["max_batch_size"] = args.trtllm_max_batch_size
        if args.trtllm_max_num_tokens is not None:
            driver_options["max_num_tokens"] = args.trtllm_max_num_tokens
        if args.trtllm_kv_cache_free_gpu_memory_fraction is not None:
            driver_options["kv_cache_free_gpu_memory_fraction"] = (
                args.trtllm_kv_cache_free_gpu_memory_fraction
            )
    else:
        driver_options = {}

    # Concurrency 0 means "no admission cap" (all submitted inferlets run wasm
    # immediately; the inference scheduler still caps via max_forward_requests).
    if args.mode == "latency":
        max_concurrent_processes: int | None = 1
    elif args.concurrency == 0:
        max_concurrent_processes = None  # serializer drops field → unlimited
    else:
        max_concurrent_processes = args.concurrency
    requested_scheduler_kwargs = {
        "default_token_limit": args.default_token_limit,
        "default_endowment_pages": args.default_endowment_pages,
        "admission_oversubscription_factor": args.admission_oversubscription_factor,
    }
    scheduler_parameters = inspect.signature(SchedulerConfig).parameters
    scheduler_kwargs = {
        key: value
        for key, value in requested_scheduler_kwargs.items()
        if key in scheduler_parameters
    }
    if args.speculation_depth is not None and "speculation_depth" in scheduler_parameters:
        scheduler_kwargs["speculation_depth"] = args.speculation_depth

    resolved_model = resolve_local_model(args.model)
    cfg = Config(
        server=ServerConfig(
            host="127.0.0.1",
            port=0,
            verbose=True,
            max_concurrent_processes=max_concurrent_processes,
        ),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        runtime=RuntimeConfig(
            wasm_max_instances=max(4096, (args.num_requests + args.warmup) * 4),
            **({"worker_threads": args.worker_threads} if args.worker_threads else {}),
        ),
        model=ModelConfig(
            name="default",
            hf_repo=resolved_model,
            scheduler=SchedulerConfig(**scheduler_kwargs),
            driver=DriverConfig(
                type=args.driver,
                device=device,
                tensor_parallel_size=args.tp_size,
                options=driver_options,
            )
        ),
    )
    config_blob = {
        "driver": args.driver,
        "resolved model": resolved_model,
        **driver_options,
    }
    if args.speculation_depth is not None:
        # Surface for the summary's "spec chain yield" derived stat —
        # yield = hits / (attempted × depth).
        config_blob["speculation depth"] = args.speculation_depth
    if args.warmup_max_tokens is not None:
        config_blob["warmup max tokens"] = args.warmup_max_tokens
    return cfg, config_blob


@asynccontextmanager
async def python_pie_client(args: argparse.Namespace):
    from pie.server import Server

    cfg, engine_config = build_config(args)
    engine_identity = embedded_engine_identity()
    if url := os.environ.get("PIE_BENCH_SERVER_URL"):
        # Connect to an externally hosted server (see PIE_BENCH_SERVE_ONLY):
        # same embedded engine, its own process. vLLM is always benched with
        # its server in a separate process; this gives pie the same topology
        # so client-interpreter interference can be isolated and measured.
        from pie_client import PieClient

        client = PieClient(url)
        await client.connect()
        try:
            yield client, {**engine_config, **engine_identity}
        finally:
            await client.close()
        return
    async with Server(cfg) as server:
        if os.environ.get("PIE_BENCH_SERVE_ONLY") == "1":
            # Host-only mode: boot the server with this invocation's exact
            # config, announce the URL, and park until killed. A second
            # invocation with PIE_BENCH_SERVER_URL runs the workload.
            print(f"PIE_BENCH_SERVER_URL={server.url}", flush=True)
            await asyncio.Event().wait()
        yield await server.connect(), {**engine_config, **engine_identity}


@asynccontextmanager
async def cli_pie_client(args: argparse.Namespace):
    from pie_client import PieClient

    cfg, engine_config = build_config(args)
    cfg.server.port = find_free_port()
    cfg_path = ROOT / ".tmp" / "benches" / f"pie-{args.driver}-{cfg.server.port}.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg.to_toml())

    pie_bin = Path(args.pie_bin)
    if not pie_bin.exists():
        raise FileNotFoundError(
            f"missing {pie_bin}; build with: cargo build -p pie-worker --release "
            "--no-default-features --features driver-cuda"
        )

    proc = await asyncio.create_subprocess_exec(
        str(pie_bin),
        "--config",
        str(cfg_path),
        "serve",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    startup_lines: list[str] = []
    server_lines: list[str] = startup_lines
    drain_task: asyncio.Task[None] | None = None
    server_ready = False
    server_log_file = None
    if server_log_path := os.environ.get("PIE_BENCH_SERVER_LOG"):
        path = Path(server_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        server_log_file = path.open("w", encoding="utf-8")

    def should_surface_server_line(txt: str) -> bool:
        return (
            txt.startswith("[fire ")
            or txt.startswith("[pie-fire-timing] ")
            or txt.startswith("[sched-fire ")
            or txt.startswith("[outer-fire ")
            or txt.startswith("[sched-batch ")
            or txt.startswith("[pie-spec] ")
            or txt.startswith("[pie-driver-cuda] sampled tokens ")
            or "[pie-driver-cuda] memory planner:" in txt
            or "[pie-driver-cuda] forward_limits:" in txt
            or "[pie-driver-cuda] kv_cache:" in txt
            or "[pie-driver-cuda] CUDA graph upfront capture:" in txt
            or " xqa_decode=" in txt
            or " WARN " in txt
            or " ERROR " in txt
            or "Batch response count mismatch" in txt
            or "fire_batch failed" in txt
            or "exceeds workspace" in txt
            or "graph captured" in txt
        )

    async def drain_stdout() -> None:
        assert proc.stdout is not None
        import sys
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            txt = line.decode("utf-8", errors="replace")
            server_lines.append(txt)
            del server_lines[:-200]
            if server_log_file is not None:
                server_log_file.write(txt)
                server_log_file.flush()
            # Surface per-fire timing and server diagnostics the moment
            # they land; otherwise keep the server log buffered for
            # startup/failure messages.
            if should_surface_server_line(txt):
                sys.stderr.write(txt)
                sys.stderr.flush()

    try:
        assert proc.stdout is not None
        deadline = time.perf_counter() + args.server_startup_timeout
        while time.perf_counter() < deadline:
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=max(0.1, deadline - time.perf_counter()),
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    "timed out waiting for pie serve startup:\n"
                    + "".join(startup_lines[-80:])
                ) from exc
            if not line:
                raise RuntimeError(
                    "pie serve exited before startup completed:\n"
                    + "".join(startup_lines[-80:])
                )
            text = line.decode("utf-8", errors="replace")
            startup_lines.append(text)
            if server_log_file is not None:
                server_log_file.write(text)
                server_log_file.flush()
            if should_surface_server_line(text):
                sys.stderr.write(text)
                sys.stderr.flush()
            if "Server ready at ws://" in text:
                server_ready = True
                break
            if proc.returncode is not None:
                raise RuntimeError(
                    f"pie serve exited with {proc.returncode}:\n"
                    + "".join(startup_lines[-80:])
                )
        if not server_ready:
            raise TimeoutError(
                "timed out waiting for pie serve startup:\n" + "".join(startup_lines[-80:])
            )

        drain_task = asyncio.create_task(drain_stdout())
        client = PieClient(f"ws://127.0.0.1:{cfg.server.port}")
        await client.connect()
        try:
            yield client, {**engine_config, "pie_bin": str(pie_bin)}
        finally:
            await client.close()
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if drain_task is not None:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
        if server_log_file is not None:
            server_log_file.close()


def pie_client(args: argparse.Namespace):
    if args.driver in EMBEDDED_CLI_DRIVERS:
        return cli_pie_client(args)
    return python_pie_client(args)


async def run(args: argparse.Namespace):
    from pie_client import Event

    # Pin to GPU-local CPUs before the server spawns so the pie serve
    # subprocess inherits the affinity mask (mirrors vllm/sglang benches).
    cpu_affinity = maybe_set_cpu_affinity(args, visible_cuda_devices(args.tp_size))

    n = args.requests if args.mode == "latency" else args.num_requests
    prompts = make_prompts(args, n + args.warmup)
    prompt_token_ids: list[list[int]] | None = None
    if args.pretokenized_prompts:
        prompt_token_ids, _ = hf_chat_token_ids_and_counts(
            args.model, args.system, prompts
        )
    wasm, manifest, pkg = bench_inferlet_paths(args.inferlet_dir)

    async with pie_client(args) as (client, engine_config):
        await client.install_program(wasm, manifest, force_overwrite=True)

        first_output_text: list[str | None] = [None]
        output_token_ids_by_process: dict[str, list[int]] = {}
        measured_epoch: float | None = None
        measured_epoch_unix_s: float | None = None
        measured_epoch_monotonic_ns: int | None = None

        def common_input(max_tokens: int | None = None) -> dict[str, Any]:
            return {
                "system": args.system,
                "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "ignore_eos": args.ignore_eos,
                "wasm_delay_us": args.wasm_delay_us,
                "return_text": args.dump_first_text or args.dump_all_texts,
                "report_timing": args.report_timing,
                "report_arrivals": args.report_arrivals,
                "wait_for_start": args.defer_start,
                **(
                    {"system_speculation": args.system_speculation}
                    if args.system_speculation is not None
                    else {}
                ),
            }

        async def launch_one(i: int, *, max_tokens: int | None = None):
            if max_tokens is None and getattr(args, "mixed_phase", False):
                max_tokens = request_max_tokens(args, i)
            inp = {
                **common_input(max_tokens),
                "prompt": prompts[i],
            }
            if prompt_token_ids is not None:
                inp["prompt_tokens"] = prompt_token_ids[i]
            start = time.perf_counter()
            send_monotonic_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            client_send_s = (
                (send_monotonic_ns - measured_epoch_monotonic_ns)
                / 1_000_000_000.0
                if measured_epoch_monotonic_ns is not None
                and (args.report_timing or args.report_arrivals)
                else None
            )
            try:
                proc = await client.launch_process(pkg, input=inp)
                return i, start, proc, client_send_s
            except Exception as e:
                return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")

        async def wait_one(launched) -> RequestResult:
            if isinstance(launched, RequestResult):
                return launched
            i, start, proc, client_send_s = launched
            ttft_s: float | None = None
            first_arrival_s: float | None = None
            try:
                while True:
                    ev, msg = await asyncio.wait_for(
                        proc.recv(), timeout=args.request_timeout
                    )
                    if ev == Event.Message and str(msg) == "t0":
                        # Launch-inclusive first-token stamp (see inferlet's
                        # report_timing contract).
                        now = time.perf_counter()
                        now_monotonic_ns = time.clock_gettime_ns(
                            time.CLOCK_MONOTONIC
                        )
                        ttft_s = now - start
                        first_arrival_s = (
                            (
                                now_monotonic_ns
                                - measured_epoch_monotonic_ns
                            )
                            / 1_000_000_000.0
                            if measured_epoch_monotonic_ns is not None
                            and (args.report_timing or args.report_arrivals)
                            else None
                        )
                        continue
                    if ev == Event.Return:
                        returned = time.perf_counter()
                        returned_monotonic_ns = time.clock_gettime_ns(
                            time.CLOCK_MONOTONIC
                        )
                        obj = json.loads(msg)
                        if i == args.warmup and first_output_text[0] is None:
                            first_output_text[0] = obj.get("text", "")
                        output_tokens = int(obj["num_output_tokens"])
                        token_ids = [int(token) for token in obj.get("token_ids") or []]
                        if len(token_ids) != output_tokens:
                            raise ValueError(
                                f"output token count {len(token_ids)}, expected {output_tokens}"
                            )
                        output_token_ids_by_process[str(proc.process_id)] = token_ids
                        gaps = obj.get("intertoken_us") or []
                        arrivals = None
                        arrival_monotonic_ns = [
                            int(value)
                            for value in (obj.get("token_monotonic_ns") or [])
                        ]
                        if (
                            measured_epoch_monotonic_ns is not None
                            and arrival_monotonic_ns
                            and arrival_monotonic_ns[-1]
                            < measured_epoch_monotonic_ns
                        ):
                            # The guest clock may return measured-epoch-relative
                            # marks; normalize the persisted field to absolute
                            # CLOCK_MONOTONIC like the client stamps.
                            arrival_monotonic_ns = [
                                measured_epoch_monotonic_ns + value
                                for value in arrival_monotonic_ns
                            ]
                        if measured_epoch_monotonic_ns is not None:
                            if first_arrival_s is not None:
                                arrivals = reconstruct_token_arrivals(
                                    first_arrival_s, gaps, output_tokens
                                )
                            elif args.report_arrivals:
                                if len(arrival_monotonic_ns) != output_tokens:
                                    raise ValueError(
                                        "shared-clock token timing count is "
                                        f"{len(arrival_monotonic_ns)}, "
                                        f"expected {output_tokens}"
                                    )
                                arrivals = [
                                    (
                                        int(value)
                                        - measured_epoch_monotonic_ns
                                    )
                                    / 1_000_000_000.0
                                    for value in arrival_monotonic_ns
                                ]
                        return RequestResult(
                            True,
                            returned - start,
                            output_tokens,
                            int(obj["num_prompt_tokens"]),
                            ttft_s=ttft_s,
                            intertoken_us=gaps or None,
                            client_send_s=client_send_s,
                            token_arrival_s=arrivals,
                            token_arrival_monotonic_ns=(
                                arrival_monotonic_ns or None
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
                            process_id=str(proc.process_id),
                            prologue_us=obj.get("prologue_us") or None,
                            output_text=(
                                obj.get("text", "") if args.dump_all_texts else None
                            ),
                        )
                    if ev == Event.Error:
                        return RequestResult(False, time.perf_counter() - start, 0, error=str(msg))
            except Exception as e:
                return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")

        async def batch(indices, *, max_tokens: int | None = None) -> list[RequestResult]:
            indices = list(indices)
            inp = {
                **common_input(max_tokens),
                "prompt": prompts[indices[0]] if indices else args.prompt,
                "prompts": [prompts[i] for i in indices],
            }
            if args.concurrency and args.concurrency > 0:
                inp["batch_concurrency"] = args.concurrency
            if prompt_token_ids is not None:
                inp["prompt_tokens_batch"] = [prompt_token_ids[i] for i in indices]
            start = time.perf_counter()
            try:
                proc = await client.launch_process(pkg, input=inp)
                if args.defer_start:
                    while True:
                        ev, msg = await asyncio.wait_for(
                            proc.recv(), timeout=args.request_timeout
                        )
                        if ev == Event.Message and str(msg) == "ready":
                            break
                        if ev == Event.Return:
                            return [
                                RequestResult(
                                    False, 0.0, 0, error="returned before start"
                                )
                                for _ in indices
                            ]
                        if ev == Event.Error:
                            return [
                                RequestResult(False, 0.0, 0, error=str(msg))
                                for _ in indices
                            ]
                    start = time.perf_counter()
                    await proc.signal("start")
                while True:
                    ev, msg = await asyncio.wait_for(
                        proc.recv(), timeout=args.request_timeout
                    )
                    if ev == Event.Return:
                        obj = json.loads(msg)
                        if first_output_text[0] is None:
                            first_output_text[0] = obj.get("text", "")
                        req_out = obj.get("request_output_tokens") or []
                        req_prompt = obj.get("request_prompt_tokens") or []
                        elapsed = time.perf_counter() - start
                        if len(req_out) == len(indices) and len(req_prompt) == len(indices):
                            return [
                                RequestResult(True, elapsed, int(out), int(prompt))
                                for out, prompt in zip(req_out, req_prompt)
                            ]
                        total_out = int(obj["num_output_tokens"])
                        total_prompt = int(obj["num_prompt_tokens"])
                        per_out = total_out // max(1, len(indices))
                        per_prompt = total_prompt // max(1, len(indices))
                        return [
                            RequestResult(True, elapsed, per_out, per_prompt)
                            for _ in indices
                        ]
                    if ev == Event.Error:
                        return [
                            RequestResult(False, time.perf_counter() - start, 0, error=str(msg))
                            for _ in indices
                        ]
            except Exception as e:
                return [
                    RequestResult(
                        False,
                        time.perf_counter() - start,
                        0,
                        error=f"{type(e).__name__}: {e}",
                    )
                    for _ in indices
                ]

        async def one(i: int, *, max_tokens: int | None = None) -> RequestResult:
            return await wait_one(await launch_one(i, max_tokens=max_tokens))

        async def many(indices, *, max_tokens: int | None = None) -> list[RequestResult]:
            if args.single_process_batch and args.mode == "tput":
                return await batch(indices, max_tokens=max_tokens)
            launched = await asyncio.gather(
                *(launch_one(i, max_tokens=max_tokens) for i in indices)
            )
            if args.defer_start and args.mode == "tput":
                ready: list[tuple[int, Any]] = []
                failed: list[RequestResult] = []
                for item in launched:
                    if isinstance(item, RequestResult):
                        failed.append(item)
                        continue
                    i, _start, proc, _client_send_s = item
                    try:
                        while True:
                            ev, msg = await asyncio.wait_for(
                                proc.recv(), timeout=args.request_timeout
                            )
                            if ev == Event.Message and str(msg) == "ready":
                                ready.append((i, proc))
                                break
                            if ev == Event.Return:
                                failed.append(
                                    RequestResult(False, 0.0, 0, error="returned before start")
                                )
                                break
                            if ev == Event.Error:
                                failed.append(RequestResult(False, 0.0, 0, error=str(msg)))
                                break
                    except Exception as e:
                        failed.append(
                            RequestResult(False, 0.0, 0, error=f"{type(e).__name__}: {e}")
                        )
                start = time.perf_counter()
                await asyncio.gather(*(proc.signal("start") for _i, proc in ready))
                deferred = [
                    (
                        i,
                        start,
                        proc,
                        (
                            time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                            - measured_epoch_monotonic_ns
                        )
                        / 1_000_000_000.0
                        if measured_epoch_monotonic_ns is not None
                        else None,
                    )
                    for i, proc in ready
                ]
                return failed + await asyncio.gather(
                    *(wait_one(item) for item in deferred)
                )
            return await asyncio.gather(*(wait_one(item) for item in launched))

        if args.warmup:
            warmup_max_tokens = args.warmup_max_tokens or args.max_tokens
            if args.mode == "tput":
                await many(range(args.warmup), max_tokens=warmup_max_tokens)
            else:
                for i in range(args.warmup):
                    await one(i, max_tokens=warmup_max_tokens)

        start_idx = args.warmup
        # Snapshot cumulative stats after warmup so the final diff
        # reflects only the measured window, not warmup fires.
        pre_stats: dict[str, Any] = {}
        try:
            ok, body = await client.query("model_status", "")
            if ok:
                pre_stats = json.loads(body)
        except Exception:
            pass
        print("[pie-bench] measured-start", flush=True)
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
        if (
            args.report_timing
            or args.report_arrivals
            or args.report_wall_clock
        ):
            measured_epoch_unix_s = time.time()
            measured_epoch_monotonic_ns = time.clock_gettime_ns(
                time.CLOCK_MONOTONIC
            )
            measured_epoch = time.perf_counter()
            start = measured_epoch
        else:
            start = time.perf_counter()
        try:
            if args.mode == "latency":
                results = [await one(start_idx + i) for i in range(n)]
            else:
                results = await many(range(start_idx, start_idx + n))
        finally:
            wall = time.perf_counter() - start
            if profiler_task is None:
                cuda_profiler_stop(args.cuda_profiler_capture)
            else:
                if not profiler_task.done():
                    profiler_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await profiler_task
            print("[pie-bench] measured-end", flush=True)
        for result in results:
            if result.process_id is not None:
                token_ids = output_token_ids_by_process[result.process_id]
                result.output_token_sha256 = hash_output_tokens(token_ids)
                if args.dump_all_token_ids:
                    result.output_token_ids = token_ids
        if args.mode == "tput" and args.defer_start:
            measured = [r.latency_s for r in results if r.ok]
            if measured:
                wall = max(measured)

        # Pull speculation counters out of the server's model status
        # so the bench output reflects what actually happened. Zero
        # on devices without speculation capability or with the
        # operator override disabled.
        try:
            ok, body = await client.query("model_status", "")
            if ok:
                model_status_raw = json.loads(body)
                # Diff cumulative counters against pre-warmup snapshot
                # so the output only reflects the measured window.
                model_status: dict[str, Any] = {}
                for k, v in model_status_raw.items():
                    pre = pre_stats.get(k)
                    if (
                        is_cumulative_status_key(k)
                        and isinstance(v, (int, float))
                        and isinstance(pre, (int, float))
                    ):
                        model_status[k] = v - pre
                    elif (
                        is_cumulative_status_key(k)
                        and isinstance(v, list)
                        and isinstance(pre, list)
                        and len(v) == len(pre)
                    ):
                        model_status[k] = [a - b for a, b in zip(v, pre)]
                    else:
                        model_status[k] = v
                measured_batches = model_status.get("default.total_batches", 0)
                if isinstance(measured_batches, (int, float)):
                    measured_average(
                        model_status,
                        "default.cumulative_batch_latency_us",
                        measured_batches,
                        "default.avg_batch_latency_us",
                    )
                    for sum_key, average_key in (
                        ("default.fire.accumulate.accum_loop_us_sum",
                         "default.fire.accumulate.accum_loop_us"),
                        ("default.fire.pre_dispatch.fire_prepare_us_sum",
                         "default.fire.pre_dispatch.fire_prepare_us"),
                        ("default.fire.execute.total_us_sum",
                         "default.fire.execute.total_us"),
                        ("default.fire.execute.batch_build_us_sum",
                         "default.fire.execute.batch_build_us"),
                        ("default.fire.execute.driver_fire_us_sum",
                         "default.fire.execute.driver_fire_us"),
                        ("default.fire.post_dispatch.context_tick_us_sum",
                         "default.fire.post_dispatch.context_tick_us"),
                        ("default.fire.post_dispatch.stats_update_us_sum",
                         "default.fire.post_dispatch.stats_update_us"),
                        ("default.fire.quorum.inter_batch_bubble_us_sum",
                         "default.fire.quorum.inter_batch_bubble_us"),
                        ("default.fire.quorum.quorum_latency_us_sum",
                         "default.fire.quorum.quorum_latency_us"),
                    ):
                        measured_average(
                            model_status,
                            sum_key,
                            measured_batches,
                            average_key,
                        )
                    pre_batches = pre_stats.get("default.total_batches", 0)
                    inter_fire_samples = (
                        measured_batches
                        if isinstance(pre_batches, (int, float)) and pre_batches > 0
                        else max(measured_batches - 1, 0)
                    )
                    for sum_key, average_key in (
                        ("default.fire.inter_fire_us_sum",
                         "default.fire.inter_fire_us"),
                        ("default.fire.post_dispatch_to_fire_us_sum",
                         "default.fire.post_dispatch_to_fire_us"),
                        ("default.fire.recv_block_wait_us_sum",
                         "default.fire.recv_block_wait_us"),
                    ):
                        measured_average(
                            model_status,
                            sum_key,
                            inter_fire_samples,
                            average_key,
                        )
                cold_hold_fires = model_status.get(
                    "default.fire.quorum.cold_hold_fires", 0
                )
                if isinstance(cold_hold_fires, (int, float)):
                    measured_average(
                        model_status,
                        "default.fire.quorum.cold_hold_us_sum",
                        cold_hold_fires,
                        "default.fire.quorum.cold_hold_us",
                    )
                wave_fires = model_status.get("default.fire.quorum.wave_fires", 0)
                if isinstance(wave_fires, (int, float)):
                    measured_average(
                        model_status,
                        "default.fire.quorum.wave_active_sum",
                        wave_fires,
                        "default.fire.quorum.avg_active_pipelines_at_fire",
                    )
                    measured_average(
                        model_status,
                        "default.fire.quorum.wave_missing_sum",
                        wave_fires,
                        "default.fire.quorum.avg_missing_at_fire",
                    )
                for key, label in (
                    ("default.spec_attempted", "spec attempted"),
                    ("default.spec_hits", "spec hits"),
                    ("default.spec_misses", "spec misses"),
                    ("default.spec_rule_skipped", "spec rule skipped"),
                    ("default.spec_budget_skipped", "spec budget skipped"),
                    ("default.spec_dropped_orphan", "spec dropped orphan"),
                    ("default.spec_need_pages", "spec need pages"),
                    ("default.spec_chain_entries", "spec chain now"),
                    ("default.spec_chain_entries_high_water", "spec chain peak"),
                    ("default.spec_longest_chain", "spec longest chain"),
                    ("default.total_batches", "total batches"),
                    ("default.avg_batch_latency_us", "avg batch latency us"),
                    # Fire-domain probes. Mirror runtime/src/probe/fire.rs
                    # hierarchy. All-zero unless server built with
                    # --features profile-fire (or profile-hot-path / profile-all).
                    ("default.fire.inter_fire_us", "fire.inter_fire_us"),
                    ("default.fire.post_dispatch_to_fire_us", "fire.post_dispatch_to_fire_us"),
                    ("default.fire.accumulate.accum_loop_us", "fire.accumulate.accum_loop_us"),
                    ("default.fire.pre_dispatch.fire_prepare_us", "fire.pre_dispatch.fire_prepare_us"),
                    ("default.fire.execute.total_us", "fire.execute.total_us"),
                    ("default.fire.execute.batch_build_us", "fire.execute.batch_build_us"),
                    ("default.fire.execute.driver_fire_us", "fire.execute.driver_fire_us"),
                    (
                        "default.fire.quorum.avg_active_pipelines_at_fire",
                        "wave avg active pipelines",
                    ),
                    (
                        "default.fire.quorum.avg_missing_at_fire",
                        "wave avg missing pipelines",
                    ),
                    ("default.fire.quorum.wave_fires", "wave fires"),
                    ("default.cumulative_batch_latency_us", "cumulative_batch_latency_us"),
                    ("default.fire.post_dispatch.context_tick_us", "fire.post_dispatch.context_tick_us"),
                    ("default.fire.post_dispatch.stats_update_us", "fire.post_dispatch.stats_update_us"),
                    ("default.last_batch_latency_us", "last batch latency us"),
                    ("default.bypass_hits", "bypass hits"),
                    ("default.chain_submits", "chain submits"),
                    ("default.chain_ext_avg_wake_us", "chain ext avg wake us"),
                    ("default.chain_ext_avg_work_us", "chain ext avg work us"),
                    ("default.chain_drops", "chain drops"),
                    ("default.total_requests_processed", "total requests"),
                    ("default.max_forward_requests_observed", "max forward requests"),
                    ("default.batch_size_hist", "batch size hist"),
                ):
                    if key in model_status:
                        engine_config[label] = model_status[key]
                for key, value in model_status.items():
                    if (
                        "wave" in key
                        or "active_pipelines" in key
                        or "missing_at_fire" in key
                        or "straggler" in key
                        or "cold_hold" in key
                    ):
                        engine_config[key] = value
        except Exception:  # noqa: BLE001
            # Stats are advisory — never break a bench on a failed query.
            pass

    if args.dump_first_text and first_output_text[0] is not None:
        import hashlib
        sha = hashlib.sha256(first_output_text[0].encode()).hexdigest()[:16]
        print(f"\nFIRST REQUEST OUTPUT (sha256[:16]={sha}):")
        print(first_output_text[0])
        print(f"END OUTPUT (chars={len(first_output_text[0])})")

    summary = summarize(
        mode=args.mode,
        engine="pie",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "system": args.system,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
            "cuda profiler capture": args.cuda_profiler_capture,
            **(
                {
                    "client timing epoch unix s": measured_epoch_unix_s,
                    "client timing epoch monotonic ns": (
                        measured_epoch_monotonic_ns
                    ),
                }
                if (
                    args.report_timing
                    or args.report_arrivals
                    or args.report_wall_clock
                )
                else {}
            ),
            "cpu affinity": cpu_affinity,
            **engine_config,
        },
    )
    return summary, results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pie canonical latency/throughput benchmark")
    add_mode_subcommands(p)
    for sp in p._subparsers._group_actions[0].choices.values():
        sp.add_argument(
            "--inferlet-dir",
            default=os.environ.get("PIE_BENCH_INFERLET_DIR"),
            help="Path to a built text-completion-bench inferlet project "
                 "(or set PIE_BENCH_INFERLET_DIR).",
        )
        sp.add_argument("--device", default="cuda:0")
        sp.add_argument("--driver", default="cuda_native",
                        choices=["cuda_native", "vllm", "sglang", "tensorrt_llm", "dummy"])
        sp.add_argument("--default-token-limit", type=int, default=200_000)
        sp.add_argument("--default-endowment-pages", type=int, default=64)
        sp.add_argument("--admission-oversubscription-factor", type=float, default=4.0)
        sp.add_argument("--cpu-mem-budget", type=int, default=0)
        sp.add_argument(
            "--memory-profile",
            default="auto",
            choices=["auto", "latency", "balanced", "throughput", "capacity"],
        )
        sp.add_argument("--kv-pages", type=int, default=2048)
        sp.add_argument("--kv-cache-dtype", choices=KV_CACHE_DTYPES, default="auto")
        sp.add_argument("--max-forward-tokens", type=int, default=10240)
        sp.add_argument("--max-forward-requests", type=int, default=512)
        sp.add_argument("--runtime-quant", choices=["fp8", "int8"], default=None)
        sp.add_argument(
            "--mxfp4-moe",
            choices=["auto", "routed_dequant", "packed", "bf16", "dequant", "eager_bf16", "native"],
            default=None,
        )
        sp.add_argument("--worker-threads", type=int, default=None)
        sp.add_argument(
            "--speculation-depth",
            type=int,
            default=None,
            help="Per-ctx depth of pass-level speculative execution (0..=64). "
                 "0 disables speculation; 1 is piggyback (default). Forwards "
                 "to scheduler.speculation_depth in the generated toml.",
        )
        sp.add_argument(
            "--dump-first-text",
            action="store_true",
            help="Print the first request's full output text + its sha256 prefix. "
                 "Use to A/B-compare spec vs no-spec runs at temperature=0.",
        )
        sp.add_argument(
            "--dump-all-token-ids",
            action="store_true",
            help="Include every measured request's emitted token IDs in JSON output.",
        )
        sp.add_argument(
            "--dump-all-texts",
            action="store_true",
            help="Include every measured request's decoded output in JSON output.",
        )
        sp.add_argument(
            "--report-timing",
            action="store_true",
            help="Collect per-request TTFT (launch-inclusive, client-stamped on "
                 "the inferlet's t0 message) and inter-token gap distributions; "
                 "adds ttft/intertoken summaries to the output.",
        )
        sp.add_argument(
            "--report-arrivals",
            action="store_true",
            help="Collect per-token guest-drain timestamps from the shared "
            "host monotonic clock without sending a live first-token client "
            "message. Intended for non-perturbing occupancy reconstruction.",
        )
        sp.add_argument(
            "--report-wall-clock",
            action="store_true",
            help="Record only the measured wall's shared monotonic epoch; "
            "adds no per-request or per-token client timing.",
        )
        sp.add_argument(
            "--pretokenized-prompts",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Pre-render chat prompts to token IDs before timing and send raw tokens "
                 "to the benchmark inferlet.",
        )
        sp.add_argument(
            "--single-process-batch",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="For throughput mode, launch one benchmark inferlet that drives all "
                 "requests concurrently inside one WASM process.",
        )
        sp.add_argument(
            "--defer-start",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="For throughput mode, prelaunch inferlets and start timed generation "
                 "after each process reports ready.",
        )
        sp.add_argument(
            "--system-speculation",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Override system speculation. Omit to use the model default; "
                 "--no-system-speculation forces the no-spec baseline.",
        )
        sp.add_argument(
            "--mtp-assistant-snapshot-dir",
            default=None,
            help="cuda_native Gemma4 MTP assistant snapshot path used by .system_speculation(); "
                 "auto-discovered from the HF cache when omitted.",
        )
        sp.add_argument(
            "--mtp-num-drafts",
            type=int,
            default=None,
            help="Number of native MTP draft tokens per accepted token.",
        )
        sp.add_argument(
            "--enable-system-speculation",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="cuda_native deployment opt-in for system speculation (MTP). "
                 "Sets the driver config [model].enable_system_speculation; the "
                 "runtime drives the auto-drafter only when this is on. Default "
                 "off (latency-regime feature).",
        )
        sp.add_argument("--vllm-attention-backend", default=None)
        sp.add_argument("--vllm-max-num-seqs", type=int, default=None)
        sp.add_argument("--vllm-max-num-batched-tokens", type=int, default=None)
        sp.add_argument("--vllm-max-model-len", type=int, default=None)
        sp.add_argument("--vllm-spec-ngram", action=argparse.BooleanOptionalAction, default=False)
        sp.add_argument("--vllm-spec-ngram-num-drafts", type=int, default=4)
        sp.add_argument("--vllm-spec-ngram-min-n", type=int, default=2)
        sp.add_argument("--vllm-spec-ngram-max-n", type=int, default=4)
        sp.add_argument("--trtllm-backend", default=None)
        sp.add_argument("--trtllm-attn-backend", default=None)
        sp.add_argument("--trtllm-lookahead-tokens", type=int, default=None)
        sp.add_argument(
            "--trtllm-execution-mode",
            choices=["generate", "pyexecutor"],
            default=None,
        )
        sp.add_argument("--trtllm-pyexecutor-max-tokens", type=int, default=None)
        sp.add_argument("--trtllm-pyexecutor-lookahead", action="store_true")
        sp.add_argument(
            "--trtllm-pyexecutor-lookahead-min-batch-size",
            type=int,
            default=None,
        )
        sp.add_argument("--trtllm-pyexecutor-direct-token-limit", type=int, default=None)
        sp.add_argument(
            "--trtllm-pyexecutor-speculative-lookahead",
            action="store_true",
        )
        sp.add_argument("--trtllm-max-seq-len", type=int, default=None)
        sp.add_argument("--trtllm-max-batch-size", type=int, default=None)
        sp.add_argument("--trtllm-max-num-tokens", type=int, default=None)
        sp.add_argument(
            "--trtllm-kv-cache-free-gpu-memory-fraction",
            type=float,
            default=None,
        )
        sp.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
        sp.add_argument("--server-startup-timeout", type=float, default=300.0)
        sp.add_argument("--venv", default=None,
                        help="Path to a Python venv for subprocess drivers (vllm/sglang/tensorrt_llm/dev)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary, results = asyncio.run(run(args))
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
