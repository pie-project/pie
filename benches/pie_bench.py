#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import os
import socket
import statistics
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
    finish,
    make_prompts,
    summarize,
)

SERVER_SDK = ROOT / "sdk" / "python-server" / "python"
if str(SERVER_SDK) not in sys.path:
    sys.path.insert(0, str(SERVER_SDK))


BENCH_INFERLET = "text-completion-bench"
EMBEDDED_CLI_DRIVERS: set[str] = {
    "cuda_native",
    "portable",
    "dummy",
    "vllm",
    "sglang",
    "tensorrt_llm",
    "dev",
}
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


def bench_inferlet_paths(name: str = BENCH_INFERLET) -> tuple[Path, Path, str]:
    inferlet_dir = ROOT / "inferlets" / name
    wasm_name = name.replace("-", "_")
    wasm = (
        inferlet_dir / "target" / "wasm32-wasip2" / "release"
        / f"{wasm_name}.wasm"
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
    if args.driver == "dev":
        driver_options = {
            "gpu_mem_utilization": args.gpu_mem_util,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
            "kv_cache_dtype": args.kv_cache_dtype,
        }
    elif args.driver == "cuda_native":
        driver_options = {
            "gpu_mem_utilization": args.gpu_mem_util,
            "memory_profile": args.memory_profile,
            "kv_cache_dtype": args.kv_cache_dtype,
        }
        if args.runtime_quant:
            driver_options["runtime_quant"] = args.runtime_quant
        if args.mxfp4_moe:
            driver_options["mxfp4_moe"] = args.mxfp4_moe
    elif args.driver == "portable":
        driver_options = {
            "max_forward_tokens": args.max_forward_tokens,
            "max_forward_requests": args.max_forward_requests,
            "total_pages": args.kv_pages,
            "kv_cache_dtype": args.kv_cache_dtype,
        }
    elif args.driver == "vllm":
        driver_options = {
            "gpu_memory_utilization": args.gpu_mem_util,
        }
        if getattr(args, "venv", None):
            driver_options["venv"] = args.venv
        if args.vllm_attention_backend:
            driver_options["attention_backend"] = args.vllm_attention_backend
    elif args.driver == "sglang":
        driver_options = {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": True,
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
    scheduler = args.batch_policy or ("greedy" if args.mode == "latency" else "adaptive")
    scheduler_kwargs = {
        "batch_policy": scheduler,
        "default_token_limit": args.default_token_limit,
        "default_endowment_pages": args.default_endowment_pages,
        "admission_oversubscription_factor": args.admission_oversubscription_factor,
    }
    if (
        args.speculation_depth is not None
        and "speculation_depth" in inspect.signature(SchedulerConfig).parameters
    ):
        scheduler_kwargs["speculation_depth"] = args.speculation_depth

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
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                scheduler=SchedulerConfig(**scheduler_kwargs),
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    tensor_parallel_size=args.tp_size,
                    ipc_profile=args.ipc_profile,
                    spin_budget_us=args.spin_budget_us,
                    options=driver_options,
                ),
            )
        ],
    )
    config_blob = {
        "driver": args.driver,
        "scheduler": scheduler,
        **driver_options,
    }
    if args.token_budget is not None:
        config_blob["token budget"] = args.token_budget
    elif args.auto_token_budget:
        config_blob["token budget"] = args.max_tokens + args.token_budget_prompt_margin
    if args.speculation_depth is not None:
        # Surface for the summary's "spec chain yield" derived stat —
        # yield = hits / (attempted × depth).
        config_blob["speculation depth"] = args.speculation_depth
    return cfg, config_blob


@asynccontextmanager
async def python_pie_client(args: argparse.Namespace):
    from pie.server import Server

    cfg, engine_config = build_config(args)
    async with Server(cfg) as server:
        yield await server.connect(), engine_config


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
            f"missing {pie_bin}; build with: cargo build -p pie-server --release "
            "--no-default-features --features driver-cuda"
        )

    proc = await asyncio.create_subprocess_exec(
        str(pie_bin),
        "serve",
        "--config",
        str(cfg_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    startup_lines: list[str] = []
    server_lines: list[str] = startup_lines
    drain_task: asyncio.Task[None] | None = None
    token: str | None = None
    server_log_file = None
    if server_log_path := os.environ.get("PIE_BENCH_SERVER_LOG"):
        path = Path(server_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        server_log_file = path.open("w", encoding="utf-8")

    def should_surface_server_line(txt: str) -> bool:
        return (
            txt.startswith("[fire ")
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
            marker = "internal token: "
            if marker in text:
                token = text.split(marker, 1)[1].strip()
                break
            if proc.returncode is not None:
                raise RuntimeError(
                    f"pie serve exited with {proc.returncode}:\n"
                    + "".join(startup_lines[-80:])
                )
        if token is None:
            raise TimeoutError(
                "timed out waiting for pie serve startup:\n" + "".join(startup_lines[-80:])
            )

        drain_task = asyncio.create_task(drain_stdout())
        client = PieClient(f"ws://127.0.0.1:{cfg.server.port}")
        await client.connect()
        await client.auth_by_token(token)
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

    n = args.requests if args.mode == "latency" else args.num_requests
    prompts = make_prompts(args, n + args.warmup)
    wasm, manifest, pkg = bench_inferlet_paths(args.bench_inferlet)
    runtime_events: dict[int, dict[str, float]] = {}

    async with pie_client(args) as (client, engine_config):
        await client.install_program(wasm, manifest, force_overwrite=True)

        first_output_text: list[str | None] = [None]

        async def launch_one(
            i: int,
            *,
            max_tokens: int | None = None,
            start_signal: bool = False,
        ):
            inp = {
                "prompt": prompts[i],
                "system": args.system,
                "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "ignore_eos": args.ignore_eos,
                "wasm_delay_us": args.wasm_delay_us,
                "decode_output": args.decode_output or args.dump_first_text,
                "start_signal": start_signal,
                "compact_output": not args.dump_first_text,
            }
            start = time.perf_counter()
            if args.profile_runtime_overhead:
                runtime_events[i] = {"launch_start": start}
            try:
                token_budget = args.token_budget
                if token_budget is None and args.auto_token_budget:
                    budget_tokens = args.max_tokens if max_tokens is None else max_tokens
                    token_budget = budget_tokens + args.token_budget_prompt_margin
                proc = await client.launch_process(pkg, input=inp, token_budget=token_budget)
                if args.profile_runtime_overhead:
                    runtime_events.setdefault(i, {})["launch_done"] = time.perf_counter()
                return i, start, proc
            except Exception as e:
                if args.profile_runtime_overhead:
                    runtime_events.setdefault(i, {})["launch_done"] = time.perf_counter()
                return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")

        async def wait_one(launched) -> RequestResult:
            if isinstance(launched, RequestResult):
                return launched
            i, start, proc = launched
            try:
                while True:
                    ev, msg = await asyncio.wait_for(
                        proc.recv(), timeout=args.request_timeout
                    )
                    if ev == Event.Return:
                        return_time = time.perf_counter()
                        if args.profile_runtime_overhead:
                            runtime_events.setdefault(i, {})["return"] = return_time
                        obj = json.loads(msg)
                        if i == 0 and first_output_text[0] is None:
                            first_output_text[0] = obj.get("text", "")
                        output_tokens = obj.get("num_output_tokens")
                        if output_tokens is None:
                            output_tokens = obj.get("generated_tokens", 0)
                        prompt_tokens = obj.get("num_prompt_tokens")
                        if prompt_tokens is None:
                            prompt_tokens = obj.get("prompt_tokens", 0)
                        return RequestResult(
                            True,
                            return_time - start,
                            int(output_tokens),
                            int(prompt_tokens),
                        )
                    if ev == Event.Error:
                        if args.profile_runtime_overhead:
                            runtime_events.setdefault(i, {})["return"] = time.perf_counter()
                        return RequestResult(False, time.perf_counter() - start, 0, error=str(msg))
            except Exception as e:
                if args.profile_runtime_overhead:
                    runtime_events.setdefault(i, {})["return"] = time.perf_counter()
                return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")

        async def one(i: int, *, max_tokens: int | None = None) -> RequestResult:
            return await wait_one(await launch_one(i, max_tokens=max_tokens))

        async def many(indices, *, max_tokens: int | None = None) -> list[RequestResult]:
            launched = await asyncio.gather(
                *(launch_one(i, max_tokens=max_tokens) for i in indices)
            )
            return await asyncio.gather(*(wait_one(item) for item in launched))

        async def wait_ready(launched):
            if isinstance(launched, RequestResult):
                return launched
            i, start, proc = launched
            try:
                while True:
                    ev, msg = await asyncio.wait_for(
                        proc.recv(), timeout=args.request_timeout
                    )
                    if ev == Event.Message and msg == "ready":
                        if args.profile_runtime_overhead:
                            runtime_events.setdefault(i, {})["ready"] = time.perf_counter()
                        return launched
                    if ev == Event.Return:
                        return RequestResult(
                            False,
                            time.perf_counter() - start,
                            0,
                            error="process returned before start barrier",
                        )
                    if ev == Event.Error:
                        return RequestResult(
                            False,
                            time.perf_counter() - start,
                            0,
                            error=str(msg),
                        )
            except Exception as e:
                return RequestResult(
                    False,
                    time.perf_counter() - start,
                    0,
                    error=f"{type(e).__name__}: {e}",
                )

        async def many_with_prelaunch_barrier(indices) -> tuple[float, float, list[RequestResult]]:
            indices = list(indices)
            wave_size = len(indices) if args.concurrency == 0 else max(1, args.concurrency)
            all_results: list[RequestResult] = []
            first_start = time.perf_counter()
            total_wall = 0.0

            for offset in range(0, len(indices), wave_size):
                wave_indices = indices[offset : offset + wave_size]
                launched = await asyncio.gather(
                    *(launch_one(i, start_signal=True) for i in wave_indices)
                )
                ready = await asyncio.gather(*(wait_ready(item) for item in launched))
                failures = [item for item in ready if isinstance(item, RequestResult)]
                if failures:
                    all_results.extend(failures)
                    continue

                start = time.perf_counter()
                if offset == 0:
                    first_start = start
                await asyncio.gather(*(item[2].signal("start") for item in ready))
                signaled = [(item[0], start, item[2]) for item in ready]
                all_results.extend(await asyncio.gather(*(wait_one(item) for item in signaled)))
                total_wall += time.perf_counter() - start

            return first_start, total_wall, all_results

        if args.warmup:
            warmup_max_tokens = args.warmup_max_tokens or args.max_tokens
            if args.mode == "tput":
                await many(range(args.warmup), max_tokens=warmup_max_tokens)
            else:
                for i in range(args.warmup):
                    await one(i, max_tokens=warmup_max_tokens)

        start_idx = args.warmup
        if args.profile_runtime_overhead:
            runtime_events.clear()
        if args.mode == "latency":
            start = time.perf_counter()
            results = [await one(start_idx + i) for i in range(n)]
            wall = time.perf_counter() - start
        elif args.prelaunch_barrier:
            start, wall, results = await many_with_prelaunch_barrier(
                range(start_idx, start_idx + n)
            )
        else:
            start = time.perf_counter()
            results = await many(range(start_idx, start_idx + n))
            wall = time.perf_counter() - start

        # Pull speculation counters out of the server's model status
        # so the bench output reflects what actually happened. Zero
        # on devices without speculation capability or with the
        # operator override disabled.
        try:
            ok, body = await client.query("model_status", "")
            if ok:
                model_status = json.loads(body)
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
                    ("default.cumulative_batch_latency_us", "cumulative batch latency us"),
                    ("default.avg_batch_latency_us", "avg batch latency us"),
                    ("default.last_batch_latency_us", "last batch latency us"),
                    ("default.bypass_hits", "bypass hits"),
                    ("default.chain_submits", "chain submits"),
                    ("default.chain_drops", "chain drops"),
                    ("default.execute_profile_calls", "execute profile calls"),
                    ("default.execute_profile_hits", "execute profile hits"),
                    ("default.execute_profile_misses", "execute profile misses"),
                    ("default.execute_profile_total_mean_us", "execute profile total mean us"),
                    ("default.execute_profile_prepare_mean_us", "execute profile prepare mean us"),
                    ("default.execute_profile_try_hit_mean_us", "execute profile try_hit mean us"),
                    ("default.execute_profile_hit_wait_mean_us", "execute profile hit_wait mean us"),
                    ("default.execute_profile_cold_prepare_mean_us", "execute profile cold_prepare mean us"),
                    ("default.execute_profile_pin_mean_us", "execute profile pin mean us"),
                    ("default.execute_profile_submit_wait_mean_us", "execute profile submit_wait mean us"),
                    ("default.execute_profile_postprocess_mean_us", "execute profile postprocess mean us"),
                    ("default.total_requests_processed", "total requests"),
                    ("default.max_forward_requests_observed", "max forward requests"),
                    ("default.batch_size_hist", "batch size hist"),
                ):
                    if key in model_status:
                        engine_config[label] = model_status[key]
        except Exception:  # noqa: BLE001
            # Stats are advisory — never break a bench on a failed query.
            pass

        if args.profile_runtime_overhead:
            add_runtime_overhead_profile(engine_config, runtime_events, start, wall)

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
            "bench_inferlet": args.bench_inferlet,
            **engine_config,
        },
    )
    return summary, results


def add_runtime_overhead_profile(
    engine_config: dict[str, Any],
    events: dict[int, dict[str, float]],
    run_start: float,
    wall_s: float,
) -> None:
    def percentile(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = (len(s) - 1) * q
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    launch_latencies = [
        e["launch_done"] - e["launch_start"]
        for e in events.values()
        if "launch_start" in e and "launch_done" in e
    ]
    launch_dones = [e["launch_done"] for e in events.values() if "launch_done" in e]
    ready_times = [e["ready"] for e in events.values() if "ready" in e]
    returns = [e["return"] for e in events.values() if "return" in e]
    driver_us = int(engine_config.get("cumulative batch latency us") or 0)
    driver_s = driver_us / 1_000_000.0

    if launch_latencies:
        engine_config["runtime launch ack mean ms"] = round(
            statistics.fmean(launch_latencies) * 1000.0, 3
        )
        engine_config["runtime launch ack p50 ms"] = round(
            percentile(launch_latencies, 0.50) * 1000.0, 3
        )
        engine_config["runtime launch ack p95 ms"] = round(
            percentile(launch_latencies, 0.95) * 1000.0, 3
        )
        engine_config["runtime launch ack max ms"] = round(
            max(launch_latencies) * 1000.0, 3
        )
    if launch_dones:
        if max(launch_dones) <= run_start:
            engine_config["runtime launch ack before start ms"] = round(
                (run_start - max(launch_dones)) * 1000.0, 3
            )
        else:
            engine_config["runtime first launch ack ms"] = round(
                (min(launch_dones) - run_start) * 1000.0, 3
            )
            engine_config["runtime all launch ack ms"] = round(
                (max(launch_dones) - run_start) * 1000.0, 3
            )
    if ready_times:
        if max(ready_times) <= run_start:
            engine_config["runtime ready before start ms"] = round(
                (run_start - max(ready_times)) * 1000.0, 3
            )
        else:
            engine_config["runtime all ready ms"] = round(
                (max(ready_times) - run_start) * 1000.0, 3
            )
    if returns:
        engine_config["runtime first return ms"] = round(
            (min(returns) - run_start) * 1000.0, 3
        )
        engine_config["runtime last return ms"] = round(
            (max(returns) - run_start) * 1000.0, 3
        )
    if driver_us:
        engine_config["runtime driver cumulative ms"] = round(driver_s * 1000.0, 3)
        engine_config["runtime wall minus driver ms"] = round(
            (wall_s - driver_s) * 1000.0, 3
        )
        if launch_dones and max(launch_dones) > run_start:
            engine_config["runtime non-driver after launch ms"] = round(
                (wall_s - (max(launch_dones) - run_start) - driver_s) * 1000.0,
                3,
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pie canonical latency/throughput benchmark")
    add_mode_subcommands(p)
    for sp in p._subparsers._group_actions[0].choices.values():
        sp.add_argument("--device", default="cuda:0")
        sp.add_argument("--driver", default="cuda_native",
                        choices=["dev", "cuda_native", "portable", "vllm", "sglang", "tensorrt_llm", "dummy"])
        sp.add_argument("--bench-inferlet", default=BENCH_INFERLET)
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
        sp.add_argument("--portable-n-gpu-layers", type=int, default=-1)
        sp.add_argument("--worker-threads", type=int, default=None)
        sp.add_argument("--token-budget", type=int, default=None)
        sp.add_argument("--auto-token-budget", action=argparse.BooleanOptionalAction, default=False)
        sp.add_argument("--token-budget-prompt-margin", type=int, default=64)
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
            "--decode-output",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Ask benchmark inferlets to detokenize generated tokens before returning. "
                 "Disabled by default so Pie throughput matches standalone baselines that "
                 "run with detokenize=False; implied by --dump-first-text.",
        )
        sp.add_argument(
            "--profile-runtime-overhead",
            action="store_true",
            help="Record benchmark-side launch/return timing and compare wall time "
                 "with cumulative scheduler batch latency from model_status.",
        )
        sp.add_argument(
            "--prelaunch-barrier",
            action="store_true",
            help="Throughput mode only: launch timed inferlets, wait for each to "
                 "finish prompt setup and report ready, then start the timer and "
                 "release them together via session signal. This excludes "
                 "request-as-process launch/tokenization overhead from the "
                 "measured generation window.",
        )
        sp.add_argument(
            "--batch-policy",
            default=None,
            choices=["adaptive", "eager", "greedy"],
            help="Override scheduler.batch_policy. Default: greedy (latency) "
                 "or adaptive (tput).",
        )
        sp.add_argument("--vllm-attention-backend", default=None)
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
        sp.add_argument(
            "--ipc-profile",
            default=None,
            choices=["latency", "balanced", "power"],
            help="Driver IPC wait profile. latency uses the polling in-process channel.",
        )
        sp.add_argument("--spin-budget-us", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary, results = asyncio.run(run(args))
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
