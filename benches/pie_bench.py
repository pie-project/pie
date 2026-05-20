#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
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
    finish,
    make_prompts,
    summarize,
)

SERVER_SDK = ROOT / "sdk" / "python-server" / "python"
if str(SERVER_SDK) not in sys.path:
    sys.path.insert(0, str(SERVER_SDK))


BENCH_INFERLET = "text-completion-bench"
EMBEDDED_CLI_DRIVERS: set[str] = {"cuda_native", "portable", "dummy", "vllm", "sglang", "dev"}
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


def bench_inferlet_paths() -> tuple[Path, Path, str]:
    inferlet_dir = ROOT / "inferlets" / BENCH_INFERLET
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
    else:
        driver_options = {}

    max_concurrent_processes = args.concurrency if args.mode == "tput" else 1
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
    wasm, manifest, pkg = bench_inferlet_paths()

    async with pie_client(args) as (client, engine_config):
        await client.install_program(wasm, manifest, force_overwrite=True)

        first_output_text: list[str | None] = [None]

        async def launch_one(i: int, *, max_tokens: int | None = None):
            inp = {
                "prompt": prompts[i],
                "system": args.system,
                "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "ignore_eos": args.ignore_eos,
                "wasm_delay_us": args.wasm_delay_us,
            }
            start = time.perf_counter()
            try:
                token_budget = args.token_budget
                if token_budget is None and args.auto_token_budget:
                    budget_tokens = args.max_tokens if max_tokens is None else max_tokens
                    token_budget = budget_tokens + args.token_budget_prompt_margin
                proc = await client.launch_process(pkg, input=inp, token_budget=token_budget)
                return i, start, proc
            except Exception as e:
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
                        obj = json.loads(msg)
                        if i == 0 and first_output_text[0] is None:
                            first_output_text[0] = obj.get("text", "")
                        return RequestResult(
                            True,
                            time.perf_counter() - start,
                            int(obj["num_output_tokens"]),
                            int(obj["num_prompt_tokens"]),
                        )
                    if ev == Event.Error:
                        return RequestResult(False, time.perf_counter() - start, 0, error=str(msg))
            except Exception as e:
                return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")

        async def one(i: int, *, max_tokens: int | None = None) -> RequestResult:
            return await wait_one(await launch_one(i, max_tokens=max_tokens))

        async def many(indices, *, max_tokens: int | None = None) -> list[RequestResult]:
            launched = await asyncio.gather(
                *(launch_one(i, max_tokens=max_tokens) for i in indices)
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
        start = time.perf_counter()
        if args.mode == "latency":
            results = [await one(start_idx + i) for i in range(n)]
        else:
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
                    ("default.avg_batch_latency_us", "avg batch latency us"),
                    ("default.last_batch_latency_us", "last batch latency us"),
                    ("default.bypass_hits", "bypass hits"),
                    ("default.chain_submits", "chain submits"),
                    ("default.chain_drops", "chain drops"),
                    ("default.total_requests_processed", "total requests"),
                    ("default.max_forward_requests_observed", "max forward requests"),
                    ("default.batch_size_hist", "batch size hist"),
                ):
                    if key in model_status:
                        engine_config[label] = model_status[key]
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
            **engine_config,
        },
    )
    return summary, results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pie canonical latency/throughput benchmark")
    add_mode_subcommands(p)
    for sp in p._subparsers._group_actions[0].choices.values():
        sp.add_argument("--device", default="cuda:0")
        sp.add_argument("--driver", default="cuda_native",
                        choices=["dev", "cuda_native", "portable", "vllm", "sglang", "dummy"])
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
            "--batch-policy",
            default=None,
            choices=["adaptive", "eager", "greedy"],
            help="Override scheduler.batch_policy. Default: greedy (latency) "
                 "or adaptive (tput).",
        )
        sp.add_argument("--vllm-attention-backend", default=None)
        sp.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
        sp.add_argument("--server-startup-timeout", type=float, default=300.0)
        sp.add_argument("--venv", default=None,
                        help="Path to a Python venv for subprocess drivers (vllm/sglang/dev)")
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
