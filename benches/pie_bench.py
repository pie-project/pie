#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import socket
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


BENCH_INFERLET = "text-completion-bench"
EMBEDDED_CLI_DRIVERS: set[str] = set()
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
            "max_batch_size": args.max_batch_size,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
            "kv_cache_dtype": args.kv_cache_dtype,
        }
    elif args.driver == "cuda_native":
        driver_options = {
            "max_batch_size": args.max_batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "max_num_kv_pages": args.kv_pages,
            "kv_cache_dtype": args.kv_cache_dtype,
        }
    elif args.driver == "portable":
        driver_options = {
            "max_batch_size": args.max_batch_size,
            "max_num_kv_pages": args.kv_pages,
            "n_gpu_layers": args.portable_n_gpu_layers,
            "kv_cache_dtype": args.kv_cache_dtype,
        }
    elif args.driver == "vllm":
        driver_options = {
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_batch_size,
        }
        if args.vllm_attention_backend:
            driver_options["attention_backend"] = args.vllm_attention_backend
    elif args.driver == "sglang":
        driver_options = {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
        }
        if args.sglang_attention_backend:
            driver_options["attention_backend"] = args.sglang_attention_backend
    else:
        driver_options = {}

    scheduler = "greedy" if args.mode == "latency" else "adaptive"
    cfg = Config(
        server=ServerConfig(
            host="127.0.0.1",
            port=0,
            max_concurrent_processes=args.concurrency if args.mode == "tput" else 1,
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
                scheduler=SchedulerConfig(
                    batch_policy=scheduler,
                    default_token_limit=args.default_token_limit,
                    default_endowment_pages=args.default_endowment_pages,
                    admission_oversubscription_factor=args.admission_oversubscription_factor,
                ),
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    tensor_parallel_size=args.tp_size,
                    options=driver_options,
                ),
            )
        ],
    )
    return cfg, {"driver": args.driver, "scheduler": scheduler, **driver_options}


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

    async def drain_stdout() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            server_lines.append(line.decode("utf-8", errors="replace"))
            del server_lines[:-200]

    try:
        assert proc.stdout is not None
        deadline = time.perf_counter() + args.server_startup_timeout
        while time.perf_counter() < deadline:
            line = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=max(0.1, deadline - time.perf_counter()),
            )
            if not line:
                raise RuntimeError(
                    "pie serve exited before startup completed:\n"
                    + "".join(startup_lines[-80:])
                )
            text = line.decode("utf-8", errors="replace")
            startup_lines.append(text)
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

        async def one(i: int) -> RequestResult:
            inp = {
                "prompt": prompts[i],
                "system": args.system,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "ignore_eos": args.ignore_eos,
            }
            start = time.perf_counter()
            try:
                proc = await client.launch_process(pkg, input=inp)
                while True:
                    ev, msg = await asyncio.wait_for(
                        proc.recv(), timeout=args.request_timeout
                    )
                    if ev == Event.Return:
                        obj = json.loads(msg)
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

        for i in range(args.warmup):
            await one(i)

        start_idx = args.warmup
        start = time.perf_counter()
        if args.mode == "latency":
            results = [await one(start_idx + i) for i in range(n)]
        else:
            sem = asyncio.Semaphore(args.concurrency)

            async def guarded(i: int) -> RequestResult:
                async with sem:
                    return await one(start_idx + i)

            results = await asyncio.gather(*(guarded(i) for i in range(n)))
        wall = time.perf_counter() - start

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
        sp.add_argument("--max-batch-size", type=int, default=512)
        sp.add_argument("--max-batch-tokens", type=int, default=10_240)
        sp.add_argument("--default-token-limit", type=int, default=200_000)
        sp.add_argument("--default-endowment-pages", type=int, default=64)
        sp.add_argument("--admission-oversubscription-factor", type=float, default=1000.0)
        sp.add_argument("--cpu-mem-budget", type=int, default=0)
        sp.add_argument("--kv-pages", type=int, default=2048)
        sp.add_argument("--kv-cache-dtype", choices=KV_CACHE_DTYPES, default="auto")
        sp.add_argument("--portable-n-gpu-layers", type=int, default=-1)
        sp.add_argument("--worker-threads", type=int, default=None)
        sp.add_argument("--vllm-attention-backend", default=None)
        sp.add_argument("--sglang-attention-backend", default=None)
        sp.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
        sp.add_argument("--server-startup-timeout", type=float, default=300.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary, results = asyncio.run(run(args))
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
