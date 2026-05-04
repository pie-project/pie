"""High-concurrency overload sweep for the workhorse text-completion inferlet.

Sweeps concurrency from 1 → N (e.g. 32, 64, 128) on a single driver to find
the saturation point. Reports per-step latency, p99, and aggregate tok/s.

Usage::

    uv run python benches/overload.py --driver native --concurrencies 1,4,16,32,64,128
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from pie_client import Event


ROOT = Path(__file__).resolve().parent.parent


def find_wasm(name: str) -> tuple[Path | None, Path | None]:
    wasm_name = name.replace("-", "_")
    inferlet_dir = ROOT / "inferlets" / name
    candidates = [
        inferlet_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
    ]
    wasm = next((p for p in candidates if p.exists()), None)
    return wasm, inferlet_dir / "Pie.toml"


def package_id(name: str) -> str | None:
    import tomllib
    manifest = ROOT / "inferlets" / name / "Pie.toml"
    if not manifest.exists():
        return None
    with open(manifest, "rb") as f:
        data = tomllib.load(f)
    pkg = data.get("package", {})
    return f"{pkg['name']}@{pkg['version']}"


def driver_options(driver: str, args) -> dict:
    if driver == "dev":
        return {
            "gpu_mem_utilization": args.gpu_mem_util,
            "max_batch_size": args.max_batch_size,
        }
    if driver == "sglang":
        opts = {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": not args.use_cuda_graphs,
            "attention_backend": getattr(args, "sglang_attention_backend", None) or "triton",
        }
        if getattr(args, "sglang_cuda_graph_max_bs", None):
            opts["cuda_graph_max_bs"] = args.sglang_cuda_graph_max_bs
        return opts
    if driver == "vllm":
        return {
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_batch_size,
            "enforce_eager": not args.use_cuda_graphs,
            "attention_backend": "FLASHINFER",
        }
    return {}


async def run_one(client, pkg, inputs, timeout):
    t0 = time.time()
    try:
        proc = await client.launch_process(pkg, input=inputs)
    except Exception as e:
        return False, time.time() - t0, str(e)[:60]
    try:
        while True:
            if time.time() - t0 > timeout:
                return False, time.time() - t0, "TIMEOUT"
            ev, msg = await asyncio.wait_for(proc.recv(), timeout=timeout)
            if ev == Event.Return:
                return True, time.time() - t0, ""
            if ev == Event.Error:
                return False, time.time() - t0, str(msg)[:60]
    except asyncio.TimeoutError:
        return False, time.time() - t0, "TIMEOUT"


async def main_async(args):
    from pie.server import Server
    from pie.config import (
        AuthConfig, Config, DriverConfig, ModelConfig, SchedulerConfig,
        ServerConfig, TelemetryConfig,
    )

    inferlet_name = args.inferlet
    pkg = package_id(inferlet_name)
    wasm, manifest = find_wasm(inferlet_name)
    if not wasm or not pkg:
        print(f"Error: no built WASM for {inferlet_name}")
        sys.exit(1)

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else [args.device]
    max_c = max(args.concurrencies)
    inputs = {"prompt": args.prompt, "max_tokens": args.max_tokens}

    cfg = Config(
        server=ServerConfig(port=0, max_concurrent_processes=max(max_c * 2, 32)),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                scheduler=SchedulerConfig(default_token_limit=args.default_token_limit),
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    options=driver_options(args.driver, args),
                ),
            ),
        ],
    )

    print(f"Driver: {args.driver}")
    print(f"Inferlet: {inferlet_name}")
    print(f"Workload: prompt={args.prompt!r} max_tokens={args.max_tokens}")
    print(f"Concurrencies: {args.concurrencies}")
    print()

    async with Server(cfg) as server:
        client = await server.connect()
        await client.install_program(wasm, manifest, force_overwrite=True)

        # Warm-up
        await run_one(client, pkg, inputs, 60.0)

        print(f"{'concurrency':>11} {'success':>8} {'fail':>5} "
              f"{'wall(s)':>8} {'p50(s)':>7} {'p99(s)':>7} {'req/s':>7} {'tok/s':>8}")
        print("-" * 70)

        for c in args.concurrencies:
            t_start = time.time()
            results = await asyncio.gather(*[
                run_one(client, pkg, inputs, args.timeout) for _ in range(c)
            ])
            wall = time.time() - t_start

            successes = [r for r in results if r[0]]
            failures = [r for r in results if not r[0]]
            times = sorted(r[1] for r in successes) or [0.0]
            n = len(times)
            p50 = times[n // 2] if n else 0.0
            p99 = times[max(0, int(n * 0.99) - 1)] if n else 0.0
            tps_req = len(successes) / wall if wall > 0 else 0
            tps_tok = (len(successes) * args.max_tokens) / wall if wall > 0 else 0

            print(f"{c:>11} {len(successes):>8} {len(failures):>5} "
                  f"{wall:>8.2f} {p50:>7.2f} {p99:>7.2f} {tps_req:>7.2f} {tps_tok:>8.1f}")
            if failures:
                fail_msg = next((r[2] for r in failures if r[2]), "")
                print(f"            failures: {fail_msg[:80]}")


def main():
    parser = argparse.ArgumentParser(description="High-concurrency overload sweep")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--driver", default="dev")
    parser.add_argument("--inferlet", default="text-completion")
    parser.add_argument("--prompt", default="Write a haiku about distributed systems.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrencies", default="1,4,16,32,64",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--default-token-limit", type=int, default=200_000)
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--sglang-attention-backend", default=None,
                        help="Override sglang's attention_backend (triton, flashinfer, ...)")
    parser.add_argument("--sglang-cuda-graph-max-bs", type=int, default=None,
                        help="Override sglang's cuda_graph_max_bs (default auto, ~24)")
    args = parser.parse_args()
    args.concurrencies = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
