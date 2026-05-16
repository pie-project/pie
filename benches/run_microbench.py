#!/usr/bin/env python3
"""Runs the wit-microbench inferlet to measure WIT call overhead.

Reuses pie_bench.py's server-boot machinery; only swaps the wasm + I/O
so we can launch the microbench program and parse its timing output."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pie_bench  # noqa: E402


def microbench_paths():
    root = pie_bench.ROOT
    inferlet_dir = root / "inferlets" / "wit-microbench"
    wasm = (
        inferlet_dir / "target" / "wasm32-wasip2" / "release"
        / "wit_microbench.wasm"
    )
    manifest = inferlet_dir / "Pie.toml"
    if not wasm.exists():
        raise FileNotFoundError(
            f"missing {wasm}; build with: cd {inferlet_dir} && "
            "cargo build --target wasm32-wasip2 --release"
        )
    pkg = tomllib.loads(manifest.read_text())["package"]
    return wasm, manifest, f"{pkg['name']}@{pkg['version']}"


async def run(args: argparse.Namespace):
    from pie_client import Event

    wasm, manifest, pkg = microbench_paths()

    async with pie_bench.pie_client(args) as (client, _engine_config):
        await client.install_program(wasm, manifest, force_overwrite=True)
        proc = await client.launch_process(
            pkg, input={"iterations": args.iterations}
        )
        while True:
            ev, msg = await asyncio.wait_for(proc.recv(), timeout=120)
            if ev == Event.Return:
                obj = json.loads(msg)
                print()
                print(f"{'Benchmark':<48} {'iters':>10} {'total µs':>12} {'per call':>14}")
                print("-" * 86)
                for b in obj["benches"]:
                    per = b["per_call_ns"]
                    if per >= 1_000_000:
                        per_str = f"{per / 1_000_000:.2f} ms"
                    elif per >= 1_000:
                        per_str = f"{per / 1_000:.2f} µs"
                    else:
                        per_str = f"{per} ns"
                    print(
                        f"{b['name']:<48} {b['iterations']:>10} "
                        f"{b['total_us']:>12} {per_str:>14}"
                    )
                return
            if ev == Event.Error:
                print(f"ERROR: {msg}")
                return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--driver", default="cuda_native")
    p.add_argument("--iterations", type=int, default=200_000)
    p.add_argument("--gpu-mem-util", type=float, default=0.30)
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--max-batch-tokens", type=int, default=2048)
    p.add_argument("--kv-pages", type=int, default=256)
    p.add_argument("--cpu-mem-budget", type=float, default=4.0)
    p.add_argument("--portable-n-gpu-layers", type=int, default=99)
    p.add_argument("--vllm-attention-backend", default=None)
    p.add_argument("--sglang-attention-backend", default=None)
    p.add_argument("--batch-policy", default=None)
    p.add_argument("--speculation-depth", type=int, default=None)
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--dump-first-text", action="store_true")
    p.add_argument("--worker-threads", type=int, default=None)
    p.add_argument("--default-token-limit", type=int, default=None)
    p.add_argument("--default-endowment-pages", type=int, default=64)
    p.add_argument("--admission-oversubscription-factor", type=float, default=2.0)
    args = p.parse_args()
    args.mode = "latency"  # pie_bench.build_config uses this
    args.num_requests = 1
    args.warmup = 0

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
