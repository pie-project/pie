"""Comprehensive cross-driver benchmark.

Runs a curated set of inferlets against each available driver at
multiple concurrency levels and records throughput, latency, and
success/fail counts.

Usage::

    uv run python benches/all_drivers.py --model Qwen/Qwen3-1.7B \
      --device cuda:0 --drivers native,sglang \
      --concurrencies 1,4,8 --output-dir /tmp/bench-runs

Output: per-driver markdown + a combined comparison table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from pie_client import Event


# ---------------------------------------------------------------------------
# Inferlet workload table
# ---------------------------------------------------------------------------
# Each entry: (inferlet_name, package_id, input_dict, time_budget_seconds)
# All workloads are sized to be moderate so we can sweep concurrency without
# the per-process times exploding.

# (inferlet_name, input_dict, time_budget_seconds). The package id
# (`name@version`) is read at runtime from the inferlet's Pie.toml.
WORKLOADS: list[tuple[str, dict, float]] = [
    ("helloworld", {}, 30.0),
    ("text-completion",
     {"prompt": "Write a haiku about distributed systems.", "max_tokens": 64}, 60.0),
    ("text-completion-spec",
     {"prompt": "Tell me a story about a robot.", "max_tokens": 64}, 90.0),
    ("parallel-generation",
     {"prompt": "Capital of France?", "max_tokens": 32, "n": 4}, 60.0),
    ("skeleton-of-thought",
     {"question": "How does photosynthesis work?"}, 120.0),
    ("tree-of-thought",
     {"question": "What is 8 * 7?"}, 120.0),
    ("recursion-of-thought",
     {"problem": "Add 12 + 24"}, 120.0),
    ("graph-of-thought",
     {"task": "Plan a small picnic."}, 120.0),
    ("best-of-n",
     {"prompt": "Best one-line python tip:", "n": 4, "max_tokens": 32}, 90.0),
    ("agent-react", {"task": "What is 2+2?"}, 60.0),
    ("attention-sink",
     {"max_tokens": 64, "sink_size": 4, "window_size": 32}, 60.0),
    ("windowed-attention",
     {"max_tokens": 64, "window_size": 32}, 60.0),
    ("page-trim-bench",
     {"prompt_tokens": 256, "decode_steps": 32, "use_mask": True}, 60.0),
    ("constrained-decoding", {}, 60.0),
    ("json-schema-validation", {}, 60.0),
    ("template-generation", {}, 60.0),
    ("output-validation", {}, 90.0),
    ("cacheback-decoding",
     {"prompt": "Hello world", "max_tokens": 32}, 90.0),
    ("prefix-tree", {}, 180.0),
    ("sampler-suite", {}, 60.0),
    ("raw-logits-demo", {"iters": 16}, 90.0),
    ("watermarking",
     {"prompt": "Once upon a time", "max_tokens": 64}, 60.0),
]


def package_id(name: str) -> str | None:
    """Read inferlet's Pie.toml and return `name@version`."""
    manifest = ROOT / "inferlets" / name / "Pie.toml"
    if not manifest.exists():
        return None
    try:
        with open(manifest, "rb") as f:
            data = tomllib.load(f)
        pkg = data.get("package", {})
        return f"{pkg['name']}@{pkg['version']}"
    except Exception:
        return None


@dataclass
class RunResult:
    inferlet: str
    concurrency: int
    driver: str
    success: int
    fail: int
    timeout: int
    wall_seconds: float
    p50_seconds: float
    p99_seconds: float
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Driver config
# ---------------------------------------------------------------------------

def build_driver_options(driver: str, args) -> dict:
    if driver == "dev":
        return {
            "gpu_mem_utilization": args.gpu_mem_util,
            "max_batch_size": args.max_batch_size,
        }
    if driver == "sglang":
        return {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": not args.use_cuda_graphs,
            "attention_backend": "triton",
        }
    if driver == "vllm":
        return {
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_batch_size,
            "enforce_eager": not args.use_cuda_graphs,
            # FLASHINFER is the only backend pie_driver_vllm's mask-routing
            # supports. FlashAttention V1 silently drops the BRLE custom mask.
            "attention_backend": "FLASHINFER",
        }
    if driver == "dummy":
        return {}
    raise ValueError(f"Unknown driver: {driver}")


# ---------------------------------------------------------------------------
# Helpers: locate WASM, install, run
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def find_wasm(name: str) -> tuple[Path | None, Path | None]:
    """Return (wasm_path, manifest_path) for an inferlet, or (None, None)."""
    wasm_name = name.replace("-", "_")
    inferlet_dir = ROOT / "inferlets" / name
    candidates = [
        inferlet_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / f"{wasm_name}.wasm",
    ]
    wasm = next((p for p in candidates if p.exists()), None)
    manifest = inferlet_dir / "Pie.toml"
    if not wasm or not manifest.exists():
        return None, None
    return wasm, manifest


async def run_one(client, pkg: str, inputs: dict, timeout: float) -> tuple[bool, float, str]:
    """Launch one inferlet, wait for completion. Returns (success, seconds, reason)."""
    t0 = time.time()
    try:
        proc = await client.launch_process(pkg, input=inputs)
    except Exception as e:
        return False, time.time() - t0, f"launch: {type(e).__name__}: {str(e)[:80]}"

    try:
        while True:
            if time.time() - t0 > timeout:
                return False, time.time() - t0, "TIMEOUT"
            ev, msg = await asyncio.wait_for(proc.recv(), timeout=timeout)
            if ev == Event.Return:
                return True, time.time() - t0, ""
            if ev == Event.Error:
                return False, time.time() - t0, f"error: {str(msg)[:80]}"
    except asyncio.TimeoutError:
        return False, time.time() - t0, "TIMEOUT"
    except Exception as e:
        return False, time.time() - t0, f"recv: {type(e).__name__}: {str(e)[:80]}"


async def run_concurrent(
    client, pkg: str, inputs: dict, concurrency: int, timeout: float
) -> RunResult:
    """Launch `concurrency` inferlets in parallel, gather stats."""
    t_start = time.time()
    tasks = [run_one(client, pkg, inputs, timeout) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    wall = time.time() - t_start

    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    timeouts = [r for r in failures if "TIMEOUT" in r[2]]

    times = [r[1] for r in successes] or [0.0]
    times.sort()
    n = len(times)
    p50 = times[n // 2] if n else 0.0
    p99 = times[max(0, int(n * 0.99) - 1)] if n else 0.0

    fail_msgs = list({r[2] for r in failures if r[2]})

    return RunResult(
        inferlet="",
        concurrency=concurrency,
        driver="",
        success=len(successes),
        fail=len(failures),
        timeout=len(timeouts),
        wall_seconds=wall,
        p50_seconds=p50,
        p99_seconds=p99,
        failures=fail_msgs[:5],
    )


# ---------------------------------------------------------------------------
# Bench loop per driver
# ---------------------------------------------------------------------------

async def bench_driver(driver: str, args, output_dir: Path) -> list[RunResult]:
    from pie.server import Server
    from pie.config import (
        AuthConfig, Config, DriverConfig, ModelConfig, SchedulerConfig,
        ServerConfig, TelemetryConfig,
    )

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else [args.device]
    driver_opts = build_driver_options(driver, args)
    max_concurrency = max(args.concurrencies)

    cfg = Config(
        server=ServerConfig(
            port=0,
            max_concurrent_processes=max(max_concurrency * 2, 32),
        ),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                scheduler=SchedulerConfig(default_token_limit=args.default_token_limit),
                driver=DriverConfig(
                    type=driver,
                    device=device,
                    options=driver_opts,
                ),
            ),
        ],
    )

    print(f"\n{'#' * 70}\n# Driver: {driver}\n{'#' * 70}", flush=True)

    results: list[RunResult] = []

    async with Server(cfg) as server:
        client = await server.connect()

        # Install all inferlets we plan to run.
        installed: dict[str, str] = {}
        for name, _, _ in WORKLOADS:
            wasm, manifest = find_wasm(name)
            pkg = package_id(name)
            if not wasm or not pkg:
                print(f"  [skip] {name}: no built WASM or manifest")
                continue
            try:
                await client.install_program(wasm, manifest, force_overwrite=True)
                installed[name] = pkg
            except Exception as e:
                print(f"  [skip] {name}: install failed: {type(e).__name__}: {str(e)[:80]}")

        # Warm up: one launch of each unique inferlet to amortize JIT/load.
        print(f"\n  Warming up ({len(installed)} inferlets)...", flush=True)
        for name, pkg in installed.items():
            inputs = next(w[1] for w in WORKLOADS if w[0] == name)
            _, _, reason = await run_one(client, pkg, inputs, 60.0)
            if reason and "error" in reason.lower():
                print(f"    [warmup-fail] {name}: {reason}")

        # Run each (workload × concurrency) combination.
        for name, inputs, budget in WORKLOADS:
            if name not in installed:
                continue
            pkg = installed[name]
            print(f"\n  --- {name} ---", flush=True)
            for c in args.concurrencies:
                # Cap concurrency by what each workload reasonably supports.
                eff_c = min(c, args.max_concurrency_per_workload)
                t = budget * 1.5  # generous timeout per request
                res = await run_concurrent(client, pkg, inputs, eff_c, t)
                res.inferlet = name
                res.driver = driver
                res.concurrency = eff_c
                results.append(res)
                tps = res.success / res.wall_seconds if res.wall_seconds > 0 else 0.0
                fail_note = f" fail={res.fail}" if res.fail else ""
                fail_detail = f" {res.failures[0][:40]!r}" if res.failures else ""
                print(
                    f"    c={eff_c:>3}  ok={res.success:>3}{fail_note}  "
                    f"wall={res.wall_seconds:>6.2f}s  p50={res.p50_seconds:>5.2f}s  "
                    f"p99={res.p99_seconds:>5.2f}s  tps={tps:>5.2f}/s{fail_detail}",
                    flush=True
                )

    # Save per-driver JSON.
    out = output_dir / f"{driver}_results.json"
    with open(out, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Saved: {out}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_markdown_report(
    all_results: dict[str, list[RunResult]], output_dir: Path, args
):
    out = output_dir / "REPORT.md"
    drivers = list(all_results.keys())

    inferlets = sorted({r.inferlet for rs in all_results.values() for r in rs})
    concurrencies = sorted({r.concurrency for rs in all_results.values() for r in rs})

    with open(out, "w") as f:
        f.write(f"# Cross-driver inferlet benchmark\n\n")
        f.write(f"- Model: `{args.model}`\n")
        f.write(f"- Device: `{args.device}`\n")
        f.write(f"- Drivers: {', '.join(f'`{d}`' for d in drivers)}\n")
        f.write(f"- Concurrencies: {concurrencies}\n")
        f.write(f"- max_batch_size: {args.max_batch_size}\n")
        f.write(f"- gpu_mem_util: {args.gpu_mem_util}\n\n")

        # Per-driver summary table
        for driver in drivers:
            rs = all_results[driver]
            ok_total = sum(r.success for r in rs)
            fail_total = sum(r.fail for r in rs)
            f.write(f"## Driver: `{driver}`\n\n")
            f.write(f"- Total runs: {ok_total + fail_total}, "
                    f"success: {ok_total}, fail: {fail_total}\n\n")

            f.write("| inferlet | c |  ok | fail |  wall(s) | p50(s) | p99(s) | tps |\n")
            f.write("|----------|--:|----:|----:|---------:|-------:|-------:|----:|\n")
            for r in rs:
                tps = r.success / r.wall_seconds if r.wall_seconds > 0 else 0
                f.write(
                    f"| {r.inferlet} | {r.concurrency} | {r.success} | {r.fail} | "
                    f"{r.wall_seconds:.2f} | {r.p50_seconds:.2f} | {r.p99_seconds:.2f} | {tps:.2f} |\n"
                )
            f.write("\n")

        # Cross-driver throughput comparison: most representative single
        # workload per (inferlet, concurrency)
        if len(drivers) > 1:
            f.write("## Cross-driver throughput comparison (req/s)\n\n")
            f.write("Higher is better. Empty cell = workload not run for that driver.\n\n")
            header = "| inferlet | c | " + " | ".join(drivers) + " |"
            sep = "|----------|--:|" + "|".join("----:" for _ in drivers) + "|"
            f.write(header + "\n" + sep + "\n")
            for inf in inferlets:
                for c in concurrencies:
                    row = [inf, str(c)]
                    for d in drivers:
                        match = next(
                            (r for r in all_results[d] if r.inferlet == inf and r.concurrency == c),
                            None,
                        )
                        if match and match.wall_seconds > 0:
                            tps = match.success / match.wall_seconds
                            row.append(f"{tps:.2f}")
                        else:
                            row.append("—")
                    f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

        # Failures section
        f.write("## Failures observed\n\n")
        any_fail = False
        for driver in drivers:
            for r in all_results[driver]:
                if r.fail and r.failures:
                    any_fail = True
                    f.write(f"- **{driver}** / {r.inferlet} c={r.concurrency}: "
                            f"{r.fail} failed — {r.failures[0]}\n")
        if not any_fail:
            f.write("None.\n")

    print(f"\nReport: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drivers = [d.strip() for d in args.drivers.split(",") if d.strip()]
    all_results: dict[str, list[RunResult]] = {}

    for driver in drivers:
        try:
            rs = await bench_driver(driver, args, output_dir)
            all_results[driver] = rs
        except Exception as e:
            print(f"\n  [driver-fail] {driver}: {type(e).__name__}: {e}", flush=True)
            all_results[driver] = []

    write_markdown_report(all_results, output_dir, args)


def main():
    parser = argparse.ArgumentParser(description="Cross-driver inferlet benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--drivers", default="native,sglang",
                        help="Comma-separated list. Available: native, sglang, vllm, dummy")
    parser.add_argument("--concurrencies", default="1,4,8",
                        help="Comma-separated concurrency levels to sweep")
    parser.add_argument("--output-dir", default="/tmp/bench-runs")
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--default-token-limit", type=int, default=200_000)
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--max-concurrency-per-workload", type=int, default=16,
                        help="Cap concurrency per workload to avoid total OOM")
    args = parser.parse_args()
    args.concurrencies = [int(c) for c in args.concurrencies.split(",") if c.strip()]

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
