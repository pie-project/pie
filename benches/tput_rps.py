"""Pie throughput/latency benchmark for *varying request rates*.

Adapted from ``tput.py``. The original benchmark is closed-loop: it
fires ``--num-requests`` requests and lets ``concurrency`` workers pull
from a queue as fast as possible. That only exercises the steady-state /
saturation regime — exactly where the adaptive batch scheduler looks
healthy.

This script is open-loop: requests are launched at a target RPS using
Poisson (exponential inter-arrival) timing, regardless of how fast the
server drains them. It runs the workload for a fixed wall-clock window
and reports throughput + latency percentiles, so the scheduler's
behavior at low / mid / high arrival rates is directly comparable.

You can sweep multiple RPS points in a single server boot via
``--rps-sweep`` to amortize the (expensive) model load.

Usage::

    uv run python benches/tput_rps.py --rps 4 --duration 15 --default-token-limit 256
    uv run python benches/tput_rps.py --rps-sweep 1,4,16,64 --duration 12 --default-token-limit 256
"""

import argparse
import asyncio
import math
import random
import statistics
import sys
import time
from pathlib import Path

from pie_client import Event


def _percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def _run_churn(client, inferlet_name, args, phases: list[tuple[float, float]]):
    """Drive the server through a sequence of (rps, duration_s) phases.

    Each request is tagged with the phase index it was *launched* in
    (not when it completed) so we can see how `ref_l`-style state from
    a busy phase poisons a subsequent idle phase, or vice versa.
    """
    inferlet_input_base = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "system": "You are a helpful benchmarking assistant.",
    }

    rng = random.Random(args.seed)
    # Each entry: (launch_offset_within_phase, latency_seconds)
    per_phase_records: list[list[tuple[float, float]]] = [[] for _ in phases]
    in_flight: list[asyncio.Task] = []

    async def one_request(req_id: int, phase_idx: int, launch_offset: float):
        req_input = inferlet_input_base
        if args.unique_prompts:
            req_input = {**inferlet_input_base, "prompt": f"{inferlet_input_base['prompt']} (Request #{req_id})"}
        t0 = time.perf_counter()
        try:
            process = await client.launch_process(inferlet_name, input=req_input)
            while True:
                event, msg = await process.recv()
                if event == Event.Return:
                    per_phase_records[phase_idx].append((launch_offset, time.perf_counter() - t0))
                    return
                if event == Event.Error:
                    return
        except Exception:
            pass

    print(f"\n>>> CHURN: {len(phases)} phases — " + ", ".join(
        f"({rps:g}rps, {dur}s)" for rps, dur in phases
    ))
    bench_start = time.perf_counter()
    req_id = 0
    phase_start = bench_start

    for phase_idx, (rps, duration) in enumerate(phases):
        deadline = phase_start + duration
        next_arrival = max(phase_start, time.perf_counter())
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            if now < next_arrival:
                await asyncio.sleep(min(next_arrival - now, deadline - now))
                continue
            launch_offset = now - phase_start
            in_flight.append(asyncio.create_task(one_request(req_id, phase_idx, launch_offset)))
            req_id += 1
            gap = rng.expovariate(rps) if rps > 0 else float("inf")
            next_arrival += gap
        phase_start = deadline

    drive_duration = time.perf_counter() - bench_start

    if in_flight:
        try:
            await asyncio.wait_for(asyncio.gather(*in_flight, return_exceptions=True),
                                    timeout=args.drain_timeout)
        except asyncio.TimeoutError:
            for t in in_flight:
                if not t.done():
                    t.cancel()
    total_duration = time.perf_counter() - bench_start

    print(f"  drive_t:   {drive_duration:.2f}s   total_t: {total_duration:.2f}s")
    # Split each phase into "early half" and "late half" based on launch
    # time within the phase. The spiral, if it exists, manifests as
    # elevated latency in the early half of a high-RPS phase that follows
    # an idle phase, settling in the late half as `last_latency` adapts.
    print(f"  {'phase':>5} {'rps':>5} {'dur':>5} {'half':>5} {'n':>4} "
          f"{'mean':>7} {'p50':>7} {'p95':>7}")
    print("  " + "─" * 56)
    results = []
    for i, ((rps, dur), recs) in enumerate(zip(phases, per_phase_records)):
        if not recs:
            print(f"  {i:>5} {rps:>5g} {dur:>5g} {'-':>5} {0:>4} {'-':>7} {'-':>7} {'-':>7}")
            results.append({"phase": i, "rps": rps, "duration": dur, "completed": 0})
            continue
        # Split by launch offset relative to phase start.
        midpoint = dur / 2.0
        early = [l for off, l in recs if off < midpoint]
        late = [l for off, l in recs if off >= midpoint]
        for half_name, lats in (("early", early), ("late", late), ("all", [l for _, l in recs])):
            if not lats:
                print(f"  {i:>5} {rps:>5g} {dur:>5g} {half_name:>5} {0:>4} "
                      f"{'-':>7} {'-':>7} {'-':>7}")
                continue
            m = statistics.fmean(lats)
            p50 = _percentile(lats, 0.50)
            p95 = _percentile(lats, 0.95)
            print(f"  {i:>5} {rps:>5g} {dur:>5g} {half_name:>5} {len(lats):>4} "
                  f"{m*1000:>7.1f} {p50*1000:>7.1f} {p95*1000:>7.1f}")
        all_lats = [l for _, l in recs]
        results.append({
            "phase": i, "rps": rps, "duration": dur, "completed": len(all_lats),
            "mean_ms": statistics.fmean(all_lats) * 1000,
            "p50_ms": _percentile(all_lats, 0.50) * 1000,
            "p95_ms": _percentile(all_lats, 0.95) * 1000,
            "p99_ms": _percentile(all_lats, 0.99) * 1000,
        })
    return results


async def _run_one_rate(client, inferlet_name, args, rps: float):
    """Drive the server at ``rps`` requests/sec for ``args.duration`` seconds.

    Each arrival launches its inferlet in its own task and records
    start/end timestamps. After the driving window closes we wait for
    in-flight requests up to ``args.drain_timeout`` seconds.
    """
    inferlet_input_base = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "system": "You are a helpful benchmarking assistant.",
    }

    rng = random.Random(args.seed)
    latencies: list[float] = []
    completed = 0
    failed = 0
    in_flight: list[asyncio.Task] = []

    async def one_request(req_id: int):
        nonlocal completed, failed
        req_input = inferlet_input_base
        if args.unique_prompts:
            req_input = {**inferlet_input_base, "prompt": f"{inferlet_input_base['prompt']} (Request #{req_id})"}
        t0 = time.perf_counter()
        try:
            process = await client.launch_process(inferlet_name, input=req_input)
            while True:
                event, msg = await process.recv()
                if event == Event.Return:
                    latencies.append(time.perf_counter() - t0)
                    completed += 1
                    return
                if event == Event.Error:
                    failed += 1
                    return
        except Exception:
            failed += 1

    print(f"\n>>> RPS={rps:g}  duration={args.duration}s  (target arrivals ≈ {int(rps * args.duration)})")
    bench_start = time.perf_counter()
    deadline = bench_start + args.duration
    next_arrival = bench_start
    req_id = 0

    # Open-loop driver. Sleep until the next scheduled arrival, fire it, repeat.
    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        if now < next_arrival:
            await asyncio.sleep(next_arrival - now)
        in_flight.append(asyncio.create_task(one_request(req_id)))
        req_id += 1
        # Exponential inter-arrival ⇒ Poisson process.
        gap = rng.expovariate(rps)
        next_arrival += gap

    drive_duration = time.perf_counter() - bench_start
    launched = req_id

    # Drain in-flight requests with a deadline so a misbehaving config
    # cannot stall the whole sweep.
    if in_flight:
        try:
            await asyncio.wait_for(asyncio.gather(*in_flight, return_exceptions=True),
                                    timeout=args.drain_timeout)
        except asyncio.TimeoutError:
            for t in in_flight:
                if not t.done():
                    t.cancel()
    total_duration = time.perf_counter() - bench_start

    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    p99 = _percentile(latencies, 0.99)
    mean = statistics.fmean(latencies) if latencies else float("nan")

    print(f"  launched:  {launched}  completed: {completed}  failed: {failed}")
    print(f"  drive_t:   {drive_duration:.2f}s   total_t: {total_duration:.2f}s")
    print(f"  achieved:  {completed / total_duration:.2f} req/s  (target {rps:g})")
    print(f"  latency:   mean={mean*1000:.1f}ms  p50={p50*1000:.1f}ms  "
          f"p95={p95*1000:.1f}ms  p99={p99*1000:.1f}ms")

    return {
        "rps_target": rps,
        "launched": launched,
        "completed": completed,
        "failed": failed,
        "duration_s": total_duration,
        "rps_achieved": completed / total_duration if total_duration > 0 else 0.0,
        "mean_ms": mean * 1000,
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "p99_ms": p99 * 1000,
    }


async def run_benchmark(args):
    from pie.server import Server
    from pie.config import (
        Config, ModelConfig, AuthConfig,
        ServerConfig, TelemetryConfig, DriverConfig, SchedulerConfig,
    )

    script_dir = Path(__file__).parent.resolve()
    wasm_path = (
        script_dir.parent / "inferlets" / "text-completion"
        / "target" / "wasm32-wasip2" / "release" / "text_completion.wasm"
    )
    manifest_path = script_dir.parent / "inferlets" / "text-completion" / "Pie.toml"
    if not wasm_path.exists():
        print(f"Error: WASM binary not found at {wasm_path}")
        sys.exit(1)

    import tomllib
    manifest = tomllib.loads(manifest_path.read_text())
    inferlet_name = f"{manifest['package']['name']}@{manifest['package']['version']}"

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else [args.device]

    # Driver subsection — keep the same vocabulary translation as tput.py.
    if args.driver == "vllm":
        driver_subsection: dict = {
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_batch_size,
            "enforce_eager": not args.use_cuda_graphs,
        }
        if args.vllm_attention_backend is not None:
            driver_subsection["attention_backend"] = args.vllm_attention_backend
    elif args.driver == "dev":
        driver_subsection = {
            "gpu_mem_utilization": args.gpu_mem_util,
            "max_batch_size": args.max_batch_size,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
        }
    elif args.driver == "sglang":
        driver_subsection = {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": not args.use_cuda_graphs,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
        }
        if args.sglang_attention_backend is not None:
            driver_subsection["attention_backend"] = args.sglang_attention_backend
    else:
        driver_subsection = {}

    cfg = Config(
        server=ServerConfig(port=0, max_concurrent_processes=args.max_concurrent_processes),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                scheduler=SchedulerConfig(
                    batch_policy=args.batch_policy,
                    default_token_limit=args.default_token_limit,
                ),
                driver=DriverConfig(type=args.driver, device=device, options=driver_subsection),
            ),
        ],
    )

    churn_phases: list[tuple[float, float]] | None = None
    if args.churn:
        # Parse "rps:dur,rps:dur,..."
        churn_phases = []
        for spec in args.churn.split(","):
            spec = spec.strip()
            if not spec:
                continue
            r, d = spec.split(":")
            churn_phases.append((float(r), float(d)))
    rates: list[float]
    if args.rps_sweep:
        rates = [float(x) for x in args.rps_sweep.split(",") if x.strip()]
    elif churn_phases is None:
        rates = [float(args.rps)]
    else:
        rates = []

    print(f"Model:       {args.model}")
    print(f"Device:      {device}   driver={args.driver}")
    print(f"Max Tokens:  {args.max_tokens}    duration/point: {args.duration}s")
    if churn_phases:
        print(f"Churn:       {churn_phases}")
    if rates:
        print(f"RPS sweep:   {rates}")

    async with Server(cfg) as server:
        client = await server.connect()
        print("Installing program...")
        await client.install_program(wasm_path, manifest_path, force_overwrite=True)

        # Optional warmup so the first sweep point doesn't pay model JIT /
        # CUDA graph capture costs (the policy itself also skips its first
        # batch, so without warmup the first RPS point is doubly penalized).
        if args.warmup_requests > 0:
            # Warmup at the *measurement* token count so JIT / CUDA-graph /
            # FlashInfer planning costs are paid here, not on the first
            # measured request.
            print(f"Warmup: {args.warmup_requests} requests at {args.max_tokens} tokens...")
            warm_input = {
                "prompt": args.prompt, "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "system": "You are a helpful benchmarking assistant.",
            }
            warm = [asyncio.create_task(_consume_one(client, inferlet_name, warm_input))
                    for _ in range(args.warmup_requests)]
            await asyncio.gather(*warm, return_exceptions=True)
            print("Warmup done.")

        results = []
        for rps in rates:
            r = await _run_one_rate(client, inferlet_name, args, rps)
            results.append(r)
            # Brief pause so the policy's ref_l can decay between points
            # (otherwise a high-RPS point poisons the next low-RPS point).
            await asyncio.sleep(args.cool_down)

        if churn_phases:
            await _run_churn(client, inferlet_name, args, churn_phases)

    # Summary table
    print(f"\n{'─' * 78}")
    print(f"{'rps_target':>10} {'rps_achieved':>13} {'completed':>10} {'mean_ms':>9} "
          f"{'p50_ms':>8} {'p95_ms':>9} {'p99_ms':>9}")
    print("─" * 78)
    for r in results:
        print(f"{r['rps_target']:>10.2f} {r['rps_achieved']:>13.2f} {r['completed']:>10} "
              f"{r['mean_ms']:>9.1f} {r['p50_ms']:>8.1f} {r['p95_ms']:>9.1f} {r['p99_ms']:>9.1f}")
    print("─" * 78)


async def _consume_one(client, inferlet_name, req_input):
    process = await client.launch_process(inferlet_name, input=req_input)
    while True:
        event, _ = await process.recv()
        if event in (Event.Return, Event.Error):
            return


def main():
    parser = argparse.ArgumentParser(description="Pie open-loop RPS benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="Write a short story about a robot.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-mem-util", type=float, default=0.8)
    parser.add_argument("--cpu-mem-budget", type=int, default=0)
    parser.add_argument("--unique-prompts", action="store_true")
    parser.add_argument("--default-token-limit", type=int, required=True)
    parser.add_argument("--max-concurrent-processes", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--driver", default="dev", choices=["dev", "vllm", "sglang", "dummy"])
    parser.add_argument("--vllm-attention-backend", default=None)
    parser.add_argument("--sglang-attention-backend", default=None)
    parser.add_argument("--use-cuda-graphs", action="store_true")

    # Open-loop driving controls
    parser.add_argument("--rps", type=float, default=4.0,
                        help="Target arrival rate (Poisson). Ignored if --rps-sweep is set.")
    parser.add_argument("--rps-sweep", type=str, default=None,
                        help="Comma-separated RPS values to sweep in one server boot, e.g. '1,4,16,64'.")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Driving window per RPS point (seconds).")
    parser.add_argument("--drain-timeout", type=float, default=30.0,
                        help="Max seconds to wait for in-flight requests after driving stops.")
    parser.add_argument("--cool-down", type=float, default=2.0,
                        help="Idle pause between sweep points so ref_l can settle.")
    parser.add_argument("--warmup-requests", type=int, default=4,
                        help="Number of small warmup requests before the first measurement.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--churn", type=str, default=None,
                        help="Phase-alternating workload, e.g. '32:6,1:6,32:6,1:6'.")
    parser.add_argument("--batch-policy", type=str, default="adaptive",
                        choices=["adaptive", "eager", "greedy"],
                        help="Batch-firing policy (default: adaptive).")

    args = parser.parse_args()
    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()
