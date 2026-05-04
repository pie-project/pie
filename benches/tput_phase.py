"""Phase-instrumented variant of tput.py — splits per-request lifetime into:
  launch  : client.launch_process round-trip
  ttft    : time-to-first-event after launch (cold spin-up + first inferlet output)
  stream  : first-event → Return wall time (steady-state token streaming)

Reports p50/p95/avg per phase plus aggregate throughput."""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

from pie_client import Event


async def run(args):
    from pie.server import Server
    from pie.config import (
        Config, ModelConfig, AuthConfig, ServerConfig, TelemetryConfig,
        DriverConfig, SchedulerConfig,
    )

    script_dir = Path(__file__).parent.resolve()
    wasm_path = (
        script_dir.parent / "inferlets" / "text-completion"
        / "target" / "wasm32-wasip2" / "release" / "text_completion.wasm"
    )
    manifest_path = script_dir.parent / "inferlets" / "text-completion" / "Pie.toml"

    import tomllib
    manifest = tomllib.loads(manifest_path.read_text())
    inferlet_name = f"{manifest['package']['name']}@{manifest['package']['version']}"

    cfg = Config(
        server=ServerConfig(port=0, max_concurrent_processes=args.max_concurrent_processes),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models=[ModelConfig(
            name="default", hf_repo=args.model,
            scheduler=SchedulerConfig(default_token_limit=args.default_token_limit),
            driver=DriverConfig(type=args.driver, device=["cuda:0"], options={}),
        )],
    )

    async with Server(cfg) as server:
        client = await server.connect()
        await client.install_program(wasm_path, manifest_path, force_overwrite=True)

        inferlet_input = {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system": "You are a helpful benchmarking assistant.",
        }

        # per-phase samples
        launch_ms, ttft_ms, stream_ms, total_ms = [], [], [], []
        event_counts = []

        async def one(req_id: int):
            t0 = time.perf_counter()
            process = await client.launch_process(inferlet_name, input=inferlet_input)
            t_launch = time.perf_counter()
            launch_ms.append((t_launch - t0) * 1000)

            t_first = None
            n_events = 0
            while True:
                event, msg = await process.recv()
                n_events += 1
                if t_first is None:
                    t_first = time.perf_counter()
                    ttft_ms.append((t_first - t_launch) * 1000)
                if event == Event.Return:
                    t_end = time.perf_counter()
                    stream_ms.append((t_end - t_first) * 1000)
                    total_ms.append((t_end - t0) * 1000)
                    event_counts.append(n_events)
                    return
                if event == Event.Error:
                    return

        print(f"Driver: {args.driver}  Reqs: {args.num_requests}  Max-tok: {args.max_tokens}")
        t_bench = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(args.num_requests)])
        wall = time.perf_counter() - t_bench

        def pct(xs, p):
            xs = sorted(xs)
            return xs[min(int(len(xs) * p), len(xs) - 1)]

        def fmt(name, xs):
            if not xs:
                print(f"{name:<10} (no samples)")
                return
            print(f"{name:<10} p50={pct(xs,0.5):7.2f}ms  p95={pct(xs,0.95):7.2f}ms  "
                  f"avg={statistics.mean(xs):7.2f}ms  max={max(xs):7.2f}ms")

        print()
        print(f"Wall: {wall:.3f}s  Req/s: {args.num_requests/wall:.2f}")
        print(f"Total events received: {sum(event_counts)}  "
              f"(avg {statistics.mean(event_counts):.1f}/req)")
        print()
        fmt("launch:", launch_ms)
        fmt("ttft:",   ttft_ms)
        fmt("stream:", stream_ms)
        fmt("total:",  total_ms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--driver", default="dummy", choices=["dev", "vllm", "sglang", "dummy"])
    p.add_argument("--num-requests", type=int, default=128)
    p.add_argument("--prompt", default="Write a short story about a robot.")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--default-token-limit", type=int, default=256)
    p.add_argument("--max-concurrent-processes", type=int, default=None)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
