"""Pie throughput benchmark.

Spins up a local Pie server using the ``Server`` API, installs the
text-completion inferlet, then fires concurrent requests and reports
throughput.

Usage::

    uv run python benches/tput.py
    uv run python benches/tput.py --num-requests 128 --concurrency 32
    uv run python benches/tput.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda:0,cuda:1
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from pie_client import Event


async def run_benchmark(args):
    from pie.server import Server

    # -- Resolve paths --------------------------------------------------------

    script_dir = Path(__file__).parent.resolve()
    wasm_path = (
        script_dir.parent
        / "inferlets"
        / "text-completion"
        / "target"
        / "wasm32-wasip2"
        / "release"
        / "text_completion.wasm"
    )
    manifest_path = script_dir.parent / "inferlets" / "text-completion" / "Pie.toml"

    if not wasm_path.exists():
        print(f"Error: WASM binary not found at {wasm_path}")
        print("Run `cargo build --target wasm32-wasip2 --release` in text-completion first.")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    import tomllib

    manifest = tomllib.loads(manifest_path.read_text())
    pkg_name = manifest["package"]["name"]
    version = manifest["package"]["version"]
    inferlet_name = f"{pkg_name}@{version}"

    # -- Parse device list ----------------------------------------------------

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else args.device

    # -- Start server ---------------------------------------------------------

    print(f"Model:       {args.model}")
    print(f"Device:      {device}")
    print(f"Requests:    {args.num_requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max Tokens:  {args.max_tokens}")
    print(f"Prompt:      {args.prompt!r}")
    print()

    async with Server(
        model=args.model,
        device=device,
        dummy=args.dummy,
    ) as client:
        # -- Install program --------------------------------------------------

        if not await client.check_program(inferlet_name, wasm_path, manifest_path):
            print("Installing program...")
            await client.install_program(wasm_path, manifest_path)
        else:
            print("Program already installed.")

        # -- Build workload ---------------------------------------------------

        inferlet_args = [
            "--prompt", args.prompt,
            "--max-tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--system", "You are a helpful benchmarking assistant.",
        ]

        queue = asyncio.Queue()
        for i in range(args.num_requests):
            queue.put_nowait(i)

        completed = 0
        total_chars = 0
        total_tokens_est = 0

        # -- Workers ----------------------------------------------------------

        async def worker(worker_id: int):
            nonlocal completed, total_chars, total_tokens_est
            while not queue.empty():
                try:
                    req_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    process = await client.launch_process(
                        inferlet_name, arguments=inferlet_args,
                    )
                    req_chars = 0
                    while True:
                        event, msg = await process.recv()
                        if event == Event.Stdout:
                            req_chars += len(msg)
                        elif event == Event.Return:
                            req_chars += len(msg)
                            total_chars += req_chars
                            total_tokens_est += req_chars / 4.0
                            completed += 1
                            print(".", end="", flush=True)
                            break
                        elif event == Event.Error:
                            print(f"\n[{worker_id}] Req {req_id} failed: {msg}")
                            break
                except Exception as e:
                    print(f"\n[{worker_id}] Error: {e}")
                finally:
                    queue.task_done()

        # -- Run --------------------------------------------------------------

        print("Running", end="", flush=True)
        start = time.time()

        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        await asyncio.wait(workers)

        duration = time.time() - start

        # -- Report -----------------------------------------------------------

        print(f"\n\n{'─' * 40}")
        print(f"{'Total Time:':<25} {duration:.2f} s")
        print(f"{'Completed:':<25} {completed}/{args.num_requests}")
        print(f"{'Total Chars:':<25} {total_chars}")
        print(f"{'Est. Total Tokens:':<25} {total_tokens_est:.0f}")
        print(f"{'Requests/sec:':<25} {completed / duration:.2f}")
        print(f"{'Est. Tokens/sec:':<25} {total_tokens_est / duration:.2f}")
        print(f"{'─' * 40}")


def main():
    parser = argparse.ArgumentParser(description="Pie Throughput Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--device", default="cuda:0", help="Device(s), comma-separated (e.g. cuda:0,cuda:1)")
    parser.add_argument("--num-requests", type=int, default=64, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=64, help="Concurrent requests")
    parser.add_argument("--prompt", default="Write a short story about a robot.", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--dummy", action="store_true", help="Use dummy mode (no GPU)")

    args = parser.parse_args()

    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()
