import asyncio
import argparse
import time
import sys
import tomllib
from pathlib import Path
from blake3 import blake3
from pie_client import PieClient, Event


async def run_benchmark(args):
    # 1. Setup paths
    script_dir = Path(__file__).parent.resolve()
    # Default WASM path assuming standard layout
    wasm_path = (
        script_dir.parent
        / "std"
        / "text-completion"
        / "target"
        / "wasm32-wasip2"
        / "release"
        / "text_completion.wasm"
    )
    # Manifest path
    manifest_path = script_dir.parent / "std" / "text-completion" / "Pie.toml"

    if not wasm_path.exists():
        print(f"Error: WASM binary not found at {wasm_path}")
        print(
            "Please run `cargo build --target wasm32-wasip2 --release` in `std/text-completion` first."
        )
        sys.exit(1)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    print(f"Using WASM: {wasm_path}")
    print(f"Using Manifest: {manifest_path}")
    program_bytes = wasm_path.read_bytes()
    manifest_content = manifest_path.read_text()
    manifest = tomllib.loads(manifest_content)
    namespace, name = manifest["package"]["name"].split("/", 1)
    version = manifest["package"]["version"]
    wasm_hash = blake3(program_bytes).hexdigest()
    toml_hash = blake3(manifest_content.encode()).hexdigest()
    inferlet_name = f"{namespace}/{name}@{version}"
    print(f"Inferlet: {inferlet_name} (wasm: {wasm_hash}, toml: {toml_hash})")

    # 2. Connect to server
    print(f"Connecting to {args.server}...")
    async with PieClient(args.server) as client:
        await client.authenticate("benchmark-user")

        # 3. Upload program (check both name and hashes match)
        if not await client.program_exists(inferlet_name, wasm_hash, toml_hash):
            print("Uploading program...")
            await client.upload_program(program_bytes, manifest_content)
        else:
            print("Program already exists on server.")

        # 4. Prepare workload
        print(
            f"Starting benchmark: {args.num_requests} requests, concurrency {args.concurrency}"
        )
        print(f"Prompt: {args.prompt}")
        print(f"Max Tokens: {args.max_tokens}")

        inferlet_args = [
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--system",
            "You are a helpful benchmarking assistant.",
        ]

        # 5. Execution Loop
        start_time = time.time()

        tasks = []
        completed = 0
        total_chars = 0
        total_tokens_est = 0

        queue = asyncio.Queue()
        for i in range(args.num_requests):
            queue.put_nowait(i)

        async def worker(worker_id):
            nonlocal completed, total_chars, total_tokens_est
            while not queue.empty():
                try:
                    req_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                # Launch instance
                try:
                    inferlet_name = f"{namespace}/{name}@{version}"
                    instance = await client.launch_instance(
                        inferlet_name, arguments=inferlet_args
                    )
                    while True:
                        event, msg = await instance.recv()
                        if event == Event.Completed:
                            text = msg
                            chars = len(text)
                            tokens = chars / 4.0

                            total_chars += chars
                            total_tokens_est += tokens
                            completed += 1
                            print(".", end="", flush=True)
                            break
                        elif event == Event.Exception:
                            print(f"[{worker_id}] Req {req_id} failed: {msg}")
                            break
                        # Handle other potential closing events
                        elif event in (
                            Event.Aborted,
                            Event.ServerError,
                            Event.OutOfResources,
                        ):
                            print(
                                f"[{worker_id}] Req {req_id} aborted/failed: {event} {msg}"
                            )
                            break
                except Exception as e:
                    print(f"[{worker_id}] Error: {e}")
                finally:
                    queue.task_done()

        # Creates workers
        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        await asyncio.wait(workers)

        duration = time.time() - start_time

        print("\n--- Benchmark Results ---")
        print(f"Total Time: {duration:.2f} s")
        print(f"Total Requests: {completed}/{args.num_requests}")
        print(f"Total Chars: {total_chars}")
        print(f"Est. Total Tokens: {total_tokens_est:.0f}")
        print(f"Throughput (Requests/sec): {completed / duration:.2f}")
        print(f"Throughput (Est. Tokens/sec): {total_tokens_est / duration:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Pie Throughput Benchmark")
    parser.add_argument("--server", default="ws://127.0.0.1:8080", help="Server URI")
    parser.add_argument(
        "--num-requests", type=int, default=64, help="Total number of requests"
    )
    parser.add_argument(
        "--concurrency", type=int, default=64, help="Concurrent requests"
    )
    parser.add_argument(
        "--prompt", default="Write a short story about a robot.", help="Prompt to use"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate per request"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")

    args = parser.parse_args()

    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()
