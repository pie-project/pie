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
    from pie.config import Config, ModelConfig, AuthConfig

    if args.driver == "native" and args.use_cuda_graphs:
        print("ERROR: --use-cuda-graphs is not supported on the native driver.",
              file=sys.stderr)
        sys.exit(1)

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

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else [args.device]

    # -- Start server ---------------------------------------------------------

    print(f"Model:       {args.model}")
    print(f"Device:      {device}")
    print(f"Requests:    {args.num_requests}")
    print(f"Max Tokens:  {args.max_tokens}")
    print(f"GPU Mem:     {args.gpu_mem_util}")
    print(f"Prompt:      {args.prompt!r}")
    print()

    from pie.config import (
        ServerConfig, TelemetryConfig, DriverConfig,
    )

    # Build the [model.X.driver.<type>] subsection. Each driver expresses
    # its budgets in its own vocabulary — translate CLI flags accordingly.
    if args.driver == "vllm":
        driver_subsection: dict = {
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_batch_size,
            "enforce_eager": not args.use_cuda_graphs,
        }
        if args.vllm_attention_backend is not None:
            driver_subsection["attention_backend"] = args.vllm_attention_backend
    elif args.driver == "native":
        driver_subsection = {
            "gpu_mem_utilization": args.gpu_mem_util,
            "max_batch_size": args.max_batch_size,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
        }
    elif args.driver == "sglang":
        # sglang's vocabulary differs: `mem_fraction_static` instead of
        # `gpu_memory_utilization`, `disable_cuda_graph` (negation),
        # `cpu_mem_budget_in_gb` is pie-universal (filtered out before
        # ServerArgs splat in pie_driver_sgl/loader.py).
        driver_subsection = {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": not args.use_cuda_graphs,
            "cpu_mem_budget_in_gb": args.cpu_mem_budget,
        }
        if args.sglang_attention_backend is not None:
            driver_subsection["attention_backend"] = args.sglang_attention_backend
    else:  # dummy
        driver_subsection = {}

    cfg = Config(
        server=ServerConfig(
            port=0,
            max_concurrent_processes=args.max_concurrent_processes,
        ),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models={
            "default": ModelConfig(
                name="default",
                hf_repo=args.model,
                default_token_budget=args.default_token_budget,
                default_endowment_pages=args.default_endowment_pages,
                oversubscription_factor=args.oversubscription_factor,
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    options=driver_subsection,
                ),
            )
        },
    )
    async with Server(cfg) as server:
        client = await server.connect()
        # -- Install program --------------------------------------------------

        # Always install to pick up latest build
        print("Installing program...")
        await client.install_program(wasm_path, manifest_path, force_overwrite=True)

        # -- Build workload ---------------------------------------------------

        inferlet_input = {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system": "You are a helpful benchmarking assistant.",
        }

        queue = asyncio.Queue()
        # Items are pushed inside the warmup / timed-run blocks below so the
        # warmup and timed phases don't share queue items.

        completed = 0
        total_chars = 0
        total_tokens_est = 0
        output_samples = []  # Collect (req_id, text) tuples
        output_lock = asyncio.Lock()

        # -- Workers ----------------------------------------------------------

        async def worker(worker_id: int):
            nonlocal completed, total_chars, total_tokens_est
            while not queue.empty():
                try:
                    req_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    req_input = inferlet_input
                    if getattr(args, 'unique_prompts', False):
                        req_input = {**inferlet_input, "prompt": f"{inferlet_input['prompt']} (Request #{req_id})"}
                    process = await client.launch_process(
                        inferlet_name, input=req_input,
                    )
                    req_chars = 0
                    req_text = []
                    while True:
                        event, msg = await process.recv()
                        if event == Event.Stdout:
                            req_chars += len(msg)
                            req_text.append(msg)
                        elif event == Event.Stderr:
                            req_chars += len(msg)
                        elif event == Event.Return:
                            req_chars += len(msg)
                            req_text.append(msg)
                            total_chars += req_chars
                            total_tokens_est += req_chars / 4.0
                            completed += 1
                            # Save output
                            async with output_lock:
                                output_samples.append((req_id, "".join(req_text)))
                            print(".", end="", flush=True)
                            break
                        elif event == Event.Error:
                            print(f"\n[{worker_id}] Req {req_id} failed: {msg}")
                            break
                except Exception as e:
                    print(f"\n[{worker_id}] Error: {e}")
                finally:
                    queue.task_done()

        # -- Warmup -----------------------------------------------------------

        if args.warmup_requests > 0:
            print(f"Warmup ({args.warmup_requests} reqs)", end="", flush=True)
            for i in range(args.warmup_requests):
                queue.put_nowait(args.num_requests + i)
            warmup_workers = [
                asyncio.create_task(worker(-1 - i)) for i in range(args.warmup_requests)
            ]
            await asyncio.wait(warmup_workers)
            # Reset counters so warmup doesn't pollute the timed measurement.
            completed = 0
            total_chars = 0
            total_tokens_est = 0
            output_samples.clear()
            print(" done")

        # -- Run --------------------------------------------------------------

        for i in range(args.num_requests):
            queue.put_nowait(i)

        print("Running", end="", flush=True)
        start = time.time()

        workers = [asyncio.create_task(worker(i)) for i in range(args.num_requests)]
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

        # -- Save output samples ----------------------------------------------

        if args.save_outputs and output_samples:
            out_path = Path(args.save_outputs)
            with open(out_path, "w") as f:
                for req_id, text in sorted(output_samples):
                    f.write(f"=== Request {req_id} ===\n")
                    f.write(text)
                    f.write("\n\n")
            print(f"Saved {len(output_samples)} output samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pie Throughput Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--device", default="cuda:0", help="Device(s), comma-separated (e.g. cuda:0,cuda:1)")
    parser.add_argument("--num-requests", type=int, default=64, help="Total number of concurrent requests")
    parser.add_argument("--prompt", default="Write a short story about a robot.", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    # Dummy mode is now its own driver: pass --driver dummy (no separate flag needed)
    parser.add_argument("--gpu-mem-util", type=float, default=0.8, help="GPU memory utilization for KV cache (lower = fewer pages = more contention)")
    parser.add_argument("--cpu-mem-budget", type=int, default=0, help="CPU memory budget in GB for working page swap (0 = disabled)")
    parser.add_argument("--save-outputs", type=str, default=None, help="Save output samples to this file path")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of output samples to save (default: 10)")
    parser.add_argument("--unique-prompts", action="store_true", help="Make each request's prompt unique (append request #N)")
    parser.add_argument("--default-token-budget", type=int, required=True, help="Default token budget per process (required)")
    parser.add_argument("--max-concurrent-processes", type=int, default=None,
                        help="Maximum number of concurrent processes (default: None — uncapped, saturate the GPU)")
    parser.add_argument("--max-batch-size", type=int, default=2048,
                        help="Maximum batch size for inference (default: 2048 — let the GPU dictate).")
    parser.add_argument("--driver", default="native", choices=["native", "vllm", "sglang", "dummy"],
                        help="Inference driver: 'native', 'vllm', 'sglang', or 'dummy'")
    parser.add_argument("--vllm-attention-backend", default=None,
                        help="vLLM attention backend (FLASH_ATTN / FLASHINFER / etc.). Only used when --driver=vllm")
    parser.add_argument("--sglang-attention-backend", default=None,
                        help="SGLang attention backend (triton / flashinfer / flex_attention / fa3). Only used when --driver=sglang")
    parser.add_argument("--use-cuda-graphs", action="store_true",
                        help="Enable CUDA graphs (vllm/sglang only — native driver does not support this).")
    parser.add_argument("--default-endowment-pages", type=int, default=64,
                        help="Per-process KV-page endowment used by the admission gate (lower = more concurrent processes admitted)")
    parser.add_argument("--oversubscription-factor", type=float, default=1000.0,
                        help="Admission overbook factor (Σ endowment ≤ capacity × factor). "
                             "Default 1000.0 effectively disables the gate; lower it to study admission behavior.")
    parser.add_argument("--warmup-requests", type=int, default=0,
                        help="Number of warmup requests to run (and discard) before timing")

    args = parser.parse_args()

    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()
