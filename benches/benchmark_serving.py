import asyncio
import argparse
import time
import sys
import json
import statistics
from pathlib import Path
from blake3 import blake3
from dataclasses import dataclass
from typing import List, Optional
from pie_client import PieClient, Event

# Import dataset loaders
sys.path.insert(0, str(Path(__file__).parent))
from datasets import get_dataset, DATASETS, Request as DatasetRequest


@dataclass
class RequestMetrics:
    request_id: int
    prompt: str
    expected_output_len: int
    start_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    output_len: int = 0
    success: bool = False
    error: str = ""

    @property
    def ttft(self) -> float:
        if self.start_time and self.first_token_time:
            return self.first_token_time - self.start_time
        return 0.0

    @property
    def e2e_latency(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def tpot(self) -> float:
        # Time Per Output Token: (E2E - TTFT) / (tokens - 1)
        if self.output_len < 2:
            return 0.0
        tokens = self.output_len / 4.0
        if tokens <= 1:
            return 0.0
        latency_decoding = self.e2e_latency - self.ttft
        return latency_decoding / (tokens - 1)

    @property
    def output_tokens(self) -> float:
        return self.output_len / 4.0


async def run_benchmark(args, requests: List[DatasetRequest]):
    # 1. Setup paths and WASM
    script_dir = Path(__file__).parent.resolve()
    wasm_path = (
        script_dir.parent
        / "std"
        / "text-completion"
        / "target"
        / "wasm32-wasip2"
        / "release"
        / "text_completion.wasm"
    )

    if not wasm_path.exists():
        print(f"Error: WASM binary not found at {wasm_path}")
        print("Please build the text-completion inferlet first.")
        sys.exit(1)

    print(f"Using WASM: {wasm_path}")
    program_bytes = wasm_path.read_bytes()
    program_hash = blake3(program_bytes).hexdigest()

    # 2. Connect and Upload
    print(f"Connecting to {args.server}...")
    async with PieClient(args.server) as client:
        try:
            await client.authenticate("benchmark-user")
        except Exception as e:
            print(f"Authentication warning (may be disabled on server): {e}")

        if not await client.program_exists(program_hash):
            print("Uploading program...")
            await client.upload_program(program_bytes)
        else:
            print("Program already exists on server.")

        # 3. Preparation
        print(f"\nStarting benchmark:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Requests: {len(requests)}")
        print(f"  Concurrency: {args.concurrency}")

        # 4. Workload Execution
        metrics: List[RequestMetrics] = []
        queue = asyncio.Queue()
        for i, req in enumerate(requests):
            queue.put_nowait((i, req))

        pbar_update_interval = max(1, len(requests) // 20)
        completed_count = 0
        
        start_time_global = time.time()

        async def worker(worker_id):
            nonlocal completed_count
            while not queue.empty():
                try:
                    req_id, dataset_req = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                req_metric = RequestMetrics(
                    request_id=req_id,
                    prompt=dataset_req.prompt[:100],  # Truncate for storage
                    expected_output_len=dataset_req.expected_output_len,
                    start_time=time.time(),
                )
                
                # Build inferlet args for this specific request
                inferlet_args = [
                    "--prompt", dataset_req.prompt,
                    "--max-tokens", str(dataset_req.expected_output_len),
                    "--temperature", str(args.temperature),
                    "--system", "You are a helpful assistant.",
                ]
                
                try:
                    instance = await client.launch_instance(
                        program_hash, arguments=inferlet_args
                    )
                    
                    first_token_received = False
                    
                    while True:
                        event, msg = await instance.recv()
                        current_time = time.time()
                        
                        if event == Event.Stdout:
                            text = msg
                            if text:
                                if not first_token_received:
                                    req_metric.first_token_time = current_time
                                    first_token_received = True
                                req_metric.output_len += len(text)
                                
                        elif event == Event.Completed:
                            req_metric.end_time = current_time
                            final_text = msg
                            if final_text and not req_metric.output_len:
                                req_metric.output_len = len(final_text)
                            if not req_metric.first_token_time:
                                req_metric.first_token_time = current_time
                            req_metric.success = True
                            break
                            
                        elif event in (Event.Exception, Event.ServerError, Event.OutOfResources, Event.Aborted):
                            req_metric.end_time = current_time
                            req_metric.error = f"{event.name}: {msg}"
                            break
                            
                except Exception as e:
                    req_metric.end_time = time.time()
                    req_metric.error = str(e)
                finally:
                    queue.task_done()
                    metrics.append(req_metric)
                    completed_count += 1
                    if completed_count % pbar_update_interval == 0:
                        print(".", end="", flush=True)

        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        await asyncio.wait(workers)
        
        duration_global = time.time() - start_time_global
        print("\n")

    # 5. Analysis
    successful_reqs = [m for m in metrics if m.success]
    failed_reqs = [m for m in metrics if not m.success]

    if not successful_reqs:
        print("All requests failed!")
        for m in failed_reqs[:5]:
            print(f"  Error: {m.error}")
        return

    # Throughput
    total_output_tokens = sum(m.output_tokens for m in successful_reqs)
    req_throughput = len(successful_reqs) / duration_global
    token_throughput = total_output_tokens / duration_global

    # Latency
    ttfts = [m.ttft * 1000 for m in successful_reqs]
    e2es = [m.e2e_latency * 1000 for m in successful_reqs]
    tpots = [m.tpot * 1000 for m in successful_reqs if m.tpot > 0]

    def print_stat(name, data):
        if not data:
            print(f"{name:<20}: N/A")
            return
        avg = statistics.mean(data)
        p50 = statistics.median(data)
        p90 = statistics.quantiles(data, n=10)[8] if len(data) >= 10 else max(data)
        p99 = statistics.quantiles(data, n=100)[98] if len(data) >= 100 else max(data)
        print(f"{name:<20}: Avg={avg:.2f}ms, P50={p50:.2f}ms, P90={p90:.2f}ms, P99={p99:.2f}ms")

    print("\n--- Benchmark Results ---")
    print(f"Total Time          : {duration_global:.2f} s")
    print(f"Successful Requests : {len(successful_reqs)}")
    print(f"Failed Requests     : {len(failed_reqs)}")
    print(f"Request Throughput  : {req_throughput:.2f} req/s")
    print(f"Token Throughput    : {token_throughput:.2f} tokens/s (est)")
    
    print("-" * 40)
    print_stat("TTFT", ttfts)
    print_stat("TPOT (Inter-tok)", tpots)
    print_stat("E2E Latency", e2es)
    print("-" * 40)

    if args.output_json:
        results = {
            "config": {
                "dataset": args.dataset,
                "num_requests": len(requests),
                "concurrency": args.concurrency,
            },
            "results": {
                "duration": duration_global,
                "successful": len(successful_reqs),
                "failed": len(failed_reqs),
                "req_throughput": req_throughput,
                "token_throughput": token_throughput,
                "ttft_ms": {"avg": statistics.mean(ttfts), "p50": statistics.median(ttfts)},
                "e2e_ms": {"avg": statistics.mean(e2es), "p50": statistics.median(e2es)},
            }
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Pie Serving Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random synthetic workload
  python benchmark_serving.py --dataset random --input-len 512 --output-len 128

  # ShareGPT (realistic chat)
  python benchmark_serving.py --dataset sharegpt --num-requests 100

  # High concurrency stress test
  python benchmark_serving.py --dataset random --num-requests 500 --input-len 256 --output-len 64
"""
    )
    
    # Server options
    parser.add_argument("--server", default="ws://127.0.0.1:8080", help="Server URI")
    parser.add_argument("--output-json", help="Path to save results JSON")
    
    # Workload options
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="random",
                        help="Dataset to use for benchmarking")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=None, 
                        help="Concurrent requests (defaults to num-requests)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    
    # Random dataset options
    parser.add_argument("--input-len", type=int, default=512, 
                        help="[random] Input length in tokens")
    parser.add_argument("--output-len", type=int, default=128, 
                        help="[random] Output length in tokens")
    parser.add_argument("--range-ratio", type=float, default=0.0,
                        help="[random] Length variation ratio (0.0=fixed, 0.5=±50%%)")
    
    # ShareGPT dataset options
    parser.add_argument("--sharegpt-path", type=str, default=None,
                        help="[sharegpt] Path to ShareGPT JSON (auto-downloads if not provided)")
    parser.add_argument("--max-input-len", type=int, default=1024,
                        help="[sharegpt] Max input length in tokens")
    parser.add_argument("--max-output-len", type=int, default=512,
                        help="[sharegpt] Max output length in tokens")
    
    args = parser.parse_args()
    
    # Default concurrency to num_requests
    if args.concurrency is None:
        args.concurrency = args.num_requests

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "random":
        dataset = get_dataset(
            "random",
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.range_ratio,
        )
    elif args.dataset == "sharegpt":
        dataset = get_dataset(
            "sharegpt",
            path=args.sharegpt_path,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
        )
    else:
        dataset = get_dataset(args.dataset)
    
    requests = dataset.load(args.num_requests)
    print(f"Loaded {len(requests)} requests")

    try:
        asyncio.run(run_benchmark(args, requests))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()

