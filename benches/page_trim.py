"""End-to-end benchmark for the page-trim optimization.

Spins up a local Pie server, installs the ``page-trim-bench`` inferlet, and
sweeps prompt lengths with a sink+window mask. Reports per-step decode
latency and aggregate throughput.

Usage::

    uv run python benches/page_trim.py
    uv run python benches/page_trim.py --prompt-tokens 512,2048,8192 --decode-steps 128
    uv run python benches/page_trim.py --model meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path

from pie_client import Event


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)=([-+0-9.eE]+|true|false)$")


def parse_summary(output: str) -> dict[str, str]:
    """Extract key=value lines from the inferlet's stdout."""
    pairs: dict[str, str] = {}
    for line in output.splitlines():
        m = _KV_RE.match(line.strip())
        if m:
            pairs[m.group(1)] = m.group(2)
    return pairs


# ---------------------------------------------------------------------------
# Driver config (mirrors benches/tput.py)
# ---------------------------------------------------------------------------

def build_driver_config(args, device: list[str]) -> dict:
    if args.driver == "vllm":
        opts: dict = {
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_batch_size,
            "enforce_eager": not args.use_cuda_graphs,
        }
        if args.vllm_attention_backend is not None:
            opts["attention_backend"] = args.vllm_attention_backend
    elif args.driver == "native":
        opts = {
            "gpu_mem_utilization": args.gpu_mem_util,
            "max_batch_size": args.max_batch_size,
        }
    elif args.driver == "sglang":
        opts = {
            "mem_fraction_static": args.gpu_mem_util,
            "disable_cuda_graph": not args.use_cuda_graphs,
        }
        if args.sglang_attention_backend is not None:
            opts["attention_backend"] = args.sglang_attention_backend
    else:
        opts = {}
    return opts


# ---------------------------------------------------------------------------
# Build inferlet (lazy, only if missing)
# ---------------------------------------------------------------------------

INFERLET_NAME = "page-trim-bench"
INFERLET_PKG = "page-trim-bench@0.1.0"


def find_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).parent.resolve()
    inferlet_dir = script_dir.parent / "inferlets" / INFERLET_NAME
    wasm_path = (
        inferlet_dir
        / "target"
        / "wasm32-wasip2"
        / "release"
        / "page_trim_bench.wasm"
    )
    manifest_path = inferlet_dir / "Pie.toml"
    return wasm_path, manifest_path


def ensure_built(wasm_path: Path, inferlet_dir: Path) -> None:
    if wasm_path.exists():
        return
    print(f"Building {INFERLET_NAME} (release)...", flush=True)
    subprocess.run(
        ["cargo", "build", "--target", "wasm32-wasip2", "--release"],
        cwd=inferlet_dir,
        check=True,
    )


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

async def run_once(
    client,
    *,
    prompt_tokens: int,
    decode_steps: int,
    sink: int,
    window: int,
    use_mask: bool,
    timeout: float,
) -> dict[str, str]:
    process = await client.launch_process(
        INFERLET_PKG,
        input={
            "prompt_tokens": prompt_tokens,
            "decode_steps": decode_steps,
            "sink_size": sink,
            "window_size": window,
            "use_mask": use_mask,
        },
    )
    parts: list[str] = []
    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise RuntimeError(f"timeout after {timeout}s")
        event, msg = await asyncio.wait_for(process.recv(), timeout=timeout)
        if event in (Event.Stdout, Event.Message):
            parts.append(msg)
        elif event == Event.Return:
            parts.append(msg)
            return parse_summary("".join(parts))
        elif event == Event.Error:
            raise RuntimeError(f"inferlet error: {msg}")


# ---------------------------------------------------------------------------
# Main bench
# ---------------------------------------------------------------------------

async def run_benchmark(args):
    from pie.server import Server
    from pie.config import (
        AuthConfig, Config, DriverConfig, ModelConfig, ServerConfig,
        TelemetryConfig,
    )

    wasm_path, manifest_path = find_paths()
    inferlet_dir = wasm_path.parent.parent.parent.parent
    ensure_built(wasm_path, inferlet_dir)

    if not manifest_path.exists():
        print(f"Error: missing Pie.toml at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else [args.device]
    driver_opts = build_driver_config(args, device)

    cfg = Config(
        server=ServerConfig(port=0, max_concurrent_processes=4),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models={
            "default": ModelConfig(
                name="default",
                hf_repo=args.model,
                default_token_budget=args.default_token_budget,
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    options=driver_opts,
                ),
            )
        },
    )

    prompt_lengths = [int(x) for x in args.prompt_tokens.split(",") if x.strip()]

    print(f"Model:          {args.model}")
    print(f"Device:         {device}")
    print(f"Driver:         {args.driver}")
    print(f"Decode steps:   {args.decode_steps}")
    print(f"Sink / window:  {args.sink_size} / {args.window_size}")
    print(f"Prompt sweep:   {prompt_lengths}")
    print()

    async with Server(cfg) as server:
        client = await server.connect()
        await client.install_program(wasm_path, manifest_path, force_overwrite=True)

        # Warm-up: discard the first run's numbers — JIT, cuda-graph capture,
        # FlashInfer plan caches all populate during the first forward pass.
        if not args.skip_warmup:
            print("Warm-up...", flush=True)
            await run_once(
                client,
                prompt_tokens=min(prompt_lengths),
                decode_steps=8,
                sink=args.sink_size,
                window=args.window_size,
                use_mask=True,
                timeout=args.timeout,
            )
            await run_once(
                client,
                prompt_tokens=min(prompt_lengths),
                decode_steps=8,
                sink=args.sink_size,
                window=args.window_size,
                use_mask=False,
                timeout=args.timeout,
            )

        header = (
            f"{'prompt':>8} {'mode':<10} {'prefill_ms':>11} "
            f"{'decode_ms':>10} {'ms/step':>8} {'tok/s':>8}"
        )
        print(header)
        print("-" * len(header))

        results: list[tuple[int, dict[str, dict[str, float]]]] = []
        for n_prompt in prompt_lengths:
            row: dict[str, dict[str, float]] = {}
            for mode_label, use_mask in (("with_mask", True), ("no_mask", False)):
                summary = await run_once(
                    client,
                    prompt_tokens=n_prompt,
                    decode_steps=args.decode_steps,
                    sink=args.sink_size,
                    window=args.window_size,
                    use_mask=use_mask,
                    timeout=args.timeout,
                )
                m = {
                    "prefill_ms": float(summary["prefill_ms"]),
                    "decode_ms": float(summary["decode_ms"]),
                    "ms_per_step": float(summary["decode_per_step_ms"]),
                    "tps": float(summary["decode_tokens_per_sec"]),
                }
                row[mode_label] = m
                print(
                    f"{n_prompt:>8} {mode_label:<10} {m['prefill_ms']:>11.2f} "
                    f"{m['decode_ms']:>10.2f} {m['ms_per_step']:>8.2f} {m['tps']:>8.1f}"
                )
            results.append((n_prompt, row))

        print()
        print(f"{'prompt':>8} {'with_mask ms/step':>20} {'no_mask ms/step':>18} {'speedup':>9}")
        print("-" * 60)
        for n_prompt, row in results:
            with_ms = row["with_mask"]["ms_per_step"]
            no_ms = row["no_mask"]["ms_per_step"]
            speedup = no_ms / with_ms if with_ms > 0 else float("nan")
            print(
                f"{n_prompt:>8} {with_ms:>20.3f} {no_ms:>18.3f} {speedup:>8.2f}x"
            )


def main():
    parser = argparse.ArgumentParser(description="Pie page-trim end-to-end benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--device", default="cuda:0", help="Device(s), comma-separated")
    parser.add_argument(
        "--prompt-tokens",
        default="512,2048,4096",
        help="Comma-separated list of prompt lengths (in tokens) to sweep",
    )
    parser.add_argument(
        "--decode-steps", type=int, default=128,
        help="Decode iterations per run (more = lower noise)",
    )
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--driver", default="native",
        choices=["native", "vllm", "sglang", "dummy"],
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.8)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--default-token-budget", type=int, default=200_000)
    parser.add_argument("--use-cuda-graphs", action="store_true")
    parser.add_argument("--vllm-attention-backend", default=None)
    parser.add_argument("--sglang-attention-backend", default=None)
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip the warm-up runs (faster but noisier first data point)",
    )

    args = parser.parse_args()
    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
