#!/usr/bin/env python3
"""Pie multimodal (image) latency/throughput benchmark.

The Pie-side counterpart to `vllm_mm_bench.py`. Launches `pie serve` with the
`cuda_native` driver, installs the `image-qa-bench` inferlet, and drives it with
a fixed local image + question per request. The inferlet returns exact prompt
and output token counts (prompt = text tokens + image soft-token rows), so the
accounting matches vLLM's.

Timed path mirrors vLLM's: host-side decode/resize/patchify + vision encode +
text prefill + decode. The image arrives base64 in the launch input — no network
fetch in the measured window.

Modes (from common.add_mode_subcommands):
  latency  one request at a time (max_concurrent_processes = 1).
  tput     all requests launched concurrently (capped by --concurrency).

Run:
  python pie_mm_bench.py latency --model Qwen/Qwen3-VL-2B-Instruct \
      --image assets/bench_image.png --requests 16 --max-tokens 128 \
      --json-out out/pie_latency.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import socket
import sys
import time
from pathlib import Path

from common import (
    ROOT,
    RequestResult,
    add_mode_subcommands,
    cuda_profiler_start,
    cuda_profiler_stop,
    finish,
    summarize,
)

SERVER_SDK = ROOT / "sdk" / "python-server" / "python"
if str(SERVER_SDK) not in sys.path:
    sys.path.insert(0, str(SERVER_SDK))

BENCH_INFERLET = "image-qa-bench"
WASM_NAME = "image_qa_bench.wasm"


def bench_inferlet_paths() -> tuple[Path, Path, str]:
    import tomllib

    d = ROOT / "inferlets" / BENCH_INFERLET
    wasm = d / "target" / "wasm32-wasip2" / "release" / WASM_NAME
    manifest = d / "Pie.toml"
    if not wasm.exists():
        raise FileNotFoundError(
            f"missing {wasm}; build with: cd {d} && "
            "cargo build --target wasm32-wasip2 --release"
        )
    pkg = tomllib.loads(manifest.read_text())["package"]
    return wasm, manifest, f"{pkg['name']}@{pkg['version']}"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_config(args: argparse.Namespace, port: int):
    from pie.config import (
        AuthConfig,
        Config,
        DriverConfig,
        ModelConfig,
        RuntimeConfig,
        SchedulerConfig,
        ServerConfig,
        TelemetryConfig,
    )

    device = [d.strip() for d in args.device.split(",")]
    if args.mode == "latency":
        max_concurrent: int | None = 1
    else:
        max_concurrent = None if args.concurrency == 0 else args.concurrency

    driver_options = {
        "gpu_mem_utilization": args.gpu_mem_util,
        "ready_timeout_s": float(args.server_startup_timeout),
    }
    cfg = Config(
        server=ServerConfig(
            host="127.0.0.1",
            port=port,
            verbose=True,
            max_concurrent_processes=max_concurrent,
        ),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        runtime=RuntimeConfig(
            wasm_max_instances=max(4096, (args.num_requests + args.warmup) * 4),
        ),
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                scheduler=SchedulerConfig(
                    default_token_limit=args.default_token_limit,
                    default_endowment_pages=args.endowment_pages,
                    admission_oversubscription_factor=args.admission_oversubscription_factor,
                    **({"speculation_depth": args.speculation_depth}
                       if args.speculation_depth is not None else {}),
                ),
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    tensor_parallel_size=args.tp_size,
                    activation_dtype=args.activation_dtype,
                    options=driver_options,
                ),
            )
        ],
    )
    config_blob = {
        "modality": "image",
        "image": args.image,
        "driver": args.driver,
        "endowment_pages": args.endowment_pages,
        "gpu_mem_utilization": args.gpu_mem_util,
        "max_concurrent_processes": max_concurrent,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ignore_eos": args.ignore_eos,
        "unique_prompts": args.unique_prompts,
    }
    return cfg, config_blob


async def launch_server(args: argparse.Namespace):
    """Start `pie serve`, return (proc, port, token, drain_task, lines)."""
    port = find_free_port()
    cfg, config_blob = build_config(args, port)
    cfg_path = ROOT / ".tmp" / "benches" / f"pie-mm-{port}.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg.to_toml())

    pie_bin = Path(args.pie_bin)
    if not pie_bin.exists():
        raise FileNotFoundError(
            f"missing {pie_bin}; build with: cargo build -p pie-worker --release "
            "--no-default-features --features driver-cuda"
        )

    # Force flashinfer split-KV (flash-decoding) on for small-batch decode: the
    # driver's default heuristic disables it for batch<=512 on sm>=8, which makes
    # long-KV (multimodal) single-stream decode scale linearly with context. With
    # split-KV the long-KV decode stays flat, matching vLLM/SGLang.
    serve_env = {**os.environ, "PIE_FLASHINFER_FORCE_SPLIT_KV_SMALL": "1"}
    proc = await asyncio.create_subprocess_exec(
        str(pie_bin), "serve", "--config", str(cfg_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=serve_env,
    )
    lines: list[str] = []

    def surface(txt: str) -> bool:
        return (" WARN " in txt or " ERROR " in txt or "panic" in txt
                or "[vtim]" in txt or "[fire]" in txt)

    # Read startup until the direct client server is ready.
    server_ready = False
    deadline = time.perf_counter() + args.server_startup_timeout
    assert proc.stdout is not None
    while time.perf_counter() < deadline:
        try:
            line = await asyncio.wait_for(
                proc.stdout.readline(), timeout=max(0.1, deadline - time.perf_counter())
            )
        except asyncio.TimeoutError:
            break
        if not line:
            raise RuntimeError("pie serve exited before startup:\n" + "".join(lines[-60:]))
        txt = line.decode("utf-8", errors="replace")
        lines.append(txt)
        if surface(txt):
            sys.stderr.write(txt)
            sys.stderr.flush()
        if "Server ready at ws://" in txt:
            server_ready = True
            break
        if proc.returncode is not None:
            raise RuntimeError(f"pie serve exited {proc.returncode}:\n" + "".join(lines[-60:]))
    if not server_ready:
        raise TimeoutError("timed out waiting for pie serve startup:\n" + "".join(lines[-60:]))

    async def drain() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            txt = line.decode("utf-8", errors="replace")
            lines.append(txt)
            del lines[:-200]
            if surface(txt):
                sys.stderr.write(txt)
                sys.stderr.flush()

    drain_task = asyncio.create_task(drain())
    return proc, port, drain_task, config_blob


async def run(args: argparse.Namespace):
    from pie_client import Event, PieClient

    image_b64 = base64.b64encode(Path(args.image).read_bytes()).decode("ascii")
    n = args.requests if args.mode == "latency" else args.num_requests
    total = n + args.warmup

    def question(i: int) -> str:
        return f"{args.question} (Request #{i})" if args.unique_prompts else args.question

    def make_input(i: int, *, max_tokens: int | None = None) -> dict:
        return {
            "image_b64": image_b64,
            "question": question(i),
            "system": args.system,
            "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "return_text": args.dump_first_text,
        }

    wasm, manifest, pkg = bench_inferlet_paths()
    proc, port, drain_task, config_blob = await launch_server(args)
    first_text: list[str | None] = [None]
    sem = asyncio.Semaphore(args.concurrency) if (args.mode == "tput" and args.concurrency > 0) else None

    try:
        client = PieClient(f"ws://127.0.0.1:{port}")
        await client.connect()
        try:
            await client.install_program(wasm, manifest, force_overwrite=True)

            async def one(i: int, *, max_tokens: int | None = None) -> RequestResult:
                start = time.perf_counter()
                try:
                    p = await client.launch_process(
                        pkg, input=make_input(i, max_tokens=max_tokens),
                    )
                    while True:
                        ev, msg = await asyncio.wait_for(p.recv(), timeout=args.request_timeout)
                        if ev == Event.Return:
                            obj = json.loads(msg)
                            if first_text[0] is None:
                                first_text[0] = obj.get("text", "")
                            return RequestResult(
                                True, time.perf_counter() - start,
                                int(obj["num_output_tokens"]), int(obj["num_prompt_tokens"]),
                            )
                        if ev == Event.Error:
                            return RequestResult(False, time.perf_counter() - start, 0, error=str(msg))
                except Exception as e:
                    return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")

            async def one_capped(i: int) -> RequestResult:
                if sem is None:
                    return await one(i)
                async with sem:
                    return await one(i)

            # Warmup (not measured).
            if args.warmup:
                wmt = args.warmup_max_tokens or args.max_tokens
                if args.mode == "tput":
                    await asyncio.gather(*(one(i, max_tokens=wmt) for i in range(args.warmup)))
                else:
                    for i in range(args.warmup):
                        await one(i, max_tokens=wmt)

            cuda_profiler_start(args.cuda_profiler_capture)
            start = time.perf_counter()
            try:
                if args.mode == "latency":
                    results = [await one(args.warmup + i) for i in range(n)]
                else:
                    results = await asyncio.gather(
                        *(one_capped(args.warmup + i) for i in range(n))
                    )
            finally:
                wall = time.perf_counter() - start
                cuda_profiler_stop(args.cuda_profiler_capture)

            # Diagnostic: did the scheduler coalesce concurrent contexts into
            # batched forward passes, and where does per-batch time go?
            #   avg_batch_latency_us ≈ time INSIDE the forward pass (driver).
            #   wall/total_batches   ≈ time per batch INCLUDING gaps (runtime).
            # If wall/batch >> avg_batch_latency_us, the cost is between passes
            # (dispatch / per-process round-trips), not in the GPU forward pass.
            try:
                ok, body = await client.query("model_status", "")
                if ok:
                    ms = json.loads(body)
                    Path("out/mm").mkdir(parents=True, exist_ok=True)
                    Path("out/mm/pie_model_status.json").write_text(json.dumps(ms, indent=2))
                    for k in (
                        "default.total_batches",
                        "default.total_requests_processed",
                        "default.max_forward_requests_observed",
                        "default.batch_size_hist",
                        "default.avg_batch_latency_us",
                        "default.last_batch_latency_us",
                        "default.cumulative_batch_latency_us",
                    ):
                        if k in ms:
                            sys.stderr.write(f"[stat] {k} = {ms[k]}\n")
                    tb = ms.get("default.total_batches") or 0
                    cum = ms.get("default.cumulative_batch_latency_us") or 0
                    if tb:
                        sys.stderr.write(
                            f"[stat] wall/batch = {wall / tb * 1e3:.1f} ms | "
                            f"driver/batch = {cum / tb / 1e3:.1f} ms | "
                            f"gap/batch = {(wall - cum / 1e6) / tb * 1e3:.1f} ms\n"
                        )
                    sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[stat] model_status failed: {e}\n")
        finally:
            await client.close()
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task

    if args.dump_first_text and first_text[0]:
        print("\n--- first output ---\n" + first_text[0].strip() + "\n--------------------")

    summary = summarize(
        mode=args.mode, engine="pie", model=args.model,
        results=results, wall_s=wall, config={**config_blob, "pie_bin": str(args.pie_bin)},
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pie multimodal (image) benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument("--image", default="assets/bench_image.png",
                        help="Local image fed (base64) to every request.")
        sp.add_argument("--question",
                        default="What is in this image? Answer in one sentence.")
        sp.add_argument("--driver", default="cuda_native")
        sp.add_argument("--device", default="cuda:0")
        sp.add_argument("--activation-dtype", default="bfloat16")
        sp.add_argument("--endowment-pages", type=int, default=64,
                        help="Initial KV page grant per context (covers image soft tokens).")
        sp.add_argument("--default-token-limit", type=int, default=200_000,
                        help="Scheduler token-admission limit (match pie_bench).")
        sp.add_argument("--admission-oversubscription-factor", type=float, default=4.0,
                        help="Scheduler admission oversubscription (match pie_bench).")
        sp.add_argument("--speculation-depth", type=int, default=None,
                        help="Pass-level chain-firing depth (hides guest<->host per-step round-trip).")
        sp.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
        sp.add_argument("--server-startup-timeout", type=float, default=300.0)
        sp.add_argument("--dump-first-text", action="store_true",
                        help="Decode + print the first request's output (spot-check).")
    args = parser.parse_args()
    if args.model == "Qwen/Qwen3-0.6B":
        args.model = "Qwen/Qwen3-VL-2B-Instruct"
    summary, results = asyncio.run(run(args))
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
