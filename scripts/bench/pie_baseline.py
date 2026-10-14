#!/usr/bin/env python3
"""pie at the fixed points `vllm_baseline.py` holds vLLM to.

    offline   prefill: 32 prompts x 4096 tokens, 1 output token, one batch
              decode:  batch 1, 8, 64; 512-token prompts, 256 output tokens,
                       timed once with 256 outputs and once with 1, so the
                       per-step decode time is (t256 - t1) / 255
    serve     256 requests, 1024 in / 256 out, every one offered at once

One `pie serve` boot (`pie_bench.cli_pie_client`, with `pie_bench`'s own
config) serves every point. A request is one `text-completion-bench` process
fed random token ids (`common.fixed_prompt_ids`, seeded as the vLLM baseline
seeds it), greedy, `ignore_eos`, with no prefix cache: every prompt is new, as
the vLLM baseline's fresh ids make it. A batch is that many processes launched
at once and timed from the first launch to the last return. Each offline
point is warmed once and timed `--iters` times; the median is reported. The
serving burst is timed once after a one-request warmup, as `vllm bench serve`
runs its one test prompt first; TTFT is the client's stamp on the guest's
first-token message, TPOT per request is (latency - TTFT) / (outputs - 1).

pie's serving client speaks the engine's own process protocol, not OpenAI
HTTP: what vLLM's numbers spend on HTTP, SSE and detokenization, pie's do not.

    python scripts/bench/pie_baseline.py offline --model ARTIFACT.zt --hf SNAPSHOT --out DIR
    python scripts/bench/pie_baseline.py serve   --model ARTIFACT.zt --hf SNAPSHOT --out DIR

Run with an interpreter that has `pie_client`'s dependencies
(websockets, msgpack, blake3); `packages/python/client/src` is put on the path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from common import (
    FIXED_DECODE_BATCHES,
    FIXED_DECODE_INPUT,
    FIXED_DECODE_OUTPUT,
    FIXED_PREFILL,
    FIXED_SERVE,
    ROOT,
    MemSampler,
    fixed_prompt_ids,
    percentile,
    tokenizer_vocab_size,
)

CLIENT_SRC = ROOT / "packages" / "python" / "client" / "src"
if str(CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(CLIENT_SRC))

import pie_bench  # noqa: E402


def bench_args(args: argparse.Namespace) -> argparse.Namespace:
    """`pie_bench`'s own argument namespace for this run, so the server is
    configured by `pie_bench.build_config` and nothing here restates it."""
    argv = [
        "tput", "--model", args.model,
        "--engine", "cuda_native",
        "--max-model-len", str(args.max_model_len),
        "--gpu-mem-util", str(args.gpu_mem_util),
        "--inferlet-dir", args.inferlet_dir,
        "--pie-bin", args.pie_bin,
        "--server-startup-timeout", str(args.startup_timeout),
        "--cpu-affinity", "none",
    ]
    for opt in args.engine_option:
        argv += ["--engine-option", opt]
    if getattr(args, "precision", None):
        argv += ["--precision", args.precision]
    if getattr(args, "frame_dispatch_depth", None):
        argv += ["--frame-dispatch-depth", str(args.frame_dispatch_depth)]
    if getattr(args, "device_weight_budget", None):
        argv += ["--device-weight-budget", args.device_weight_budget]
    return pie_bench.build_parser().parse_args(argv)


@contextlib.asynccontextmanager
async def pie_session(args: argparse.Namespace, server_log: str):
    """One `pie serve` boot; yields `(burst, engine_config)`, where
    `await burst(prompts, output_len, timing)` runs one process a prompt, all
    launched at once, and returns the seconds from the first launch to the
    last return with each request's latency, TTFT and output count."""
    from pie_client import Event

    wasm = pie_bench.bench_inferlet_wasm(args.inferlet_dir)
    os.environ["PIE_BENCH_SERVER_LOG"] = server_log
    async with pie_bench.cli_pie_client(bench_args(args)) as (client, engine_config):
        pkg = await client.install_program(wasm, force_overwrite=True)

        async def one(ids: list[int], output_len: int, timing: bool) -> dict:
            start = time.perf_counter()
            proc = await client.launch_process(pkg, input={
                "prompt": "", "prompt_tokens": ids, "max_tokens": output_len,
                "temperature": 0.0, "top_p": 1.0, "ignore_eos": True,
                "return_text": False, "no_prefix_cache": True, "report_timing": timing,
            })
            ttft = None
            while True:
                ev, msg = await asyncio.wait_for(proc.recv(), timeout=args.request_timeout)
                if ev == Event.Message and str(msg) == "t0":
                    ttft = time.perf_counter() - start
                elif ev == Event.Return:
                    obj = json.loads(msg)
                    out = int(obj.get("num_output_tokens", 0))
                    if out != output_len:
                        raise RuntimeError(f"{out} output tokens, {output_len} asked")
                    return dict(latency_s=time.perf_counter() - start, ttft_s=ttft, output_tokens=out)
                elif ev == Event.Error:
                    raise RuntimeError(f"request failed: {msg}")

        async def burst(prompts: list[list[int]], output_len: int, timing: bool = False) -> tuple[float, list[dict]]:
            t0 = time.perf_counter()
            reqs = await asyncio.gather(*(one(ids, output_len, timing) for ids in prompts))
            return time.perf_counter() - t0, reqs

        yield burst, engine_config


async def run(args: argparse.Namespace, mem: MemSampler) -> dict:
    vocab = tokenizer_vocab_size(args.hf)
    rng = random.Random(0)
    async with pie_session(args, os.path.join(args.out, f"{args.mode}_engine.log")) as (burst, engine_config):
        mem.measuring()

        async def timed(batch: int, input_len: int, output_len: int) -> dict:
            await burst(fixed_prompt_ids(rng, vocab, batch, input_len), output_len)
            times = []
            for _ in range(args.iters):
                seconds, _ = await burst(fixed_prompt_ids(rng, vocab, batch, input_len), output_len)
                times.append(seconds)
            return dict(batch=batch, input_len=input_len, output_len=output_len,
                        seconds=times, median_s=statistics.median(times))

        res: dict = dict(engine_config=engine_config)
        if args.mode == "offline":
            pf = await timed(**FIXED_PREFILL)
            pf["prefill_tok_s"] = FIXED_PREFILL["batch"] * FIXED_PREFILL["input_len"] / pf["median_s"]
            res["prefill"] = pf
            res["decode"] = []
            for b in FIXED_DECODE_BATCHES:
                full = await timed(b, FIXED_DECODE_INPUT, FIXED_DECODE_OUTPUT)
                first = await timed(b, FIXED_DECODE_INPUT, 1)
                step = (full["median_s"] - first["median_s"]) / (FIXED_DECODE_OUTPUT - 1)
                res["decode"].append(dict(
                    batch=b, e2e=full, prefill_only=first,
                    tpot_ms=step * 1e3,
                    decode_tok_s=b / step,
                    e2e_output_tok_s=b * FIXED_DECODE_OUTPUT / full["median_s"],
                ))
        else:
            n, inp, out = FIXED_SERVE["num_prompts"], FIXED_SERVE["input_len"], FIXED_SERVE["output_len"]
            await burst(fixed_prompt_ids(rng, vocab, 1, inp), out, timing=True)
            duration, reqs = await burst(fixed_prompt_ids(rng, vocab, n, inp), out, timing=True)
            ttft = [r["ttft_s"] * 1e3 for r in reqs if r["ttft_s"] is not None]
            tpot = [(r["latency_s"] - r["ttft_s"]) * 1e3 / (out - 1) for r in reqs if r["ttft_s"] is not None]
            e2el = [r["latency_s"] * 1e3 for r in reqs]
            stats = lambda name, xs: {  # noqa: E731
                f"mean_{name}_ms": statistics.fmean(xs), f"median_{name}_ms": statistics.median(xs),
                f"p99_{name}_ms": percentile(xs, 0.99)}
            res["result"] = dict(
                duration=duration, completed=len(reqs),
                total_input_tokens=n * inp, total_output_tokens=sum(r["output_tokens"] for r in reqs),
                request_throughput=n / duration,
                output_throughput=sum(r["output_tokens"] for r in reqs) / duration,
                total_token_throughput=(n * inp + sum(r["output_tokens"] for r in reqs)) / duration,
                **stats("ttft", ttft), **stats("tpot", tpot), **stats("e2el", e2el),
            )
        return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["offline", "serve"])
    ap.add_argument("--model", required=True, help="what `[model] model` serves: an artifact path or store name")
    ap.add_argument("--hf", required=True, help="the checkpoint snapshot, for its tokenizer's vocabulary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--startup-timeout", type=float, default=3600)
    ap.add_argument("--inferlet-dir", default=str(ROOT / "examples" / "text-completion-bench"))
    ap.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
    ap.add_argument("--engine-option", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--precision", default=None, help="[model] precision, e.g. bf16+nvfp4a4")
    ap.add_argument("--frame-dispatch-depth", type=int, default=None,
                    help="[runtime] frame_dispatch_depth: frames a lane keeps posted (its state slots)")
    ap.add_argument("--device-weight-budget", default=None,
                    help="[model] device_weight_budget, e.g. 80GiB: what of the weights stays on the device")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with MemSampler(os.path.join(args.out, f"gpu_mem_{args.mode}.csv")) as mem:
        t0 = time.time()
        res = asyncio.run(run(args, mem))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    res.update(mode=args.mode, model=args.model, hf=args.hf, argv=shlex.join(sys.argv), pie_commit=commit,
               wall_s=time.time() - t0, peak_gpu_mem_mib=mem.peak,
               contended=bool(mem.foreign), foreign_pids=sorted(mem.foreign), foreign_samples=mem.foreign_samples,
               foreign_pids_at_load=sorted(mem.foreign_at_load))
    path = os.path.join(args.out, f"{args.mode}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "engine_config"}, indent=2)[:4000])
    print("wrote", path)


if __name__ == "__main__":
    main()
