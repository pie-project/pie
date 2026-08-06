#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import Any

from common import (
    RequestResult,
    add_mode_subcommands,
    finish,
    hf_chat_prompts_and_counts,
    make_prompts,
    request_max_tokens,
    summarize,
)


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(url: str) -> None:
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5.0) as resp:
        resp.read()


@asynccontextmanager
async def maybe_server(args: argparse.Namespace, slot_ctx: int | None = None):
    proc: subprocess.Popen[str] | None = None
    url = args.url
    if args.server_bin:
        if not args.gguf_model:
            raise ValueError("--gguf-model is required with --server-bin")
        url = f"http://127.0.0.1:{args.port}"
        parallel = args.num_requests if args.mode == "tput" else 1
        cmd = [
            args.server_bin,
            "--model", args.gguf_model,
            "--host", "127.0.0.1",
            "--port", str(args.port),
            # llama.cpp splits `--ctx-size` ACROSS the parallel slots, so
            # passing the per-request context here gave each slot
            # `max_model_len / parallel` tokens -- 128 at the defaults, which
            # silently truncated every request in a tput run and made the
            # engine look like it stopped early. Scale by the slot count so a
            # slot gets the context it needs.
            #
            # What it NEEDS, not `max_model_len`. llama.cpp preallocates this
            # whole budget where pie and mlx-lm page theirs, so asking for
            # `max_model_len` a slot asks the machine for something the other
            # two never take: at the 16384 `three_way.py` passes, sixteen slots
            # is 262144 tokens of KV and the server dies with
            # `kIOGPUCommandBufferCallbackErrorOutOfMemory` before answering a
            # request. The workload's own widest prompt plus its output budget
            # is the honest number, capped by `max_model_len` so the flag still
            # means what it says.
            "--ctx-size", str(min(args.max_model_len, slot_ctx or args.max_model_len)
                              * parallel),
            "--parallel", str(parallel),
            "--n-gpu-layers", "all",
            # On, because the comparison is against engines running their own
            # optimized attention; turning llama.cpp's off benchmarks a
            # handicap rather than the engine.
            "--flash-attn", "on",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                health(url)
                break
            except Exception:
                await asyncio.sleep(0.5)
        else:
            proc.terminate()
            raise RuntimeError("llama.cpp server did not become healthy")
    else:
        health(url)
    try:
        yield url
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


async def run(args: argparse.Namespace):
    n = args.requests if args.mode == "latency" else args.num_requests
    prompts, prompt_counts = hf_chat_prompts_and_counts(
        args.model, args.system, make_prompts(args, n + args.warmup)
    )
    # `--concurrency` has to mean the same thing it means to the harnesses it is
    # compared against, and here it meant nothing: the client gathered every
    # request at once and only the server's `--parallel` bounded the batch. A
    # row labelled "concurrency 1" was therefore eight-way concurrent for
    # llama.cpp and serial for the other two -- the engines were not being asked
    # the same question. 0 keeps the old behaviour, which is "no cap".
    gate = asyncio.Semaphore(args.concurrency) if getattr(args, "concurrency", 0) else None

    # Rounded up to a power of two so a prompt a few tokens longer does not
    # re-tune the server, and floored so a tiny workload still leaves room.
    needed = max(prompt_counts) + args.max_tokens + 64
    slot_ctx = 512
    while slot_ctx < needed:
        slot_ctx *= 2
    async with maybe_server(args, slot_ctx) as base_url:
        endpoint = base_url.rstrip("/") + "/v1/completions"

        async def one(prompt: str, prompt_count: int, max_tokens: int) -> RequestResult:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "ignore_eos": args.ignore_eos,
                "cache_prompt": False,
                "stream": False,
                "stream_options": {"include_usage": True},
            }
            if gate is not None:
                async with gate:
                    return await send(payload, prompt_count)
            return await send(payload, prompt_count)

        async def send(payload: dict[str, Any], prompt_count: int) -> RequestResult:
            start = time.perf_counter()
            try:
                obj = await asyncio.to_thread(http_json, endpoint, payload, args.request_timeout)
                usage = obj.get("usage", {})
                return RequestResult(
                    True,
                    time.perf_counter() - start,
                    int(usage.get("completion_tokens", 0)),
                    int(usage.get("prompt_tokens", prompt_count)),
                )
            except Exception as e:
                return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")

        # Indexed by ABSOLUTE prompt index and sliced with the same
        # `[warmup:]` slice as `prompts`; see the note in `mlx_bench.py`. A
        # flat budget here against pie's per-request one compared two
        # different amounts of work under every unequal-budget shape.
        budgets = [request_max_tokens(args, i) for i in range(len(prompts))]

        for i in range(args.warmup):
            await one(prompts[i], prompt_counts[i], budgets[i])

        run_prompts = prompts[args.warmup:]
        run_counts = prompt_counts[args.warmup:]
        run_budgets = budgets[args.warmup:]
        start = time.perf_counter()
        if args.mode == "latency":
            results = [
                await one(p, c, m)
                for p, c, m in zip(run_prompts, run_counts, run_budgets)
            ]
        else:
            results = await asyncio.gather(
                *(one(p, c, m)
                  for p, c, m in zip(run_prompts, run_counts, run_budgets))
            )
        wall = time.perf_counter() - start

    summary = summarize(
        mode=args.mode,
        engine="llamacpp",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "cache_prompt": False,
            "flash_attn": "on when spawned by benches",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
        },
    )
    return summary, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="llama.cpp canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument("--url", default="http://127.0.0.1:8080")
        sp.add_argument("--server-bin", default=None)
        sp.add_argument("--gguf-model", default=None)
        sp.add_argument("--port", type=int, default=8080)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, results = asyncio.run(run(args))
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
