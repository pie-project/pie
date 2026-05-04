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
async def maybe_server(args: argparse.Namespace):
    proc: subprocess.Popen[str] | None = None
    url = args.url
    if args.server_bin:
        if not args.gguf_model:
            raise ValueError("--gguf-model is required with --server-bin")
        url = f"http://127.0.0.1:{args.port}"
        cmd = [
            args.server_bin,
            "--model", args.gguf_model,
            "--host", "127.0.0.1",
            "--port", str(args.port),
            "--ctx-size", str(args.max_model_len),
            "--parallel", str(args.concurrency),
            "--n-gpu-layers", "all",
            "--flash-attn", "off",
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
    async with maybe_server(args) as base_url:
        endpoint = base_url.rstrip("/") + "/v1/completions"

        async def one(prompt: str, prompt_count: int) -> RequestResult:
            payload = {
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "ignore_eos": args.ignore_eos,
                "cache_prompt": False,
                "stream": False,
                "stream_options": {"include_usage": True},
            }
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

        for i in range(args.warmup):
            await one(prompts[i], prompt_counts[i])

        run_prompts = prompts[args.warmup:]
        run_counts = prompt_counts[args.warmup:]
        start = time.perf_counter()
        if args.mode == "latency":
            results = [await one(p, c) for p, c in zip(run_prompts, run_counts)]
        else:
            sem = asyncio.Semaphore(args.concurrency)

            async def guarded(p: str, c: int) -> RequestResult:
                async with sem:
                    return await one(p, c)

            results = await asyncio.gather(*(guarded(p, c) for p, c in zip(run_prompts, run_counts)))
        wall = time.perf_counter() - start

    summary = summarize(
        mode=args.mode,
        engine="llamacpp",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "cache_prompt": False,
            "flash_attn": "off when spawned by benches",
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
