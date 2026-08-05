#!/usr/bin/env python3
"""mlx-lm, through the same client every other engine here is measured by.

The reference Apple Silicon serves against. `driver/metal` already diffs
against mlx-lm twice -- `tests/parity/*_mlx_taps.py` compares activations and
`llama_bench`'s gates compare tokens -- but both ask whether the ANSWER agrees.
Neither says how long it took, and the throughput comparisons that decided
Metal's batching were run by hand and thrown away. This is that comparison,
kept.

Two things about mlx-lm's server the numbers depend on, stated rather than
discovered:

* **It has no `ignore_eos`.** Every other harness here pins output length so
  both engines do identical work; mlx-lm stops when the model does. Passing the
  field anyway would be worse than not supporting it -- the server ignores
  unknown keys, so the run would silently measure a different shape. So it is
  refused, and the refusal names `--no-ignore-eos`, which makes the *other*
  engine stop at EOS too. Throughput is still computed from the tokens actually
  produced, which `usage` reports.

* **The socket layer is not the engine.** `--concurrency` maps to mlx-lm's
  `--decode-concurrency`, the way it maps to vLLM's `max_num_seqs` next door.
  But mlx-lm serves from a `ThreadingHTTPServer` whose listen backlog is five
  deep and whose accept loop starves under load: fired all at once, 16 requests
  complete 10 here and 32 complete 8, the rest answered `Connection reset by
  peer`. A refused request was never admitted, so it is offered again rather
  than counted as a failure — and the retries are counted, because a run that
  spent itself at the accept queue is not the measurement it looks like.

The prompt cache is off by default here for the reason the other harnesses turn
it off: a closed-loop client that reuses prompts would otherwise measure the
cache.
"""
from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
import subprocess
import sys
import time
import urllib.error
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

# Where `tests/parity/requirements.txt` says the pinned mlx-lm lives. The same
# interpreter that produced every recorded gate answer, so a throughput number
# and a correctness answer come from one build.
PINNED_ENV = os.path.expanduser("~/.cache/pie-metal/mlxenv/bin/python")


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def refused(err: Exception) -> bool:
    """Was this a refusal at the socket, rather than an answer?

    Only these are worth offering again. A refused connection means the server
    never saw the request and did no work for it; an HTTP error or a timeout
    both mean it did, and retrying those would measure the engine twice and
    report it once.

    `ConnectionError` covers the reset, the refusal and the broken pipe that a
    starved accept loop produces. `EINVAL` is here because macOS reports it too
    when a socket is written to after the peer dropped it mid-accept — observed
    from this server at 128 requests, alongside the `EPIPE` for the same cause.
    """
    if isinstance(err, urllib.error.HTTPError):
        return False
    cause = err.reason if isinstance(err, urllib.error.URLError) else err
    if isinstance(cause, TimeoutError):
        return False
    if isinstance(cause, ConnectionError):
        return True
    return isinstance(cause, OSError) and cause.errno in (errno.EINVAL, errno.EPIPE)


def health(url: str) -> None:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=5.0) as resp:
        resp.read()


@asynccontextmanager
async def maybe_server(args: argparse.Namespace, batch: int):
    proc: subprocess.Popen[str] | None = None
    url = args.url
    if args.spawn:
        python = args.python or PINNED_ENV
        if not os.path.exists(python):
            raise RuntimeError(
                f"no interpreter at {python}. The pinned mlx-lm is what every "
                "recorded answer in this tree came from:\n"
                "  python3.14 -m venv ~/.cache/pie-metal/mlxenv\n"
                "  ~/.cache/pie-metal/mlxenv/bin/pip install -r "
                "driver/metal/tests/parity/requirements.txt\n"
                "or point --python at another one."
            )
        url = f"http://127.0.0.1:{args.port}"
        cmd = [
            python, "-m", "mlx_lm", "server",
            "--model", args.model,
            "--host", "127.0.0.1",
            "--port", str(args.port),
            "--decode-concurrency", str(batch),
            "--prompt-concurrency", str(args.prompt_concurrency),
            "--prompt-cache-size", str(args.prompt_cache_size),
            "--log-level", "ERROR",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + args.startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"mlx-lm server exited during startup:\n{out}")
            try:
                health(url)
                break
            except Exception:
                await asyncio.sleep(0.5)
        else:
            proc.terminate()
            raise RuntimeError(
                f"mlx-lm server did not answer /v1/models within {args.startup_timeout}s"
            )
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
    if args.ignore_eos:
        sys.exit(
            "mlx-lm's server has no `ignore_eos`, so it cannot be held to a fixed\n"
            "output length. Sending the field anyway would be measured as if it\n"
            "had been honoured. Run both engines with --no-ignore-eos."
        )
    n = args.requests if args.mode == "latency" else args.num_requests
    # `--concurrency` means the same thing here as it does next door: the
    # server's batch cap, with 0 reading as "no cap, so as wide as the run".
    if args.mode == "latency":
        batch = 1
    elif args.concurrency == 0:
        batch = max(1, args.num_requests)
    else:
        batch = args.concurrency
    prompts, prompt_counts = hf_chat_prompts_and_counts(
        args.model, args.system, make_prompts(args, n + args.warmup)
    )
    gate = asyncio.Semaphore(batch)
    retries = [0]
    async with maybe_server(args, batch) as base_url:
        endpoint = base_url.rstrip("/") + "/v1/completions"

        async def one(prompt: str, prompt_count: int) -> RequestResult:
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": False,
            }
            # In flight, not merely admitted. Beyond the server's batch there
            # is nothing to gain by holding a socket open.
            async with gate:
                start = time.perf_counter()
                for attempt in range(args.connect_retries + 1):
                    try:
                        obj = await asyncio.to_thread(
                            http_json, endpoint, payload, args.request_timeout
                        )
                        usage = obj.get("usage", {})
                        return RequestResult(
                            True,
                            time.perf_counter() - start,
                            int(usage.get("completion_tokens", 0)),
                            int(usage.get("prompt_tokens", prompt_count)),
                        )
                    except Exception as e:
                        # `ThreadingHTTPServer` accepts into a five-deep
                        # backlog, and under load its accept loop starves: 16
                        # sockets at once complete 10 here, 32 complete 8, the
                        # rest answered `Connection reset by peer`. Those
                        # requests were never admitted -- the server did no
                        # work for them -- so offering them again measures the
                        # engine rather than the accept queue. Bounded, and
                        # counted, because a run that spent itself retrying is
                        # not the same measurement as one that did not.
                        if attempt == args.connect_retries or not refused(e):
                            return RequestResult(
                                False, time.perf_counter() - start, 0,
                                error=f"{type(e).__name__}: {e}",
                            )
                        retries[0] += 1
                        # Exponential, because the starvation lasts as long as
                        # the burst does: a fixed wait clears 32 requests and
                        # not 128, which is how a constant tuned against one
                        # run becomes wrong on the next.
                        wait = min(
                            args.connect_backoff_ms * (2 ** attempt),
                            args.connect_backoff_cap_ms,
                        )
                        await asyncio.sleep(wait / 1000.0)
                raise AssertionError("unreachable: the loop returns on every path")

        for i in range(args.warmup):
            await one(prompts[i], prompt_counts[i])

        run_prompts = prompts[args.warmup:]
        run_counts = prompt_counts[args.warmup:]
        start = time.perf_counter()
        if args.mode == "latency":
            results = [await one(p, c) for p, c in zip(run_prompts, run_counts)]
        else:
            results = await asyncio.gather(
                *(one(p, c) for p, c in zip(run_prompts, run_counts))
            )
        wall = time.perf_counter() - start

    summary = summarize(
        mode=args.mode,
        engine="mlx-lm",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            # What the server was actually told to batch, and what the client
            # was allowed to have in flight. A tput number is only comparable
            # with another taken under the same pair.
            "decode_concurrency": batch,
            "client_in_flight": batch,
            "connect_retries_spent": retries[0],
            "prompt_concurrency": args.prompt_concurrency,
            "prompt_cache_size": args.prompt_cache_size,
            "ignore_eos": "unsupported by mlx-lm; every request stopped at EOS "
                          "or --max-tokens",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "unique_prompts": args.unique_prompts,
        },
    )
    return summary, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="mlx-lm canonical latency/throughput benchmark"
    )
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument("--url", default="http://127.0.0.1:8080")
        sp.add_argument(
            "--spawn",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Start the server here. --no-spawn measures one already running "
                 "at --url, whose flags this cannot then report.",
        )
        sp.add_argument("--port", type=int, default=8080)
        sp.add_argument(
            "--python",
            default=None,
            help=f"Interpreter with mlx-lm. Defaults to {PINNED_ENV}, the pinned "
                 "environment every recorded answer in this tree came from.",
        )
        sp.add_argument("--prompt-concurrency", type=int, default=8)
        sp.add_argument(
            "--connect-retries",
            type=int,
            default=12,
            help="How many times a request refused at the socket may be "
                 "offered again. mlx-lm's listen backlog is five deep and its "
                 "accept loop starves under load; a refused request was never "
                 "admitted, so retrying it measures the engine.",
        )
        sp.add_argument("--connect-backoff-ms", type=float, default=50.0)
        sp.add_argument("--connect-backoff-cap-ms", type=float, default=2000.0)
        sp.add_argument(
            "--prompt-cache-size",
            type=int,
            default=0,
            help="0 keeps the cache out of the measurement, which is what the "
                 "other harnesses do with theirs.",
        )
        sp.add_argument("--startup-timeout", type=float, default=300.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, results = asyncio.run(run(args))
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
