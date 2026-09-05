"""Throughput of the Rust / Python / JavaScript twins on a running server.

For each program and concurrency `n`, launches `n` processes at once with the
same input and reports wall time, aggregate output tokens/s and latency
percentiles, so a language-level performance regression shows up as a gap
between the three rows of one program. Output-token counts come from the
returned JSON (`count`). The three twins trace byte-identical containers, so
a gap here is a host round-trip or instance-startup cost, never a kernel.

    uv run python tests/inferlets/bench_twins.py --attach ws://127.0.0.1:8080 \
        --which text-completion,naive-baseline --langs rust,py,js --n 16,64

Attaches to a live server (start one with `pie serve`); artifacts must
already be built (`test_twins.py` builds them, or `PIE_INFERLETS_NO_BUILD`
is irrelevant here -- this script never builds).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

from pie_client import Event, PieClient

from conftest import find_artifact, inferlet_id

SUFFIX = {"rust": "", "py": "-py", "js": "-js"}

# Inputs each program takes; `count` in the returned JSON is what is measured.
INPUTS = {
    "text-completion": {"prompt": "The capital of France is"},
    "naive-baseline": {"prompt": "The capital of France is", "temperature": 0.7},
    "top-a-sampling": {"prompt": "The capital of France is"},
}


async def run_one(client, iid: str, args: dict, timeout: float) -> tuple[int, float]:
    t0 = time.perf_counter()
    proc = await client.launch_process(iid, input=args)
    parts: list[str] = []
    while True:
        ev, msg = await asyncio.wait_for(proc.recv(), timeout=timeout)
        if ev in (Event.Stdout, Event.Message, Event.Return):
            parts.append(msg if isinstance(msg, str) else msg.decode())
            if ev == Event.Return:
                break
        elif ev == Event.Error:
            raise RuntimeError(msg)
    out = "".join(parts)
    start = out.find("{")
    count = json.loads(out[start:])["count"] if start >= 0 else 0
    return count, time.perf_counter() - t0


async def bench(client, label: str, iid: str, n: int, args: dict, timeout: float) -> dict:
    t0 = time.perf_counter()
    res = await asyncio.gather(*[run_one(client, iid, args, timeout) for _ in range(n)], return_exceptions=True)
    wall = time.perf_counter() - t0
    errs = [r for r in res if isinstance(r, BaseException)]
    ok = [r for r in res if not isinstance(r, BaseException)]
    toks = sum(c for c, _ in ok)
    lats = sorted(lat for _, lat in ok)
    p50 = lats[len(lats) // 2] if lats else 0.0
    p99 = lats[int(len(lats) * 0.99)] if lats else 0.0
    per_req = statistics.mean(c / lat for c, lat in ok) if ok else 0.0
    print(
        f"{label:26s} n={n:3d}  ok={len(ok):3d}/{n:<3d} wall={wall:7.2f}s  tok/s={toks / wall:8.1f}  "
        f"lat p50={p50:6.2f}s p99={p99:6.2f}s  per-req tok/s={per_req:6.1f}"
    )
    for e in errs[:2]:
        print("   error:", str(e)[:300])
    return {"program": label, "n": n, "ok": len(ok), "wall": wall, "tokens": toks, "p50": p50, "p99": p99}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--attach", default="ws://127.0.0.1:8080", help="the running server")
    ap.add_argument("--which", default="text-completion,naive-baseline", help="programs, comma-separated")
    ap.add_argument("--langs", default="rust,py,js")
    ap.add_argument("--n", default="1,8,32", help="concurrency levels, comma-separated")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--json", action="store_true", help="also print the rows as JSON on stderr")
    a = ap.parse_args()
    ns = [int(x) for x in a.n.split(",")]
    rows = []
    async with PieClient(a.attach) as client:
        await client.authenticate("default")
        for which in a.which.split(","):
            for lang in a.langs.split(","):
                name = which + SUFFIX[lang]
                try:
                    wasm, manifest = find_artifact(name)
                except FileNotFoundError as e:
                    print(f"{name}: skipped ({e})")
                    continue
                await client.install_program(wasm, manifest, force_overwrite=True)
                iid = inferlet_id(manifest)
                args = {**INPUTS.get(which, {"prompt": "The capital of France is"}), "max_tokens": a.max_tokens}
                try:
                    await run_one(client, iid, args, a.timeout)  # warm-up: compile + program cache
                except Exception as e:
                    print(f"{name}: skipped ({str(e)[:160]})")
                    continue
                for n in ns:
                    for _ in range(a.repeat):
                        rows.append(await bench(client, name, iid, n, args, a.timeout))
    if a.json:
        print(json.dumps(rows), file=sys.stderr)
    return 0 if rows and all(r["ok"] == r["n"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
