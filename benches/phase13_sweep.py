"""Phase 13 sweep — config-parity audit + per-step host overhead.

Boots one pie.server with PIE_LATENCY_LOG=1 and runs three concurrencies
(c=4, c=16, c=64) at max_tokens=200 over the throughput-shape workload
that Phase 11 used. Per-step CSV is written to /tmp/pie-latency.csv;
this harness collects tps + serializes a summary JSON per cell.

Sub-task A (config parity): we run TWO variants:
  * baseline  — pie's vllm-bridge default config (FLASHINFER pinned,
                FULL_DECODE_ONLY, max_num_batched_tokens widened to MPE)
  * parity    — strip the pie-specific overrides where possible
                without breaking pie correctness; see PHASE13_REPORT.md
                §1 for the parity matrix.

Sub-task B (host overhead): both variants log to PIE_LATENCY_LOG=1 so
the CSV captures `inter_call_gap_ms` per step. The harness aggregates
mean / p95 per cell.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# The dev image ships a pre-built `text_completion.wasm`; the bench
# variant is not built. We therefore use the standard `text_completion`
# inferlet plus chars/4 token estimation (same approach as
# `pie/benches/tput.py`). For Sub-task B's per-step measurements this
# difference is irrelevant — `inter_call_gap_ms` is read directly from
# the CSV, not derived from token counts.
WASM = Path("/opt/pie-runtime/inferlets/text_completion.wasm")
MANIFEST = Path("/workspace/pie-deva/inferlets/text-completion/Pie.toml")


def _build_config(model: str, max_tok: int, parity: bool):
    """Return a pie.config.Config tuned for this run.

    `parity=True` strips the pie-specific bridge overrides where pie
    correctness allows. The audit found that FLASHINFER + FULL_DECODE_ONLY
    are correctness-required when cap-on; the only remaining knobs that
    can be flipped without breaking pie are gpu_memory_utilization (align
    pie's 0.9 default to vllm-direct's 0.85 used in Phase 11) and
    max_num_seqs (align pie's 256 default to the cell concurrency).
    """
    from pie.config import (
        Config, ModelConfig, AuthConfig, RuntimeConfig, ServerConfig,
        DriverConfig, SchedulerConfig,
    )

    driver_options = {
        "gpu_memory_utilization": 0.85 if parity else 0.9,
        # max_num_seqs: pie default 256 vs vllm-direct's
        # max_num_seqs=concurrency. parity=True aligns to a small cap;
        # since we sweep multiple concurrencies on one boot, we use 64
        # (max concurrency in the sweep).
        "max_num_seqs": 64 if parity else 256,
        "enforce_eager": False,  # cap-on for both
    }
    return Config(
        server=ServerConfig(host="127.0.0.1", port=0, max_concurrent_processes=None),
        auth=AuthConfig(enabled=False),
        runtime=RuntimeConfig(allow_fs=False, allow_network=True),
        models={"default": ModelConfig(
            name="default",
            hf_repo=model,
            default_token_budget=max_tok + 256,
            default_endowment_pages=64,
            oversubscription_factor=4.0,
            scheduler=SchedulerConfig(policy="adaptive", request_timeout_secs=1200),
            driver=DriverConfig(
                type="vllm",
                device=["cuda:0"],
                options=driver_options,
            ),
        )},
    )


async def _run_cell(client, pkg, c, n, w, tok, label):
    from pie_client import Event

    async def one(idx):
        # Standard text_completion inferlet (no ignore_eos in v0.2.14
        # parameters); chat-template stop tokens may end early. We size
        # max_tokens generously and count tokens from the streamed
        # output (chars/4 approximation, matches tput.py).
        inp = {"prompt": f"Write a short story about a robot. (variant {idx})",
               "system": "You are a helpful benchmarking assistant.",
               "max_tokens": tok, "temperature": 0.0, "top_p": 1.0}
        start = time.perf_counter()
        try:
            pr = await client.launch_process(pkg, input=inp)
            req_chars = 0
            while True:
                ev, msg = await asyncio.wait_for(pr.recv(), timeout=1200)
                if ev == Event.Stdout:
                    req_chars += len(msg)
                    continue
                if ev == Event.Stderr:
                    continue
                if ev == Event.Return:
                    req_chars += len(msg)
                    return {"ok": True, "lat": time.perf_counter() - start,
                            "out": int(req_chars / 4.0)}
                if ev == Event.Error:
                    return {"ok": False, "lat": time.perf_counter() - start, "err": str(msg)[:100]}
        except Exception as e:
            return {"ok": False, "lat": time.perf_counter() - start, "err": f"{type(e).__name__}: {e}"}

    sem_w = asyncio.Semaphore(c)
    async def gw(i):
        async with sem_w: return await one(i)
    if w:
        await asyncio.gather(*(gw(i) for i in range(w)))

    sem = asyncio.Semaphore(c)
    async def gt(i):
        async with sem: return await one(w + i)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(gt(i) for i in range(n)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.get("ok")]
    lats = sorted([r["lat"] * 1000 for r in ok])
    out_tokens = sum(r.get("out", 0) for r in ok)
    return {
        "label": label,
        "concurrency": c, "num_requests": n, "warmup": w, "max_tokens": tok,
        "completed": len(ok), "wall_s": wall,
        "out_tokens": out_tokens, "tps": out_tokens / wall if wall else 0,
        "p50_ms": lats[len(lats) // 2] if lats else 0,
        "p95_ms": lats[int(0.95 * len(lats))] if lats else 0,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="baseline | parity")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cells", required=True,
                   help="c:N:warmup:tok,c:N:warmup:tok,...")
    p.add_argument("--latency-csv", default="/tmp/pie-latency.csv")
    args = p.parse_args()

    cells = [tuple(int(x) for x in c.split(":")) for c in args.cells.split(",")]
    max_tok = max(t for *_, t in cells)

    # Per-cell CSV split — give each cell its own latency CSV so
    # mean/p95 stats are clean. Set PIE_LATENCY_LOG_PATH dynamically by
    # symlink swapping the engine's open fh — but simpler: rotate the
    # whole file before each cell. Latency.py opens /tmp/pie-latency.csv
    # in append mode so a fresh file works.
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    parity = (args.label == "parity")
    cfg = _build_config(args.model, max_tok, parity)

    from pie.server import Server
    print(f"[{args.label}] booting pie.server (parity={parity})", file=sys.stderr)

    async with Server(cfg) as server:
        client = await server.connect()

        import tomllib
        pkg_meta = tomllib.loads(MANIFEST.read_text())["package"]
        pkg = f"{pkg_meta['name']}@{pkg_meta['version']}"
        await client.install_program(WASM, MANIFEST, force_overwrite=True)

        # First cell: 4 warmup → server warm-up
        for c, n, w, tok in cells:
            # Rotate the latency CSV so per-cell stats don't conflate.
            cell_csv = Path(args.out_dir) / f"{args.label}_c{c}_t{tok}_latency.csv"
            try:
                Path(args.latency_csv).unlink(missing_ok=True)
            except Exception:
                pass

            print(f"[{args.label}] cell c={c} N={n} w={w} tok={tok}", file=sys.stderr)
            summary = await _run_cell(client, pkg, c, n, w, tok, args.label)

            # Copy the per-cell latency CSV into the out-dir so it travels
            # with the JSON.
            try:
                Path(args.latency_csv).rename(cell_csv)
            except FileNotFoundError:
                pass

            out_path = Path(args.out_dir) / f"{args.label}_c{c}_t{tok}.json"
            out_path.write_text(json.dumps(summary, indent=2))
            print(f"[{args.label}] {out_path.name}: tps={summary['tps']:.1f} "
                  f"ok={summary['completed']}/{n} wall={summary['wall_s']:.1f}s "
                  f"p50={summary['p50_ms']:.0f}ms",
                  file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
