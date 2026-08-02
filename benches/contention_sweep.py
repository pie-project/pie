#!/usr/bin/env python3
"""Run pie and vLLM over the contention cells and report the ratio per cell.

Adds no measurement of its own: it invokes `pie_bench.py` and `vllm_bench.py`,
which already agree on a client and on `common.py`'s prompt and output-budget
construction, and collects the JSON each writes.

Two things this exists to get right, both of which are easy to get wrong by
hand and were, repeatedly, before it existed.

**Order.** Box variance between runs is a few percent, comfortably larger than
most effects worth chasing, and it drifts. Engines therefore alternate which
goes first every round (ABBA), and per-round ratios are printed alongside the
mean so a result carried by one outlier is visible instead of averaged away.

**Output-length uniformity.** By default every request gets the same
`--max-tokens`, and that single property dominates the comparison: it is the
one condition under which continuous batching packs perfectly and never
re-plans. Measured on 4096x64 @c256, pie/vLLM is 0.96 at uniform length and
1.15 once `--output-spread` fans the budgets — because vLLM loses ~25% the
moment lengths vary at all while pie loses ~12%. `--spread` therefore runs BOTH
and reports them side by side; a contention number quoted from only the uniform
column is describing a shape, not an engine.

`common.py:request_max_tokens` fans the budget on a deterministic sawtooth
whose mean over each period is exactly 1, so the total output-token count is
preserved and both engines are handed identical per-request budgets. The
same-work guard in each harness still applies.

Example:

    python contention_sweep.py --out /tmp/sweep --spread 0.5 --reps 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL = "Qwen/Qwen3-0.6B"
BLOCK = 16  # tokens per KV page/block, applied to both engines

# The four contention cells, named `(num_requests x max_tokens)` the way
# `.wiki/contention/analysis.md` names them. Concurrency is 256 on every cell.
#
# `X` carries a shared instruction prefix, which is the one cell whose ratio
# depends on prefix handling; vLLM's automatic prefix caching measures the same
# with it on or off on this shape, so `--no-prefix-caching` below is not
# distorting it.
CELLS: dict[str, dict] = {
    "c256": {"num_requests": 256, "concurrency": 256, "max_tokens": 256},
    "S": {"num_requests": 4096, "concurrency": 256, "max_tokens": 64},
    "D": {"num_requests": 768, "concurrency": 256, "max_tokens": 512},
    "X": {
        "num_requests": 1024,
        "concurrency": 256,
        "max_tokens": 128,
        "shared_prefix_words": 768,
    },
}


def common_args(cell: dict, max_model_len: int, spread: float) -> list[str]:
    args = [
        "tput",
        "--model", MODEL,
        "--num-requests", str(cell["num_requests"]),
        "--concurrency", str(cell["concurrency"]),
        "--max-tokens", str(cell["max_tokens"]),
        "--max-model-len", str(max_model_len),
        "--warmup", "2",
    ]
    if cell.get("shared_prefix_words"):
        args += ["--shared-prefix-words", str(cell["shared_prefix_words"])]
    if spread > 0.0:
        args += ["--output-spread", str(spread)]
    return args


def pie_cmd(cell, out, max_model_len, pages, swap, spread, python) -> list[str]:
    return [
        python, str(HERE / "pie_bench.py"),
        *common_args(cell, max_model_len, spread),
        "--driver", "cuda_native",
        "--total-pages", str(pages),
        "--swap-pool-size", str(swap),
        "--inferlet-dir", str(ROOT / "tests/inferlets/text-completion-bench"),
        "--json-out", str(out),
    ]


def vllm_cmd(cell, out, max_model_len, pages, spread, python) -> list[str]:
    # Same KV budget by exact block count, so neither engine is sized against a
    # different pool: pie takes `--total-pages`, vLLM takes the override.
    return [
        python, str(HERE / "vllm_bench.py"),
        *common_args(cell, max_model_len, spread),
        "--num-gpu-blocks-override", str(pages),
        "--block-size", str(BLOCK),
        "--no-prefix-caching",
        "--json-out", str(out),
    ]


def run_one(engine, cell_name, cell, out_dir, rep, spread, args) -> dict | None:
    tag = f"{cell_name}_{engine}_s{spread}_r{rep}"
    out = out_dir / f"{tag}.json"
    log = out_dir / f"{tag}.log"
    if engine == "pie":
        cmd = pie_cmd(cell, out, args.max_model_len, args.pages,
                      args.swap_pool_size, spread, args.pie_python)
    else:
        cmd = vllm_cmd(cell, out, args.max_model_len, args.pages, spread,
                       args.vllm_python)

    env = dict(os.environ)
    if args.cuda_compat:
        env["LD_LIBRARY_PATH"] = (
            args.cuda_compat + ":" + env.get("LD_LIBRARY_PATH", "")
        )
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=HERE, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, timeout=args.timeout)
    elapsed = time.time() - t0
    if proc.returncode != 0 or not out.exists():
        print(f"    !! {engine:5s} {cell_name:5s} spread={spread} r{rep} "
              f"FAILED rc={proc.returncode} ({elapsed:.0f}s) -> {log}",
              flush=True)
        return None

    s = json.loads(out.read_text())["summary"]
    print(f"    {engine:5s} {cell_name:5s} spread={spread} r{rep}  "
          f"{s['output_tok_per_s']:9.1f} tok/s  "
          f"ok={s['completed']}/{s['requests']} failed={s['failed']}  "
          f"wall={s['wall_s']:.2f}s  boot+run={elapsed:.0f}s", flush=True)
    return s


def report(rows: list[dict], cells: list[str], spreads: list[float]) -> None:
    width = 6 + 35 * len(spreads)
    print("\n" + "=" * width)
    print(f"{'cell':<6}" + "".join(
        f"{'--- output-spread ' + str(sp) + ' ---':^35}" for sp in spreads))
    print(f"{'':<6}" + "".join(
        f"{'pie':>10}{'vllm':>10}{'ratio':>8}{'var':>7}" for _ in spreads))
    print("-" * width)
    for cell in cells:
        line = f"{cell:<6}"
        for sp in spreads:
            p = [r["output_tok_per_s"] for r in rows
                 if r["cell"] == cell and r["engine"] == "pie"
                 and r["spread"] == sp]
            v = [r["output_tok_per_s"] for r in rows
                 if r["cell"] == cell and r["engine"] == "vllm"
                 and r["spread"] == sp]
            if not p or not v:
                line += f"{'-':>10}{'-':>10}{'-':>8}{'-':>7}"
                continue
            pm, vm = statistics.fmean(p), statistics.fmean(v)
            var = (max(p) - min(p)) / pm * 100 if len(p) > 1 else 0.0
            line += f"{pm:>10.0f}{vm:>10.0f}{pm / vm:>8.3f}{var:>6.1f}%"
        print(line)
    print("=" * width)

    # Per-round ratios: a mean whose rounds disagree in sign is not a result.
    for cell in cells:
        for sp in spreads:
            p = [r["output_tok_per_s"] for r in sorted(
                (r for r in rows if r["cell"] == cell
                 and r["engine"] == "pie" and r["spread"] == sp),
                key=lambda r: r["rep"])]
            v = [r["output_tok_per_s"] for r in sorted(
                (r for r in rows if r["cell"] == cell
                 and r["engine"] == "vllm" and r["spread"] == sp),
                key=lambda r: r["rep"])]
            if len(p) > 1 and len(v) == len(p):
                per = "  ".join(f"{a / b:.3f}" for a, b in zip(p, v))
                print(f"  {cell:<5} spread={sp}: {per}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="directory for per-run JSON and logs")
    ap.add_argument("--cells", default="c256,S,D,X")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--spread", type=float, default=None,
                    help="also run every cell with this --output-spread and "
                         "report both columns. Uniform-only is the DEFAULT "
                         "shape and vLLM's best case; see the module docstring.")
    ap.add_argument("--pages", type=int, default=22039,
                    help="KV pages, applied identically to both engines")
    ap.add_argument("--swap-pool-size", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--pie-python",
                    default=str(ROOT / "sdk/python-server/.venv/bin/python"))
    ap.add_argument("--vllm-python", default="/root/vllm-venv/bin/python")
    ap.add_argument("--cuda-compat", default="",
                    help="prepended to LD_LIBRARY_PATH; needed when running "
                         "cu13 binaries on an older driver via forward compat")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [c for c in args.cells.split(",") if c]
    spreads = [0.0] + ([args.spread] if args.spread else [])
    engines = ["pie", "vllm"]

    rows: list[dict] = []
    for cell_name in cells:
        cell = CELLS[cell_name]
        for spread in spreads:
            print(f"\n=== {cell_name}  {cell['num_requests']}x"
                  f"{cell['max_tokens']} conc={cell['concurrency']} "
                  f"spread={spread} ===", flush=True)
            for rep in range(args.reps):
                order = engines if rep % 2 == 0 else list(reversed(engines))
                for engine in order:
                    s = run_one(engine, cell_name, cell, out_dir, rep,
                                spread, args)
                    if s is not None:
                        rows.append({**s, "cell": cell_name, "engine": engine,
                                     "rep": rep, "spread": spread})

    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2))
    report(rows, cells, spreads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
