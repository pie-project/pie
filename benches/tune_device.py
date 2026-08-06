#!/usr/bin/env python3
"""Sweep this driver's per-device crossovers and print a `tuning_for()` block.

Every crossover in `driver/metal/src/device_tuning.hpp` was measured on one
machine. The file's rule is that a default-constructed `DeviceTuning`
reproduces those numbers exactly and a new device gets an override carrying
its own measurement -- which means somebody has to run the measurement, and
until now that meant assembling env overrides by hand and remembering the
method.

The method is the part worth automating, because it is easy to get wrong:

  * ONE binary, two arms. A rebuild between arms is a different binary and a
    different binary is a different measurement, so every knob is an env
    override and this never rebuilds.
  * ARMS ALTERNATED, not batched. A machine that warms up or throttles
    partway through biases whichever arm ran second; interleaving them costs
    nothing and removes it.
  * MEDIANS, and every run printed. A mean hides the one run where something
    else woke up; the spread is what says whether a 2% difference is real.
  * A shape where the knob DECIDES something. A sweep at a batch where two
    settings take the same path measures the weather, so each knob below names
    the model and row count where its two answers actually diverge -- and
    `--control` re-runs one knob at a shape where they should NOT diverge,
    which is the check that the rest of the table means anything.

What it cannot do is tell you the answer is worth taking. Read the spread.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field

# `llama_bench <checkpoint> <rows>` prints one of these per fire.
PREFILL = re.compile(r"prefill:\s+\d+ tok in [\d.]+ s\s+=\s+([\d.]+) tok/s")


@dataclass
class Knob:
    """One tuned field, and the shape at which its settings diverge."""

    field: str          # the `DeviceTuning` member, for the printed block
    env: str            # the override
    values: list[int]   # candidates, default first
    model: str          # substring matching a checkpoint directory
    rows: int
    why: str
    # A shape where the settings should NOT diverge. Run under `--control`.
    control_rows: int = 0


KNOBS: list[Knob] = [
    Knob("fp16_qmm", "PIE_METAL_FP16_QMM", [1, 0], "gemma-4-e2b", 448,
         "M1 and M2 emulate bfloat16 matrix ops and Apple9 does not, so this "
         "is the one field whose default may be wrong rather than unmeasured"),
    Knob("qmm_min_batch", "PIE_METAL_QMM_MIN_BATCH", [12, 8, 16],
         "Llama-3.2-1B-Instruct-4bit", 16,
         "matvec against GEMM; measured at the batch where the settings pick "
         "different paths"),
    Knob("qmm_bn_crossover_tg", "PIE_METAL_QMM_BN_CROSSOVER_TG",
         [160, 96, 256], "Llama-3.2-1B-Instruct-4bit", 448,
         "a threadgroup count, so a wider GPU saturates later and wants it up",
         control_rows=128),
    Knob("moe_tile_wide_per", "PIE_METAL_MOE_TILE_WIDE_PER", [88, 56, 128],
         "gpt-oss-20b", 1024,
         "rows an expert needs before a 64-row tile pays; end to end because "
         "the probe has misled this one three times"),
    Knob("moe_tile_mid_per", "PIE_METAL_MOE_TILE_MID_PER", [12, 8, 24],
         "gemma-4-26B-A4B", 448, "the same trade one rung down"),
    Knob("sdpa_tile_min_rows_per_request", "PIE_METAL_SDPA_TILE_MIN_ROWS",
         [32, 16, 64], "Llama-3.2-1B-Instruct-4bit", 448,
         "tiled attention against per-row"),
    Knob("moe_batch_min_per_expert", "PIE_METAL_MOE_BATCH_MIN_PER_EXPERT",
         [8, 4, 16], "gpt-oss-20b", 128,
         "sorting the mixture against leaving it a matvec"),
]


def find_checkpoint(needle: str, roots: list[str]) -> str | None:
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "config.json" in filenames and needle.lower() in dirpath.lower():
                return dirpath
            # Snapshots are two deep; do not walk whole model directories.
            if dirpath.count(os.sep) - root.count(os.sep) > 4:
                dirnames.clear()
    return None


def run_once(bench: str, ckpt: str, rows: int, env_name: str, value: int) -> float | None:
    env = dict(os.environ)
    env[env_name] = str(value)
    try:
        out = subprocess.run([bench, ckpt, str(rows)], env=env, capture_output=True,
                             text=True, timeout=900).stdout
    except subprocess.TimeoutExpired:
        return None
    hit = PREFILL.search(out)
    return float(hit.group(1)) if hit else None


def sweep(knob: Knob, bench: str, ckpt: str, rows: int, repeats: int) -> dict[int, list[float]]:
    runs: dict[int, list[float]] = {v: [] for v in knob.values}
    # Alternated: one pass over every value, `repeats` times.
    for _ in range(repeats):
        for v in knob.values:
            got = run_once(bench, ckpt, rows, knob.env, v)
            if got is not None:
                runs[v].append(got)
    return runs


def report(knob: Knob, runs: dict[int, list[float]], rows: int, control: bool) -> int | None:
    print(f"\n{knob.field}  ({knob.env})")
    print(f"  {knob.why}")
    print(f"  {'control ' if control else ''}shape: {knob.model} at {rows} rows")
    best, best_med = None, 0.0
    for v in knob.values:
        got = runs[v]
        if not got:
            print(f"    {v:>6}  no measurement")
            continue
        med = statistics.median(got)
        spread = (max(got) - min(got)) / med * 100 if med else 0
        marker = ""
        if med > best_med:
            best, best_med = v, med
            marker = ""
        print(f"    {v:>6}  {med:8.1f} tok/s   spread {spread:4.1f}%   "
              + " ".join(f"{g:.1f}" for g in got) + marker)
    if best is None:
        return None
    default = knob.values[0]
    gain = (best_med / statistics.median(runs[default]) - 1) * 100 if runs[default] else 0
    if best == default:
        print(f"    -> keeps the default {default}")
    elif control:
        print(f"    -> {best} leads by {gain:+.1f}% AT THE CONTROL SHAPE, where the "
              "settings should agree. Treat the whole table as suspect.")
    else:
        print(f"    -> {best} beats the default {default} by {gain:+.1f}%")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="path to the built llama_bench")
    ap.add_argument("--checkpoint-root", action="append", default=[],
                    help="where to look for checkpoints; repeatable")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", action="append", default=[],
                    help="run only these fields; repeatable")
    ap.add_argument("--control", action="store_true",
                    help="also run each knob at a shape where its settings "
                         "should NOT diverge, which is the check that the rest "
                         "of the table means anything")
    args = ap.parse_args()

    roots = args.checkpoint_root or [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.pie-bench"),
    ]
    if not os.path.isfile(args.bench):
        print(f"no llama_bench at {args.bench}", file=sys.stderr)
        return 2

    chosen: dict[str, int] = {}
    for knob in KNOBS:
        if args.only and knob.field not in args.only:
            continue
        ckpt = find_checkpoint(knob.model, roots)
        if ckpt is None:
            print(f"\n{knob.field}: SKIP (no checkpoint matching '{knob.model}')")
            continue
        best = report(knob, sweep(knob, args.bench, ckpt, knob.rows, args.repeats),
                      knob.rows, control=False)
        if best is not None and best != knob.values[0]:
            chosen[knob.field] = best
        if args.control and knob.control_rows:
            report(knob, sweep(knob, args.bench, ckpt, knob.control_rows, args.repeats),
                   knob.control_rows, control=True)

    print("\n" + "=" * 68)
    if not chosen:
        print("Every knob keeps its default. This machine wants the M1 numbers,")
        print("or the shapes above do not separate them here -- which `--control`")
        print("is the way to tell apart.")
        return 0
    print("Paste into `tuning_for()` in driver/metal/src/device_tuning.cpp,")
    print("under the case for this device's family, WITH the runs above as the")
    print("comment. A value without its measurement is the thing this file exists")
    print("to stop.\n")
    for field_name, value in chosen.items():
        print(f"    t.{field_name} = {value};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
