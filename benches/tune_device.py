#!/usr/bin/env python3
"""Sweep this driver's per-device crossovers and print a `tuning_for()` block.

Every crossover in `driver/metal/src/device_tuning.hpp` was measured on one
machine. The file's rule is that a default-constructed `DeviceTuning`
reproduces those numbers exactly and a new device gets an override carrying
its own measurement -- which means somebody has to run the measurement, and
until now that meant assembling env overrides by hand and remembering the
method.

The method is easy to get wrong, and the first version of this script got it
wrong in the way that matters, so the failure is worth stating plainly.

A threshold only does something at batches that STRADDLE it. Sweeping
`sdpa_tile_min_rows_per_request` over 32/16/64 at 448 rows asks whether
448>=32, 448>=16, 448>=64 -- three yeses, one code path, three runs of the
same binary doing the same work. It duly returned 2666.4 / 2662.2 / 2664.7
tok/s and the script called it "keeps the default", which reads like a
measurement and is not one. Same for `qmm_min_batch` at 16 rows and for all
three MoE knobs at the row counts first chosen here.

So this does not take a row count on faith. Each knob carries the driver's
own rule as a predicate, and the script picks the batch: it walks candidate
row counts until it finds one where the settings provably choose DIFFERENT
paths, prints what each arm decides there, and refuses to sweep a knob whose
settings it cannot make diverge. `--control` then runs the same arms at a
batch where they provably agree; the numbers there should land on top of each
other, and if they do not, the machine is too noisy for the rest to mean
anything.

The rest of the method:

  * ONE binary, two arms. A rebuild is a different binary and a different
    binary is a different measurement, so every knob is an env override.
  * ARMS ALTERNATED, not batched, so a machine that warms up partway does not
    bias whichever arm ran second.
  * MEDIANS, every run printed, and a winner declared only when the gap
    between arms clears the noise WITHIN them.

Rules mirrored from the driver -- `qmm_bn`/`qmm_bn_unsplit` and
`sdpa_should_tile` in model/qwen3_5/decode_dispatch_mb.hpp, `moe_should_batch`
and `moe_tile_rows` in model/shared_kernels.hpp. If those move, `--control`
is what notices.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PREFILL = re.compile(r"prefill:\s+\d+ tok in [\d.]+ s\s+=\s+([\d.]+) tok/s")


@dataclass
class Model:
    """What a checkpoint's config says about the shapes the rules see."""

    path: str
    n_experts: int = 0
    top_k: int = 0
    widths: tuple[int, ...] = ()  # distinct projection output widths

    @staticmethod
    def load(path: str) -> "Model":
        cfg = json.load(open(os.path.join(path, "config.json")))
        cfg = cfg.get("text_config", cfg)
        experts = cfg.get("num_local_experts") or cfg.get("num_experts") or 0
        top_k = cfg.get("num_experts_per_tok") or cfg.get("top_k_experts") or 0

        # `qmm_bn*` is decided per projection, so the crossover's effect depends
        # on the set of output widths a layer dispatches, not on one of them.
        hidden = cfg.get("hidden_size") or 0
        heads = cfg.get("num_attention_heads") or 0
        kv = cfg.get("num_key_value_heads") or heads
        head_dim = cfg.get("head_dim") or (hidden // heads if heads else 0)
        inter = cfg.get("intermediate_size") or 0
        if isinstance(inter, list):
            inter = inter[0] if inter else 0
        widths = {heads * head_dim, kv * head_dim, hidden, inter,
                  cfg.get("vocab_size") or 0}
        return Model(path, int(experts), int(top_k),
                     tuple(sorted(w for w in widths if w > 0)))


# Each rule returns what the driver would CHOOSE for a setting at a batch.
# Two settings diverge exactly when their choices differ.

def rule_always(value: int, rows: int, m: Model):
    """For a boolean knob: the setting is the choice."""
    return value


def rule_qmm_min_batch(value: int, rows: int, m: Model):
    # `<family>_qmm_rows`: below the threshold the batch is left unpadded, so
    # `qmm_bn`'s `N % bm != 0` drops it to the matvec; at or above it the batch
    # is rounded up to a `bm` multiple and takes the GEMM. So the threshold is
    # the whole decision -- the modulo never independently rejects anything.
    return "gemm" if rows >= value else "matvec"


def rule_sdpa_tile(value: int, rows: int, m: Model):
    # sdpa_should_tile, one request: rows/r >= sdpa_tile_min_rows_per_request().
    return "tiled" if rows >= value else "per-row"


QMM_BMS = (16, 32, 64)  # kQmmBMs


def qmm_bm(n: int) -> int:
    best = QMM_BMS[0]
    for bm in QMM_BMS[1:]:
        if n >= bm:
            best = bm
    return best


def rule_qmm_bn_crossover(value: int, rows: int, m: Model):
    """`qmm_bn_unsplit`: BN=32 once a projection's grid clears the crossover.

    Decided per projection -- `(out_vec/32) * row_tiles >= crossover` -- so
    two settings diverge when they disagree about ANY width the model
    dispatches, which is why this needs the config's widths and not just a
    batch. The signature is the tile chosen for each distinct width.
    """
    if not m.widths:
        return None
    n = rows if rows < 12 else ((rows + qmm_bm(rows) - 1) // qmm_bm(rows)) * qmm_bm(rows)
    bm = qmm_bm(n)
    if n < 12 or n % bm != 0:
        return "matvec"
    row_tiles = n // bm
    picks = []
    for w in m.widths:
        if w % 16 != 0:
            picks.append("-")
        elif w % 32 == 0 and (w // 32) * row_tiles >= value:
            picks.append("32")
        else:
            picks.append("16")
    return "/".join(picks)


def rule_moe_batch(value: int, rows: int, m: Model):
    # moe_should_batch: n_pairs >= n_experts * moe_batch_min_per_expert().
    if not m.n_experts:
        return None
    return "sorted" if rows * m.top_k >= m.n_experts * value else "matvec"


def moe_tile(rows: int, m: Model, mid: int, wide: int):
    if not m.n_experts or rows * m.top_k < m.n_experts * 8:
        return 1
    per = (rows * m.top_k) // m.n_experts
    if per >= wide:
        return 64
    return 32 if per >= mid else 16


def rule_moe_mid(value: int, rows: int, m: Model):
    return moe_tile(rows, m, mid=value, wide=88)


def rule_moe_wide(value: int, rows: int, m: Model):
    return moe_tile(rows, m, mid=12, wide=value)


@dataclass
class Knob:
    field: str          # the `DeviceTuning` member, for the printed block
    env: str            # the override
    values: list[int]   # candidates, default first
    models: list[str]   # substrings matching checkpoint directories; a value
                        # only moves the family default if it wins on ALL of
                        # them, which is the standard the M2 entry in
                        # `device_tuning.hpp` was already held to and the
                        # script was not
    rule: Callable      # what the driver chooses, for finding a batch that splits
    why: str
    rows: list[int] = field(default_factory=list)  # batches to consider


# `values[0]` is scored as "the default", so a stale literal here does not
# produce a wrong number -- it produces a wrong SENTENCE, which is worse,
# because the number is right and the reader has no way to tell.
#
# It happened. `moe_batch_min_per_expert` moved from 8 to 4 in
# `device_tuning.hpp` and this list kept the 8, so the sweep reported "4 beats
# the default 8 by +90.4%" and printed `t.moe_batch_min_per_expert = 4;` as
# something to paste into `tuning_for()`. Both halves are false: 4 was already
# the default, the +90.4% was the cost of the value nobody runs, and the paste
# was a no-op dressed as the largest win in the table. A file whose whole
# purpose is that a value must arrive with its measurement had produced a
# measurement of nothing.
#
# So the defaults are not transcribed any more. They are read from the header
# that owns them, and a knob whose first candidate is not the default is a hard
# error rather than a footnote.
_TUNING_HPP = (Path(__file__).resolve().parent.parent
               / "driver/metal/src/device_tuning.hpp")


def header_defaults() -> dict[str, int]:
    """`DeviceTuning`'s field initializers, by name.

    Only the plain `type name = literal;` members, which is all of them; a
    field that stops being one will simply be absent and trip the check below
    rather than be guessed at.
    """
    out: dict[str, int] = {}
    text = _TUNING_HPP.read_text()
    for m in re.finditer(r"^\s{4}(?:int|bool)\s+(\w+)\s*=\s*(\w+)\s*;", text, re.M):
        name, raw = m.group(1), m.group(2)
        if raw in ("true", "false"):
            out[name] = 1 if raw == "true" else 0
        elif raw.isdigit():
            out[name] = int(raw)
    return out


def check_defaults_match_header(knobs: list["Knob"]) -> list[str]:
    """The knobs whose first candidate is not what the header ships."""
    have = header_defaults()
    bad = []
    for k in knobs:
        if k.field not in have:
            bad.append(f"{k.field}: not found in {_TUNING_HPP.name}")
        elif have[k.field] != k.values[0]:
            bad.append(f"{k.field}: header says {have[k.field]}, "
                       f"KNOBS says {k.values[0]}")
    return bad


# Row counts worth considering. A threshold's two sides are by construction
# close in cost near the crossover, so these lean toward the smallest batch
# that still separates -- but not so small the run is all fixed overhead.
COMMON_ROWS = [32, 48, 64, 96, 128, 192, 256, 320, 448, 512, 704, 1024]

KNOBS: list[Knob] = [
    Knob("fp16_qmm", "PIE_METAL_FP16_QMM", [1, 0], ["gemma-4-e2b", "gemma-4-31b-it", "Qwen3.6-27B"], rule_always,
         "M1 and M2 emulate bfloat16 matrix ops and Apple9 does not, so this "
         "is the one field whose default may be wrong rather than unmeasured",
         rows=[448]),
    Knob("qmm_min_batch", "PIE_METAL_QMM_MIN_BATCH", [12, 8, 16],
         ["Llama-3.2-1B-Instruct-4bit"], rule_qmm_min_batch,
         "matvec against GEMM. Its crossover is small by construction, so the "
         "two sides are close there and the noise guard earns its keep",
         rows=[12, 8, 16, 24, 32]),
    Knob("qmm_bn_crossover_tg", "PIE_METAL_QMM_BN_CROSSOVER_TG",
         [160, 96, 256], ["Llama-3.2-1B-Instruct-4bit"], rule_qmm_bn_crossover,
         "a threadgroup count, so a wider GPU saturates later and wants it up. "
         "Decided per projection, so the batch has to be one where the "
         "candidates disagree about at least one of the model's widths",
         rows=[64, 96, 128, 192, 256, 448, 1024]),
    Knob("moe_tile_wide_per", "PIE_METAL_MOE_TILE_WIDE_PER", [88, 56, 128],
         ["gpt-oss-20b"], rule_moe_wide,
         "rows an expert needs before a 64-row tile pays; end to end because "
         "the probe has misled this one three times"),
    Knob("moe_tile_mid_per", "PIE_METAL_MOE_TILE_MID_PER", [12, 8, 24],
         ["gemma-4-26B-A4B"], rule_moe_mid, "the same trade one rung down"),
    Knob("sdpa_tile_min_rows_per_request", "PIE_METAL_SDPA_TILE_MIN_ROWS",
         [32, 16, 64], ["Llama-3.2-1B-Instruct-4bit"], rule_sdpa_tile,
         "tiled attention against per-row"),
    Knob("moe_batch_min_per_expert", "PIE_METAL_MOE_BATCH_MIN_PER_EXPERT",
         [4, 8, 2], ["gpt-oss-20b"], rule_moe_batch,
         "sorting the mixture against leaving it a matvec"),
]


def find_checkpoint(needle: str, roots: list[str]) -> str | None:
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "config.json" in filenames and needle.lower() in dirpath.lower():
                return dirpath
            if dirpath.count(os.sep) - root.count(os.sep) > 4:
                dirnames.clear()
    return None


def display_name(path: str) -> str:
    """HF snapshot directories are named by hash; the model name is a level up."""
    for part in reversed(path.rstrip("/").split(os.sep)):
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/")
        if not re.fullmatch(r"[0-9a-f]{16,}|snapshots", part):
            return part
    return path


def choices_at(knob: Knob, rows: int, m: Model) -> list | None:
    if knob.rule is None:
        return None
    return [knob.rule(v, rows, m) for v in knob.values]


def pick_rows(knob: Knob, m: Model, want_split: bool) -> tuple[int, list | None] | None:
    """A batch where the settings provably differ (or provably agree).

    Splits are searched smallest-first, because a threshold's two sides are
    only both plausible near it. Controls are searched largest-first, so the
    harness check runs on a batch doing real work rather than on whatever
    degenerate path a tiny batch happens to share.
    """
    if knob.rule is None:
        # Cannot be shown to diverge from the config, so there is no control
        # either: an agreement we cannot predict proves nothing.
        return (knob.rows[0], None) if want_split else None
    order = knob.rows or COMMON_ROWS
    for rows in (order if want_split else list(reversed(order))):
        got = choices_at(knob, rows, m)
        if got is None or None in got:
            continue
        if (len(set(got)) > 1) == want_split:
            return rows, got
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
    for _ in range(repeats):  # alternated: one pass over every arm, `repeats` times
        for v in knob.values:
            got = run_once(bench, ckpt, rows, knob.env, v)
            if got is not None:
                runs[v].append(got)
    return runs


def report(knob: Knob, runs: dict[int, list[float]], rows: int,
           choices: list | None, control: bool) -> int | None:
    meds: dict[int, float] = {}
    noise = 0.0
    for v in knob.values:
        if runs[v]:
            meds[v] = statistics.median(runs[v])
            if meds[v]:
                noise = max(noise, (max(runs[v]) - min(runs[v])) / meds[v] * 100)
    for v in knob.values:
        got = runs[v]
        if not got:
            print(f"    {v:>6}  no measurement")
            continue
        med = meds[v]
        spread = (max(got) - min(got)) / med * 100 if med else 0
        picks = f"  -> {choices[knob.values.index(v)]}" if choices else ""
        print(f"    {v:>6}  {med:8.1f} tok/s   spread {spread:4.1f}%   "
              + " ".join(f"{g:.1f}" for g in got) + picks)
    if not meds:
        return None

    default = knob.values[0]
    best = max(meds, key=lambda v: meds[v])

    if control:
        # These settings take the same path here, so the arms are the same
        # work. A gap that clears the noise means the noise is understated.
        worst = min(meds, key=lambda v: meds[v])
        split = (meds[best] / meds[worst] - 1) * 100 if meds[worst] else 0
        if split > max(noise, 1.0):
            print(f"    -> arms that take the SAME path differ by {split:.1f}% "
                  f"against {noise:.1f}% noise. This machine is too unsettled "
                  "for the numbers above to mean anything.")
        else:
            print(f"    -> same path, {split:.1f}% apart, noise {noise:.1f}%. "
                  "The harness is sound.")
        return None

    # Only arms that CHOSE differently are rivals. Two settings either side of
    # the same decision run the same code, so scoring the default against one
    # of those measures the machine and calls it a tuning result.
    if choices:
        mine = choices[knob.values.index(default)]
        rivals = [v for v in meds if choices[knob.values.index(v)] != mine]
    else:
        rivals = [v for v in meds if v != default]
    if not rivals or default not in meds:
        print("    -> nothing to compare: no arm took a different path.")
        return None

    top = max(rivals, key=lambda v: meds[v])
    gap = (meds[top] / meds[default] - 1) * 100
    if gap > noise:
        print(f"    -> {top} beats the default {default} by {gap:+.1f}% "
              f"(noise {noise:.1f}%)")
        return top
    # Both branches below return the DEFAULT rather than None. A checkpoint
    # that measured and declined to move is not a checkpoint that was never
    # asked, and the caller needs to tell them apart: a family default may only
    # move when every checkpoint asked for it, so an abstention that reads as
    # an absence would let one model carry the vote.
    if -gap > noise:
        print(f"    -> keeps the default {default}: {top} comes in {-gap:.1f}% "
              f"below it (noise {noise:.1f}%)")
        return default
    print(f"    -> no winner: {top} is {gap:+.1f}% against the default and "
          f"the runs themselves move {noise:.1f}%. Raise --repeats, or take "
          "it that the paths cost the same here.")
    return default
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="path to the built llama_bench")
    ap.add_argument("--checkpoint-root", action="append", default=[])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--control", action="store_true",
                    help="also run each knob at a batch where its settings "
                         "provably take the SAME path; the arms should land on "
                         "top of each other, and that is what says the rest of "
                         "the table is measurement rather than weather")
    args = ap.parse_args()

    # Before anything is measured: a sweep that scores against the wrong
    # baseline is not slower to notice, it is impossible to notice.
    if drift := check_defaults_match_header(KNOBS):
        print("KNOBS disagrees with device_tuning.hpp about what ships:",
              file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        print("Fix the candidate list; `values[0]` is scored as the default.",
              file=sys.stderr)
        return 2

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
        print(f"\n{knob.field}  ({knob.env})")
        print(f"  {knob.why}")
        # Every checkpoint this knob names, because one is not enough to move
        # a FAMILY default: `fp16_qmm` reverses sign with model size on an M4
        # Pro, and a sweep that had asked only the small one would have
        # recommended a 1.7-4.7% regression on the four models anybody runs.
        verdicts: dict[str, int | None] = {}
        for model_name in knob.models:
            ckpt = find_checkpoint(model_name, roots)
            if ckpt is None:
                print(f"  SKIP: no checkpoint matching '{model_name}'")
                verdicts[model_name] = None
                continue
            model = Model.load(ckpt)

            split = pick_rows(knob, model, want_split=True)
            if split is None:
                print(f"  SKIP: no batch in {knob.rows or COMMON_ROWS} makes "
                      "these settings take different paths, so a sweep here "
                      "would time the same work three times.")
                verdicts[model_name] = None
                continue
            rows, choices = split
            print(f"  {display_name(ckpt)} at {rows} rows"
                  + (f", where the arms choose {choices}" if choices else
                     ", divergence not predictable from config"))
            best = report(knob, sweep(knob, args.bench, ckpt, rows, args.repeats),
                          rows, choices, control=False)
            verdicts[model_name] = best

            if args.control:
                same = pick_rows(knob, model, want_split=False)
                if same is None:
                    print("  (no control: nothing here makes these settings "
                          "agree, or their choice is not predictable from the "
                          "config)")
                else:
                    print(f"  control: {same[0]} rows, all arms choose {same[1][0]}")
                    report(knob, sweep(knob, args.bench, ckpt, same[0], args.repeats),
                           same[0], same[1], control=True)

        measured = {m: v for m, v in verdicts.items() if v is not None}
        movers = {m: v for m, v in measured.items() if v != knob.values[0]}
        if not measured or not movers:
            continue
        winners = set(movers.values())
        if len(movers) == len(measured) and len(winners) == 1:
            chosen[knob.field] = winners.pop()
        else:
            kept = [m for m in measured if m not in movers]
            print(f"    -> NOT RECOMMENDED: "
                  + "; ".join(f"{m} wants {v}" for m, v in movers.items())
                  + (f", but {', '.join(kept)} keeps {knob.values[0]}" if kept else "")
                  + ". A family default needs every checkpoint that was asked, "
                    "or the value is a property of one model rather than of "
                    "this device.")


    print("\n" + "=" * 70)
    if not chosen:
        print("Every knob keeps its default: at the batches where these settings")
        print("provably diverge, nothing beat the M1 numbers by more than the runs")
        print("moved on their own.")
        return 0
    print("Paste into `tuning_for()` in driver/metal/src/device_tuning.cpp, under")
    print("the case for this device's family, WITH the runs above as the comment.")
    print("A value without its measurement is the thing that file exists to stop.\n")
    for field_name, value in chosen.items():
        print(f"    t.{field_name} = {value};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
