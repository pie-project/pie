"""E2E test for text-completion-spec inferlet.

Two modes:

  1. Default — runs the inferlet once against whatever driver/config the
     user passed via CLI. Asserts non-empty output and a parseable
     `SPEC_STATS` line. Reports throughput; the user can compare two
     runs (with and without `--spec-ngram`) by hand.

  2. `--compare-spec` — *only* when `--driver sglang`: spawns two
     subprocesses (one with NGRAM off, one with NGRAM on) each running
     this script in mode (1), parses the SPEC_STATS line from each, and
     asserts the NGRAM run's decode throughput is at least as high as
     the baseline. Subprocesses are required because the pie Rust runtime
     installs a global tracing subscriber that can't be re-initialized
     in the same process.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time

from conftest import (
    INFERLETS_DIR,
    make_parser,
    run_inferlet,
    run_tests,
    _clear_wasmtime_cache,
)

# Greedy decoding (temperature=0) gives the highest spec acceptance because
# the model's predictions are deterministic — every matching draft is
# accepted. The prompt is engineered for NGRAM: it asks the model to
# enumerate a long list with a strong template (`N. **NAME** — short
# description.`), so common subsequences (the numbering, the asterisks,
# the dashes, frequent vocabulary like "the", "a", "and") repeat heavily
# and the trie keeps proposing useful continuations. Free-form prose
# would barely benefit from NGRAM (per-iteration trie overhead exceeds
# the savings); structured / repetitive output is the NGRAM sweet spot.
_DEFAULT_PROMPT_INPUT = {
    "prompt": (
        "List the first 20 elements of the periodic table. "
        "For each element, give the format: "
        "`N. **NAME** (Symbol) — atomic mass: M.MMM amu, group: G, period: P.` "
        "Output exactly 20 numbered lines and nothing else."
    ),
    "max_tokens": 384,
    "temperature": 0.0,
}

_STATS_RE = re.compile(
    r"SPEC_STATS\s+"
    r"prompt_tokens=(?P<prompt>\d+)\s+"
    r"generated_tokens=(?P<gen>\d+)\s+"
    r"elapsed_ms=(?P<elapsed_ms>\d+)\s+"
    r"prefill_ms=(?P<prefill_ms>\d+)\s+"
    r"decode_ms=(?P<decode_ms>\d+)\s+"
    r"tokens_per_sec=(?P<tps>[0-9.]+)\s+"
    r"decode_tokens_per_sec=(?P<dtps>[0-9.]+)\s+"
    r"steps=(?P<steps>\d+)\s+"
    r"avg_tokens_per_step=(?P<avg>[0-9.]+)"
)


def _parse_stats(stdout: str) -> dict:
    """Return the last SPEC_STATS line as a dict of floats/ints."""
    last_match = None
    for line in stdout.splitlines():
        m = _STATS_RE.search(line)
        if m:
            last_match = m
    if last_match is None:
        raise AssertionError(
            f"No SPEC_STATS line found in inferlet stdout. Tail:\n"
            + "\n".join(stdout.splitlines()[-20:])
        )
    g = last_match.groupdict()
    return {
        "prompt_tokens": int(g["prompt"]),
        "generated_tokens": int(g["gen"]),
        "elapsed_ms": int(g["elapsed_ms"]),
        "prefill_ms": int(g["prefill_ms"]),
        "decode_ms": int(g["decode_ms"]),
        "tokens_per_sec": float(g["tps"]),
        "decode_tokens_per_sec": float(g["dtps"]),
        "steps": int(g["steps"]),
        "avg_tokens_per_step": float(g["avg"]),
    }


# ---------------------------------------------------------------------------
# Default test: single run, sanity-check the SPEC_STATS line.
# ---------------------------------------------------------------------------

async def test_text_completion_spec(client, args):
    output = await run_inferlet(
        client, "text-completion-spec",
        _DEFAULT_PROMPT_INPUT,
        timeout=args.timeout,
    )
    stats = _parse_stats(output)
    assert stats["generated_tokens"] > 0, f"no tokens generated: {stats}"
    assert stats["tokens_per_sec"] > 0, f"zero throughput: {stats}"
    # Echo the inferlet's SPEC_STATS line on our own stdout so the
    # comparison runner (subprocess mode) can grep it back out.
    for line in output.splitlines():
        if line.startswith("SPEC_STATS"):
            print(line)
    print(
        f"\n  → generated {stats['generated_tokens']} tokens in "
        f"{stats['elapsed_ms']} ms = {stats['tokens_per_sec']:.1f} tok/s "
        f"(decode-only: {stats['decode_tokens_per_sec']:.1f} tok/s, "
        f"{stats['steps']} steps, {stats['avg_tokens_per_step']:.2f} tok/step)"
    )


# ---------------------------------------------------------------------------
# Comparison runner: spawns two subprocesses (NGRAM off, NGRAM on), each
# running this script in single-server mode, then parses SPEC_STATS from
# each. Subprocesses isolate the pie Rust runtime's global tracing init.
# ---------------------------------------------------------------------------

def _run_subprocess(args, *, ngram: bool) -> dict:
    """Spawn this script as a subprocess in single-test mode and return
    the parsed SPEC_STATS from its stdout."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--driver", args.driver,
        "--model", args.model,
        "--device", args.device,
        "--timeout", str(args.timeout),
        "--sglang-attention-backend", args.sglang_attention_backend,
    ]
    if args.cpu_mem_gb > 0:
        cmd += ["--cpu-mem-gb", str(args.cpu_mem_gb)]
    if ngram:
        cmd += ["--spec-ngram", "--spec-num-drafts", str(args.spec_num_drafts)]

    label = "NGRAM-on" if ngram else "NGRAM-off"
    print(f"\n[{label}] spawning subprocess …", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # Print the tail of the subprocess output so the failure is debuggable.
        sys.stderr.write(f"[{label}] subprocess exited {proc.returncode}\n")
        sys.stderr.write("--- stdout tail ---\n")
        sys.stderr.write("\n".join(proc.stdout.splitlines()[-30:]) + "\n")
        sys.stderr.write("--- stderr tail ---\n")
        sys.stderr.write("\n".join(proc.stderr.splitlines()[-30:]) + "\n")
        raise RuntimeError(f"[{label}] test subprocess failed")
    stats = _parse_stats(proc.stdout)
    print(
        f"[{label}] {stats['generated_tokens']} tokens in "
        f"{stats['elapsed_ms']} ms (prefill {stats['prefill_ms']} ms, "
        f"decode {stats['decode_ms']} ms) → "
        f"decode {stats['decode_tokens_per_sec']:.1f} tok/s "
        f"({stats['steps']} steps, {stats['avg_tokens_per_step']:.2f} tok/step)"
    )
    return stats


async def _compare(args) -> int:
    if args.driver != "sglang":
        print("--compare-spec requires --driver sglang (NGRAM is sglang-only).")
        return 1

    baseline = _run_subprocess(args, ngram=False)
    spec = _run_subprocess(args, ngram=True)

    print("\n" + "─" * 78)
    print(f"{'metric':30s} {'baseline':>15s} {'NGRAM':>15s} {'ratio':>8s}")
    print("─" * 78)
    for key in (
        "decode_tokens_per_sec", "tokens_per_sec", "decode_ms", "prefill_ms",
        "steps", "avg_tokens_per_step",
    ):
        b, s = baseline[key], spec[key]
        ratio = (s / b) if b else float("nan")
        print(f"{key:30s} {b:>15.2f} {s:>15.2f} {ratio:>8.2f}x")
    print("─" * 78)

    # Soft assertion: NGRAM should not regress meaningfully on the decode
    # path. We compare decode_tokens_per_sec — prefill is identical between
    # runs (NGRAM doesn't touch prefill) and including it would dampen the
    # ratio.
    speedup = spec["decode_tokens_per_sec"] / baseline["decode_tokens_per_sec"]
    if speedup < 0.90:
        print(
            f"\n❌ NGRAM decode is {speedup:.2f}x baseline — expected ≥0.90x. "
            "Either trie didn't warm up enough or NGRAM overhead is too high."
        )
        return 1
    if spec["avg_tokens_per_step"] < 1.05:
        print(
            f"\n⚠️  avg_tokens_per_step={spec['avg_tokens_per_step']:.2f} — "
            "NGRAM didn't accept any drafts in the timed run. The trie may "
            "still be cold; try a longer max_tokens or repeat the prompt."
        )
        # Don't fail on this — it's diagnostic, not a strict requirement.
    print(f"\n✅ NGRAM speedup: {speedup:.2f}x")
    return 0


# ---------------------------------------------------------------------------
# Entry point: dispatch to comparison runner or default test runner.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = make_parser("text-completion-spec test")
    parser.add_argument(
        "--compare-spec", action="store_true",
        help="Run the inferlet under two server configs (NGRAM off, NGRAM on) "
             "and report the speedup. Requires --driver sglang."
    )
    parsed_args = parser.parse_args()
    if parsed_args.compare_spec:
        try:
            rc = asyncio.run(_compare(parsed_args))
        except KeyboardInterrupt:
            rc = 1
        sys.exit(rc)
    else:
        run_tests([test_text_completion_spec], "text-completion-spec test")
