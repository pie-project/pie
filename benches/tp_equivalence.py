#!/usr/bin/env python3
"""Assert that a model answers the same way at tp=1 and at tp>1.

Tensor parallelism is a pure implementation detail: sharding changes who
multiplies what, not what the model says. When a shard is wrong the engine
does not crash — it keeps running at full speed and returns fluent nonsense,
so throughput and latency numbers look healthy while the model is broken.
gemma-4-E4B shipped in exactly that state: every full-attention layer ran at
half its head width at tp=2 for as long as tp=2 existed, and no benchmark
noticed, because nothing compared the tokens.

Bit-identical output is NOT the bar. A different reduction order is a
different (equally valid) floating-point sum, so a near-tie between two
candidate tokens can legitimately flip. This compares by prefix: the runs
must agree for the first `--min-agree` tokens of decoded text. A real
sharding bug diverges immediately and completely; reduction-order noise
diverges late if at all.

Usage:
    tp_equivalence.py --model google/gemma-4-E4B --tp-sizes 1,2
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHA_RE = re.compile(r"sha256\[:16\]=([0-9a-f]+)")


def run_once(python: str, model: str, tp: int, max_tokens: int,
             prompt: str | None, timeout: float) -> tuple[str, str]:
    """Return (sha, decoded text) for one single-request latency run."""
    cmd = [python, str(HERE / "pie_bench.py"), "latency",
           "--model", model, "--tp-size", str(tp),
           "--requests", "1", "--max-tokens", str(max_tokens),
           "--dump-first-text"]
    if prompt:
        cmd += ["--prompt", prompt]
    proc = subprocess.run(cmd, cwd=HERE.parent, capture_output=True,
                          text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"tp={tp} run failed ({proc.returncode}):\n{out[-2000:]}")
    m = SHA_RE.search(out)
    lines = out.splitlines()
    text = ""
    for i, line in enumerate(lines):
        if "FIRST REQUEST OUTPUT" in line:
            body = []
            for follow in lines[i + 1:]:
                if follow.startswith("END OUTPUT"):
                    break
                body.append(follow)
            text = "\n".join(body)
            break
    if not text:
        raise RuntimeError(f"tp={tp}: no decoded text in output")
    return (m.group(1) if m else ""), text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp-sizes", default="1,2",
                    help="comma-separated; the first is the reference")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--min-agree", type=int, default=12,
                    help="characters of decoded text that must match. A "
                         "broken shard diverges at character 0 with "
                         "non-fluent output; reduction-order noise flips a "
                         "near-tie well into a fluent sentence, so a short "
                         "prefix separates the two cleanly.")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    tps = [int(x) for x in args.tp_sizes.split(",")]
    ref_tp = tps[0]
    ref_sha, ref_text = run_once(args.python, args.model, ref_tp,
                                 args.max_tokens, args.prompt, args.timeout)
    print(f"tp={ref_tp} (reference) sha={ref_sha}\n  {ref_text!r}", flush=True)

    failures = []
    for tp in tps[1:]:
        sha, text = run_once(args.python, args.model, tp, args.max_tokens,
                             args.prompt, args.timeout)
        head_ref, head = ref_text[:args.min_agree], text[:args.min_agree]
        exact = sha == ref_sha and ref_sha != ""
        agree = head_ref == head
        status = ("IDENTICAL" if exact
                  else "AGREES" if agree
                  else "*** DIVERGES ***")
        print(f"tp={tp} sha={sha} {status}\n  {text!r}", flush=True)
        common = 0
        while (common < min(len(ref_text), len(text))
               and ref_text[common] == text[common]):
            common += 1
        if not agree:
            failures.append(tp)
        if not exact:
            print(f"  first difference at character {common}"
                  + (f"; tp={ref_tp} continues "
                     f"{ref_text[common:common + 40]!r}, tp={tp} continues "
                     f"{text[common:common + 40]!r}"
                     if common < min(len(ref_text), len(text)) else ""),
                  flush=True)

    if failures:
        print(f"\nFAIL: tp={failures} disagree with tp={ref_tp} within the "
              f"first {args.min_agree} characters. That is where a wrong "
              f"sharded dimension shows up, not where reduction-order noise "
              f"does -- dump the per-layer residual stream for both and find "
              f"the first layer that diverges "
              f"(PIE_GEMMA4_DUMP_DIR / PIE_GEMMA4_DUMP_LAYERS for gemma-4).",
              flush=True)
        return 1
    print(f"\nOK: every topology agrees with tp={ref_tp} for the first "
          f"{args.min_agree} characters.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
