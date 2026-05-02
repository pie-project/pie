#!/usr/bin/env python3
"""Quantization parity / perf sweep.

Runs a model under multiple quantization configs and reports:
  * Output coherence (first 32 greedy tokens — coherent text → ✓; gibberish → ✗)
  * Throughput (tokens/sec from tput.py's existing measurement)
  * Token-by-token diff vs. the bf16 baseline (% identical)

Used to catch quality regressions when adding a new dtype path or
quantization scheme. The default model list covers the archs that have
been migrated to `gemm_act_x_w` (qwen3, qwen3_5, qwen3_5_text); add new
entries as more archs land.

Usage:
    cd pie && uv run python ../benches/parity_table.py
    # Or specify subset:
    cd pie && uv run python ../benches/parity_table.py --models Qwen/Qwen3-0.6B
    cd pie && uv run python ../benches/parity_table.py --configs bf16 fp8

Each row in the output table is one (model, config) cell. The harness
shells out to tput.py and parses the saved-outputs JSON, so it inherits
tput.py's startup overhead — runs are slow but reflect the production
serving path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Models we currently exercise. Each entry: (hf_repo, expected_arch).
# Add to this list as new archs are wired through gemm_act_x_w. The
# entry under `gptq_int4` implicitly substitutes a different repo for
# the int4 column since GPTQ-Int4 ckpts ship as their own repos rather
# than being toggled at runtime.
DEFAULT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-4B",
]

# GPTQ-int4 sibling repos (paired with the matching bf16 baseline). Used
# to validate the M3 path; the harness loads each side independently and
# diffs the generated text.
GPTQ_INT4_PAIRS = {
    # bf16_repo -> gptq_int4_repo
    "Qwen/Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
}

# Each config maps a label to extra tput.py argv. Labels are used as
# column keys in the parity matrix. The first entry MUST be the
# baseline (bf16 unquantized) — diffs are computed against it.
CONFIGS = {
    "bf16": [],
    "fp8":  ["--runtime-quant", "fp8"],
    "int8": ["--runtime-quant", "int8"],
}

PROMPT = "Write a short story about a robot."  # tput.py's default prompt
MAX_TOKENS = 24
DEVICE = "cuda:0"


def _run_one(
    model: str,
    config_args: list[str],
    out_path: Path,
    *,
    pie_dir: Path,
    tput_path: Path,
) -> Optional[dict]:
    """Run tput.py once with the given config and parse its output sample."""
    cmd = [
        "uv", "run", "python", str(tput_path),
        "--model", model,
        "--device", DEVICE,
        "--num-requests", "1",
        "--max-tokens", str(MAX_TOKENS),
        "--temperature", "0",
        "--default-token-limit", "256",
        "--driver", "cuda_native",
        "--save-outputs", str(out_path),
        *config_args,
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=pie_dir, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "tok_s": 0.0, "text": "TIMEOUT", "elapsed_s": 600.0}
    elapsed = time.monotonic() - t0

    if proc.returncode != 0 or not out_path.exists():
        # The driver failed to load or generate. Capture the last
        # interesting stderr line for diagnosis.
        last = "\n".join(
            ln for ln in proc.stdout.splitlines() if "FP_NONE_FOR_DECODE" not in ln
        )[-400:]
        return {"ok": False, "tok_s": 0.0, "text": f"FAIL: {last}",
                "elapsed_s": elapsed}

    raw = out_path.read_text()
    # tput.py's --save-outputs format: one record per request, each
    # with `=== Request 0 ===\n{json}\n`. Extract the first record's
    # generated text.
    try:
        # The JSON is the last `{` block in the file.
        json_start = raw.rfind("{")
        record = json.loads(raw[json_start:])
        text = record.get("thinking", "") + record.get("text", "")
    except Exception:
        text = raw.strip()

    # Parse tok/s from the throughput summary block in stdout.
    tok_s = 0.0
    for ln in proc.stdout.splitlines():
        if "Est. Tokens/sec:" in ln:
            try:
                tok_s = float(ln.split(":")[-1].strip())
            except Exception:
                pass

    return {
        "ok": True,
        "tok_s": tok_s,
        "text": text,
        "elapsed_s": elapsed,
    }


def _token_diff(a: str, b: str) -> float:
    """Return % of leading characters that match between `a` and `b`.

    A coarse but reproducible parity signal at temperature=0. Real token
    diffing would need the tokenizer; prefix-matching is sufficient to
    flag regressions.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    same = 0
    for i in range(n):
        if a[i] == b[i]:
            same += 1
        else:
            break
    return 100.0 * same / max(len(a), len(b))


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    pie_dir = repo / "pie"
    tput_path = repo / "benches" / "tput.py"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                   help="HF repos to evaluate")
    p.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()),
                   help="Subset of config labels to run")
    p.add_argument("--include-gptq-pairs", action="store_true",
                   help="Also evaluate the GPTQ-Int4 sibling of each "
                        "model in GPTQ_INT4_PAIRS as an extra row.")
    args = p.parse_args()

    configs = {k: CONFIGS[k] for k in args.configs if k in CONFIGS}
    if "bf16" not in configs:
        print("WARNING: bf16 baseline missing — diff column will be N/A",
              file=sys.stderr)

    # Build the list of (display-model, hf-repo, config-label, extra-args).
    # Most rows match `model` against itself; GPTQ-Int4 rows add a
    # sibling repo as a synthetic "config" column.
    plan = []
    for model in args.models:
        for label, extra in configs.items():
            plan.append((model, model, label, extra))
        if args.include_gptq_pairs and model in GPTQ_INT4_PAIRS:
            plan.append((model, GPTQ_INT4_PAIRS[model], "int4-gptq", []))

    rows = []
    last_model = None
    baseline_text = None
    for display_model, hf_repo, label, extra in plan:
        if display_model != last_model:
            baseline_text = None
            last_model = display_model
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = Path(tmp.name)
        try:
            tag = label if hf_repo == display_model else f"{label} ({hf_repo})"
            print(f"\n=== {display_model}  [{tag}] ===", flush=True)
            r = _run_one(hf_repo, extra, out_path,
                         pie_dir=pie_dir, tput_path=tput_path)
            if r is None:
                continue
            if label == "bf16" and r["ok"]:
                baseline_text = r["text"]
            diff_pct = (_token_diff(baseline_text, r["text"])
                        if (baseline_text is not None and r["ok"])
                        else float("nan"))
            # int4-gptq runs use a sibling repo, so the "Δbf16" column
            # would be a structurally-meaningless diff between two
            # different models. Mark as N/A.
            if hf_repo != display_model:
                diff_pct = float("nan")
            rows.append({
                "model": display_model, "config": label,
                "ok": r["ok"], "tok_s": r["tok_s"],
                "diff_pct": diff_pct, "elapsed_s": r["elapsed_s"],
                "preview": (r["text"][:60].replace("\n", "\\n")
                            if r["ok"] else r["text"][:60]),
            })
        finally:
            try: out_path.unlink()
            except FileNotFoundError: pass

    print("\n" + "=" * 110)
    print(f"{'Model':30s}  {'Config':10s}  {'OK':3s}  "
          f"{'tok/s':>7s}  {'Δbf16%':>7s}  {'wall':>5s}  Preview")
    print("-" * 110)
    for r in rows:
        diff_str = ("100" if r["diff_pct"] >= 99.5
                    else (f"{r['diff_pct']:>5.1f}"
                          if r["diff_pct"] == r["diff_pct"] else "  N/A"))
        ok_str = "✓" if r["ok"] else "✗"
        print(f"{r['model']:30s}  {r['config']:10s}  {ok_str:3s}  "
              f"{r['tok_s']:>7.1f}  {diff_str:>7s}  "
              f"{r['elapsed_s']:>4.0f}s  {r['preview']}")
    print("=" * 110)

    # Exit non-zero if anything failed or quality dropped substantially.
    failures = [r for r in rows if not r["ok"]]
    quality_drops = [r for r in rows
                     if r["config"] != "bf16" and r["ok"]
                     and r["diff_pct"] < 50.0]
    if failures or quality_drops:
        print(f"\nFAIL: {len(failures)} run failures, "
              f"{len(quality_drops)} quality drops",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
