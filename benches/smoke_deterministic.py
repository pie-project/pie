#!/usr/bin/env python3
"""Per-model deterministic smoke check.

Runs a single 1x8 latency request at temperature 0 / ignore_eos / argmax-only
and prints the sha256[:16] of the generated text. Drift in this sha means
either:
  (a) a real numerical change to the forward path (intended → update the
      ledger; unintended → bisect), or
  (b) prompt-template drift (the bench's chat template is the prompt
      formatter — check the model's tokenizer config didn't change).

Usage:
    python benches/smoke_deterministic.py                 # run full ledger
    python benches/smoke_deterministic.py --model nemotron_h  # one model
    python benches/smoke_deterministic.py --record         # update ledger

The ledger lives at `benches/smoke_deterministic_ledger.json` and is the
authoritative source of canonical shas. Phase 2 of the refactor (per-model
IModel migrations) uses this script as its non-regression gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benches" / "pie_bench.py"
LEDGER = ROOT / "benches" / "smoke_deterministic_ledger.json"


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # short name used on the CLI
    hf_repo: str             # HuggingFace repo
    tp_size: int             # tp_size to run with
    devices: str             # CUDA devices string
    max_model_len: int = 2048
    gpu_mem_util: float = 0.92
    trust_remote_code: bool = False
    pie_env: tuple[tuple[str, str], ...] = ()


# Each ModelSpec is the smallest representative for its arch that the
# pie cuda_native driver can serve. Sizing chosen to fit on 2x L40 (45GB
# each) at gpu_mem_util=0.92 with max_model_len=2048.
SPECS: dict[str, ModelSpec] = {
    "qwen3": ModelSpec(
        key="qwen3",
        hf_repo="Qwen/Qwen3-1.7B",
        tp_size=1,
        devices="cuda:0",
    ),
    "qwen3_6_moe": ModelSpec(
        # Qwen3.6-35B uses the qwen3_5_moe arch path. The older
        # Qwen3-30B-A3B repo trips a weight-name mismatch on this Pie rev,
        # so smoke-matrix coverage of this arch goes through 3.6 only.
        key="qwen3_6_moe",
        hf_repo="Qwen/Qwen3.6-35B-A3B",
        tp_size=2,
        devices="cuda:0,cuda:1",
    ),
    "gemma4_e2b": ModelSpec(
        # Smallest gemma4 (mixture-of-depths). Fits on 1 GPU.
        key="gemma4_e2b",
        hf_repo="google/gemma-4-E2B-it",
        tp_size=1,
        devices="cuda:0",
    ),
    "nemotron_h": ModelSpec(
        key="nemotron_h",
        hf_repo="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        tp_size=2,
        devices="cuda:0,cuda:1",
        trust_remote_code=True,
        pie_env=(
            ("PIE_NEMOTRON_FLASHINFER_MOE", "1"),
            ("PIE_NEMOTRON_FLASHINFER_MOE_DECODE", "1"),
        ),
    ),
}


SHA_RE = re.compile(r"sha256\[:16\]=([0-9a-f]{16})")


def run_one(spec: ModelSpec) -> str:
    """Run the 1x8 deterministic bench and return the captured sha256[:16]."""
    env = os.environ.copy()
    for k, v in spec.pie_env:
        env[k] = v

    cmd = [
        sys.executable,
        str(BENCH),
        "latency",
        "--driver", "cuda_native",
        "--device", spec.devices,
        "--model", spec.hf_repo,
        "--tp-size", str(spec.tp_size),
        "--gpu-mem-util", f"{spec.gpu_mem_util}",
        "--max-model-len", str(spec.max_model_len),
        "--requests", "1",
        "--warmup", "0",
        "--max-tokens", "8",
        "--temperature", "0",
        "--ignore-eos",
        "--ipc-profile", "latency",
        "--dump-first-text",
        "--no-unique-prompts",
    ]
    if spec.trust_remote_code:
        # Older Pie revs supported --trust-remote-code; the lima-evolved
        # bench dropped it (the runtime auto-trusts now). Try both forms.
        cmd_with = cmd + ["--trust-remote-code"]
        proc = subprocess.run(cmd_with, env=env, capture_output=True, text=True,
                              cwd=str(ROOT))
        if proc.returncode != 0 and "--trust-remote-code" in (proc.stderr or ""):
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  cwd=str(ROOT))
    else:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              cwd=str(ROOT))

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = SHA_RE.search(output)
    if proc.returncode != 0 and not m:
        raise RuntimeError(
            f"pie_bench.py exited {proc.returncode} for {spec.key}\n"
            f"---stdout---\n{proc.stdout[-2000:] if proc.stdout else ''}\n"
            f"---stderr---\n{proc.stderr[-2000:] if proc.stderr else ''}"
        )
    if not m:
        raise RuntimeError(
            f"no sha256[:16] line in bench output for {spec.key}\n"
            f"{output[-2000:]}"
        )
    return m.group(1)


def load_ledger() -> dict:
    if not LEDGER.exists():
        return {}
    return json.loads(LEDGER.read_text())


def save_ledger(ledger: dict) -> None:
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None,
                        help="model key (repeatable). Default: all in SPECS.")
    parser.add_argument("--record", action="store_true",
                        help="record captured shas into the ledger as the "
                             "new canonical baseline (overwrites). Use this "
                             "after an intentional numerical change.")
    args = parser.parse_args()

    keys = args.model if args.model else list(SPECS)
    unknown = [k for k in keys if k not in SPECS]
    if unknown:
        print(f"unknown model key(s): {unknown}\n"
              f"available: {sorted(SPECS)}", file=sys.stderr)
        return 2

    ledger = load_ledger()
    ok = True
    captured: dict[str, str] = {}
    for k in keys:
        spec = SPECS[k]
        try:
            sha = run_one(spec)
        except Exception as e:
            print(f"FAIL  {k:<14} (run error: {e})")
            ok = False
            continue
        captured[k] = sha
        expected = ledger.get(k)
        if expected is None:
            label = "NEW   " if not args.record else "RECORD"
            print(f"{label}{k:<14} sha256[:16]={sha}")
        elif expected == sha:
            print(f"OK    {k:<14} sha256[:16]={sha}")
        else:
            print(f"DRIFT {k:<14} sha256[:16]={sha}  "
                  f"(canonical: {expected})")
            ok = False

    if args.record:
        ledger.update(captured)
        save_ledger(ledger)
        print(f"\nLedger written: {LEDGER}")
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
