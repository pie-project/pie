#!/usr/bin/env python3
"""Quantization logit-parity check.

Runs `pie_driver_cuda` in parity mode twice — once with the bf16
baseline config, once with a quantized config — on the same fixed
prompt, then compares the dumped last-token logits via cosine
similarity and max-abs error.

This is a stricter quality signal than `parity_table.py`'s greedy-text
prefix match: it catches subtle numerical drift in the lm_head output
that may not affect token sampling but does indicate quantization
quality issues.

Usage:
    uv run python benches/logit_parity.py \
        --binary driver/cuda/build/bin/pie_driver_cuda \
        --model Qwen/Qwen3-0.6B \
        --quant fp8

Exit non-zero if cosine sim drops below `--cosine-min` (default 0.999)
or max-abs error exceeds `--max-abs-err` (default 0.1).
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# Each variant is one row in the comparison: a label + the toml fragment
# that customizes [model] beyond the bf16 default.
QUANT_TOMLS = {
    "bf16":      "",                                 # baseline
    "fp8":       'runtime_quant = "fp8"\n',          # runtime W8A8 fp8
    "int8":      'runtime_quant = "int8"\n',         # runtime W8A8 int8
}

PROMPT = "Hello world!"


def find_snapshot(hf_repo: str) -> Path:
    cache = (Path(os.environ.get("HF_HOME",
                                  Path.home() / ".cache" / "huggingface"))
             / "hub")
    snap_dir = cache / ("models--" + hf_repo.replace("/", "--")) / "snapshots"
    snaps = sorted(snap_dir.iterdir())
    if not snaps:
        raise SystemExit(
            f"no snapshots cached for {hf_repo}; "
            f"run `hf download {hf_repo}` first")
    return snaps[-1]


def write_dev_toml(path: Path, hf_repo: str, snapshot_dir: Path,
                   model_extra: str) -> None:
    path.write_text(
        '[shmem]\n'
        'name = "/pie_shmem_logit_parity"\n'
        '[model]\n'
        f'hf_repo = "{hf_repo}"\n'
        f'snapshot_dir = "{snapshot_dir}"\n'
        'device = "cuda:0"\n'
        'dtype = "bfloat16"\n'
        + model_extra +
        '[batching]\n'
        'kv_page_size = 32\n'
        'max_num_kv_pages = 1024\n'
        'max_batch_tokens = 4096\n'
        'max_batch_size = 8\n'
    )


def run_cpp(binary: Path, dev_toml: Path, tokens_path: Path,
            logits_path: Path) -> None:
    """Run the binary's parity entry. Logits get dumped to `logits_path`
    as raw bf16 [vocab]."""
    cmd = [str(binary),
           "--config", str(dev_toml),
           "--parity-tokens", str(tokens_path),
           "--parity-out", str(logits_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"pie_driver_cuda parity exited with {proc.returncode}")


def load_bf16_logits(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    # Inflate bf16 → fp32 by placing bf16 bits in the high 16 of an fp32.
    return (raw.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    default_binary = repo / "driver" / "cuda" / "build" / "bin" / "pie_driver_cuda"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=default_binary)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--quant", default="fp8",
                    choices=list(QUANT_TOMLS.keys()),
                    help="Variant to compare against bf16")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--cosine-min", type=float, default=0.999,
                    help="Fail if cosine(bf16, quant) < this")
    ap.add_argument("--max-abs-err", type=float, default=0.1,
                    help="Fail if max|bf16 - quant| > this")
    args = ap.parse_args()

    if args.quant == "bf16":
        print("--quant must differ from bf16; nothing to compare against")
        return 0

    snap = find_snapshot(args.model)
    print(f"[logit-parity] model      = {args.model}")
    print(f"[logit-parity] snapshot   = {snap}")
    print(f"[logit-parity] variant    = {args.quant}")
    print(f"[logit-parity] prompt     = {args.prompt!r}")

    # Tokenize once via the snapshot's tokenizer (lazy import — keeps the
    # script runnable even when transformers is missing for the bf16-only
    # case). For our quant comparison we always need it.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok(args.prompt, add_special_tokens=False,
              return_tensors="pt").input_ids[0]
    print(f"[logit-parity] tokens     = {ids.tolist()}")

    with tempfile.TemporaryDirectory(prefix="pie-logit-parity-") as td:
        td = Path(td)
        tokens_path = td / "tokens.bin"
        ids.numpy().astype("<i4").tofile(tokens_path)

        results = {}
        for label, extra in [("bf16", QUANT_TOMLS["bf16"]),
                             (args.quant, QUANT_TOMLS[args.quant])]:
            dev_toml = td / f"dev_{label}.toml"
            logits_path = td / f"logits_{label}.bin"
            write_dev_toml(dev_toml, args.model, snap, extra)
            print(f"[logit-parity] running [{label}]…", flush=True)
            run_cpp(args.binary, dev_toml, tokens_path, logits_path)
            results[label] = load_bf16_logits(logits_path)

        a = results["bf16"]
        b = results[args.quant]
        if a.shape != b.shape:
            print(f"shape mismatch: bf16={a.shape} {args.quant}={b.shape}",
                  file=sys.stderr)
            return 1

        diff = a - b
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        argmax_match = int(a.argmax()) == int(b.argmax())

        # Top-K overlap on logit ranking — informative diagnostic.
        k = 5
        top_a = set(a.argsort()[-k:].tolist())
        top_b = set(b.argsort()[-k:].tolist())
        topk_overlap = len(top_a & top_b)

        print("\n=== last-token logit parity ===")
        print(f"  vocab           = {a.size}")
        print(f"  cosine sim      = {cos:.6f}")
        print(f"  max |diff|      = {max_abs:.4f}")
        print(f"  mean |diff|     = {mean_abs:.4f}")
        print(f"  argmax match    = {'YES' if argmax_match else 'NO'}")
        print(f"  top-{k} overlap   = {topk_overlap}/{k}")

        ok = (cos >= args.cosine_min) and (max_abs <= args.max_abs_err)
        if not ok:
            print(f"\nFAIL: cosine={cos:.6f} (>= {args.cosine_min}?) "
                  f"max_abs={max_abs:.4f} (<= {args.max_abs_err}?)",
                  file=sys.stderr)
            return 1
        print("\nPASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
