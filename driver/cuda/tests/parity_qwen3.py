"""Numeric-parity check: pie_driver_cuda's prefill vs HF Qwen3 / PyTorch.

Tokenizes a fixed prompt with HF tokenizer, dumps i32 ids to disk, runs the
C++ binary in parity mode, then runs the same prompt through transformers
on the same device. Reports max-abs diff, mean-abs diff, and cosine
similarity on the last-token logits.

Usage:
    uv run python driver/cuda/tests/parity_qwen3.py \
        --binary driver/cuda/build/bin/pie_driver_cuda \
        --hf-repo Qwen/Qwen3-0.6B \
        [--prompt "Hello world!"]
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_snapshot(hf_repo: str) -> Path:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    name = "models--" + hf_repo.replace("/", "--")
    snap_dir = cache / name / "snapshots"
    snaps = sorted(snap_dir.iterdir())
    if not snaps:
        raise SystemExit(f"no snapshots cached for {hf_repo}; download it first via `hf download {hf_repo}`")
    return snaps[-1]


def write_dev_toml(path: Path, hf_repo: str, snapshot_dir: Path, kv_cache_dtype: str) -> None:
    path.write_text(
        '[shmem]\n'
        'name = "/pie_shmem_parity"\n'
        '[model]\n'
        f'hf_repo = "{hf_repo}"\n'
        f'snapshot_dir = "{snapshot_dir}"\n'
        'device = "cuda:0"\n'
        'dtype = "bfloat16"\n'
        '[batching]\n'
        'gpu_mem_utilization = 0.90\n'
        'memory_profile = "balanced"\n'
        f'kv_cache_dtype = "{kv_cache_dtype}"\n'
    )


def run_cpp(binary: Path, dev_toml: Path, tokens_path: Path, logits_path: Path,
            paged: bool = False, decode_after_prefill: bool = False) -> int | None:
    """Run the C++ parity entry. Returns the GPU-sampled last-token id (or None
    if the binary didn't print one)."""
    cmd = [str(binary),
           "--config", str(dev_toml),
           "--parity-tokens", str(tokens_path),
           "--parity-out", str(logits_path)]
    if paged:
        cmd.append("--parity-paged")
    if decode_after_prefill:
        cmd.append("--parity-decode-after-prefill")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"pie_driver_cuda parity exited with {proc.returncode}")
    sys.stderr.write(proc.stderr)

    sampled = None
    for line in proc.stderr.splitlines():
        if "gpu argmax last-token id" in line:
            sampled = int(line.split("=")[-1].strip())
    return sampled


def load_bf16_logits(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    # Inflate bf16 → float32: place bf16 in the high 16 bits of an fp32.
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--hf-repo", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="Hello world!")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--paged", action="store_true",
                    help="Use the paged-KV forward path on the C++ side")
    ap.add_argument("--decode-after-prefill", action="store_true",
                    help="Split the C++ paged run into prefill plus one decode step")
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="kv_cache_dtype to write into the generated C++ config")
    ap.add_argument("--min-cos", type=float, default=None,
                    help="Minimum accepted last-token logit cosine")
    args = ap.parse_args()

    snapshot = find_snapshot(args.hf_repo)
    print(f"[parity] hf snapshot: {snapshot}")

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    ids = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    print(f"[parity] {len(ids)} tokens: {ids.tolist()}")

    with tempfile.TemporaryDirectory(prefix="pie-parity-") as td:
        td = Path(td)
        tokens_path = td / "tokens.bin"
        logits_path = td / "cpp_logits.bin"
        dev_toml    = td / "dev.toml"

        # i32 little-endian, matching the C++ side.
        ids.numpy().astype("<i4").tofile(tokens_path)

        write_dev_toml(dev_toml, args.hf_repo, snapshot, args.kv_cache_dtype)
        gpu_argmax = run_cpp(args.binary, dev_toml, tokens_path, logits_path,
                             paged=args.paged,
                             decode_after_prefill=args.decode_after_prefill)
        print(f"[parity] forward path: {'paged' if args.paged else 'naive'}")
        print(f"[parity] kv_cache_dtype: {args.kv_cache_dtype}")

        cpp_logits = load_bf16_logits(logits_path)

        # PyTorch / transformers reference, same device, bf16.
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype=torch.bfloat16).eval().to("cuda")
        with torch.no_grad():
            out = model(ids.unsqueeze(0).to("cuda"))
        ref = out.logits[0, -1, :].to(torch.float32).cpu().numpy()

        if cpp_logits.shape != ref.shape:
            raise SystemExit(f"shape mismatch: cpp={cpp_logits.shape} ref={ref.shape}")

        diff = cpp_logits - ref
        cos = (cpp_logits @ ref) / (np.linalg.norm(cpp_logits) * np.linalg.norm(ref) + 1e-12)

        print("\n=== last-token logits parity ===")
        print(f"  vocab           = {ref.size}")
        print(f"  ref[argmax]     = id {ref.argmax()}  ({tokenizer.decode([int(ref.argmax())])!r})")
        print(f"  cpp[argmax]     = id {cpp_logits.argmax()}  ({tokenizer.decode([int(cpp_logits.argmax())])!r})")
        print(f"  ref top-5 ids   = {ref.argsort()[-5:][::-1].tolist()}")
        print(f"  cpp top-5 ids   = {cpp_logits.argsort()[-5:][::-1].tolist()}")
        print(f"  max |diff|      = {np.max(np.abs(diff)):.4f}")
        print(f"  mean |diff|     = {np.mean(np.abs(diff)):.4f}")
        print(f"  cosine          = {cos:.6f}")

        host_argmax = int(cpp_logits.argmax())
        if gpu_argmax is None:
            print("  gpu argmax      = (binary did not report)")
        else:
            ok = "OK" if gpu_argmax == host_argmax else "MISMATCH"
            print(f"  gpu argmax      = {gpu_argmax}  (host argmax = {host_argmax})  [{ok}]")
            if gpu_argmax != host_argmax:
                return 2

        if args.keep_tmp:
            keep = Path(tempfile.gettempdir()) / "pie-parity-keep"
            keep.mkdir(exist_ok=True)
            for p in [tokens_path, logits_path, dev_toml]:
                (keep / p.name).write_bytes(p.read_bytes())
            print(f"[parity] kept tmp at {keep}")

    if args.min_cos is not None:
        min_cos = args.min_cos
    elif args.kv_cache_dtype in ("auto", "bf16", "bfloat16"):
        min_cos = 0.9999
    elif args.kv_cache_dtype in ("fp4_e2m1", "nvfp4"):
        min_cos = 0.90
    else:
        min_cos = 0.95
    return 0 if cos >= min_cos else 1


if __name__ == "__main__":
    sys.exit(main())
