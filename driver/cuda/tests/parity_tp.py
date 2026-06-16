"""TP self-consistency parity test.

Runs the same prompt through `pie_driver_cuda` once at tp_size=1 (reference)
and once at tp_size=2 (two ranks coordinating via NCCL). Compares rank 0's
last-token logits against the reference. Bit-exact agreement isn't expected
because bf16 reductions don't commute with the partition; we instead require
high cosine similarity, identical argmax, and identical top-5.

Usage:
    uv run python driver/cuda/tests/parity_tp.py \
        --binary driver/cuda/build/bin/pie_driver_cuda \
        --hf-repo Qwen/Qwen3-0.6B \
        [--prompt "..."] \
        [--decode-after-prefill]
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def find_snapshot(hf_repo: str) -> Path:
    local = Path(hf_repo).expanduser()
    if local.exists():
        return local.resolve()
    cache = Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    name = "models--" + hf_repo.replace("/", "--")
    snap_dir = cache / name / "snapshots"
    snaps = sorted(snap_dir.iterdir())
    if not snaps:
        raise SystemExit(f"no snapshots cached for {hf_repo}; "
                         f"download via `hf download {hf_repo}`")
    return snaps[-1]


def _toml(snapshot: Path, hf_repo: str, device: str, shmem_name: str) -> str:
    return f"""[shmem]
name = "{shmem_name}"
[model]
hf_repo = "{hf_repo}"
snapshot_dir = "{snapshot}"
device = "{device}"
dtype = "bfloat16"
[batching]
gpu_mem_utilization = 0.90
memory_profile = "balanced"
"""


def tokenize(prompt: str, hf_repo: str) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_repo)
    ids = tok(prompt, return_tensors="np")["input_ids"][0].tolist()
    return [int(x) for x in ids]


def write_tokens(ids: list[int]) -> Path:
    p = Path(tempfile.mkstemp(suffix=".bin", prefix="pie_tokens_")[1])
    with open(p, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", t))
    return p


def load_bf16(p: Path) -> np.ndarray:
    raw = np.fromfile(p, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def run_single(binary: Path, snapshot: Path, hf_repo: str,
               tokens_path: Path, *, decode_after_prefill: bool) -> Path:
    toml = Path(tempfile.mkstemp(suffix=".toml", prefix="pie_tp1_")[1])
    toml.write_text(_toml(snapshot, hf_repo, "cuda:0", "/pie_parity_tp1"))
    out = Path(tempfile.mkstemp(suffix=".bin", prefix="pie_logits_tp1_")[1])
    cmd = [str(binary), "--config", str(toml),
           "--parity-tokens", str(tokens_path),
           "--parity-out", str(out), "--parity-paged"]
    if decode_after_prefill:
        cmd.append("--parity-decode-after-prefill")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout); sys.stderr.write(proc.stderr)
        raise SystemExit(f"single-GPU parity failed (rc={proc.returncode})")
    return out


def run_tp(binary: Path, snapshot: Path, hf_repo: str,
           tokens_path: Path, tp_size: int, *,
           decode_after_prefill: bool) -> Path:
    """Spawn `tp_size` binaries; rank 0 generates the NCCL unique-id, the
    others receive it via the wrapper. Returns rank 0's logits path."""
    tomls, outs, procs = [], [], []
    rank0_log = Path(tempfile.mkstemp(suffix=".log",
                                      prefix=f"pie_tp{tp_size}_r0_")[1])
    rank0_out = Path(tempfile.mkstemp(suffix=".bin",
                                      prefix=f"pie_logits_tp{tp_size}_")[1])
    try:
        # Rank 0
        toml = Path(tempfile.mkstemp(suffix=".toml",
                                     prefix=f"pie_tp{tp_size}_r0_")[1])
        toml.write_text(_toml(snapshot, hf_repo, "cuda:0",
                              f"/pie_parity_tp{tp_size}_g0"))
        tomls.append(toml)
        outs.append(rank0_out)
        cmd = [str(binary), "--config", str(toml),
               "--tp-size", str(tp_size), "--tp-rank", "0",
               "--parity-tokens", str(tokens_path),
               "--parity-out", str(rank0_out), "--parity-paged"]
        if decode_after_prefill:
            cmd.append("--parity-decode-after-prefill")
        procs.append(subprocess.Popen(
            cmd, stdout=open(rank0_log, "w"), stderr=subprocess.STDOUT))

        # Wait for rank 0 to print its NCCL_UID line.
        deadline = time.monotonic() + 30
        uid_hex = None
        while time.monotonic() < deadline:
            try:
                for line in rank0_log.read_text().splitlines():
                    if line.startswith("NCCL_UID "):
                        uid_hex = line.split(maxsplit=1)[1].strip()
                        break
            except FileNotFoundError:
                pass
            if uid_hex is not None:
                break
            time.sleep(0.1)
        if uid_hex is None:
            procs[0].kill()
            sys.stderr.write(rank0_log.read_text() if rank0_log.exists() else "")
            raise SystemExit("rank 0 didn't emit NCCL_UID within 30s")

        # Followers
        for r in range(1, tp_size):
            t = Path(tempfile.mkstemp(suffix=".toml",
                                      prefix=f"pie_tp{tp_size}_r{r}_")[1])
            t.write_text(_toml(snapshot, hf_repo, f"cuda:{r}",
                               f"/pie_parity_tp{tp_size}_g{r}"))
            tomls.append(t)
            o = Path(tempfile.mkstemp(suffix=".bin",
                                      prefix=f"pie_logits_tp{tp_size}_r{r}_")[1])
            outs.append(o)
            cmd = [str(binary), "--config", str(t),
                   "--tp-size", str(tp_size), "--tp-rank", str(r),
                   "--nccl-unique-id-hex", uid_hex,
                   "--parity-tokens", str(tokens_path),
                   "--parity-out", str(o), "--parity-paged"]
            if decode_after_prefill:
                cmd.append("--parity-decode-after-prefill")
            procs.append(subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

        for p in procs:
            rc = p.wait(timeout=180)
            if rc != 0:
                sys.stderr.write(rank0_log.read_text())
                raise SystemExit(f"TP rank exited rc={rc}")
        return rank0_out
    finally:
        for t in tomls:
            try: t.unlink(missing_ok=True)
            except: pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, default=Path(
        __file__).resolve().parents[1] / "build/bin/pie_driver_cuda")
    p.add_argument("--hf-repo", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="The quick brown fox jumps over the "
                                       "lazy dog and runs into the forest.")
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--decode-after-prefill", action="store_true")
    p.add_argument("--cos-threshold", type=float, default=0.999,
                   help="Min cosine similarity to pass.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K must match exactly between TP and reference.")
    a = p.parse_args()

    if not a.binary.exists():
        raise SystemExit(f"binary not found: {a.binary}")
    snapshot = find_snapshot(a.hf_repo)
    print(f"[parity-tp] snapshot={snapshot}")
    print(f"[parity-tp] tokenizing prompt ({len(a.prompt)} chars)")
    ids = tokenize(a.prompt, a.hf_repo)
    print(f"[parity-tp] {len(ids)} tokens")
    tokens = write_tokens(ids)

    try:
        print("[parity-tp] single-GPU reference run...")
        ref_path = run_single(a.binary, snapshot, a.hf_repo, tokens,
                              decode_after_prefill=a.decode_after_prefill)
        print(f"[parity-tp] tp={a.tp_size} run...")
        tp_path = run_tp(a.binary, snapshot, a.hf_repo, tokens, a.tp_size,
                         decode_after_prefill=a.decode_after_prefill)

        ref = load_bf16(ref_path)
        tp = load_bf16(tp_path)
        cos = float((ref @ tp) /
                    (np.linalg.norm(ref) * np.linalg.norm(tp)))
        diff = np.abs(ref - tp)
        ref_top = np.argsort(ref)[-a.top_k:][::-1].tolist()
        tp_top  = np.argsort(tp)[-a.top_k:][::-1].tolist()
        print(f"[parity-tp] vocab={ref.size}")
        print(f"[parity-tp] max abs diff = {diff.max():.4f}")
        print(f"[parity-tp] mean abs diff = {diff.mean():.4f}")
        print(f"[parity-tp] cosine sim   = {cos:.6f}")
        print(f"[parity-tp] argmax ref={ref.argmax()}  tp={tp.argmax()}")
        print(f"[parity-tp] top-{a.top_k} ref = {ref_top}")
        print(f"[parity-tp] top-{a.top_k} tp  = {tp_top}")

        # Top-K is checked as a set, not an ordering. With bf16 logits two
        # near-tied tokens can swap positions across runs (e.g. when one
        # branch of an all-reduce sums in a slightly different order). The
        # set tells us "the model still puts the same K candidates on top";
        # the cosine + argmax checks pin down magnitude + dominant choice.
        ok = (cos >= a.cos_threshold and
              ref.argmax() == tp.argmax() and
              set(ref_top) == set(tp_top))
        if ok:
            print(f"[parity-tp] PASS (cos>={a.cos_threshold}, top-{a.top_k} set match)")
            return 0
        print(f"[parity-tp] FAIL")
        return 2
    finally:
        try: tokens.unlink(missing_ok=True)
        except: pass


if __name__ == "__main__":
    sys.exit(main())
