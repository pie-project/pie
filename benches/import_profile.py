#!/usr/bin/env python3
"""Profile `pie model import`: wall time, CPU split, and disk IO per checkpoint.

The page cache is the whole difficulty in measuring this. The box has 755 GB of
RAM and the checkpoints are 13-67 GB, so once a model has been downloaded every
subsequent read is served from memory and a "cold" import cannot be observed by
waiting. `/proc/sys/vm/drop_caches` is read-only in this container, so the cache
is evicted per file with `posix_fadvise(POSIX_FADV_DONTNEED)`, which needs no
privilege and touches only the files named.
"""

import argparse
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PIE = Path(os.environ.get("PIE_BIN", Path(__file__).resolve().parent.parent / "target/release/pie"))
HUB = Path.home() / ".cache/huggingface/hub"


def snapshot_dir(repo: str) -> Path:
    """The local snapshot for a repo id, or a clear error naming what is missing."""
    root = HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not root.is_dir():
        sys.exit(f"{repo} is not in the local HF cache ({root} missing)")
    snaps = sorted(root.iterdir())
    if not snaps:
        sys.exit(f"{repo} has no snapshot under {root}")
    return snaps[-1]


def evict(paths) -> int:
    """Drop the page cache for `paths`, returning the bytes evicted."""
    total = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
            total += size
        finally:
            os.close(fd)
    return total


def weight_files(snap: Path):
    return [p for p in snap.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".gguf"}]


def run(repo: str, cold: bool, out: Path | None):
    local = Path(repo)
    if local.exists():
        snap = local if local.is_dir() else local.parent
        sources = [local] if local.is_file() else weight_files(snap)
    else:
        snap = snapshot_dir(repo)
        sources = weight_files(snap)
    src_bytes = sum(p.stat().st_size for p in sources)

    cmd = [str(PIE), "model", "import", repo, "--force"]
    if out is not None:
        cmd += ["--out", str(out)]

    if cold:
        evicted = evict(sources)
    else:
        evicted = 0

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)

    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        sys.exit(f"import failed for {repo}: {proc.stderr[-2000:]}")

    user = after.ru_utime - before.ru_utime
    sysc = after.ru_stime - before.ru_stime
    read_gib = (after.ru_inblock - before.ru_inblock) * 512 / 2**30
    write_gib = (after.ru_oublock - before.ru_oublock) * 512 / 2**30
    src_gib = src_bytes / 2**30

    decode_line = next(
        (l for l in proc.stdout.splitlines() if "decode" in l and "copy" in l), ""
    )

    print(f"== {repo} ({'cold' if cold else 'warm'}) ==")
    if decode_line:
        print(f"   {decode_line.strip()}")
    print(f"   source        {src_gib:.1f} GiB in {len(sources)} file(s)")
    if cold:
        print(f"   evicted       {evicted / 2**30:.1f} GiB from page cache")
    print(f"   wall          {wall:.1f} s   ->  {src_gib / wall:.2f} GiB/s of source")
    print(f"   cpu           user {user:.1f}s  sys {sysc:.1f}s  "
          f"({(user + sysc) / wall * 100:.0f}% of one core)")
    print(f"   disk          read {read_gib:.1f} GiB   write {write_gib:.1f} GiB")
    print(f"   peak rss      {after.ru_maxrss / 2**20:.2f} GiB")
    return wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--warm", action="store_true", help="do not evict the page cache first")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    for repo in args.repos:
        out = None
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out = args.out_dir / (repo.replace("/", "--") + ".zt")
        run(repo, cold=not args.warm, out=out)


if __name__ == "__main__":
    main()
