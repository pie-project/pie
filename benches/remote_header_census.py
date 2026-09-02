#!/usr/bin/env python3
"""Read a remote MLX checkpoint's METADATA without downloading the weights.

A census asks three questions of an artifact — which planes are in it, what
rectangle each one is, and at which `(bits, group)` point it was written — and
all three live in the safetensors HEADER, which is the first few kilobytes of
each shard. The payload is the other sixty-eight gigabytes and answers none of
them.

So this fetches the headers and nothing else. `config.json`, the shard index,
one ranged read per shard, and a handful of named small buffers pulled by their
own byte offsets: a few megabytes against an artifact that would not fit on the
disk that is asking. The output is a MANIFEST — every tensor's stored dtype and
shape, the config verbatim, and the requested buffers as integer lists — in the
shape `models`' census tests already read a local snapshot in, so the SAME test
runs here against the remote metadata and there against the downloaded bytes:

    python3 benches/remote_header_census.py --out /tmp/q38-full.json
    PIE_HEADER_MANIFEST=/tmp/q38-full.json cargo test -p models --test \\
        the_qwen4_text_reads_the_full_two_bit_flash

The range machinery is `shrink_checkpoint.py`'s, restated at the size this
needs rather than imported: that script is a carving tool with a plan and a
writer behind it, and a header read wants none of that.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# One ranged read per shard, sized so a header fits in it. The largest shard
# header in this family is under twenty kilobytes; a shard that exceeds this
# gets a second, exact request rather than a wrong answer.
HEAD_WINDOW = 1 << 18

# The buffers a census reads the CONTENTS of rather than the shape: the PLE's
# three published hash tables, which every text of this family derives instead
# of reading and holds its derivation against. Matched as name suffixes.
BUFFER_SUFFIXES = (
    ".ple_embedding.layer_multipliers",
    ".ple_embedding.ngram_heads_offsets",
    ".ple_embedding.ngram_heads_vocab_sizes",
)

BUFFER_UNPACK = {"I64": ("<q", 8), "I32": ("<i", 4), "U32": ("<I", 4)}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _token() -> str | None:
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    path = Path.home() / ".cache/huggingface/token"
    if path.is_file():
        tok = path.read_text().strip()
        if tok:
            return tok
    return None


def _get(url: str, span: tuple[int, int] | None = None, retries: int = 6) -> bytes:
    headers = {"User-Agent": "pie-remote-header-census/1.0"}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if span is not None:
        headers["Range"] = f"bytes={span[0]}-{span[1] - 1}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def _file_url(repo: str, revision: str, name: str) -> str:
    return f"{HF_ENDPOINT}/{repo}/resolve/{revision}/{name}"


# --------------------------------------------------------------------------- #
# safetensors headers
# --------------------------------------------------------------------------- #
def shard_header(repo: str, revision: str, shard: str) -> tuple[dict, int]:
    """`(header, payload_base)` for one remote shard, in one or two requests."""
    url = _file_url(repo, revision, shard)
    window = _get(url, (0, HEAD_WINDOW))
    if len(window) < 8:
        raise RuntimeError(f"{shard}: short read on the header length")
    (length,) = struct.unpack("<Q", window[:8])
    if 8 + length <= len(window):
        blob = window[8 : 8 + length]
    else:
        blob = _get(url, (8, 8 + length))
    return json.loads(blob), 8 + length


def census(repo: str, revision: str, workers: int) -> dict:
    config = json.loads(_get(_file_url(repo, revision, "config.json")))
    index = json.loads(
        _get(_file_url(repo, revision, "model.safetensors.index.json"))
    )
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"{len(weight_map)} tensors over {len(shards)} shards", file=sys.stderr)

    tensors: dict[str, dict] = {}
    spans: dict[str, tuple[str, str, int, int]] = {}

    def one(shard: str) -> tuple[str, dict, int]:
        header, base = shard_header(repo, revision, shard)
        return shard, header, base

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for shard, header, base in pool.map(one, shards):
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                tensors[name] = {
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                }
                start, end = entry["data_offsets"]
                if name.endswith(BUFFER_SUFFIXES):
                    spans[name] = (shard, entry["dtype"], base + start, base + end)
            done += 1
            if done % 16 == 0 or done == len(shards):
                print(f"  headers {done}/{len(shards)}", file=sys.stderr)

    buffers: dict[str, list[int]] = {}
    for name, (shard, dtype, start, end) in sorted(spans.items()):
        fmt, width = BUFFER_UNPACK.get(dtype, (None, 0))
        if fmt is None:
            print(f"  skipping buffer `{name}`: stored {dtype}", file=sys.stderr)
            continue
        raw = _get(_file_url(repo, revision, shard), (start, end))
        buffers[name] = [
            struct.unpack(fmt, raw[i : i + width])[0] for i in range(0, len(raw), width)
        ]
        print(f"  buffer {name}: {len(buffers[name])} values", file=sys.stderr)

    return {
        "repo": repo,
        "revision": revision,
        "config": config,
        "total_size": index.get("metadata", {}).get("total_size"),
        "tensors": tensors,
        "buffers": buffers,
    }


# --------------------------------------------------------------------------- #
# the summary a run prints
# --------------------------------------------------------------------------- #
def summarize(manifest: dict) -> None:
    config = manifest["config"]
    text = config.get("text_config", config)
    quant = config.get("quantization", {})
    tensors = manifest["tensors"]

    scalar = {k: v for k, v in quant.items() if not isinstance(v, dict)}
    per_plane = {k: v for k, v in quant.items() if isinstance(v, dict)}
    points: dict[tuple[int, int], int] = {}
    for spec in per_plane.values():
        key = (spec.get("bits"), spec.get("group_size"))
        points[key] = points.get(key, 0) + 1

    quantized = sorted(n[: -len(".scales")] for n in tensors if n.endswith(".scales"))
    bare = sorted(
        n
        for n in tensors
        if n.endswith(".weight") and n[: -len(".weight")] + ".scales" not in tensors
    )

    print(f"\n== {manifest['repo']} @ {manifest['revision']} ==")
    print(f"total_size          {manifest['total_size']}")
    print(f"tensors             {len(tensors)}")
    print(f"quantized triplets  {len(quantized)}")
    print(f"bare .weight planes {len(bare)}")
    print(f"quantization default {scalar}")
    print(f"per-plane overrides  {len(per_plane)}  {points}")
    for key in (
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "hidden_size",
        "vocab_size",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
        "split_ngram_parts",
        "heads_per_ngram",
        "ngram_size",
        "ple_layer_ids",
        "full_attention_interval",
    ):
        if key in text:
            print(f"  {key:38} {text[key]}")
    for name, values in manifest["buffers"].items():
        head = values[: min(4, len(values))]
        print(f"  buffer {name.rsplit('.', 1)[-1]:26} n={len(values)} {head}...")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = census(args.repo, args.revision, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest))
    summarize(manifest)
    print(f"\nmanifest -> {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
