#!/usr/bin/env python3
"""Write Engram's tokenizer-compressed id map beside a DeepSeek-V4.1 checkpoint.

Engram (DeepSeek-V4.1-Flash) hashes n-grams over a *compressed* id space in
which tokens that normalise alike (" The", "the", "THE") share one id; every
hash multiplier is derived from the size of that space, so the map has to be
exactly the one the reference derives — `build_compressed_token_map` in the
release's `inference/engram.py`, reproduced here over the same HF `tokenizers`
normalizers. pie reads the result as the plane `engram.token_map` (one i32 per
vocabulary row) from `engram_token_map.safetensors` in the snapshot directory.

    python scripts/bench/engram_token_map.py /path/to/DeepSeek-V4.1-Flash-snapshot

Checks the compressed vocabulary size against `engram_compressed_vocab_size`
in config.json when the config states one.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

PLANE = "engram.token_map"
FILE = "engram_token_map.safetensors"


def build_compressed_token_map(tokenizer_json: Path) -> tuple[list[int], int]:
    from tokenizers import Regex, Tokenizer, normalizers

    tok = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * vocab_size
    for token_id in range(vocab_size):
        text = tok.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(token_id) or ""
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def write_plane(path: Path, lookup: list[int]) -> None:
    data = struct.pack(f"<{len(lookup)}i", *lookup)
    header = json.dumps(
        {PLANE: {"dtype": "I32", "shape": [len(lookup)], "data_offsets": [0, len(data)]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((-len(header)) % 8)
    with path.open("wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
        f.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", help="checkpoint directory holding tokenizer.json (and config.json)")
    ap.add_argument("--out", default=None, help=f"output file (default: <snapshot>/{FILE})")
    args = ap.parse_args()
    snap = Path(args.snapshot).expanduser()
    tokenizer_json = snap / "tokenizer.json"
    if not tokenizer_json.exists():
        print(f"{tokenizer_json}: not found", file=sys.stderr)
        return 2
    lookup, compressed = build_compressed_token_map(tokenizer_json)
    cfg_path = snap / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        tc = cfg.get("text_config", cfg)
        want = tc.get("engram_compressed_vocab_size")
        if isinstance(want, int) and want != compressed:
            print(
                f"compressed vocabulary is {compressed} tokens and config.json states "
                f"{want}; the hashing would not be the checkpoint's",
                file=sys.stderr,
            )
            return 1
    out = Path(args.out) if args.out else snap / FILE
    write_plane(out, lookup)
    print(f"{out}: {len(lookup)} ids -> {compressed} compressed (pad {lookup[2] if len(lookup) > 2 else '?'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
