#!/usr/bin/env python3
"""Pin TP's threaded NCCL communicator initialization to blocking mode."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else Path(__file__).resolve().parents[1] / "src" / "distributed.cpp"
    )
    source = path.read_text(encoding="utf-8")
    start = source.index("NcclComm::NcclComm(int world_size")
    end = source.index("\nNcclComm::~NcclComm()", start)
    body = source[start:end]
    modes = re.findall(r"config\.blocking\s*=\s*([01])\s*;", body)
    if modes != ["1"]:
        print(
            "FAIL: threaded TP communicator initialization must use blocking mode; "
            f"observed assignments={modes}",
            file=sys.stderr,
        )
        return 1
    print("PASS: threaded TP communicator initialization uses blocking mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
