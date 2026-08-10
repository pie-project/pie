#!/usr/bin/env python3
"""Guard the NCCL init handle against unsequenced sibling evaluation."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def constructor(source: str) -> str:
    start = source.index("NcclComm::NcclComm(int world_size")
    end = source.index("\nNcclComm::~NcclComm()", start)
    return source[start:end]


def main() -> int:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else Path(__file__).resolve().parents[1] / "src" / "distributed.cpp"
    )
    body = constructor(path.read_text(encoding="utf-8"))
    bad = re.compile(
        r"NCCL_CHECK_ASYNC\s*\(\s*"
        r"ncclCommInitRankConfig\s*\(\s*&comm_.*?\)\s*,\s*comm_\s*\)",
        re.DOTALL,
    )
    if bad.search(body):
        print(
            "FAIL: ncclCommInitRankConfig output and comm_ are sibling "
            "arguments; the checker can observe the old handle",
            file=sys.stderr,
        )
        return 1

    init = re.search(
        r"const\s+ncclResult_t\s+init_result\s*=\s*"
        r"ncclCommInitRankConfig\s*\(\s*&comm_.*?;",
        body,
        re.DOTALL,
    )
    check = re.search(
        r"NCCL_CHECK_ASYNC\s*\(\s*init_result\s*,\s*comm_\s*\)\s*;",
        body,
    )
    if init is None or check is None or init.end() > check.start():
        print(
            "FAIL: communicator init result is not stored before comm_ is checked",
            file=sys.stderr,
        )
        return 1

    print("PASS: communicator handle write is sequenced before async checking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
