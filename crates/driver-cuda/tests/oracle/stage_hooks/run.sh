#!/bin/bash
# Regenerate the golden for tests/stage_hooks_parity.rs — gate-stage-hooks.
#
# The header is the whole implementation, so the oracle is one TU: the REAL
# stage_hooks.hpp over a one-line CUDA stub, driven by a recording execute.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/model"
cp "$SRC/model/stage_hooks.hpp" "$WORK/model/"
cp -r "$HERE/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wextra \
    -I"$WORK" \
    -o "$WORK/oracle" "$HERE/oracle.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${SH_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$SH_ORACLE_OUT"
    echo "transcript written to $SH_ORACLE_OUT" >&2
fi

python3 - "$OUT" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
h = 0xcbf29ce484222325
for b in data:
    h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
print(f"GOLDEN_FNV1A64 = 0x{h:016x}")
print(f"GOLDEN_ROWS    = {sum(1 for _ in open(sys.argv[1]))}")
PY
