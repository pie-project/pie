#!/bin/bash
# Regenerate the golden for tests/sideband_arena_parity.rs.
#
# Compiles the REAL model/hook_sideband_arena.cpp and drives it through a
# scripted sequence of acquire/release/begin_fire calls, reporting for each the
# block it returned, the generation, and the device-allocator traffic it caused.
#
# The only replaced implementations are `cudaMalloc`, `cudaFree` and
# `cudaStreamSynchronize`, defined in oracle.cpp. They are replaced not merely
# to drop the GPU dependency but because the growth path frees the old block
# before it learns whether a replacement exists — an ordering that can only be
# observed if the allocator can be told to fail on cue.
#
# Growth logging goes to stderr in the shipping code and is discarded here; the
# transcript is stdout, so a changed log message does not churn the golden while
# a changed ALLOCATION does.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# `hook_sideband_arena.cpp` includes its own header as "model/...", i.e.
# relative to `src/`, so the file under test is COPIED into a tree whose
# neighbours are the stubs. The copy happens at build time from the shipping
# source, so it cannot drift the way a checked-in duplicate would.
mkdir -p "$WORK/model"
cp "$SRC/model/hook_sideband_arena.cpp" "$SRC/model/hook_sideband_arena.hpp" \
   "$WORK/model/"
cp -r "$HERE/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wextra \
    -I"$WORK" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$WORK/model/hook_sideband_arena.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" 2>/dev/null > "$OUT"

if [[ -n "${SB_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$SB_ORACLE_OUT"
    echo "transcript written to $SB_ORACLE_OUT" >&2
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
