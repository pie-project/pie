#!/bin/bash
# Regenerate the golden for tests/recurrent_parity.rs.
#
# Compiles the REAL store/recurrent_state_cache.cpp and drives it over a grid
# of hybrid layer stacks, recording every allocation, every stream operation
# and every accessor offset.
#
# Two implementations are replaced, both of them the layer where the answer
# would otherwise be thrown away:
#
#   * <cuda_runtime.h> -- a recording stand-in. The memsets and memcpys ARE
#     the behaviour under test, and a real one returns cudaSuccess whatever
#     geometry it is handed.
#   * kernels::layout::zero_slots_if_fresh -- a device kernel, recorded by its
#     geometry arguments.
#
# Everything that decides WHAT to issue -- the dense linear-layer compaction,
# the slot strides, the pitch arithmetic, the guards, the exception messages --
# is the shipping source, copied verbatim at build time.
#
# `KSRC` below points into the ARCHIVE crate `kernels-cuda` -- CMake+nvcc, a
# `csrc/` of host `.hpp` and `.cpp` -- deleted whole at `85c6c674b`, and the
# `tensor.hpp` this copies out of it went with it. The line is left unchanged
# because it records the command that took the golden, not a path anyone can
# follow.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Relative includes ("../device_buffer.hpp") resolve against the including
# file's own directory, so -I cannot shadow them: the files under test are
# COPIED into a tree whose neighbours are the stubs. The copy happens at build
# time from the shipping source, so it cannot drift the way a checked-in
# duplicate would.
mkdir -p "$WORK/store" "$WORK/layout"
cp "$SRC/store/recurrent_state_cache.cpp" \
   "$SRC/store/recurrent_state_cache.hpp" "$WORK/store/"
cp "$SRC/device_buffer.hpp" "$SRC/runahead.hpp" "$WORK/"
cp "$KSRC/tensor.hpp" "$WORK/"
cp -r "$HERE/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -I"$WORK" -I"$WORK/store" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$WORK/store/recurrent_state_cache.cpp" \
    "$WORK/cuda_recorder.cpp" "$WORK/layout/slot_ops_recorder.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${RS_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$RS_ORACLE_OUT"
    echo "transcript written to $RS_ORACLE_OUT" >&2
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
