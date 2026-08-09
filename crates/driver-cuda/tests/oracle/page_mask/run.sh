#!/bin/bash
# Regenerate the golden for tests/page_mask_parity.rs.
#
# Compiles the REAL model/attn_page_mask.cu together with the REAL
# model/hook_sideband_arena.cpp and drives them over a grid of fire geometries.
#
# `attn_page_mask.cu` carries no device code — it calls the compaction kernel
# through the `kernels::attn` wrapper — so it builds with g++ once
# <cuda_runtime.h> is stubbed. That matters: the file under test is the
# shipping one, not a transcription of it.
#
# Replaced: the four CUDA entry points and `compact_page_csr`, all defined in
# oracle.cpp. The kernel is replaced by a RECORDER rather than a no-op, because
# half of what this oracle checks is which carved buffer lands in which of the
# kernel's ten pointer parameters.
#
# The `-x c++` is because g++ reads a bare `.cu` as a linker script. The file
# needs no nvcc: it contains no device code.
#
# `attn/page_compact.hpp` is copied from kernels-cuda at build time rather than
# kept as a stub, so a change to the kernel's signature breaks this build
# instead of being silently absorbed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Both files include their neighbours as "model/...", i.e. relative to `src/`,
# so they are COPIED into a tree whose other neighbours are the stubs. Copying
# at build time from the shipping source is what stops this from drifting the
# way a checked-in duplicate would.
mkdir -p "$WORK/model" "$WORK/attn"
cp "$SRC/model/attn_page_mask.cu" "$SRC/model/attn_page_mask.hpp" \
   "$SRC/model/hook_sideband_arena.cpp" "$SRC/model/hook_sideband_arena.hpp" \
   "$WORK/model/"
cp "$KSRC/attn/page_compact.hpp" "$WORK/attn/"
cp -r "$HERE/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wextra \
    -I"$WORK" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" \
    -x c++ "$WORK/model/attn_page_mask.cu" -x none \
    "$WORK/model/hook_sideband_arena.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" 2>/dev/null > "$OUT"

if [[ -n "${PM_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$PM_ORACLE_OUT"
    echo "transcript written to $PM_ORACLE_OUT" >&2
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
