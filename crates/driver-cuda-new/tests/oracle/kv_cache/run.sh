#!/bin/bash
# Regenerate the golden for tests/kv_cache_parity.rs.
#
# Compiles the REAL store/kv_cache.cpp and store/kv_cache_format.cpp and builds
# a KV cache over a grid of layer stacks, formats and sharing patterns,
# reporting the exact sequence of tensor allocations each one makes.
#
# The only replaced implementation is `DeviceTensor::allocate`, whose real body
# ends in a `cudaMalloc`. Its DECLARATION comes from the verbatim
# `kernels-cuda/csrc/src/tensor.hpp`, so the shipping code still computes every
# shape; the recorder simply writes down what it was handed instead of throwing
# it away. Everything about WHICH tensors exist -- the aliasing, the scale
# tier, the dequantisation mirror, the envelope guard -- is shipping code.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Relative includes ("../model/config.hpp", "attn/kv_cache_view.hpp") resolve
# against the including file's own directory, so -I cannot shadow them: the
# files under test are COPIED into a tree whose neighbours are the stubs. The
# copy happens at build time from the shipping source, so it cannot drift the
# way a checked-in duplicate would.
mkdir -p "$WORK/store" "$WORK/attn"
cp "$SRC/store/kv_cache.cpp" "$SRC/store/kv_cache.hpp" \
   "$SRC/store/kv_cache_format.cpp" "$SRC/store/kv_cache_format.hpp" "$WORK/store/"
cp "$KSRC/tensor.hpp" "$WORK/"
cp "$KSRC/attn/kv_cache_view.hpp" "$WORK/attn/"
cp -r "$HERE/stub/." "$WORK/"

# `store/` files include their neighbours as "attn/..." and "tensor.hpp",
# i.e. relative to `src/`, so the copied store/ files need the parent on the
# include path as well as their own directory.
g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -I"$WORK" -I"$WORK/store" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$WORK/store/kv_cache.cpp" \
    "$WORK/store/kv_cache_format.cpp" "$WORK/tensor_recorder.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${KV_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$KV_ORACLE_OUT"
    echo "transcript written to $KV_ORACLE_OUT" >&2
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
