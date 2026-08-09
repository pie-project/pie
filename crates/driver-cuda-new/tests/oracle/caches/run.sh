#!/bin/bash
# Regenerate the golden for tests/caches_parity.rs.
#
# Compiles the REAL store/mla_cache.cpp, store/dsv4_compress_cache.cpp and
# store/swap_pool.cpp (plus store/kv_cache.cpp, which the swap pool's
# cache-driven constructor needs) and reports exactly what memory each
# allocation path asks for, in what order.
#
# Two implementations are replaced, both of them the point where a request
# leaves for the driver and stops being observable:
#
#   * `DeviceTensor::allocate`, whose real body ends in a `cudaMalloc`. Its
#     DECLARATION comes from the verbatim `kernels-cuda/csrc/src/tensor.hpp`.
#   * `<cuda_runtime.h>` itself, replaced by a recorder found first on the
#     include path. `cudaMallocHost` records its request size; `cudaMemset`
#     records its target and can be made to fail, which is the only way to
#     exercise the dsv4 cache's best-effort zeroing.
#
# Everything about WHICH allocations happen, how large they are and in what
# order is shipping code.
#
# The COPY half of swap_pool.cpp is proved separately by tests/oracle/store;
# this covers the two constructors, which that oracle does not reach.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Relative includes ("model/config.hpp", "attn/kv_cache_view.hpp") resolve
# against the including file's own directory, so -I cannot shadow them: the
# files under test are COPIED into a tree whose neighbours are the stubs. The
# copy happens at build time from the shipping source, so it cannot drift the
# way a checked-in duplicate would.
mkdir -p "$WORK/store" "$WORK/attn"
cp "$SRC/store/mla_cache.cpp" "$SRC/store/mla_cache.hpp" \
   "$SRC/store/dsv4_compress_cache.cpp" "$SRC/store/dsv4_compress_cache.hpp" \
   "$SRC/store/swap_pool.cpp" "$SRC/store/swap_pool.hpp" \
   "$SRC/store/kv_cache.cpp" "$SRC/store/kv_cache.hpp" \
   "$SRC/store/kv_cache_format.cpp" "$SRC/store/kv_cache_format.hpp" \
   "$WORK/store/"
cp "$KSRC/tensor.hpp" "$WORK/"
cp "$KSRC/attn/kv_cache_view.hpp" "$KSRC/attn/mla_cache_view.hpp" "$WORK/attn/"
cp -r "$HERE/stub/." "$WORK/"

# `store/` files include their neighbours as "attn/..." and "tensor.hpp", i.e.
# relative to `src/`, so the copied store/ files need the parent on the include
# path as well as their own directory. $WORK comes first so the stub
# `cuda_runtime.h` is found ahead of any real one.
g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -I"$WORK" -I"$WORK/store" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" \
    "$WORK/store/mla_cache.cpp" \
    "$WORK/store/dsv4_compress_cache.cpp" \
    "$WORK/store/swap_pool.cpp" \
    "$WORK/store/kv_cache.cpp" \
    "$WORK/store/kv_cache_format.cpp" \
    "$WORK/tensor_recorder.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${CACHES_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$CACHES_ORACLE_OUT"
    echo "transcript written to $CACHES_ORACLE_OUT" >&2
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
