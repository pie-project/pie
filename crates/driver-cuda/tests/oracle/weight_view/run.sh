#!/bin/bash
# Regenerate the golden for tests/weight_view_parity.rs.
#
# Compiles the REAL kernels-cuda headers with NO STUBS: `weight_view.hpp`,
# `quant_meta.hpp` and `tensor.hpp` include only the standard library, and
# `DeviceTensor::view` builds a tensor without touching CUDA. Nothing here is
# mediated by a stand-in, so what the transcript measures is the shipping type
# exactly.
#
# The real `tensor.cpp` IS linked: `DeviceTensor::view` and `free_()` are
# out-of-line, and reimplementing them in the oracle would turn this into a
# test of the reimplementation. It names `cudaMalloc` and `cudaFree` and
# nothing else, and reaches neither — every tensor here is a non-owning view —
# so a stub <cuda_runtime.h> plus five one-line definitions is the whole
# replacement.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

g++ -std=c++20 -O1 -Wall -Wextra \
    -I"$HERE/stub" -I"$KSRC" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$KSRC/tensor.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${WV_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$WV_ORACLE_OUT"
    echo "transcript written to $WV_ORACLE_OUT" >&2
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
