#!/bin/bash
# Regenerate the golden for tests/cublas_handle_parity.rs — gate-cublas.
#
# The whole real gemm.cpp over a full cublas/cublasLt declaration stub,
# -ffunction-sections + --gc-sections; only the CublasHandle members are
# driven, through the five recorders in oracle.cpp.
#
# `KSRC` below points into the ARCHIVE crate `kernels-cuda` — CMake+nvcc, a
# `csrc/` of host `.hpp` and `.cpp` — deleted whole at `85c6c674b`, and
# `gemm/gemm.{cpp,hpp}` went before it. The line is left unchanged because it
# records the command that took the golden, not a path to follow.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/gemm"
cp "$KSRC/gemm/gemm.cpp" "$KSRC/gemm/gemm.hpp" "$WORK/gemm/"
cp -r "$HERE/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$KSRC" \
    -c -o "$WORK/gemm.o" "$WORK/gemm/gemm.cpp"
g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$KSRC" \
    -c -o "$WORK/oracle.o" "$HERE/oracle.cpp"
g++ -Wl,--gc-sections -o "$WORK/oracle" "$WORK/oracle.o" "$WORK/gemm.o"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${CB_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$CB_ORACLE_OUT"
    echo "transcript written to $CB_ORACLE_OUT" >&2
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
