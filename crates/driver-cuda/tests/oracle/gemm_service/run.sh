#!/bin/bash
# Regenerate the golden for tests/gemm_service_parity.rs — gate-gemm-service.
#
# The four host launchers §45 moved out of `gemm/gemm.cpp`, as they were,
# against the real cuBLAS on the real device. Writes `golden.txt`, which the
# Rust test compares line by line against what `bind::service::*` produces
# from the same inputs.
#
# cuBLAS picks its kernel from the device, so this transcript belongs to the
# GPU that made it. `driver-cuda/build.rs` compiles for `sm_89` and nothing
# else, so that is not a new constraint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

"$CUDA_HOME/bin/nvcc" -std=c++17 -O2 \
    -gencode arch=compute_89,code=sm_89 \
    -o "$HERE/oracle.bin" "$HERE/oracle.cu" \
    -lcublas

"$HERE/oracle.bin" > "$HERE/golden.txt"
rm -f "$HERE/oracle.bin"

echo "wrote $HERE/golden.txt ($(wc -l < "$HERE/golden.txt") rows)" >&2

# The bias fold: the archive's own `gemv_bf16` against the archive's own
# `gemv_bf16` + `add_bias_bf16`, out of the real archive. This is the one
# behavioural change §45 makes, and `gemv.hpp:25-28` claimed it costs nothing.
ROOT="$(cd "$HERE/../../../../.." && pwd)"
ARCHIVE="$(find "$ROOT/target" -name libpie_kernels_cuda.a -printf '%T@ %p\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
if [[ -z "$ARCHIVE" ]]; then
    echo "no libpie_kernels_cuda.a in $ROOT/target -- build kernels-cuda --features native first" >&2
    exit 1
fi
echo "linking bias_fold against $ARCHIVE" >&2
"$CUDA_HOME/bin/nvcc" -std=c++17 -O2 \
    -gencode arch=compute_89,code=sm_89 \
    -o "$HERE/bias_fold.bin" "$HERE/bias_fold.cu" \
    "$ARCHIVE" -lcublas -lcublasLt -lnccl -lcudart
"$HERE/bias_fold.bin" > "$HERE/bias_fold.txt"
rm -f "$HERE/bias_fold.bin"

echo "wrote $HERE/bias_fold.txt ($(wc -l < "$HERE/bias_fold.txt") rows)" >&2
