#!/bin/bash
# Regenerate the golden for tests/gemm_service_parity.rs — gate-gemm-service.
#
# The four host launchers §45 moved out of `gemm/gemm.cpp`, as they were,
# against the real cuBLAS on the real device. Writes `golden.txt`, which the
# Rust test compares line by line against what `bind::service::*` produces
# from the same inputs.
#
# cuBLAS picks its kernel from the device, so this transcript belongs to the
# GPU that made it -- which is why there is now one golden per architecture
# rather than one golden, and why this builds for the card in front of it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

# A GOLDEN PER ARCHITECTURE, and this never overwrites another card's.
#
# `golden.txt` is sm_89's, the first recording, and stays that. Two of the
# twenty-eight rows -- the two `out_fp32` shapes large enough for cuBLAS to
# choose a different kernel -- differ on sm_120, and the other twenty-six do
# not. That is not drift to be re-pinned away: it is the sentence at the top
# of this file, that the transcript belongs to the GPU that made it, finally
# being true of two cards instead of one.
#
# So: build for the card that is actually here, and write beside the others.
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
if [[ -z "$CC" ]]; then
    echo "cannot read the device's compute capability; is there a GPU?" >&2
    exit 1
fi
if [[ "$CC" == "89" ]]; then
    OUT="$HERE/golden.txt"
else
    OUT="$HERE/golden.sm$CC.txt"
fi

"$CUDA_HOME/bin/nvcc" -std=c++17 -O2 \
    -gencode "arch=compute_$CC,code=sm_$CC" \
    -o "$HERE/oracle.bin" "$HERE/oracle.cu" \
    -lcublas

"$HERE/oracle.bin" > "$OUT"
rm -f "$HERE/oracle.bin"

echo "wrote $OUT ($(wc -l < "$OUT") rows, sm_$CC)" >&2
echo "  add it to gemm_service_parity.rs's golden_for_device() if it is new" >&2

# The bias fold: the archive's own `gemv_bf16` against the archive's own
# `gemv_bf16` + `add_bias_bf16`, out of the real archive. This is the one
# behavioural change §45 makes, and `gemv.hpp:25-28` claimed it costs nothing.
#
# THIS HALF CANNOT RUN ANY MORE, on any machine. `libpie_kernels_cuda.a` was
# the output of the ARCHIVE crate `kernels-cuda` — CMake+nvcc, `native`
# feature — and `85c6c674b` deleted that crate, the feature and the build
# together. `bias_fold.txt` is therefore a golden that is read and not
# re-derived, exactly like the sixteen dead oracles' (tests/oracle_census.rs);
# the lines below are kept as the description of how it was taken. The half
# ABOVE, which needs only nvcc and cuBLAS, is still re-derivable on a CUDA
# host.
ROOT="$(cd "$HERE/../../../../.." && pwd)"
ARCHIVE="$(find "$ROOT/target" -name libpie_kernels_cuda.a -printf '%T@ %p\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
if [[ -z "$ARCHIVE" ]]; then
    echo "no libpie_kernels_cuda.a in $ROOT/target -- the ARCHIVE crate that" \
         "built it (kernels-cuda, --features native) was deleted at" \
         "85c6c674b, so this half can no longer be re-derived anywhere;" \
         "bias_fold.txt beside this script is the record it took" >&2
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
