#!/bin/bash
# Regenerate the Metal parity fixtures from the REAL CUDA kernels.
#
# `cuda_reference.cu` beside this script includes
# `crates/kernels-cuda/kernels/{mlp,moe,rope,ssm}/*.cuh` -- the actual device text
# the CUDA driver JITs, not a transcription of it -- and launches each
# `__global__` with the grid, block and dynamic shared size the matching Rust
# claim body in `crates/kernels-cuda/src/` states. It writes four files here:
#
#     mlp_packed.txt   the five packed activations of mlp/packed.metal
#     moe_router.txt   the three routers of moe/route.metal
#     rope_neox.txt    the four `_mb` rotations of rope/neox.metal
#     ssm_conv.txt     the two entry points of ssm/causal_conv1d.metal
#
# The headers under `kernels/` are `#pragma once` device text with no host
# launcher and no `.cu`: NVRTC reaches them by name out of a header set carried
# in the Rust binary (`kernels-cuda/build.rs`), and nvcc reaches them through
# the ONE include path below. That is the whole build -- no CMake, no archive,
# no link against the crate.
#
# THE NUMBERS BELONG TO THE CARD THAT TOOK THEM, and the fixtures say which in
# their headers. `expf`, `tanhf`, `powf` and `__sincosf` are libdevice's, and
# libdevice ships with the toolkit; `__sincosf` in particular is the fast
# intrinsic, whose argument reduction is why every position in `rope_neox.txt`
# is small (see the case comments there). Re-running this on another
# (card, toolkit) pair may move the last bf16 bit of a few outputs. Compare
# with a tolerance of an ulp or two of bf16, not bit for bit -- the Metal side
# spells `fast::exp` where CUDA spells `expf` and `exp2(-d * log2(theta))`
# where CUDA spells `powf(theta, -d)`, so a bit-exact comparison would be
# testing the two math libraries rather than the two kernels.
#
# `driver-cuda/tests/oracle/gemm_service/run.sh` is the model for the shape of
# this script; it keys its golden on (card, toolkit) because cuBLAS picks a
# kernel from both. These fixtures do not, because there is one file per group
# and the group is the unit a Metal test reads -- but the same sentence is
# true of them, which is why the card and the nvcc are stamped INTO each file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

# The device text lives under `kernels/`, and every `#include` in it is
# relative to that directory -- `"prelude/device.cuh"`, `"mlp/swiglu.cuh"`.
# One `-I`, the same root NVRTC's carried header set is keyed on.
KERNELS="$ROOT/crates/kernels-cuda/kernels"

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
if [[ -z "$CC" ]]; then
    echo "cannot read the device's compute capability; is there a GPU?" >&2
    exit 1
fi

BIN="$(mktemp -d)/cuda_reference"
trap 'rm -rf "$(dirname "$BIN")"' EXIT

"$CUDA_HOME/bin/nvcc" -std=c++17 -O2 \
    -gencode "arch=compute_$CC,code=sm_$CC" \
    -I "$KERNELS" \
    -o "$BIN" "$HERE/cuda_reference.cu"

"$BIN" "$HERE"

echo "regenerated on sm_$CC with $("$CUDA_HOME/bin/nvcc" --version | tail -1)" >&2
