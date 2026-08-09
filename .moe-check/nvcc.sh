#!/bin/bash
# Compile one moe .cu with exactly the flags CMake gives it. Used because
# other agents' in-progress files break the whole-archive build.
set -e
B=/root/Workspace/pie-new-driver/target/debug/build/kernels-cuda-4248961c634a28af/out/kernels-cuda/build
cd "$B"
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DPIE_CUDA_HAS_MARLIN_MOE=1 \
  --options-file CMakeFiles/pie_kernels_cuda.dir/includes_CUDA.rsp \
  -g -std=c++20 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" \
  -Xcompiler=-fPIC -Xcompiler=-Wall -Xcompiler=-Wextra -Wno-unused-parameter \
  --extended-lambda --expt-relaxed-constexpr \
  -c "/root/Workspace/pie-new-driver/crates/kernels-cuda/csrc/src/moe/$1.cu" \
  -o "/root/Workspace/pie-new-driver/.moe-check/$1.o"
