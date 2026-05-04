#pragma once

// In-place elementwise add: `y[i] += x[i]` over `n` bf16 elements.
// Used by post-norm forward graphs (OLMo-3) where the residual add and
// the cuBLAS o_proj/down_proj GEMM can't be fused via `beta=1` because
// the GEMM result first goes through a post-projection norm.

#include <cstddef>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_residual_add_bf16(
    void* y,             // [n] bf16 — accumulator (in-place)
    const void* x,       // [n] bf16 — addend
    std::size_t n,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
