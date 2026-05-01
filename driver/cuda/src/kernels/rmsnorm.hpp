#pragma once

// Row-wise RMSNorm: y[n, h] = x[n, h] * rsqrt(mean(x[n,:]^2) + eps) * weight[h].
//
// Designed for the Qwen-style transformer block. Input/output are bf16 row-
// major contiguous; weight is bf16, length = hidden.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_rmsnorm_bf16(
    const void* x,        // [num_rows, hidden]
    const void* weight,   // [hidden]
    void* y,              // [num_rows, hidden]
    int num_rows,
    int hidden,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
