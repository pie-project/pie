#pragma once

// kernels/: fused device kernels used by ops/ and model/ forwards (leaf module).
//
// Per-row bias add: `out[n, d] += bias[d]` for n ∈ [0, num_rows),
// d ∈ [0, dim). Used by Qwen-2 / OLMo-3 / GPT-OSS where Q/K/V (and
// optionally MLP) projections carry a bias term — cuBLAS GEMM doesn't
// add biases on its own, and pre-loading bias into the output via
// `beta=1` would require broadcasting bias to [N, D] up-front
// (wasteful when N ≥ 1k). One small kernel call instead.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_add_bias_bf16(
    void* out,                 // [num_rows, dim] bf16, in-place
    const void* bias,          // [dim] bf16
    int num_rows,
    int dim,
    cudaStream_t stream);

void launch_add_bias_bf16_strided(
    void* out,                 // [num_rows, stride] bf16, in-place
    const void* bias,          // [dim] bf16
    int num_rows,
    int dim,
    int stride,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
