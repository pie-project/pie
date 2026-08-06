#pragma once

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Single-row bf16 GEMV: y[n] = sum_k W[n][k] * x[k], W row-major with row
// stride K.
//
// This is the M=1 decode shape. There is no weight reuse to exploit, so the
// kernel is a pure streaming read and the only thing that matters is HBM
// bandwidth. cuBLAS tiles these for an M worth filling and leaves half the
// bandwidth unused; measured against `cublasGemmEx` on A100-SXM4-80GB
// (bf16, CUBLAS_GEMM_DEFAULT_TENSOR_OP):
//
//   N=2048  K=4096   17.97 -> 9.47 us   (934 -> 1771 GB/s)
//   N=4096  K=2048   15.42 -> 9.24 us   (1088 -> 1815 GB/s)
//   N=8192  K=2048   24.91 -> 20.71 us  (1347 -> 1620 GB/s)
//   N=32    K=2048    8.19 -> 4.45 us   (launch-floor bound either way)
//
// Returns false when the shape or alignment is not supported, in which case
// the caller must fall back; nothing is enqueued in that case.
//
// `bias` is optional (nullptr = none). When present the epilogue computes
// `out[n] = bf16(bf16(dot) + bias[n])`, which is bit-identical to running
// `launch_add_bias_bf16` afterwards -- the double rounding is intentional.
// This removes a whole kernel launch per biased projection; on gpt-oss-20b
// that is 120 launches per decode step, each ~3.6 us against a 2.2 us
// empty-launch floor, for a few KB of arithmetic.
bool launch_gemv_bf16(
    const void* weight,   // bf16 [N, K], row stride K
    const void* act,      // bf16 [K]
    const void* bias,     // bf16 [N] or nullptr
    void*       out,      // bf16 [N]
    int N, int K,
    cudaStream_t stream,
    // `out = W @ act + bias + beta * out`. beta = 1 is what a projection
    // accumulating into a residual asks for, and refusing it used to push
    // those shapes onto cuBLAS: gpt-oss's o_proj measured 17.3 us there
    // against 11.2 for the same bytes through this kernel.
    float beta = 0.f);

}  // namespace pie_cuda_driver::kernels
