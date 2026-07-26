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
bool launch_gemv_bf16(
    const void* weight,   // bf16 [N, K], row stride K
    const void* act,      // bf16 [K]
    void*       out,      // bf16 [N]
    int N, int K,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
