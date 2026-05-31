#include "gemm.cuh"

#include <cuda_runtime.h>

// The cuBLAS column-major trick (gemm.cpp:5-9): act is row-major [M,K] and
// W is row-major [N,K]. We want row-major y=[M,N]. cuBLAS is column-major,
// so we compute `W * act^T` as `(W:KxN as col-major) op_T x (act:KxM) op_N`
// → column-major [N,M], which is the same bytes as row-major [M,N].
// Exactly the call at gemm.cpp:447-451 / 596-604.

namespace pie_cuda_device::ops {

cublasStatus_t gemm_act_x_wt_bf16(cublasHandle_t handle,
                                  const void* act, const void* w, void* y,
                                  int M, int N, int K, float beta) {
    const float alpha = 1.f;
    return cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w,   CUDA_R_16BF, K,
        act, CUDA_R_16BF, K,
        &beta,
        y,   CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

}  // namespace pie_cuda_device::ops
