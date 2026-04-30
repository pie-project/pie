#include "ops/gemm.hpp"

#include <stdexcept>
#include <string>

namespace pie_cuda_driver::ops {

namespace {

void check(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error (") +
                                 std::to_string(static_cast<int>(s)) + "): " + expr);
    }
}

}  // namespace

CublasHandle::CublasHandle(cudaStream_t stream) {
    check(cublasCreate(&h_), "cublasCreate");
    if (stream) check(cublasSetStream(h_, stream), "cublasSetStream");
    // Allow tensor cores; bf16 multiplies with fp32 accumulation.
    check(cublasSetMathMode(h_, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
}

CublasHandle::~CublasHandle() {
    if (h_) cublasDestroy(h_);
}

void CublasHandle::set_stream(cudaStream_t s) {
    check(cublasSetStream(h_, s), "cublasSetStream");
}

void gemm_act_x_wt_bf16(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    // We want row-major y[M,N] = act[M,K] @ W[N,K]^T.
    //
    // Memory equivalences (row-major M[R,C] is col-major M'[C,R] with lda=C):
    //   act'[K, M]  lda = K
    //   W'  [K, N]  lda = K
    //   y'  [N, M]  lda = N
    //
    // Row-major identity → col-major: y'[n,m] = sum_k W'[k,n] * act'[k,m]
    //                                          = sum_k op(A)[n,k] * op(B)[k,m]
    // where op(A) needs to be N × K (so transpose W'), op(B) needs to be
    // K × M (so leave act' as-is). Hence OP_T for A=W, OP_N for B=act.
    const float alpha = 1.f;

    check(cublasGemmEx(
              handle,
              /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
              /*m=*/N, /*n=*/M, /*k=*/K,
              &alpha,
              /*A=*/W,   CUDA_R_16BF, /*lda=*/K,
              /*B=*/act, CUDA_R_16BF, /*ldb=*/K,
              &beta,
              /*C=*/y,   CUDA_R_16BF, /*ldc=*/N,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT),
          "cublasGemmEx");
}

}  // namespace pie_cuda_driver::ops
