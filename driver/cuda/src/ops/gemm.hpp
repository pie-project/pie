#pragma once

// Thin cuBLAS wrapper for bf16 matmul.
//
// The transformer linear layers are all of shape `out = act @ W^T`, where
// `act` is row-major [M, K] and `W` is row-major [N, K] (the HF
// safetensors convention). We compute that in cuBLAS's column-major view by
// asking for `W * act^T`, so the kernel sees `(W : K x N) @ (act : M x K)^T`
// → output column-major [N, M] which is the same memory as row-major [M, N].

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>

namespace pie_cuda_driver::ops {

class CublasHandle {
public:
    explicit CublasHandle(cudaStream_t stream = 0);
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t handle() const noexcept { return h_; }
    void set_stream(cudaStream_t s);

private:
    cublasHandle_t h_ = nullptr;
};

// Computes y = alpha * act @ W^T + beta * y. All bf16, fp32 accumulation.
//   act: [M, K]
//   W:   [N, K]   (row-major; the HF storage convention)
//   y:   [M, N]
// Default beta = 0 (overwrite). Pass beta = 1 to fuse a residual add.
void gemm_act_x_wt_bf16(
    cublasHandle_t handle,
    const void* act,
    const void* W,
    void* y,
    int M, int N, int K,
    float beta = 0.f);

}  // namespace pie_cuda_driver::ops
