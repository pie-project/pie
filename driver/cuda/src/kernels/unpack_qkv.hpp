#pragma once

// Split a packed [N, Hq + 2*Hk] tensor (output of a fused QKV GEMM) into
// three contiguous slices q[N, Hq], k[N, Hk], v[N, Hk]. Used when a
// model materialises a single fused qkv weight at bind time and runs
// one cuBLAS GEMM instead of three smaller ones.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_unpack_qkv_bf16(
    const void* qkv_packed,
    void* q,
    void* k,
    void* v,
    int N,
    int Hq,
    int Hk,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
