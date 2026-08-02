#include "kernels/add_bias.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// One block per row; threads stride over `dim`. Single bf16 load + add.
__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    int num_rows, int dim)
{
    const int n = blockIdx.x;
    if (n >= num_rows) return;
    __nv_bfloat16* row = out + static_cast<long long>(n) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = __bfloat162float(row[d]) + __bfloat162float(bias[d]);
        row[d] = __float2bfloat16(v);
    }
}

__global__ void add_bias_bf16_strided_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    int num_rows, int dim, int stride)
{
    const int n = blockIdx.x;
    if (n >= num_rows) return;
    __nv_bfloat16* row = out + static_cast<long long>(n) * stride;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = __bfloat162float(row[d]) + __bfloat162float(bias[d]);
        row[d] = __float2bfloat16(v);
    }
}

}  // namespace

void launch_add_bias_bf16(
    void* out, const void* bias,
    int num_rows, int dim,
    cudaStream_t stream)
{
    if (num_rows <= 0 || dim <= 0) return;
    constexpr int BLOCK = 256;
    add_bias_bf16_kernel<<<num_rows, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(bias),
        num_rows, dim);
}

void launch_add_bias_bf16_strided(
    void* out, const void* bias,
    int num_rows, int dim, int stride,
    cudaStream_t stream)
{
    if (num_rows <= 0 || dim <= 0) return;
    if (stride < dim) return;
    constexpr int BLOCK = 256;
    add_bias_bf16_strided_kernel<<<num_rows, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(bias),
        num_rows, dim, stride);
}

}  // namespace pie_cuda_driver::kernels
