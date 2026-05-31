#include "dtype_cast.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Lifted verbatim from driver/cuda/src/kernels/dtype_cast.cu (the three
// element-wise casts only). The only changes are the namespace
// (pie_cuda_driver -> pie_cuda_device::kernels) and the dropped
// `launch_` prefix on the entry points.

namespace pie_cuda_device::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void cast_fp16_to_bf16_kernel(
    const __half*    __restrict__ src,
    __nv_bfloat16*   __restrict__ dst,
    std::size_t                   n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__half2float(src[i]));
}

__global__ void cast_fp32_to_bf16_kernel(
    const float*     __restrict__ src,
    __nv_bfloat16*   __restrict__ dst,
    std::size_t                   n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

__global__ void cast_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float*               __restrict__ dst,
    std::size_t                       n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

}  // namespace

void cast_fp16_to_bf16(
    const void* src_fp16, void* dst_bf16,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    cast_fp16_to_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __half*>(src_fp16),
        static_cast<__nv_bfloat16*>(dst_bf16), n);
}

void cast_fp32_to_bf16(
    const void* src_fp32, void* dst_bf16,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    cast_fp32_to_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const float*>(src_fp32),
        static_cast<__nv_bfloat16*>(dst_bf16), n);
}

void cast_bf16_to_fp32(
    const void* src_bf16, void* dst_fp32,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    cast_bf16_to_fp32_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src_bf16),
        static_cast<float*>(dst_fp32), n);
}

}  // namespace pie_cuda_device::kernels
