#include "kernels/dequant_fp8.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void dequant_fp8_e4m3_kernel(
    const __nv_fp8_storage_t* __restrict__ src,
    __nv_bfloat16*            __restrict__ dst,
    float                                  scale,
    std::size_t                            n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const __half h = __nv_cvt_fp8_to_halfraw(src[i], __NV_E4M3);
    const float f = __half2float(h) * scale;
    dst[i] = __float2bfloat16(f);
}

__global__ void dequant_fp8_e4m3_per_channel_kernel(
    const __nv_fp8_storage_t* __restrict__ src,
    __nv_bfloat16*            __restrict__ dst,
    const float*              __restrict__ scale_inv,
    int                                    cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float s = scale_inv[row];
    const std::size_t off = static_cast<std::size_t>(row) * cols;
    for (int j = tid; j < cols; j += BLOCK) {
        const __half h = __nv_cvt_fp8_to_halfraw(src[off + j], __NV_E4M3);
        dst[off + j] = __float2bfloat16(__half2float(h) * s);
    }
}

__global__ void dequant_fp8_e4m3_blocked_kernel(
    const __nv_fp8_storage_t* __restrict__ src,
    __nv_bfloat16*            __restrict__ dst,
    const float*              __restrict__ scales,
    int                                    cols,
    int                                    row_block,
    int                                    col_block,
    int                                    scale_cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int scale_row = row / row_block;
    const std::size_t off = static_cast<std::size_t>(row) * cols;
    for (int j = tid; j < cols; j += BLOCK) {
        const int scale_col = j / col_block;
        const float s = scales[scale_row * scale_cols + scale_col];
        const __half h = __nv_cvt_fp8_to_halfraw(src[off + j], __NV_E4M3);
        dst[off + j] = __float2bfloat16(__half2float(h) * s);
    }
}

}  // namespace

void launch_dequant_fp8_e4m3_to_bf16(
    const std::uint8_t* fp8_in, void* bf16_out,
    float scale, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    dequant_fp8_e4m3_kernel<<<blocks, BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(fp8_in),
        static_cast<__nv_bfloat16*>(bf16_out),
        scale, n);
}

void launch_dequant_fp8_e4m3_to_bf16_per_channel(
    const std::uint8_t* fp8_in, void* bf16_out,
    const float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    dequant_fp8_e4m3_per_channel_kernel<<<rows, BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(fp8_in),
        static_cast<__nv_bfloat16*>(bf16_out),
        scale_inv_dev, cols);
}

void launch_dequant_fp8_e4m3_to_bf16_blocked(
    const std::uint8_t* fp8_in, void* bf16_out,
    const float* scale_dev, int rows, int cols,
    int row_block, int col_block, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    const int scale_cols = (cols + col_block - 1) / col_block;
    dequant_fp8_e4m3_blocked_kernel<<<rows, BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(fp8_in),
        static_cast<__nv_bfloat16*>(bf16_out),
        scale_dev, cols, row_block, col_block, scale_cols);
}

void launch_dequant_fp8_e4m3_to_bf16_per_group(
    const std::uint8_t* fp8_in, void* bf16_out,
    const float* scale_dev, int rows, int cols,
    int group_size, cudaStream_t stream)
{
    launch_dequant_fp8_e4m3_to_bf16_blocked(
        fp8_in, bf16_out, scale_dev, rows, cols, group_size, group_size, stream);
}

}  // namespace pie_cuda_driver::kernels
