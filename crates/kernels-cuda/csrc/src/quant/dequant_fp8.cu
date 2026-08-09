//===-- dequant_fp8.cu - the ahead-of-time entry points ------------------===//
//
// Four launchers and no device text. Every `__global__` this file fires lives
// in `dequant_fp8.cuh`, which the JIT compiles from the same bytes -- see the
// header for why the split exists and what the `<<<>>>` became.
//
//===----------------------------------------------------------------------===//
#include "quant/dequant_fp8.hpp"

#include "quant/dequant_fp8.cuh"

namespace pie_cuda_driver::kernels::quant {

namespace {

constexpr int BLOCK = 256;

}  // namespace

void dequant_fp8_e4m3_to_bf16(
    const std::uint8_t* fp8_in, void* bf16_out,
    float scale, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::dequant_fp8_e4m3<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        fp8_in, static_cast<device::bf16*>(bf16_out), scale, n);
}

void dequant_fp8_e4m3_to_bf16_per_channel(
    const std::uint8_t* fp8_in, void* bf16_out,
    const float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    device::dequant_fp8_e4m3_per_channel<device::bf16><<<rows, BLOCK, 0, stream>>>(
        fp8_in, static_cast<device::bf16*>(bf16_out), scale_inv_dev, cols);
}

void dequant_fp8_e4m3_to_bf16_blocked(
    const std::uint8_t* fp8_in, void* bf16_out,
    const float* scale_dev, int rows, int cols,
    int row_block, int col_block, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    const int scale_cols = (cols + col_block - 1) / col_block;
    device::dequant_fp8_e4m3_blocked<device::bf16><<<rows, BLOCK, 0, stream>>>(
        fp8_in, static_cast<device::bf16*>(bf16_out),
        scale_dev, cols, row_block, col_block, scale_cols);
}

void dequant_fp8_e4m3_to_bf16_per_group(
    const std::uint8_t* fp8_in, void* bf16_out,
    const float* scale_dev, int rows, int cols,
    int group_size, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0 || group_size <= 0) return;
    device::dequant_fp8_e4m3_per_group<device::bf16><<<rows, BLOCK, 0, stream>>>(
        fp8_in, static_cast<device::bf16*>(bf16_out),
        scale_dev, cols, group_size);
}

}  // namespace pie_cuda_driver::kernels::quant
