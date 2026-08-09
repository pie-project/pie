//===-- dtype_cast.cu - the ahead-of-time entry points -------------------===//
//
// Eleven launchers and no device text. Every `__global__` this file fires
// lives in `dtype_cast.cuh`, which the JIT compiles from the same bytes --
// see the header for why the split exists, what each `<<<>>>` became, and
// which four have no launch rule.
//
// `model-loader` calls `pie_k_quant_cast_fp32_to_bf16` and `scale_rows_bf16`
// by name from Rust, so none of these entry points is going away.
//
//===----------------------------------------------------------------------===//
#include "quant/dtype_cast.hpp"

#include "quant/dtype_cast.cuh"

namespace pie_cuda_driver::kernels::quant {

namespace {

constexpr int BLOCK = 256;

// Marlin requires N to be a multiple of the 64-wide tile, which is also the
// permutation's period.
constexpr int MARLIN_GROUP_PERM_LEN = 64;

}  // namespace

void cast_fp16_to_bf16(
    const void* src_fp16, void* dst_bf16,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::cast_f16_to<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const device::f16*>(src_fp16),
        static_cast<device::bf16*>(dst_bf16), n);
}

void cast_fp32_to_bf16(
    const void* src_fp32, void* dst_bf16,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::cast_f32_to<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const float*>(src_fp32),
        static_cast<device::bf16*>(dst_bf16), n);
}

void cast_bf16_to_fp32(
    const void* src_bf16, void* dst_fp32,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::cast_to_f32<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(src_bf16),
        static_cast<float*>(dst_fp32), n);
}

void cast_e8m0_to_fp32(
    const void* src_e8m0, void* dst_fp32,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::cast_e8m0_to<device::f32><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const device::u8*>(src_e8m0),
        static_cast<float*>(dst_fp32), n);
}

void scale_bf16(
    const void* src_bf16, void* dst_bf16,
    std::size_t n, float factor, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::scale<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(src_bf16),
        static_cast<device::bf16*>(dst_bf16), n, factor);
}

void scale_fp32(
    const void* src_fp32, void* dst_fp32,
    std::size_t n, float factor, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::scale<device::f32><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const float*>(src_fp32),
        static_cast<float*>(dst_fp32), n, factor);
}

void scale_fp16(
    const void* src_fp16, void* dst_fp16,
    std::size_t n, float factor, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::scale<device::f16><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const device::f16*>(src_fp16),
        static_cast<device::f16*>(dst_fp16), n, factor);
}

void marlin_permute_scales_bf16(
    void* bf16_scales,
    int groups, int size_n, int group_size, int size_k,
    cudaStream_t stream)
{
    if (groups == 0 || size_n == 0) return;
    if (size_n % MARLIN_GROUP_PERM_LEN != 0) {
        // Marlin requires N multiple of 64 (tile_n_size). Caller
        // should have validated.
        return;
    }
    const std::size_t total = static_cast<std::size_t>(groups) * size_n;
    if (total % MARLIN_GROUP_PERM_LEN != 0) return;
    const int total64 = static_cast<int>(total / MARLIN_GROUP_PERM_LEN);

    if (group_size > 0 && group_size < size_k) {
        // Per-group case (group_size=128 etc).
        device::marlin_permute_scales_per_group<<<total64, 64, 0, stream>>>(
            static_cast<device::bf16*>(bf16_scales), total64);
    }
    // Per-channel uses a different perm — skip until needed.
}

void awq_dequant_to_bf16(
    const void* qweight_in,
    const void* qzeros_in,
    const void* scales_in,
    void*       bf16_out,
    int         size_k,
    int         size_n,
    int         group_size,
    cudaStream_t stream)
{
    if (size_k == 0 || size_n == 0 || group_size == 0) return;
    constexpr int BX = 32, BY = 8;
    const dim3 block(BX, BY);
    const dim3 grid((size_n + BX - 1) / BX, (size_k + BY - 1) / BY);
    device::awq_dequant_to_bf16<<<grid, block, 0, stream>>>(
        static_cast<const device::u32*>(qweight_in),
        static_cast<const device::u32*>(qzeros_in),
        static_cast<const device::bf16*>(scales_in),
        static_cast<device::bf16*>(bf16_out),
        size_k, size_n, group_size);
}

void gptq_dequant_to_bf16(
    const void* qweight_in,
    const void* qzeros_in,
    const void* scales_in,
    const void* g_idx_in,
    void*       bf16_out,
    int         size_k,
    int         size_n,
    int         group_size,
    cudaStream_t stream)
{
    if (size_k == 0 || size_n == 0 || group_size == 0) return;
    constexpr int BX = 32, BY = 8;
    const dim3 block(BX, BY);
    const dim3 grid((size_n + BX - 1) / BX, (size_k + BY - 1) / BY);
    device::gptq_dequant_to_bf16<<<grid, block, 0, stream>>>(
        static_cast<const device::u32*>(qweight_in),
        static_cast<const device::u32*>(qzeros_in),
        static_cast<const device::bf16*>(scales_in),
        static_cast<const device::i32*>(g_idx_in),
        static_cast<device::bf16*>(bf16_out),
        size_k, size_n, group_size);
}

void scale_rows_bf16(
    void*         buf_bf16,
    const void*   l_bf16,
    int           rows,
    int           width,
    cudaStream_t  stream)
{
    if (rows == 0 || width == 0) return;
    // One block per row, columns strided -- `LaunchRule::RouteRows`' shape.
    // The block width is the launcher's to pick because the kernel reads
    // `blockDim.x`; 256 here, `ceil_warp(width)` under the rule, same answer.
    device::scale_rows<device::bf16><<<rows, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(buf_bf16),
        static_cast<const device::bf16*>(l_bf16),
        width);
}

}  // namespace pie_cuda_driver::kernels::quant
