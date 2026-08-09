//===-- dequant_wna16.cu - the ahead-of-time entry points ----------------===//
//
// Four launchers and no device text. Every `__global__` this file fires lives
// in `dequant_wna16.cuh`, which the JIT compiles from the same bytes -- see
// the header for why the split exists. Two of the four are rows; the other
// two are the decode GEMVs, whose grids divide by 8 where every ported rule
// divides by 256.
//
// `<algorithm>` stays here: `std::min` and `std::max` clamp the grid, which
// is host arithmetic.
//
//===----------------------------------------------------------------------===//
#include "quant/dequant_wna16.hpp"

#include "quant/dequant_wna16.cuh"

#include <algorithm>

namespace pie_cuda_driver::kernels::quant {

namespace {

constexpr int DECODE_BLOCK = 256;

}  // namespace

void dequant_wna16_int4b8_to_bf16(
    const std::int32_t* packed,
    const void* scale_bf16,
    void* out_bf16,
    int out_dim,
    int in_dim,
    int group_size,
    cudaStream_t stream)
{
    if (out_dim <= 0 || in_dim <= 0 || group_size <= 0) return;
    if (in_dim % 8 != 0 || in_dim % group_size != 0) return;
    constexpr int BLOCK = 256;
    const int words_per_row = in_dim / 8;
    // Rows on x, word-columns on y -- the axis order `LaunchRule::
    // ElementwiseRows` states and the kernel now reads. gridDim.x is
    // 2^31-1, so the 65535 clamp the row axis used to need is gone and a
    // block IS a row; nothing here iterates.
    dim3 grid(static_cast<unsigned>(out_dim),
              static_cast<unsigned>((words_per_row + BLOCK - 1) / BLOCK));
    device::dequant_wna16_int4b8<device::bf16><<<grid, BLOCK, 0, stream>>>(
        packed,
        static_cast<const device::bf16*>(scale_bf16),
        static_cast<device::bf16*>(out_bf16),
        in_dim,
        group_size);
}

void wna16_gate_up_decode_bf16(
    const void* act_fp16,
    const std::int32_t* topk_idx,
    const std::int32_t* const* gate_packed,
    const void* const* gate_scale,
    const std::int32_t* const* up_packed,
    const void* const* up_scale,
    void* gate_out_bf16,
    void* up_out_bf16,
    int num_tokens,
    int top_k,
    int hidden,
    int intermediate,
    int group_size,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    if (routes <= 0 || hidden <= 0 || intermediate <= 0) return;
    if (hidden % 8 != 0 || hidden % group_size != 0) return;
    constexpr int GU_WARPS = DECODE_BLOCK / 32;
    const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);
    device::wna16_gate_up_decode<<<grid, DECODE_BLOCK, 0, stream>>>(
        static_cast<const __half*>(act_fp16),
        topk_idx,
        gate_packed, gate_scale,
        up_packed, up_scale,
        static_cast<device::bf16*>(gate_out_bf16),
        static_cast<device::bf16*>(up_out_bf16),
        top_k, hidden, intermediate, group_size);
}

void wna16_down_decode_bf16(
    const void* act_fp16,
    const std::int32_t* topk_idx,
    const std::int32_t* const* down_packed,
    const void* const* down_scale,
    void* out_bf16,
    int num_tokens,
    int top_k,
    int hidden,
    int intermediate,
    int group_size,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    if (routes <= 0 || hidden <= 0 || intermediate <= 0) return;
    if (intermediate % 8 != 0 || intermediate % group_size != 0) return;
    constexpr int BS = 256;
    constexpr int WARPS = BS / 32;
    const dim3 grid((hidden + WARPS - 1) / WARPS, routes);
    device::wna16_down_decode<<<grid, BS, 0, stream>>>(
        static_cast<const __half*>(act_fp16),
        topk_idx,
        down_packed, down_scale,
        static_cast<device::bf16*>(out_bf16),
        top_k, hidden, intermediate, group_size);
}


void bf16_to_fp16(const void* in_bf16, void* out_fp16,
                         std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    constexpr int BS = 256;
    const long long n = static_cast<long long>(count);
    const long long n_vec8 = n / 8;
    const long long units = n_vec8 > 0 ? n_vec8 : n;
    const int blocks = static_cast<int>(
        std::min<long long>((units + BS - 1) / BS, 1024));
    device::bf16_to_narrow<__half><<<std::max(blocks, 1), BS, 0, stream>>>(
        static_cast<const device::bf16*>(in_bf16),
        static_cast<__half*>(out_fp16), n);
}

}  // namespace pie_cuda_driver::kernels::quant
