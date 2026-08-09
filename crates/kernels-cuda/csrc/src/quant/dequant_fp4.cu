//===-- dequant_fp4.cu - the ahead-of-time entry points ------------------===//
//
// Four launchers, the block-shape constants they were swept with, and no
// device text. Every `__global__` this file fires lives in
// `dequant_fp4.cuh`, which the JIT compiles from the same bytes -- see the
// header for why the split exists, what the `<<<>>>` became, and why only one
// of the four kernels can carry a row.
//
// `<cstdlib>` and `<type_traits>` stay here. They are the `kTok` dispatch's:
// `std::getenv`, `std::atoi` and `std::integral_constant` turn a runtime
// choice into a compile-time one, which is host machinery in a header NVRTC
// has no equivalent of.
//
//===----------------------------------------------------------------------===//
#include <cstdlib>
#include <type_traits>
#include "quant/dequant_fp4.hpp"

#include "quant/dequant_fp4.cuh"

namespace pie_cuda_driver::kernels::quant {

void dequant_mxfp4_to_bf16(
    const std::uint8_t* packed, const std::uint8_t* block_scale,
    void* out, int out_dim, int in_dim, cudaStream_t stream)
{
    if (out_dim <= 0 || in_dim <= 0) return;
    if (in_dim % 32 != 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(out_dim);
    dim3 block(BLOCK);
    device::dequant_mxfp4<device::bf16><<<grid, block, 0, stream>>>(
        packed, block_scale,
        static_cast<device::bf16*>(out), in_dim);
}


namespace {
constexpr int kMxfp4DecodeBlock = 128;
// Intermediate rows each warp of the gate/up decode GEMV owns. Swept with
// `crates/driver-cuda/csrc/bench/moe_bench.cu` at gpt-oss's shape; see the table there.
constexpr int kMxfp4GateUpPairs = 4;
// Hidden rows each warp of the down decode GEMV owns, swept the same way.
constexpr int kMxfp4DownRows = 4;  // four warps, one output row each
}  // namespace


void mxfp4_moe_gate_up_decode_bf16(
    const void* act_fp16,
    const std::int32_t* topk_idx,
    const std::uint8_t* const* gate_up_packed,
    const std::uint8_t* const* gate_up_scales,
    const void* const* gate_bias,
    const void* const* up_bias,
    void* gate_out_bf16,
    void* up_out_bf16,
    int num_tokens, int top_k, int hidden, int intermediate,
    cudaStream_t stream,
    void* act_out_fp16,
    float glu_limit,
    float glu_alpha)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0 || intermediate <= 0) {
        return;
    }
    if (hidden % 32 != 0) return;
    const int warps = kMxfp4DecodeBlock / 32;
    const int pairs_per_block = warps * kMxfp4GateUpPairs;
    dim3 grid(num_tokens * top_k,
              (intermediate + pairs_per_block - 1) / pairs_per_block);
    device::mxfp4_moe_gate_up_decode<kMxfp4GateUpPairs>
        <<<grid, kMxfp4DecodeBlock, 0, stream>>>(
        static_cast<const __half*>(act_fp16), topk_idx,
        gate_up_packed, gate_up_scales, gate_bias, up_bias,
        static_cast<device::bf16*>(gate_out_bf16),
        static_cast<device::bf16*>(up_out_bf16),
        static_cast<__half*>(act_out_fp16), glu_limit, glu_alpha,
        top_k, hidden, intermediate);
}

// `mxfp4_moe_gate_up_decode_grouped_bf16` was deleted here by §43: the
// grouped fork of the launcher above it, named in no channel at all -- no
// shim entry, no row, no C++ caller, no golden, no test. Its two ungrouped
// neighbours stay and are shim roots.

void mxfp4_moe_down_decode_bf16(
    const void* act_fp16,
    const std::int32_t* topk_idx,
    const std::uint8_t* const* down_packed,
    const std::uint8_t* const* down_scales,
    const void* const* down_bias,
    void* out_bf16,
    int num_tokens, int top_k, int hidden, int intermediate,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0 || intermediate <= 0) {
        return;
    }
    if (intermediate % 32 != 0) return;
    const int warps = kMxfp4DecodeBlock / 32;
    const int rows_per_warp = kMxfp4DownRows;
    const int rows_per_block = warps * rows_per_warp;
    dim3 grid(num_tokens * top_k,
              (hidden + rows_per_block - 1) / rows_per_block);
    device::mxfp4_moe_down_decode<kMxfp4DownRows>
        <<<grid, kMxfp4DecodeBlock, 0, stream>>>(
        static_cast<const __half*>(act_fp16), topk_idx,
        down_packed, down_scales, down_bias,
        static_cast<device::bf16*>(out_bf16),
        hidden, intermediate);
}

}  // namespace pie_cuda_driver::kernels::quant
