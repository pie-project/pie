// The sparse-MoE dispatch launchers, and nothing else.
//
// All twenty-two `__global__`s this file used to hold now live in
// `moe_dispatch.cuh`, which NVRTC compiles from a string and nvcc reads
// through the `#include` below -- one definition, two compilers. What stays
// here is what a JIT has no use for and mostly cannot have:
//
//   * the emptiness and divisibility guards, which decide whether a kernel
//     fires at all;
//   * the run-time VECTORISABILITY test -- `hidden % 8` and the 16-byte
//     alignment of three pointers, which are facts about an allocation and
//     not about a shape;
//   * the dynamic shared-memory sizes the counting sorts need, computed from
//     `num_experts`;
//   * `std::integral_constant`, which is how the microbenchmark's sweep names
//     an instantiation, and which NVRTC has no `<type_traits>` for.
//
// Those are the reasons this file is not empty, and each of them is a row in
// `families/moe.rs` that states `LaunchRule::Unstated`.
#include "pie_device.cuh"
#include "moe/moe_dispatch.cuh"
#include "moe/moe_dispatch.hpp"

#include <cstdint>
#include <type_traits>

namespace pie_cuda_driver::kernels::moe {

void scatter_add_weighted_bf16(
    void* out, const void* src,
    const std::int32_t* dst_idx, const float* row_weights,
    int num_routed, int hidden, cudaStream_t stream)
{
    if (num_routed <= 0) return;
    device::scatter_add_weighted<device::bf16>
        <<<num_routed, device::kDispatchBlock, 0, stream>>>(
            static_cast<device::bf16*>(out),
            static_cast<const device::bf16*>(src),
            dst_idx, row_weights,
            hidden);
}

// `scalar_weighted_add_bf16`, `build_dual_bf16_gemm_ptrs` and
// `batched_weighted_sum_bf16` were deleted here by §43. The first is a row in
// `families::moe` and the third in `examples/unit_probe_moe.rs`: both are
// device rows over `moe/moe_dispatch.cuh`, so the kernels stay and only the
// ahead-of-time launchers go. `build_dual_bf16_gemm_ptrs` had no mention in
// any channel at all.

// The vectorised forms need eight elements per thread to be a `uint4`, which
// needs the row to divide by eight AND both allocations to start 16-byte
// aligned. The second half is not a property of the shape, so it is tested
// here and nowhere a table could see it.
namespace {

bool moe_vectorizable(const void* a, const void* b, int hidden) {
    return (hidden % device::kMoeVecWidth) == 0 &&
           (reinterpret_cast<std::uintptr_t>(a) % 16) == 0 &&
           (reinterpret_cast<std::uintptr_t>(b) % 16) == 0;
}

}  // namespace

void token_batched_weighted_sum_bf16(
    void* out, const void* src, const float* weights,
    int num_tokens, int top_k, int hidden, cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0) return;
    constexpr int BS = device::kDispatchBlock;
    const auto* srcp = static_cast<const device::bf16*>(src);
    auto* dstp = static_cast<device::bf16*>(out);
    if (moe_vectorizable(srcp, dstp, hidden)) {
        const int hidden_vec = hidden / device::kMoeVecWidth;
        const dim3 grid_v(num_tokens, (hidden_vec + BS - 1) / BS);
        device::token_batched_weighted_sum_vec<device::bf16>
            <<<grid_v, BS, 0, stream>>>(
                dstp, srcp, weights, top_k, hidden_vec);
        return;
    }
    const dim3 grid(num_tokens, (hidden + BS - 1) / BS);
    device::token_batched_weighted_sum<device::bf16><<<grid, BS, 0, stream>>>(
        dstp, srcp, weights, top_k, hidden);
}

void token_batched_weighted_sum_add_bf16(
    void* out, const void* src, const float* weights,
    int num_tokens, int top_k, int hidden, cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0) return;
    constexpr int BS = device::kDispatchBlock;
    const auto* srcp = static_cast<const device::bf16*>(src);
    auto* dstp = static_cast<device::bf16*>(out);
    if (moe_vectorizable(srcp, dstp, hidden)) {
        const int hidden_vec = hidden / device::kMoeVecWidth;
        const dim3 grid_v(num_tokens, (hidden_vec + BS - 1) / BS);
        device::token_batched_weighted_sum_add_vec<device::bf16>
            <<<grid_v, BS, 0, stream>>>(
                dstp, srcp, weights, top_k, hidden_vec);
        return;
    }
    const dim3 grid(num_tokens, (hidden + BS - 1) / BS);
    device::token_batched_weighted_sum_add<device::bf16><<<grid, BS, 0, stream>>>(
        dstp, srcp, weights, top_k, hidden);
}

// `token_batched_weighted_sum_aligned_bf16`, `build_moe_ptrs_decode_bf16` and
// `build_moe_ptrs_decode_batched_bf16` were deleted here by §43. The two
// pointer builders are rows in `examples/unit_probe_moe.rs`'s second table,
// which is an NVRTC probe over `moe/moe_dispatch.cuh` and fires nothing --
// the device text stays and the launchers go. The aligned weighted sum had
// no row and no caller anywhere.

// `moe_decode_gemv_tuned` was deleted here by §43, with its `PIE_MOE_CASE`
// ladder. It was the sweep entry point for a microbenchmark -- deliberately
// unrowed, as the comment said -- and the microbenchmark that called it is
// gone: no bench, no example, no test, no `.cu` names it. A tuning hook whose
// tuner has left is unreachable in the strictest sense.

void moe_gate_up_decode_gemv_bf16(
    const std::int32_t* topk_idx,
    const void* norm_x,
    const void* gate_up_base,
    void* expert_gate_up,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    const int N = 2 * I_moe;
    // float4 loads need every row to start 16-byte aligned, which holds
    // iff the reduction extent is a multiple of 8 bf16.
    if (routes <= 0 || H <= 0 || N <= 0 || (H % device::kMoeVecWidth) != 0) return;
    constexpr int kWarps = device::kGemvWarps;
    const dim3 grid((N + kWarps - 1) / kWarps, routes);
    const dim3 block(32, kWarps);
    device::moe_decode_gemv_by_token<device::bf16><<<grid, block, 0, stream>>>(
        topk_idx,
        static_cast<const device::bf16*>(norm_x),
        static_cast<const device::bf16*>(gate_up_base),
        static_cast<device::bf16*>(expert_gate_up),
        top_k, H, N, static_cast<long long>(N) * H);
}

void moe_down_decode_gemv_bf16(
    const std::int32_t* topk_idx,
    const void* expert_act,
    const void* down_base,
    void* expert_out,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    if (routes <= 0 || H <= 0 || I_moe <= 0 ||
        (I_moe % device::kMoeVecWidth) != 0) {
        return;
    }
    constexpr int kWarps = device::kGemvWarps;
    const dim3 grid((H + kWarps - 1) / kWarps, routes);
    const dim3 block(32, kWarps);
    device::moe_decode_gemv_by_route<device::bf16><<<grid, block, 0, stream>>>(
        topk_idx,
        static_cast<const device::bf16*>(expert_act),
        static_cast<const device::bf16*>(down_base),
        static_cast<device::bf16*>(expert_out),
        top_k, I_moe, H, static_cast<long long>(H) * I_moe);
}

// `moe_gate_up_decode_wmma_bf16` and `moe_down_decode_wmma_bf16` were deleted
// here by §43, and `moe_wmma_smem` -- their shared dynamic-shared-memory
// sizer and nothing else's -- with them. No shim entry, no row in
// `table::moe`, no C++ caller. The wmma TEMPLATES stay in
// `moe/moe_dispatch.cuh`: `examples/unit_probe_moe.rs`'s second table
// instantiates all twenty-nine unrowed entry points under NVRTC precisely
// because a wmma fragment type does not exist until an element type is
// supplied, so a parse alone would not catch a broken one. That probe is a
// consumer of the header and never of these launchers.

void moe_align_decode(
    const std::int32_t* topk_idx,
    std::int32_t* sorted_route_ids,
    std::int32_t* expert_ids,
    std::int32_t* route_to_aligned_row,
    int num_routes,
    int num_experts,
    int block_size,
    int max_blocks,
    std::int32_t* num_tokens_past_padded,
    cudaStream_t stream)
{
    if (num_routes <= 0 || num_experts <= 0 || block_size <= 0 ||
        max_blocks <= 0) {
        return;
    }
    // ONE BLOCK, whatever the routing: the scan is block-wide and the
    // counters are in shared memory. A grid over rows would run N copies of
    // the sort, each clearing what the others are reading.
    constexpr int BS = 1024;
    // counts + offsets(+1) + fill, then 32 warp partials and one running base
    // for the block-wide scan.
    const std::size_t smem =
        static_cast<std::size_t>(3 * num_experts + 1 + 33) * sizeof(std::int32_t);
    device::moe_align_decode<device::i32><<<1, BS, smem, stream>>>(
        topk_idx, sorted_route_ids, expert_ids, route_to_aligned_row,
        num_routes, num_experts, block_size, max_blocks,
        num_tokens_past_padded);
}

void moe_bucket_exact(
    const std::int32_t* topk_idx,
    std::int32_t* sorted_route_ids,
    std::int32_t* route_to_sorted_row,
    std::int32_t* counts_out,
    int num_routes,
    int num_experts,
    cudaStream_t stream)
{
    if (num_routes <= 0 || num_experts <= 0) return;
    constexpr int BS = 1024;
    const std::size_t smem =
        static_cast<std::size_t>(3 * num_experts + 1) * sizeof(std::int32_t);
    device::moe_bucket_exact<device::i32><<<1, BS, smem, stream>>>(
        topk_idx, sorted_route_ids, route_to_sorted_row, counts_out,
        num_routes, num_experts);
}

void add_moe_route_bias_bf16(
    void* out, const void* bias, const std::int32_t* topk_idx,
    int num_routes, int cols, int out_stride, cudaStream_t stream)
{
    if (num_routes <= 0 || cols <= 0) return;
    device::add_moe_route_bias<device::bf16>
        <<<num_routes, device::kDispatchBlock, 0, stream>>>(
            static_cast<device::bf16*>(out),
            static_cast<const device::bf16*>(bias),
            topk_idx, num_routes, cols, out_stride);
}

void transpose_expert_scales_u8(
    const void* src, void* dst, int num_experts, int n, int k_groups,
    cudaStream_t stream)
{
    if (num_experts <= 0 || n <= 0 || k_groups <= 0) return;
    const dim3 block(32, 8);
    const dim3 grid((k_groups + block.x - 1) / block.x,
                    (n + block.y - 1) / block.y,
                    num_experts);
    device::transpose_expert_scales<device::u8><<<grid, block, 0, stream>>>(
        static_cast<const device::u8*>(src),
        static_cast<device::u8*>(dst), n, k_groups);
}

void gather_moe_aligned_inputs_bf16(
    const void* norm_x,
    const std::int32_t* sorted_route_ids,
    void* aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden,
    int shared_row_begin,
    int num_tokens,
    cudaStream_t stream)
{
    if (aligned_rows <= 0 || hidden <= 0) return;
    constexpr int BS = device::kDispatchBlock;
    const auto* src = static_cast<const device::bf16*>(norm_x);
    auto* dst = static_cast<device::bf16*>(aligned_in);
    if (moe_vectorizable(src, dst, hidden)) {
        const int hidden_vec = hidden / device::kMoeVecWidth;
        const dim3 grid(aligned_rows, (hidden_vec + BS - 1) / BS);
        device::gather_moe_aligned_inputs_vec<device::bf16>
            <<<grid, BS, 0, stream>>>(
                src, sorted_route_ids, dst,
                num_routes, aligned_rows, top_k, hidden_vec,
                shared_row_begin, num_tokens);
        return;
    }
    const dim3 grid(aligned_rows, (hidden + BS - 1) / BS);
    device::gather_moe_aligned_inputs<device::bf16><<<grid, BS, 0, stream>>>(
        src, sorted_route_ids, dst,
        num_routes, aligned_rows, top_k, hidden,
        shared_row_begin, num_tokens);
}

void build_moe_ptrs_aligned_bf16(
    const std::int32_t* expert_ids,
    const void* gate_up_base,
    const void* down_base,
    const void* aligned_in,
    void* aligned_gate_up,
    void* aligned_act,
    void* aligned_out,
    const void** a_gu_ptrs,
    const void** b_gu_ptrs,
    void** c_gu_ptrs,
    const void** a_dn_ptrs,
    const void** b_dn_ptrs,
    void** c_dn_ptrs,
    int max_blocks,
    int block_size,
    int H,
    int I_moe,
    int routed_blocks,
    const void* shared_gate_up_base,
    const void* shared_down_base,
    cudaStream_t stream)
{
    if (max_blocks <= 0) return;
    if (shared_gate_up_base == nullptr || shared_down_base == nullptr) {
        routed_blocks = max_blocks;
    }
    constexpr int BS = device::kDispatchBlock;
    const int grid = (max_blocks + BS - 1) / BS;
    device::build_moe_ptrs_aligned<device::bf16><<<grid, BS, 0, stream>>>(
        expert_ids,
        static_cast<const device::bf16*>(gate_up_base),
        static_cast<const device::bf16*>(down_base),
        static_cast<const device::bf16*>(aligned_in),
        static_cast<device::bf16*>(aligned_gate_up),
        static_cast<device::bf16*>(aligned_act),
        static_cast<device::bf16*>(aligned_out),
        reinterpret_cast<const device::bf16**>(a_gu_ptrs),
        reinterpret_cast<const device::bf16**>(b_gu_ptrs),
        reinterpret_cast<device::bf16**>(c_gu_ptrs),
        reinterpret_cast<const device::bf16**>(a_dn_ptrs),
        reinterpret_cast<const device::bf16**>(b_dn_ptrs),
        reinterpret_cast<device::bf16**>(c_dn_ptrs),
        max_blocks, block_size, H, I_moe, routed_blocks,
        static_cast<const device::bf16*>(shared_gate_up_base),
        static_cast<const device::bf16*>(shared_down_base));
}

void reorder_moe_aligned_output_bf16(
    const void* aligned_out,
    const std::int32_t* sorted_route_ids,
    void* route_out,
    int num_routes,
    int aligned_rows,
    int hidden,
    int shared_row_begin,
    int num_tokens,
    void* shared_out,
    cudaStream_t stream)
{
    if (aligned_rows <= 0 || hidden <= 0) return;
    if (shared_out == nullptr) shared_row_begin = -1;
    constexpr int BS = device::kDispatchBlock;
    const auto* src = static_cast<const device::bf16*>(aligned_out);
    auto* dst = static_cast<device::bf16*>(route_out);
    auto* sdst = static_cast<device::bf16*>(shared_out);
    const bool vectorizable =
        moe_vectorizable(src, dst, hidden) &&
        (reinterpret_cast<std::uintptr_t>(sdst) % 16) == 0;
    if (vectorizable) {
        const int hidden_vec = hidden / device::kMoeVecWidth;
        const dim3 grid(aligned_rows, (hidden_vec + BS - 1) / BS);
        device::reorder_moe_aligned_output_vec<device::bf16>
            <<<grid, BS, 0, stream>>>(
                src, sorted_route_ids, dst, num_routes, aligned_rows, hidden_vec,
                shared_row_begin, num_tokens, sdst);
        return;
    }
    const dim3 grid(aligned_rows, (hidden + BS - 1) / BS);
    device::reorder_moe_aligned_output<device::bf16><<<grid, BS, 0, stream>>>(
        src, sorted_route_ids, dst, num_routes, aligned_rows, hidden,
        shared_row_begin, num_tokens, sdst);
}

}  // namespace pie_cuda_driver::kernels::moe
