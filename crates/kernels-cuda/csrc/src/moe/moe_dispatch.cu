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

void scalar_weighted_add_bf16(
    void* out, const void* src, float weight, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    const int grid = (n + device::kDispatchBlock - 1) / device::kDispatchBlock;
    device::scalar_weighted_add<device::bf16>
        <<<grid, device::kDispatchBlock, 0, stream>>>(
            static_cast<device::bf16*>(out),
            static_cast<const device::bf16*>(src),
            weight, n);
}

void build_dual_bf16_gemm_ptrs(
    const void* act,
    const void* w0,
    const void* w1,
    void* out0,
    void* out1,
    const void** act_ptrs,
    const void** w_ptrs,
    void** out_ptrs,
    cudaStream_t stream)
{
    device::build_dual_gemm_ptrs<device::bf16><<<1, 1, 0, stream>>>(
        static_cast<const device::bf16*>(act),
        static_cast<const device::bf16*>(w0),
        static_cast<const device::bf16*>(w1),
        static_cast<device::bf16*>(out0),
        static_cast<device::bf16*>(out1),
        reinterpret_cast<const device::bf16**>(act_ptrs),
        reinterpret_cast<const device::bf16**>(w_ptrs),
        reinterpret_cast<device::bf16**>(out_ptrs));
}

void batched_weighted_sum_bf16(
    void* out, const void* src, const float* weights,
    int batch, int hidden, cudaStream_t stream)
{
    if (batch <= 0 || hidden <= 0) return;
    const int grid = (hidden + device::kDispatchBlock - 1) / device::kDispatchBlock;
    device::batched_weighted_sum<device::bf16>
        <<<grid, device::kDispatchBlock, 0, stream>>>(
            static_cast<device::bf16*>(out),
            static_cast<const device::bf16*>(src),
            weights, batch, hidden);
}

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

void token_batched_weighted_sum_aligned_bf16(
    void* out,
    const void* aligned_out,
    const float* weights,
    const std::int32_t* route_to_aligned_row,
    int num_tokens,
    int top_k,
    int hidden,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0) return;
    constexpr int BS = device::kDispatchBlock;
    const dim3 grid(num_tokens, (hidden + BS - 1) / BS);
    device::token_batched_weighted_sum_aligned<device::bf16>
        <<<grid, BS, 0, stream>>>(
            static_cast<device::bf16*>(out),
            static_cast<const device::bf16*>(aligned_out),
            weights,
            route_to_aligned_row,
            top_k,
            hidden);
}

void build_moe_ptrs_decode_bf16(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const void* gate_up_base, const void* down_base, const void* norm_x,
    void* expert_gate_up, void* expert_act, void* expert_out,
    const void** a_gu_ptrs, const void** b_gu_ptrs, void** c_gu_ptrs,
    const void** a_dn_ptrs, const void** b_dn_ptrs, void** c_dn_ptrs,
    float*       weights_out,
    int top_k, int H, int I_moe, cudaStream_t stream)
{
    if (top_k <= 0) return;
    device::build_moe_ptrs_decode<device::bf16><<<1, top_k, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const device::bf16*>(gate_up_base),
        static_cast<const device::bf16*>(down_base),
        static_cast<const device::bf16*>(norm_x),
        static_cast<device::bf16*>(expert_gate_up),
        static_cast<device::bf16*>(expert_act),
        static_cast<device::bf16*>(expert_out),
        reinterpret_cast<const device::bf16**>(a_gu_ptrs),
        reinterpret_cast<const device::bf16**>(b_gu_ptrs),
        reinterpret_cast<device::bf16**>(c_gu_ptrs),
        reinterpret_cast<const device::bf16**>(a_dn_ptrs),
        reinterpret_cast<const device::bf16**>(b_dn_ptrs),
        reinterpret_cast<device::bf16**>(c_dn_ptrs),
        weights_out,
        top_k, H, I_moe);
}

void build_moe_ptrs_decode_batched_bf16(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const void* gate_up_base, const void* down_base, const void* norm_x,
    void* expert_gate_up, void* expert_act, void* expert_out,
    const void** a_gu_ptrs, const void** b_gu_ptrs, void** c_gu_ptrs,
    const void** a_dn_ptrs, const void** b_dn_ptrs, void** c_dn_ptrs,
    float*       weights_out,
    int num_tokens, int top_k, int H, int I_moe, cudaStream_t stream)
{
    const int total = num_tokens * top_k;
    if (total <= 0) return;
    constexpr int BS = device::kDispatchBlock;
    const int grid = (total + BS - 1) / BS;
    device::build_moe_ptrs_decode_batched<device::bf16><<<grid, BS, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const device::bf16*>(gate_up_base),
        static_cast<const device::bf16*>(down_base),
        static_cast<const device::bf16*>(norm_x),
        static_cast<device::bf16*>(expert_gate_up),
        static_cast<device::bf16*>(expert_act),
        static_cast<device::bf16*>(expert_out),
        reinterpret_cast<const device::bf16**>(a_gu_ptrs),
        reinterpret_cast<const device::bf16**>(b_gu_ptrs),
        reinterpret_cast<device::bf16**>(c_gu_ptrs),
        reinterpret_cast<const device::bf16**>(a_dn_ptrs),
        reinterpret_cast<const device::bf16**>(b_dn_ptrs),
        reinterpret_cast<device::bf16**>(c_dn_ptrs),
        weights_out,
        num_tokens, top_k, H, I_moe);
}

// Sweep entry point for the microbenchmark: the unroll depth and warps per
// block, chosen explicitly. Not for engine use, and deliberately unrowed --
// a microbenchmark is not a statement any trace makes.
bool moe_decode_gemv_tuned(
    const std::int32_t* topk_idx, const void* act, const void* weight_base,
    void* out, int routes, int top_k, int K, int N, long long expert_stride,
    int warps, int unroll, cudaStream_t stream)
{
    if (routes <= 0 || K <= 0 || N <= 0 || (K % device::kMoeVecWidth) != 0) {
        return false;
    }
    const auto* ti = topk_idx;
    const auto* a = static_cast<const device::bf16*>(act);
    const auto* w = static_cast<const device::bf16*>(weight_base);
    auto* o = static_cast<device::bf16*>(out);
    auto go = [&](auto W, auto U) {
        constexpr int kW = decltype(W)::value, kU = decltype(U)::value;
        const dim3 grid((N + kW - 1) / kW, routes);
        const dim3 block(32, kW);
        device::moe_decode_gemv<device::bf16, true, kW, kU>
            <<<grid, block, 0, stream>>>(
                ti, a, w, o, top_k, K, N, expert_stride);
    };
#define PIE_MOE_CASE(W, U) \
    if (warps == (W) && unroll == (U)) {                                   \
        go(std::integral_constant<int, W>{},                               \
           std::integral_constant<int, U>{});                              \
        return true;                                                       \
    }
    PIE_MOE_CASE(4, 1) PIE_MOE_CASE(4, 2) PIE_MOE_CASE(4, 4)
    PIE_MOE_CASE(2, 1) PIE_MOE_CASE(2, 2) PIE_MOE_CASE(2, 4)
    PIE_MOE_CASE(8, 1) PIE_MOE_CASE(8, 2) PIE_MOE_CASE(8, 4)
    PIE_MOE_CASE(1, 2) PIE_MOE_CASE(16, 2)
#undef PIE_MOE_CASE
    return false;
}

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

// The wmma path's dynamic shared memory: one 16x16 staging tile for the
// activation row, plus four warps' 16x16 fp32 accumulators.
namespace {

std::size_t moe_wmma_smem() {
    return (16 * 16 * sizeof(device::bf16)) + (4 * 16 * 16 * sizeof(float));
}

}  // namespace

void moe_gate_up_decode_wmma_bf16(
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
    if (routes <= 0 || H <= 0 || N <= 0 || (H % 16) != 0 || (N % 64) != 0) {
        return;
    }
    const dim3 grid(N / 64, routes);
    device::moe_decode_wmma_by_token<device::bf16>
        <<<grid, 128, moe_wmma_smem(), stream>>>(
            topk_idx,
            static_cast<const device::bf16*>(norm_x),
            static_cast<const device::bf16*>(gate_up_base),
            static_cast<device::bf16*>(expert_gate_up),
            top_k, H, N, static_cast<long long>(N) * H);
}

void moe_down_decode_wmma_bf16(
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
        (I_moe % 16) != 0 || (H % 64) != 0) {
        return;
    }
    const dim3 grid(H / 64, routes);
    device::moe_decode_wmma_by_route<device::bf16>
        <<<grid, 128, moe_wmma_smem(), stream>>>(
            topk_idx,
            static_cast<const device::bf16*>(expert_act),
            static_cast<const device::bf16*>(down_base),
            static_cast<device::bf16*>(expert_out),
            top_k, I_moe, H, static_cast<long long>(H) * I_moe);
}

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
