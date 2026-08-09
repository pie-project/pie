// Host launchers for Nemotron-H and Zamba. The `__global__`s they fire live
// in `ssm/nemotron_h.cuh` and are defined ONCE, there — this file includes
// that header rather than carrying a second copy, so the archive nvcc builds
// and any cubin NVRTC builds come from the same characters.
//
// Every launcher stays. Two of the ten kernels gained JIT rows
// (`prepare_mamba_params`, `prepare_mamba_dt_da`); the other eight state
// geometry no `LaunchRule` produces, so the ahead-of-time path is still the
// only path that fires them, and the header records which refusal is which.
#include "pie_device.cuh"
#include "ssm/nemotron_h.cuh"
#include "ssm/nemotron_h.hpp"

// Host-only: `std::getenv` reads the one env toggle the SSM launcher below
// consults. NVRTC answers no standard header at all — measured 0 of 31 in
// §13 — which is why this include is here and not in the `.cuh`.
#include <cstdlib>


namespace pie_cuda_driver::kernels::ssm {

void nemotron_mamba_split_bf16(
    const void* projected,
    void* gate,
    void* conv_in,
    void* dt,
    int N,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    cudaStream_t stream)
{
    const int total = N * projection_dim;
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    if (gate == nullptr) {
        const int conv_dt_total = N * (conv_dim + num_heads);
        const int conv_dt_grid = (conv_dt_total + BLOCK - 1) / BLOCK;
        device::mamba_split_conv_dt<<<conv_dt_grid, BLOCK, 0, stream>>>(
            static_cast<const device::bf16*>(projected),
            static_cast<device::bf16*>(conv_in),
            static_cast<device::bf16*>(dt),
            projection_dim, intermediate, conv_dim, num_heads,
            conv_dt_total);
        return;
    }
    const int grid = (total + BLOCK - 1) / BLOCK;
    device::mamba_split<<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(projected),
        static_cast<device::bf16*>(gate),
        static_cast<device::bf16*>(conv_in),
        static_cast<device::bf16*>(dt),
        projection_dim, intermediate, conv_dim, num_heads, total);
}

void nemotron_prepare_mamba_params(
    const void* A_log,
    const void* D,
    const void* dt_bias,
    float* A,
    float* D_f32,
    float* dt_bias_f32,
    int num_heads,
    cudaStream_t stream)
{
    if (num_heads <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (num_heads + BLOCK - 1) / BLOCK;
    device::prepare_mamba_params<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(A_log),
        static_cast<const device::bf16*>(D),
        static_cast<const device::bf16*>(dt_bias),
        A, D_f32, dt_bias_f32, num_heads);
}

void nemotron_prepare_mamba_dt_da(
    const void* dt,
    const float* A,
    const float* dt_bias,
    float* dt_out,
    float* dA_out,
    int N,
    int num_heads,
    float time_step_min,
    cudaStream_t stream)
{
    const int total = N * num_heads;
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;
    device::prepare_mamba_dt_da<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(dt),
        A, dt_bias, dt_out, dA_out, total, num_heads, time_step_min);
}

void nemotron_mamba_ssm_batched_bf16(
    const void* conv_out,
    const void* dt,
    const float* A,
    const float* D,
    const float* dt_bias,
    const float* dt_precomputed,
    const float* dA_precomputed,
    void* ssm_state_base,
    const device::i32* slot_ids,
    const device::u32* qo_indptr,
    void* y,
    int R,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min,
    bool sequence_prefill,
    cudaStream_t stream)
{
    if (R <= 0 || num_heads <= 0 || head_dim <= 0 || state_size <= 0) return;
    constexpr int BLOCK = 256;
    if (sequence_prefill) {
        constexpr int PREFILL_BLOCK = 512;
        const int num_warps = PREFILL_BLOCK / 32;
        dim3 grid(R, num_heads, (head_dim + num_warps - 1) / num_warps);
        const std::size_t shared =
            2ull * static_cast<std::size_t>(state_size) * sizeof(float);
        device::mamba_ssm_batched_prefill_reg<<<
            grid, PREFILL_BLOCK, shared, stream>>>(
            static_cast<const device::bf16*>(conv_out),
            static_cast<const device::bf16*>(dt),
            A,
            D,
            dt_bias,
            dt_precomputed,
            dA_precomputed,
            static_cast<device::bf16*>(ssm_state_base),
            slot_ids, qo_indptr,
            static_cast<device::bf16*>(y),
            num_heads, head_dim, state_size, n_groups,
            conv_dim, intermediate, time_step_min);
        return;
    }
    if constexpr (false) {
        constexpr int DECODE_TILE_BLOCK = 128;
        const int num_warps = DECODE_TILE_BLOCK / 32;
        dim3 grid(R, num_heads, (head_dim + num_warps - 1) / num_warps);
        const std::size_t shared =
            2ull * static_cast<std::size_t>(state_size) * sizeof(float);
        device::mamba_ssm_batched_prefill_reg<<<
            grid, DECODE_TILE_BLOCK, shared, stream>>>(
            static_cast<const device::bf16*>(conv_out),
            static_cast<const device::bf16*>(dt),
            A,
            D,
            dt_bias,
            dt_precomputed,
            dA_precomputed,
            static_cast<device::bf16*>(ssm_state_base),
            slot_ids, qo_indptr,
            static_cast<device::bf16*>(y),
            num_heads, head_dim, state_size, n_groups,
            conv_dim, intermediate, time_step_min);
        return;
    }
    dim3 grid(R, num_heads);
    {
        const std::size_t shared =
            2ull * static_cast<std::size_t>(state_size) * sizeof(float);
        device::mamba_ssm_batched_warp<<<grid, BLOCK, shared, stream>>>(
            static_cast<const device::bf16*>(conv_out),
            static_cast<const device::bf16*>(dt),
            A,
            D,
            dt_bias,
            dt_precomputed,
            dA_precomputed,
            static_cast<device::bf16*>(ssm_state_base),
            slot_ids, qo_indptr,
            static_cast<device::bf16*>(y),
            num_heads, head_dim, state_size, n_groups,
            conv_dim, intermediate, time_step_min);
        return;
    }
    const std::size_t shared = static_cast<std::size_t>(head_dim) * sizeof(float);
    device::mamba_ssm_batched<<<grid, BLOCK, shared, stream>>>(
        static_cast<const device::bf16*>(conv_out),
        static_cast<const device::bf16*>(dt),
        A,
        D,
        dt_bias,
        static_cast<device::bf16*>(ssm_state_base),
        slot_ids, qo_indptr,
        static_cast<device::bf16*>(y),
        num_heads, head_dim, state_size, n_groups,
        conv_dim, intermediate, time_step_min);
}

void zamba_rmsnorm_gated_bf16(
    const void* x,
    const void* gate,
    const void* weight,
    void* y,
    int N,
    int hidden,
    int gate_stride,
    int group_size,
    float eps,
    cudaStream_t stream)
{
    if (N <= 0 || hidden <= 0 || group_size <= 0) return;
    if (gate_stride <= 0) gate_stride = hidden;
    constexpr int BLOCK = 256;
    const int groups = hidden / group_size;
    dim3 grid(N, groups);
    device::zamba_rmsnorm_gated<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(gate),
        static_cast<const device::bf16*>(weight),
        static_cast<device::bf16*>(y),
        hidden, gate_stride, group_size, eps);
}

void build_nemotron_moe_ptrs_decode_batched_bf16(
    const device::i32* topk_idx,
    const float* topk_w,
    const void* const* up_weight_ptrs,
    const void* const* down_weight_ptrs,
    const void* norm_x,
    void* expert_up,
    void* expert_act,
    void* expert_out,
    const void** a_up_ptrs,
    const void** b_up_ptrs,
    void** c_up_ptrs,
    const void** a_down_ptrs,
    const void** b_down_ptrs,
    void** c_down_ptrs,
    float* weights_out,
    int N,
    int top_k,
    int hidden,
    int intermediate,
    cudaStream_t stream)
{
    const int routes = N * top_k;
    if (routes <= 0) return;
    constexpr int BLOCK = 256;
    const int blocks = (routes + BLOCK - 1) / BLOCK;
    device::build_nemotron_moe_ptrs_decode_batched<<<blocks, BLOCK, 0, stream>>>(
        topk_idx, topk_w,
        reinterpret_cast<const device::bf16* const*>(up_weight_ptrs),
        reinterpret_cast<const device::bf16* const*>(down_weight_ptrs),
        static_cast<const device::bf16*>(norm_x),
        static_cast<device::bf16*>(expert_up),
        static_cast<device::bf16*>(expert_act),
        static_cast<device::bf16*>(expert_out),
        reinterpret_cast<const device::bf16**>(a_up_ptrs),
        reinterpret_cast<const device::bf16**>(b_up_ptrs),
        reinterpret_cast<device::bf16**>(c_up_ptrs),
        reinterpret_cast<const device::bf16**>(a_down_ptrs),
        reinterpret_cast<const device::bf16**>(b_down_ptrs),
        reinterpret_cast<device::bf16**>(c_down_ptrs),
        weights_out, routes, top_k, hidden, intermediate);
}

void build_nemotron_moe_ptrs_aligned_bf16(
    const device::i32* expert_ids,
    const void* const* up_weight_ptrs,
    const void* const* down_weight_ptrs,
    const void* aligned_in,
    void* aligned_up,
    void* aligned_act,
    void* aligned_out,
    const void** a_up_ptrs,
    const void** b_up_ptrs,
    void** c_up_ptrs,
    const void** a_down_ptrs,
    const void** b_down_ptrs,
    void** c_down_ptrs,
    int max_blocks,
    int block_size,
    int hidden,
    int intermediate,
    cudaStream_t stream)
{
    if (max_blocks <= 0 || block_size <= 0 || hidden <= 0 ||
        intermediate <= 0) {
        return;
    }
    constexpr int BLOCK = 256;
    const int blocks = (max_blocks + BLOCK - 1) / BLOCK;
    device::build_nemotron_moe_ptrs_aligned<<<blocks, BLOCK, 0, stream>>>(
        expert_ids,
        reinterpret_cast<const device::bf16* const*>(up_weight_ptrs),
        reinterpret_cast<const device::bf16* const*>(down_weight_ptrs),
        static_cast<const device::bf16*>(aligned_in),
        static_cast<device::bf16*>(aligned_up),
        static_cast<device::bf16*>(aligned_act),
        static_cast<device::bf16*>(aligned_out),
        reinterpret_cast<const device::bf16**>(a_up_ptrs),
        reinterpret_cast<const device::bf16**>(b_up_ptrs),
        reinterpret_cast<device::bf16**>(c_up_ptrs),
        reinterpret_cast<const device::bf16**>(a_down_ptrs),
        reinterpret_cast<const device::bf16**>(b_down_ptrs),
        reinterpret_cast<device::bf16**>(c_down_ptrs),
        max_blocks, block_size, hidden, intermediate);
}

}  // namespace pie_cuda_driver::kernels::ssm
