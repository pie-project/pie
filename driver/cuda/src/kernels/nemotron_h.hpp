#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_nemotron_mamba_split_bf16(
    const void* projected,  // [N, projection_dim]
    void* gate,             // [N, intermediate]
    void* conv_in,          // [N, conv_dim]
    void* dt,               // [N, num_heads]
    int N,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    cudaStream_t stream);

void launch_nemotron_prepare_mamba_params(
    const void* A_log,     // [num_heads], bf16
    const void* D,         // [num_heads], bf16
    const void* dt_bias,   // [num_heads], bf16
    float* A,              // [num_heads], fp32, stores -exp(A_log)
    float* D_f32,          // [num_heads], fp32
    float* dt_bias_f32,    // [num_heads], fp32
    int num_heads,
    cudaStream_t stream);

void launch_nemotron_prepare_mamba_dt_da(
    const void* dt,        // [N, num_heads], bf16
    const float* A,        // [num_heads], fp32
    const float* dt_bias,  // [num_heads], fp32
    float* dt_out,         // [N, num_heads], fp32
    float* dA_out,         // [N, num_heads], fp32
    int N,
    int num_heads,
    float time_step_min,
    cudaStream_t stream);

void launch_nemotron_mamba_ssm_batched_bf16(
    const void* conv_out,       // [N, conv_dim], bf16
    const void* dt,             // [N, num_heads], bf16
    const float* A,             // [num_heads], fp32, -exp(A_log)
    const float* D,             // [num_heads], fp32
    const float* dt_bias,       // [num_heads], fp32
    const float* dt_precomputed, // [N, num_heads], fp32, nullable
    const float* dA_precomputed, // [N, num_heads], fp32, nullable
    void* ssm_state_base,       // [slots, heads, head_dim, state], bf16
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    void* y,                    // [N, intermediate], bf16
    int R,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min,
    bool sequence_prefill,
    cudaStream_t stream);

void launch_zamba_rmsnorm_gated_bf16(
    const void* x,          // [N, hidden]
    const void* gate,       // [N, hidden]
    const void* weight,     // [hidden] bf16
    void* y,                // [N, hidden]
    int N,
    int hidden,
    int gate_stride,
    int group_size,
    float eps,
    cudaStream_t stream);

void launch_build_nemotron_moe_ptrs_decode_batched_bf16(
    const std::int32_t* topk_idx,
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
    cudaStream_t stream);

void launch_build_nemotron_moe_ptrs_aligned_bf16(
    const std::int32_t* expert_ids,
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
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
