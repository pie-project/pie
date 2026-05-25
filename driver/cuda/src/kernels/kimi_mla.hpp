#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_kimi_split_q_b_bf16(
    const void* q_b,
    void* q_nope,
    void* q_pe,
    int tokens,
    int heads,
    int qk_nope_dim,
    int qk_rope_dim,
    cudaStream_t stream);

void launch_kimi_split_kv_a_bf16(
    const void* kv_a,
    void* kv_c,
    void* k_pe,
    int tokens,
    int kv_lora_rank,
    int qk_rope_dim,
    cudaStream_t stream);

void launch_topk_sigmoid_bf16(
    const void* logits,
    std::int32_t* topk_idx,
    float* topk_w,
    int tokens,
    int num_experts,
    int top_k,
    bool renormalize,
    float routed_scaling_factor,
    cudaStream_t stream);

void launch_kimi_q_nope_to_latent_bf16(
    const void* q_nope,     // [tokens, heads, qk_nope_dim]
    const void* kv_b_proj,  // [heads * (qk_nope_dim + v_head_dim), kv_lora_rank]
    void* q_latent,         // [tokens, heads, kv_lora_rank]
    int tokens,
    int heads,
    int qk_nope_dim,
    int v_head_dim,
    int kv_lora_rank,
    cudaStream_t stream);

void launch_kimi_latent_to_v_bf16(
    const void* attn_latent, // [tokens, heads, kv_lora_rank]
    const void* kv_b_proj,   // [heads * (qk_nope_dim + v_head_dim), kv_lora_rank]
    void* attn_v,            // [tokens, heads, v_head_dim]
    int tokens,
    int heads,
    int qk_nope_dim,
    int v_head_dim,
    int kv_lora_rank,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
