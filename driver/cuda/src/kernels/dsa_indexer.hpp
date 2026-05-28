#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// LayerNorm (weight + bias, eps) over the `head_dim` of idx_k, then GPT-J
// interleaved RoPE on the first `rope_dim` dims. One block per token.
void launch_dsa_index_knorm_rope_bf16(
    void* idx_k,                       // [tokens, head_dim] bf16, in-place
    const void* k_norm_weight,         // [head_dim] bf16
    const void* k_norm_bias,           // [head_dim] bf16
    const std::int32_t* positions,     // [tokens]
    int tokens, int head_dim, int rope_dim, float theta, float eps,
    cudaStream_t stream);

// GPT-J interleaved RoPE on the first `rope_dim` dims of each index head of
// idx_q [tokens, n_heads, head_dim]. One block per token.
void launch_dsa_index_q_rope_bf16(
    void* idx_q,                       // [tokens, n_heads*head_dim] bf16, in-place
    const std::int32_t* positions,     // [tokens]
    int tokens, int n_heads, int head_dim, int rope_dim, float theta,
    cudaStream_t stream);

// Build the per-query top-k attention mask from the lightning-indexer logits:
//   logit[i,j] = sum_h relu(idx_q[i,h,:] . idx_k[j,:]) * idx_w[i,h]   (causal)
// then mask[i,j] = 1 for the top-`topk` keys j<=i (all of them when i+1<=topk),
// else 0. mask is [tokens, tokens] uint8. One block per query token i.
void launch_dsa_index_topk_mask(
    const void* idx_q,                 // [tokens, n_heads*head_dim] bf16
    const void* idx_k,                 // [tokens, head_dim] bf16
    const void* idx_w,                 // [tokens, n_heads] bf16
    std::uint8_t* mask,                // [tokens, tokens] uint8 out
    int tokens, int n_heads, int head_dim, int topk,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
