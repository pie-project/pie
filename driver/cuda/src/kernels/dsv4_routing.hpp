#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_topk_sqrtsoftplus_bf16(
    const void* logits,         // [tokens, E] BF16
    std::int32_t* topk_idx,     // [tokens, K] output
    float* topk_w,              // [tokens, K] output
    const float* correction_bias, // [E] or nullptr
    int tokens,
    int num_experts,
    int top_k,
    bool renormalize,
    float routed_scaling_factor,
    cudaStream_t stream);

// Hash-layer routing: expert ids come from the token-id lookup table, but the
// weights come from the router logits — w_k = sqrtsoftplus(logit[e_k]),
// normalized over the K selected experts (normalizer floored at 2^-14) and
// scaled by routed_scaling_factor.
void launch_hash_route_lookup(
    const std::int32_t* token_ids,  // [tokens]
    const std::int64_t* tid2eid,    // [vocab_size, K]
    const void* router_logits,      // [tokens, E] BF16
    std::int32_t* topk_idx,         // [tokens, K] output
    float* topk_w,                  // [tokens, K] output
    int tokens,
    int vocab_size,
    int num_experts,
    int top_k,
    float routed_scaling_factor,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
