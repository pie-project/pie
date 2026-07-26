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

// DeepSeek-V4 hash MoE routing. The expert *indices* come from the
// `tid2eid` lookup table (they are a pure function of the token id), but
// the expert *weights* still come from the router logits — vLLM's
// `_topk_softplus_sqrt_torch` gathers `sqrt(softplus(logits))` at the
// hashed indices, renormalizes across the K picks, then multiplies by
// `routed_scaling_factor`. Using a uniform `1/K` instead is wrong.
void launch_hash_route_lookup(
    const std::int32_t* token_ids,  // [tokens]
    const std::int64_t* tid2eid,    // [vocab_size, K]
    const void* logits,             // [tokens, E] BF16 router logits
    std::int32_t* topk_idx,         // [tokens, K] output
    float* topk_w,                  // [tokens, K] output
    int tokens,
    int vocab_size,
    int num_experts,
    int top_k,
    bool renormalize,
    float routed_scaling_factor,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
