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

void launch_hash_route_lookup(
    const std::int32_t* token_ids,  // [tokens]
    const std::int64_t* tid2eid,    // [vocab_size, K]
    std::int32_t* topk_idx,         // [tokens, K] output
    float* topk_w,                  // [tokens, K] output
    int tokens,
    int vocab_size,
    int top_k,
    float weight_per_expert,        // typically 1.0/K
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
