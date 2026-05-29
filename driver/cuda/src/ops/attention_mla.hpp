#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "mla_cache.hpp"

namespace pie_cuda_driver::ops {

struct MlaPlanCache;
struct MlaPlanCacheDeleter {
    void operator()(MlaPlanCache* p) const noexcept;
};
using MlaPlanCachePtr = std::unique_ptr<MlaPlanCache, MlaPlanCacheDeleter>;

MlaPlanCachePtr make_mla_plan();

void plan_attention_mla_bf16(
    MlaPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_heads,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool causal,
    float sm_scale);

void dispatch_attention_mla_bf16(
    const MlaPlanCache& cache,
    const void* q_nope,                            // [total_tokens, heads, kv_lora_rank]
    const void* q_pe,                              // [total_tokens, heads, qk_rope_head_dim]
    MlaCacheLayerView layer,
    void* o,                                       // [total_tokens, heads, kv_lora_rank]
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* lse_out = nullptr,
    // Device indptr/lens for the naive (Blackwell / sm100) MLA fallback path.
    // Ignored by the FlashInfer FA2 path. When the naive path is selected
    // these MUST be provided (else it throws).
    const std::uint32_t* qo_indptr_d = nullptr,
    const std::uint32_t* kv_page_indptr_d = nullptr,
    const std::uint32_t* kv_last_page_lens_d = nullptr,
    // DSA top-k mask for the naive path: [num_query_tokens, mask_stride] uint8
    // (1=attend). Applied to in-batch keys (j < mask_stride). Null = dense.
    // Only valid for single-request pure prefill (key j == batch token j).
    const std::uint8_t* index_mask = nullptr,
    int index_mask_stride = 0);

}  // namespace pie_cuda_driver::ops
