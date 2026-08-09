#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "attn/mla_cache_view.hpp"

namespace pie_cuda_driver::kernels::attn {

void write_mla_to_pages_bf16(
    void* ckv_pages,                               // [pages, page_size, kv_lora_rank]
    void* kpe_pages,                               // [pages, page_size, qk_rope_head_dim]
    const void* ckv_curr,                          // [total_tokens, kv_lora_rank]
    const void* kpe_curr,                          // [total_tokens, qk_rope_head_dim]
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    const std::uint32_t* kv_last_page_lens,        // [R]
    int total_tokens,
    int num_requests,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim,
    cudaStream_t stream,
    const std::uint8_t* row_valid = nullptr);

void write_mla_to_pages(
    MlaCacheLayerView layer,
    const void* ckv_curr,
    const void* kpe_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    cudaStream_t stream,
    const std::uint8_t* row_valid = nullptr);

// True when `mla_prepare_bf16` can serve this head dim.
bool mla_prepare_supported(int qk_rope_head_dim);

// Original-YaRN rope scaling (DeepSeek-V2/V3, Kimi). Pass nullptr for plain
// RoPE.
struct YarnOriginalParams {
    float factor = 1.f;
    float beta_fast = 32.f;
    float beta_slow = 1.f;
    float attention_factor = 1.f;
    int original_max_position = 4096;
};

// One launch in place of split_kv_a_norm + split_q_b + rope + write_mla.
// Produces byte-identical kv_c / k_pe / q_nope / q_pe and page contents; it
// exists purely to delete three launches per layer. Handles the interleaved
// and NeoX pairings, with or without original-YaRN scaling.
void mla_prepare_bf16(
    MlaCacheLayerView layer,
    const void* kv_a,                    // [total_tokens, kv_a_row_stride]
    const void* kv_a_norm_weight,        // [kv_lora_rank]
    const void* q_b,                     // [total_tokens, heads*(nope+rope)]
    void* kv_c,                          // [total_tokens, kv_lora_rank]
    void* k_pe,                          // [total_tokens, qk_rope_head_dim]
    void* q_nope,                        // [total_tokens, heads, nope]
    void* q_pe,                          // [total_tokens, heads, rope]
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int heads,
    int qk_nope_head_dim,
    float eps,
    float theta,
    bool interleaved,
    int kv_a_row_stride,
    const YarnOriginalParams* yarn,
    cudaStream_t stream,
    const std::uint8_t* row_valid = nullptr);

}  // namespace pie_cuda_driver::kernels::attn
