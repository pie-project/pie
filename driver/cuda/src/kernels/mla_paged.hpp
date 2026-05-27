#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "mla_cache.hpp"

namespace pie_cuda_driver::kernels {

void launch_write_mla_to_pages_bf16(
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
    cudaStream_t stream);

void launch_write_mla_to_pages(
    MlaCacheLayerView layer,
    const void* ckv_curr,
    const void* kpe_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
