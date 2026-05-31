#pragma once

// Write current-step compressed-latent KV (MLA) into the paged KV pool
// (paged-KV append / scatter — the MLA analogue of write_kv_to_pages_bf16).
//
// Launcher declaration; the kernel body lives in mla_write.cu, lifted verbatim
// from driver/cuda/src/kernels/mla_paged.cu (the RAW-POINTER bf16 variant,
// `launch_write_mla_to_pages_bf16`). The MlaCacheLayerView-based overload
// (`launch_write_mla_to_pages`) is NOT lifted here.
//
// MLA cache layout (matching ops/mla_paged.cuh — one latent vector plus one
// rotary key vector per token, shared across all query heads):
//   ckv_pages: [num_pages, page_size, kv_lora_rank]      bf16  (NoPE latent)
//   kpe_pages: [num_pages, page_size, qk_rope_head_dim]  bf16  (RoPE key)
//   ckv_curr:  [total_tokens, kv_lora_rank]              bf16
//   kpe_curr:  [total_tokens, qk_rope_head_dim]          bf16
//
// Per-token destination resolved as (matching kv_append.cuh):
//   pre_kv_len_r   = total_kv_after_r - num_new_tokens_r
//   abs_kv_pos     = pre_kv_len_r + offset_in_new_tokens
//   page_idx_in_r  = abs_kv_pos / page_size
//   offset_in_page = abs_kv_pos % page_size
//   actual_page    = kv_page_indices[kv_page_indptr[r] + page_idx_in_r]

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

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
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
