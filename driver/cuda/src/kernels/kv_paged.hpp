#pragma once

// Write current-step K/V into the paged KV pool.
//
// Per-token destination resolved as (described in the wire format):
//   pre_kv_len_r   = total_kv_after_r - num_new_tokens_r
//   abs_kv_pos     = pre_kv_len_r + offset_in_new_tokens
//   page_idx_in_r  = abs_kv_pos / page_size
//   offset_in_page = abs_kv_pos % page_size
//   actual_page    = kv_page_indices[kv_page_indptr[r] + page_idx_in_r]

#include <cstdint>
#include <cuda_runtime.h>

#include "kv_cache.hpp"

namespace pie_cuda_driver::kernels {

void launch_write_kv_to_pages_bf16(
    void* k_pages,                                 // NHD: [pages, page_size, h_kv, d]; HND: [pages, h_kv, page_size, d]
    void* v_pages,
    const void* k_curr,                            // [total_tokens, h_kv, d]
    const void* v_curr,
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    const std::uint32_t* kv_last_page_lens,        // [R]
    int total_tokens,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    cudaStream_t stream);

void launch_write_kv_to_pages(
    KvCacheLayerView layer,
    const void* k_curr,                            // [total_tokens, h_kv, d]
    const void* v_curr,
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    const std::uint32_t* kv_last_page_lens,        // [R]
    int total_tokens,
    int num_requests,
    cudaStream_t stream);

void launch_write_kv_to_pages_at_positions_bf16(
    KvCacheLayerView layer,
    const void* k_curr,                            // [total_tokens, h_kv, d]
    const void* v_curr,
    const std::int32_t* positions,                 // [total_tokens], absolute positions
    int position_delta,
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    int total_tokens,
    int num_requests,
    cudaStream_t stream);

void launch_dequant_kv_cache_layer_to_bf16_active(
    KvCacheLayerView layer,
    const std::uint32_t* kv_page_indices,
    int num_pages_in_batch,
    cudaStream_t stream);

// Beam [B,P] freeze/heir KV write (SEAM 3): each beam b writes its ONE new-token
// K/V into an EXPLICIT (physical page id `w_page[b]`, offset `w_off[b]`) target,
// consuming the epilogue's WSlot/WOff (write-offset separated from KvLen) rather
// than re-deriving the position from the page-table + last_page_len. Single-cell
// per beam → shared-page-safe (a frozen sibling's kvm masks the heir's cell).
// Requires a native-bf16 KV cache. `w_page` must already be PHYSICAL page ids
// (resolve slot→physical before the call).
void launch_write_kv_beam_bf16(
    KvCacheLayerView layer,
    const void* k_curr,                 // [B, h_kv, d]
    const void* v_curr,
    const std::uint32_t* w_page,        // [B] physical page id per beam
    const std::uint32_t* w_off,         // [B] offset-in-page per beam
    int B,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
