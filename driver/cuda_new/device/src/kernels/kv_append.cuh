#pragma once

// Write current-step K/V into the paged KV pool (paged-KV append / scatter).
//
// Launcher declaration; the kernel body lives in kv_append.cu, lifted verbatim
// from driver/cuda/src/kernels/kv_paged.cu (the RAW-POINTER bf16 variant,
// `launch_write_kv_to_pages_bf16`). The KvCacheLayerView-based overloads, the
// at-positions variant, and the dequant/quant (fp8/int8/fp4) variants are NOT
// lifted here.
//
// Per-token destination resolved as (described in the wire format):
//   pre_kv_len_r   = total_kv_after_r - num_new_tokens_r
//   abs_kv_pos     = pre_kv_len_r + offset_in_new_tokens
//   page_idx_in_r  = abs_kv_pos / page_size
//   offset_in_page = abs_kv_pos % page_size
//   actual_page    = kv_page_indices[kv_page_indptr[r] + page_idx_in_r]

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void write_kv_to_pages_bf16(
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

}  // namespace pie_cuda_device::kernels
