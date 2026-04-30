#pragma once

// Causal paged attention. Reads K/V from the per-layer paged pool, applies
// GQA broadcast, writes the output back into a contiguous [total_tokens,
// num_q_heads, head_dim] buffer. **Reference implementation** — flashinfer
// integration replaces this in M1.5+.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::ops {

void launch_attention_paged_bf16(
    const void* q,                                 // [total_tokens, h_q, d]
    const void* k_pages,                           // [num_pages, page_size, h_kv, d]
    const void* v_pages,
    void* o,                                       // [total_tokens, h_q, d]
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    const std::uint32_t* kv_last_page_lens,        // [R]
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_kv_len,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::ops
