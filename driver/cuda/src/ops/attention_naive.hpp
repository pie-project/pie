#pragma once

// Single-sequence causal attention with GQA, no paging, no batching.
// **For numeric-parity testing only** — not a hot path. M1.2.3 swaps this
// for the flashinfer paged kernels.
//
// Layout:
//   q [num_tokens, num_q_heads,  head_dim]   bf16
//   k [num_tokens, num_kv_heads, head_dim]   bf16
//   v [num_tokens, num_kv_heads, head_dim]   bf16
//   o [num_tokens, num_q_heads,  head_dim]   bf16
//
// Each query at position p attends causally to keys at positions [0..p].
// GQA broadcast: query head h attends to KV head h * num_kv_heads / num_q_heads.

#include <cuda_runtime.h>

namespace pie_cuda_driver::ops {

void launch_attention_naive_bf16(
    const void* q, const void* k, const void* v,
    void* o,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::ops
