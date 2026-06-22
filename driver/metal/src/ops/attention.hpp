#pragma once

// Attention ops.
//
// Two entry points:
//   * `sdpa`             — dense scaled-dot-product attention over contiguous
//                          K/V (prefill / non-paged path). Thin wrapper over
//                          MLX's fused `fast::scaled_dot_product_attention`.
//   * `paged_attention`  — the batched, paged-KV-cache attention used by the
//                          serving executor. Backed by a custom Metal kernel
//                          (src/kernels/paged_attention.metal).
//
// SEAM NOTE (per manager): the canonical *op signature* lives here (beta).
// The canonical *paged-KV layout struct* (k/v page buffers, page table,
// indptrs, page geometry) is owned by delta and will live in a delta-owned
// header that this file will `#include` once published. Until then the
// paged-KV buffers are passed as explicit tensors + a small geometry struct;
// the explicit form collapses into `const PagedKV&` when delta lands it.

#include <optional>

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// Per-call attention parameters (beta-owned; these are op knobs, not KV
// storage layout).
struct AttnParams {
    float scale = 0.0f;          // softmax scale; 0 => 1/sqrt(head_dim)
    int   sliding_window = 0;    // 0 => full causal; >0 => SWA window
    float softcap = 0.0f;        // 0 => disabled; else cap*tanh(logits/cap)
    int   n_heads = 0;           // query heads
    int   n_kv_heads = 0;        // key/value heads (GQA); == n_heads if MHA
    int   head_dim = 0;
};

// Dense causal SDPA for the prefill path with contiguous per-request K/V.
//   q: [head_dim, n_heads, n_tokens]
//   k: [head_dim, n_kv_heads, n_kv]
//   v: [head_dim, n_kv_heads, n_kv]
// Returns [head_dim, n_heads, n_tokens]. `mask` is an optional additive
// attention mask; when empty a causal mask is applied.
Tensor sdpa(const Tensor& q, const Tensor& k, const Tensor& v,
            const AttnParams& params,
            const std::optional<Tensor>& mask = std::nullopt);

// Paged-KV attention.
//   q                : [head_dim, n_heads, n_total]  (feature-major)
//   k_cache, v_cache : paged buffers [n_pages, page_size, n_kv_heads, head_dim]
//                      (final layout converges to delta's PagedKV struct).
//   page_table       : [total_pages_in_batch] flat physical page indices
//   kv_page_indptr   : [n_req + 1] CSR offsets into page_table per request
//   qo_indptr        : [n_req + 1] CSR offsets into the query token axis
//   last_page_lens   : [n_req] slot count used in each request's last page
// Returns the attention output [head_dim, n_heads, n_total].
Tensor paged_attention(const Tensor& q,
                       const Tensor& k_cache,
                       const Tensor& v_cache,
                       const Tensor& page_table,
                       const Tensor& qo_indptr,
                       const Tensor& kv_page_indptr,
                       const Tensor& last_page_lens,
                       int page_size,
                       const AttnParams& params);

}  // namespace pie_metal_driver::ops
