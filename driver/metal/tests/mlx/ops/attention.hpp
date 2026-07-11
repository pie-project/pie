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

namespace pie_metal_driver {
// Defined in kv_cache.hpp (delta-owned). Forward-declared so this header stays
// decoupled from the KV layout; the .cpp includes the full definition.
struct PagedKV;
}  // namespace pie_metal_driver

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
//   q: [n_tokens, n_heads, head_dim]      (token-major; head_dim last)
//   k: [n_kv, n_kv_heads, head_dim]
//   v: [n_kv, n_kv_heads, head_dim]
// Returns [n_tokens, n_heads, head_dim]. `mask` is an optional additive
// attention mask; when empty a causal mask is applied.
Tensor sdpa(const Tensor& q, const Tensor& k, const Tensor& v,
            const AttnParams& params,
            const std::optional<Tensor>& mask = std::nullopt);

// Paged-KV attention.
//   q                : [n_total, n_heads, head_dim]  (token-major)
//   k_cache, v_cache : paged buffers [n_pages, page_size, n_kv_heads, head_dim]
//                      (delta's PagedKV layout; see kv_cache.hpp).
//   page_table       : [total_pages_in_batch] flat physical page indices
//   kv_page_indptr   : [n_req + 1] CSR offsets into page_table per request
//   qo_indptr        : [n_req + 1] CSR offsets into the query token axis
//   last_page_lens   : [n_req] slot count used in each request's last page
// Returns the attention output [n_total, n_heads, head_dim].
Tensor paged_attention(const Tensor& q,
                       const Tensor& k_cache,
                       const Tensor& v_cache,
                       const Tensor& page_table,
                       const Tensor& qo_indptr,
                       const Tensor& kv_page_indptr,
                       const Tensor& last_page_lens,
                       int page_size,
                       const AttnParams& params);

// Convenience overload: identical op with the per-layer page buffers + CSR
// metadata bundled in delta's `PagedKV` view (kv_cache.hpp). Forwards to the
// explicit-tensor form above.
Tensor paged_attention(const Tensor& q, const PagedKV& kv,
                       const AttnParams& params);

// Host-readback-FREE single-stream (R=1) decode attention. Same paged buffers,
// but the per-request CSR offsets stay on-device: pages are gathered with
// `take(page_table)`, and the valid-length clip + sliding window are expressed
// as an additive device mask derived from `last_page_len` (a device scalar) —
// NO `to_host_i32`. This keeps the whole decode step inside an `mx::compile`
// trace (the paged buffers are fixed-shape; the gathered length grows only per
// page, so compile retraces once per page bucket). q must be [1, H, head_dim].
Tensor paged_attention_decode(const Tensor& q,
                              const Tensor& k_cache,
                              const Tensor& v_cache,
                              const Tensor& page_table,
                              const Tensor& last_page_len,
                              int page_size,
                              const AttnParams& params);

}  // namespace pie_metal_driver::ops
