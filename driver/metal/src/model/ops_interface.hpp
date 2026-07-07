#pragma once

// ── PROPOSAL / WISHLIST — NOT a compiled definition ──
//
// Per the manager's ruling, the canonical definitions live elsewhere:
//   * op signatures            → beta's  src/ops/*.hpp   (pie_metal_driver::ops)
//   * paged-KV layout struct   → delta's KV header       (pie_metal_driver::KvCache)
//   * per-batch executor inputs → reconciled in model_graph.hpp ForwardBatch
//
// This header is the single place that records, in one view, the seam the
// model graphs (this dir) depend on, so beta/delta/the executor can see what
// the forward pass needs. It intentionally declares nothing that would become
// a third definition; it is documentation. Keep it in sync as the canonical
// headers evolve.
//
// ===========================================================================
// LOCKED CONVENTIONS
// ===========================================================================
//   * Tensor = mlx::core::array (beta, ops/tensor.hpp).
//   * Layout = TOKEN-MAJOR, row-major, last axis = feature / head_dim:
//       - activations              [n_tokens, hidden]
//       - per-head Q/K/V           [n_tokens, n_heads, head_dim]   (head_dim last)
//       - linear(w[out,in], x[n,in]) -> [n, out]
//       - embedding(table, ids)    -> [n_tokens, hidden]
//       - logits                   [n_slots, vocab]
//   * Q/K/V head split is a ZERO-COPY reshape of the head-major linear output
//     [n_tokens, n_heads*head_dim] -> [n_tokens, n_heads, head_dim].
//
// ===========================================================================
// OPS the graphs call  (pie_metal_driver::ops::, see beta's headers)
// ===========================================================================
//   embedding(table, ids)                         -> [n_tokens, hidden]
//   gather_rows(x[n,features], idx)               -> [m, features]   (logit rows)
//   rms_norm(x, w, eps, plus_one=false)           -> same shape
//   rms_norm(x, eps)                              -> same shape       (weightless)
//   linear(w[out,in], x[n,in])                    -> [n, out]
//   add_bias(x[n,out], bias[out])                 -> [n, out]
//   rope(x[n_tokens,n_heads,head_dim], pos, rope_dims, RopeParams)
//   silu/gelu(x, tanh_approx)                     -> same shape
//   swiglu(gate, up) / geglu(gate, up, tanh)      -> same shape
//   add/mul/sub/scale/tanh/residual_add/softcap
//   paged_attention(q, k_cache, v_cache, page_table, qo_indptr,
//                   kv_page_indptr, last_page_lens, page_size, AttnParams)
//                                                 -> [n_tokens, n_heads, head_dim]
//
// ===========================================================================
// KV-CACHE seam  (delta-owned pie_metal_driver::KvCache — proposed methods)
// ===========================================================================
// The Llama-like / Gemma builders need, per decoder layer `il`:
//
//   // Scatter this batch's freshly-projected K/V (token-major,
//   // [n_tokens, n_kv_heads, head_dim]) into the paged cache at the physical
//   // slots given by `write_indices` (i32 [n_tokens]). The write is part of
//   // the lazy MLX graph (returns the post-write page buffers so the
//   // attention read is correctly ordered after the write).
//   void append(int il, const Tensor& k, const Tensor& v,
//               const Tensor& write_indices);
//
//   // Paged buffers for layer `il`, head_dim last, consumed by
//   // ops::paged_attention. Layout owned by delta.
//   const Tensor& k_pages(int il) const;
//   const Tensor& v_pages(int il) const;
//
//   int page_size() const;
//
// `append` + `k_pages/v_pages` keep the page-buffer layout entirely in delta's
// hands while letting the graph express the write->read dependency in MLX.
//
// ===========================================================================
// EXECUTOR seam  (ForwardBatch in model_graph.hpp — proposed fields)
// ===========================================================================
//   token_ids, positions, logit_rows,
//   kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr,
//   kv_write_indices, {n_total, n_requests, n_slots, pure_decode}
//
// See model_graph.hpp for the live struct; reconcile field-by-field with
// beta's executor on integration.
