#pragma once

// KvCacheView — the model→KV-cache accessor seam.
//
// SEAM NOTE: the canonical paged-KV *storage layout* (page buffers, page
// table, page geometry, Copy/KV handler) is owned by delta. This header does
// NOT define that layout — it defines only the small abstract accessor the
// model graphs (this dir) call per decoder layer. delta's concrete paged KV
// cache implements this interface (or an adapter does), so the graph code
// stays decoupled from storage internals while delta remains the single
// source of truth for the layout.
//
// All tensors are token-major, head_dim last (the locked convention):
//   k/v this batch : [n_tokens, n_kv_heads, head_dim]
//   k/v pages      : delta's paged buffers (head_dim last), consumed as-is by
//                    ops::paged_attention.

#include "ops/tensor.hpp"

namespace pie_metal_driver {

class KvCacheView {
public:
    virtual ~KvCacheView() = default;

    // Scatter this batch's freshly-projected K/V into the paged cache at the
    // physical slots given by `write_indices` (i32 [n_tokens]). Expressed as
    // part of the lazy MLX graph so the subsequent attention read is ordered
    // after the write.
    virtual void append(int layer,
                        const Tensor& k,
                        const Tensor& v,
                        const Tensor& write_indices) = 0;

    // Paged K/V buffers for `layer`, consumed by ops::paged_attention.
    virtual const Tensor& k_pages(int layer) const = 0;
    virtual const Tensor& v_pages(int layer) const = 0;

    // Tokens per page (page geometry; delta-defined).
    virtual int page_size() const = 0;
};

}  // namespace pie_metal_driver
