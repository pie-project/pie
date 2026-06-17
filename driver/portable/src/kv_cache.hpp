#pragma once

// Paged KV cache.
//
// Per layer, K and V are flat 2D buffers `[n_embd_gqa, total_pages * page_size]`.
// Pie's runtime owns page allocation: each `BatchedForwardPassRequest`
// carries `kv_page_indices` + `kv_page_indptr` (per-request page lists)
// and `kv_last_page_lens` (slack in the last page). The driver computes
// physical row indices from those:
//
//   physical_row(req r, position p)
//     = kv_page_indices[kv_page_indptr[r] + p / page_size] * page_size
//     + (p % page_size)
//
// Writes scatter via `ggml_set_rows`. Reads (per-request K/V views for
// attention) gather via `ggml_get_rows` with a per-request idx tensor.
//
// This honors page IDs the runtime sends — page-trim, prefix-sharing, and
// swap (via M7) all work because we never re-interpret what page goes
// where; we just follow the runtime's page table.
//
// There is no driver-side "concurrent context cap": as many contexts as
// fit in `total_pages * page_size` token slots can coexist.

#include <cstdint>
#include <string_view>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "kv_cache_quant.hpp"

namespace pie_portable_driver {

class KvCachePaged {
public:
    // Uniform: one head_dim across all layers (qwen3, llama3, gemma2/3 etc.).
    KvCachePaged(ggml_backend_t backend,
                 std::int32_t n_layers,
                 std::int32_t n_kv_heads,
                 std::int32_t head_dim,
                 std::int32_t total_pages,
                 std::int32_t page_size,
                 ggml_type    dtype,
                 KvCacheQuantFormat quant_format = {});
    // Per-layer: each layer carries its own head_dim. Used by Gemma 4
    // where sliding layers store [head_dim] and full layers [head_dim_global]
    // KV. `per_layer_head_dim.size()` must equal `n_layers`.
    KvCachePaged(ggml_backend_t backend,
                 std::int32_t n_kv_heads,
                 std::vector<std::int32_t> per_layer_head_dim,
                 std::int32_t total_pages,
                 std::int32_t page_size,
                 ggml_type    dtype,
                 KvCacheQuantFormat quant_format = {});
    // Per-layer kv_heads AND head_dim. Used by Gemma 4 31B / 26B-A4B
    // alternative attention (full_attention layers carry fewer kv_heads
    // than sliding layers). Both vectors must have size == n_layers.
    KvCachePaged(ggml_backend_t backend,
                 std::vector<std::int32_t> per_layer_kv_heads,
                 std::vector<std::int32_t> per_layer_head_dim,
                 std::int32_t total_pages,
                 std::int32_t page_size,
                 ggml_type    dtype,
                 KvCacheQuantFormat quant_format = {});
    ~KvCachePaged();

    KvCachePaged(const KvCachePaged&) = delete;
    KvCachePaged& operator=(const KvCachePaged&) = delete;

    // Per-layer K/V cache tensor. Shape: [n_embd_gqa_layer, total_pages * page_size].
    ggml_tensor* k(std::int32_t layer) const noexcept { return k_layers_[layer]; }
    ggml_tensor* v(std::int32_t layer) const noexcept { return v_layers_[layer]; }

    std::int32_t n_layers()    const noexcept { return n_layers_; }
    std::int32_t n_kv_heads()  const noexcept { return n_kv_heads_; }
    std::int32_t n_kv_heads_at(std::int32_t layer) const noexcept {
        return per_layer_kv_heads_.empty()
            ? n_kv_heads_
            : per_layer_kv_heads_[layer];
    }
    // head_dim of layer 0 — kept for backwards compatibility on uniform
    // archs. Per-layer code paths should use head_dim_at(layer).
    std::int32_t head_dim()    const noexcept { return per_layer_head_dim_[0]; }
    std::int32_t head_dim_at(std::int32_t layer) const noexcept {
        return per_layer_head_dim_[layer];
    }
    std::int32_t n_embd_gqa()  const noexcept {
        return n_kv_heads_ * per_layer_head_dim_[0];
    }
    std::int32_t n_embd_gqa_at(std::int32_t layer) const noexcept {
        return n_kv_heads_at(layer) * per_layer_head_dim_[layer];
    }
    std::int32_t total_pages() const noexcept { return total_pages_; }
    std::int32_t page_size()   const noexcept { return page_size_; }
    std::int32_t total_slots() const noexcept { return total_pages_ * page_size_; }
    // Bytes occupied by one page of layer `layer`'s K (or V) buffer. Derived
    // from the layer's own tensor, so it is exact for any dtype (incl.
    // quantized) AND honors Gemma 4's per-layer geometry — sliding layers
    // (more kv-heads, smaller head_dim) and full layers differ in page bytes.
    // The KV-swap / fork copy paths MUST size and offset per layer with this
    // (a single uniform value over-reads the smaller layers → ggml bounds abort).
    std::size_t page_bytes(std::int32_t layer) const noexcept {
        return ggml_nbytes(k_layers_[layer]) /
               static_cast<std::size_t>(total_pages_);
    }
    std::size_t  buffer_size() const noexcept;
    const KvCacheQuantFormat& quant_format() const noexcept { return quant_format_; }

    ggml_tensor* qdq_for_append(ggml_context* ctx,
                                std::int32_t layer,
                                ggml_tensor* tensor) const;

private:
    void allocate_();

    ggml_backend_t        backend_;
    std::int32_t          n_layers_;
    std::int32_t          n_kv_heads_;
    std::vector<std::int32_t> per_layer_head_dim_;
    // Empty for uniform archs (use n_kv_heads_); populated for Gemma 4
    // alternative attention.
    std::vector<std::int32_t> per_layer_kv_heads_;
    std::int32_t          total_pages_;
    std::int32_t          page_size_;
    ggml_type             dtype_;
    KvCacheQuantFormat    quant_format_;

    ggml_context*         ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    std::vector<ggml_tensor*> k_layers_;
    std::vector<ggml_tensor*> v_layers_;
};

}  // namespace pie_portable_driver
