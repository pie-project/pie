#pragma once

// kv_cache.hpp — the canonical paged KV-cache layout for driver/metal.
//
// DELTA OWNS THIS. It is the single source of truth for how K/V are stored,
// how logical token slots map to physical page memory, and how the model
// graphs (charlie) and the paged-attention op (beta) see that storage:
//   * charlie's graphs touch the cache ONLY through the abstract
//     `KvCacheView` (model/model_kv.hpp): `append` + `k_pages`/`v_pages` +
//     `page_size`. They never see the layout internals.
//   * beta's `ops::paged_attention` reads the per-layer page buffers plus the
//     per-batch CSR metadata (page_table / qo_indptr / kv_page_indptr /
//     last_page_lens / page_size). The `PagedKV` view below bundles exactly
//     those args; beta's explicit-tensor signature collapses into
//     `const PagedKV&` once wired.
//
// LOCKED conventions (design-driver-metal):
//   * namespace `pie_metal_driver`
//   * `using Tensor = mlx::core::array;`  (ops/tensor.hpp)
//   * token-major, head_dim LAST.
//
// Layout. Per layer, K and V are each one MLX array shaped
//   [n_pages, page_size, n_kv_heads, head_dim]
// (token-major: the page-local token axis is dim 1, head_dim is last). A
// logical cache slot is addressed as
//   slot = phys_page * page_size + within_page
// which indexes the flattened [n_pages * page_size, n_kv_heads, head_dim]
// view. This is exactly the `kv_write_indices` the executor computes
// (executor.cpp: `phys_page * page_size + within`) and the inverse of the
// gather beta's paged_attention performs (`take(k_pages, physical_pages, 0)`
// then reshape to `page*page_size + within`), so writes and reads agree with
// zero transposes.

#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "model/model_kv.hpp"
#include "ops/tensor.hpp"

namespace pie_metal_driver {

// Static geometry of the paged KV cache (delta-defined page geometry).
struct PagedKvGeometry {
    int n_layers   = 0;  // decoder layers (one K and one V buffer each)
    int n_pages    = 0;  // physical pages per layer
    int page_size  = 0;  // tokens per page
    int n_kv_heads = 0;  // key/value heads (GQA)
    int head_dim   = 0;  // per-head dimension (LAST axis)

    // Total logical slots per layer == rows of the flattened page view.
    int n_slots() const { return n_pages * page_size; }
};

// Per-batch paged-attention argument bundle. References one layer's page
// buffers plus the per-request CSR index arrays the executor stages each
// step. This is the `const PagedKV&` that beta's `ops::paged_attention`
// explicit-tensor signature collapses into; the field set (and order) mirrors
// that signature so the wiring is mechanical.
//
// `page_table` (a.k.a. block_table) holds flat physical page indices; the
// CSR arrays slice it per request.
struct PagedKV {
    const Tensor& k_pages;         // [n_pages, page_size, n_kv_heads, head_dim]
    const Tensor& v_pages;         // [n_pages, page_size, n_kv_heads, head_dim]
    const Tensor& page_table;      // i32 [total_pages_in_batch] phys page idx
    const Tensor& qo_indptr;       // i32 [n_req + 1] query-token CSR offsets
    const Tensor& kv_page_indptr;  // i32 [n_req + 1] page-table CSR offsets
    const Tensor& last_page_lens;  // i32 [n_req] slots used in each last page
    int page_size   = 0;
    int n_kv_heads  = 0;
    int head_dim    = 0;
};

// The concrete paged KV cache. Owns the per-layer K/V page buffers and
// satisfies `KvCacheView` so charlie's graphs can drive it directly.
//
// MLX arrays are immutable graph values, so `append` does not mutate buffers
// in place: it builds a scatter node and rebinds the stored array to it. The
// subsequent `k_pages`/`v_pages` read therefore depends on the scatter,
// ordering the attention read after the write within the lazy graph (exactly
// the ordering guarantee `KvCacheView::append` documents).
class PagedKvCache final : public KvCacheView {
public:
    explicit PagedKvCache(const PagedKvGeometry& geo, DType dtype = DType::BF16)
        : geo_(geo), dtype_(dtype) {
        if (geo_.n_layers <= 0 || geo_.n_pages <= 0 || geo_.page_size <= 0 ||
            geo_.n_kv_heads <= 0 || geo_.head_dim <= 0) {
            throw std::invalid_argument(
                "PagedKvCache: all geometry dimensions must be positive");
        }
        k_pages_.reserve(geo_.n_layers);
        v_pages_.reserve(geo_.n_layers);
        for (int l = 0; l < geo_.n_layers; ++l) {
            k_pages_.push_back(make_zero_buffer());
            v_pages_.push_back(make_zero_buffer());
        }
    }

    // ── KvCacheView ──────────────────────────────────────────────────────
    // Scatter this batch's K/V `[n_tokens, n_kv_heads, head_dim]` into layer
    // `layer` at the physical slots `write_indices` (i32 [n_tokens]).
    void append(int layer, const Tensor& k, const Tensor& v,
                const Tensor& write_indices) override {
        check_layer(layer);
        k_pages_[layer] = scatter_slots(k_pages_[layer], k, write_indices);
        v_pages_[layer] = scatter_slots(v_pages_[layer], v, write_indices);
    }

    const Tensor& k_pages(int layer) const override {
        check_layer(layer);
        return k_pages_[layer];
    }
    const Tensor& v_pages(int layer) const override {
        check_layer(layer);
        return v_pages_[layer];
    }
    int page_size() const override { return geo_.page_size; }

    // ── geometry accessors ───────────────────────────────────────────────
    const PagedKvGeometry& geometry() const { return geo_; }
    int  n_layers()   const { return geo_.n_layers; }
    int  n_pages()    const { return geo_.n_pages; }
    int  n_kv_heads() const { return geo_.n_kv_heads; }
    int  head_dim()   const { return geo_.head_dim; }
    DType dtype()     const { return dtype_; }

    // Bundle one layer's page buffers + per-batch CSR metadata into the
    // `PagedKV` view beta's paged_attention consumes.
    PagedKV view(int layer, const Tensor& page_table, const Tensor& qo_indptr,
                 const Tensor& kv_page_indptr,
                 const Tensor& last_page_lens) const {
        check_layer(layer);
        return PagedKV{k_pages_[layer],  v_pages_[layer], page_table,
                       qo_indptr,        kv_page_indptr,  last_page_lens,
                       geo_.page_size,   geo_.n_kv_heads, geo_.head_dim};
    }

    // Zero every layer buffer (e.g. between independent test batches).
    void reset() {
        for (int l = 0; l < geo_.n_layers; ++l) {
            k_pages_[l] = make_zero_buffer();
            v_pages_[l] = make_zero_buffer();
        }
    }

    // Force materialization of all layer buffers in one graph evaluation.
    void eval() {
        std::vector<Tensor> all;
        all.reserve(k_pages_.size() + v_pages_.size());
        for (const auto& t : k_pages_) all.push_back(t);
        for (const auto& t : v_pages_) all.push_back(t);
        mlx::core::eval(std::move(all));
    }

private:
    Tensor make_zero_buffer() const {
        return mlx::core::zeros(
            {geo_.n_pages, geo_.page_size, geo_.n_kv_heads, geo_.head_dim},
            to_mlx_dtype(dtype_));
    }

    void check_layer(int layer) const {
        if (layer < 0 || layer >= geo_.n_layers) {
            throw std::out_of_range("PagedKvCache: layer " +
                                    std::to_string(layer) + " out of range [0," +
                                    std::to_string(geo_.n_layers) + ")");
        }
    }

    // Scatter `src` [n_tokens, n_kv_heads, head_dim] into the flattened
    // [n_slots, n_kv_heads, head_dim] view of `pages` at rows `write_indices`,
    // returning the rebound [n_pages, page_size, n_kv_heads, head_dim] buffer.
    Tensor scatter_slots(const Tensor& pages, const Tensor& src,
                         const Tensor& write_indices) const {
        namespace mx = mlx::core;
        const int n_slots = geo_.n_slots();
        const int H = geo_.n_kv_heads;
        const int D = geo_.head_dim;
        const int n_tokens = src.shape(0);

        Tensor flat = mx::reshape(pages, {n_slots, H, D});
        // scatter (assignment) along axis 0: updates carry a leading
        // index axis and a size-1 placeholder for the scattered axis →
        // [n_tokens, 1, H, D].
        Tensor upd =
            mx::reshape(mx::astype(src, flat.dtype()), {n_tokens, 1, H, D});
        Tensor scattered = mx::scatter(
            flat, std::vector<Tensor>{mx::astype(write_indices, mx::uint32)},
            upd, std::vector<int>{0});
        return mx::reshape(scattered,
                           {geo_.n_pages, geo_.page_size, H, D});
    }

    PagedKvGeometry geo_;
    DType dtype_;
    std::vector<Tensor> k_pages_;  // [n_layers] × [n_pages,page_size,H,D]
    std::vector<Tensor> v_pages_;  // [n_layers] × [n_pages,page_size,H,D]
};

}  // namespace pie_metal_driver
