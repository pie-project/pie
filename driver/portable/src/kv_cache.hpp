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
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace pie_portable_driver {

class KvCachePaged {
public:
    KvCachePaged(ggml_backend_t backend,
                 std::int32_t n_layers,
                 std::int32_t n_kv_heads,
                 std::int32_t head_dim,
                 std::int32_t total_pages,
                 std::int32_t page_size,
                 ggml_type    dtype);
    ~KvCachePaged();

    KvCachePaged(const KvCachePaged&) = delete;
    KvCachePaged& operator=(const KvCachePaged&) = delete;

    // Each tensor is shape `[n_embd_gqa, total_pages * page_size]`.
    ggml_tensor* k(std::int32_t layer) const noexcept { return k_layers_[layer]; }
    ggml_tensor* v(std::int32_t layer) const noexcept { return v_layers_[layer]; }

    std::int32_t n_layers()    const noexcept { return n_layers_; }
    std::int32_t n_kv_heads()  const noexcept { return n_kv_heads_; }
    std::int32_t head_dim()    const noexcept { return head_dim_; }
    std::int32_t n_embd_gqa()  const noexcept { return n_kv_heads_ * head_dim_; }
    std::int32_t total_pages() const noexcept { return total_pages_; }
    std::int32_t page_size()   const noexcept { return page_size_; }
    std::int32_t total_slots() const noexcept { return total_pages_ * page_size_; }
    std::size_t  buffer_size() const noexcept;

private:
    ggml_backend_t        backend_;
    std::int32_t          n_layers_;
    std::int32_t          n_kv_heads_;
    std::int32_t          head_dim_;
    std::int32_t          total_pages_;
    std::int32_t          page_size_;

    ggml_context*         ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    std::vector<ggml_tensor*> k_layers_;
    std::vector<ggml_tensor*> v_layers_;
};

}  // namespace pie_portable_driver
