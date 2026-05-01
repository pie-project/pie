#pragma once

// Paged KV cache pool. One pair of [num_pages, page_size, num_kv_heads,
// head_dim] tensors per layer. The runtime hands us page-index lists in
// each BPIQ request; we read/write through that translation.

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace pie_cuda_driver {

class KvCache {
public:
    static KvCache allocate(int num_layers,
                            int num_pages,
                            int page_size,
                            int num_kv_heads,
                            int head_dim,
                            DType dtype = DType::BF16);

    KvCache() = default;
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&) noexcept = default;
    KvCache& operator=(KvCache&&) noexcept = default;

    int num_layers() const noexcept { return num_layers_; }
    int num_pages() const noexcept { return num_pages_; }
    int page_size() const noexcept { return page_size_; }
    int num_kv_heads() const noexcept { return num_kv_heads_; }
    int head_dim() const noexcept { return head_dim_; }

    // Per-layer accessors. Layout: [num_pages, page_size, num_kv_heads, head_dim].
    void* k(int layer)       { return k_layers_[layer].data(); }
    void* v(int layer)       { return v_layers_[layer].data(); }
    const void* k(int layer) const { return k_layers_[layer].data(); }
    const void* v(int layer) const { return v_layers_[layer].data(); }

private:
    int num_layers_ = 0;
    int num_pages_ = 0;
    int page_size_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    std::vector<DeviceTensor> k_layers_;
    std::vector<DeviceTensor> v_layers_;
};

}  // namespace pie_cuda_driver
