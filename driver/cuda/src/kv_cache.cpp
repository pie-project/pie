#include "kv_cache.hpp"

namespace pie_cuda_driver {

KvCache KvCache::allocate(int num_layers,
                          int num_pages,
                          int page_size,
                          int num_kv_heads,
                          int head_dim,
                          DType dtype)
{
    KvCache c;
    c.num_layers_ = num_layers;
    c.num_pages_ = num_pages;
    c.page_size_ = page_size;
    c.num_kv_heads_ = num_kv_heads;
    c.head_dim_ = head_dim;

    c.k_layers_.reserve(num_layers);
    c.v_layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        c.k_layers_.push_back(DeviceTensor::allocate(
            dtype, {num_pages, page_size, num_kv_heads, head_dim}));
        c.v_layers_.push_back(DeviceTensor::allocate(
            dtype, {num_pages, page_size, num_kv_heads, head_dim}));
    }
    return c;
}

KvCache KvCache::allocate_per_layer(int num_layers,
                                    int num_pages,
                                    int page_size,
                                    int num_kv_heads,
                                    const std::vector<int>& per_layer_head_dim,
                                    const std::vector<int>& kv_source_layer,
                                    DType dtype)
{
    KvCache c;
    c.num_layers_ = num_layers;
    c.num_pages_ = num_pages;
    c.page_size_ = page_size;
    c.num_kv_heads_ = num_kv_heads;
    c.head_dim_ = per_layer_head_dim.empty() ? 0 : per_layer_head_dim[0];
    c.per_layer_head_dim_ = per_layer_head_dim;
    c.kv_source_layer_ = kv_source_layer;

    // Allocate physical storage at every slot — even shared slots get
    // an empty placeholder so the vector index matches `layer`. Slots
    // whose `kv_source_layer != self` get a zero-byte view that we
    // never read or write through (the resolver redirects `k(L)` to
    // the source slot before any access). The placeholder keeps the
    // accessor lookup O(1).
    c.k_layers_.reserve(num_layers);
    c.v_layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const bool is_source = kv_source_layer.empty() || kv_source_layer[i] == i;
        if (is_source) {
            const int hd = per_layer_head_dim.empty() ? c.head_dim_
                                                      : per_layer_head_dim[i];
            c.k_layers_.push_back(DeviceTensor::allocate(
                dtype, {num_pages, page_size, num_kv_heads, hd}));
            c.v_layers_.push_back(DeviceTensor::allocate(
                dtype, {num_pages, page_size, num_kv_heads, hd}));
        } else {
            c.k_layers_.emplace_back();  // empty
            c.v_layers_.emplace_back();
        }
    }
    return c;
}

}  // namespace pie_cuda_driver
