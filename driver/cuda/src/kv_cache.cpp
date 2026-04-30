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

}  // namespace pie_cuda_driver
