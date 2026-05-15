#include "host_swap_pool.hpp"

namespace pie_portable_driver {

HostSwapPool::HostSwapPool(std::int32_t n_layers,
                           std::int32_t n_kv_heads,
                           std::int32_t head_dim,
                           std::int32_t cpu_pages,
                           std::int32_t page_size,
                           std::size_t  dtype_size)
    : n_layers_(n_layers),
      cpu_pages_(cpu_pages),
      page_size_(page_size),
      page_bytes_(static_cast<std::size_t>(n_kv_heads) * head_dim
                  * page_size * dtype_size) {
    if (cpu_pages <= 0) return;
    const std::size_t per_buf = static_cast<std::size_t>(cpu_pages) * page_bytes_;
    k_buffers_.resize(n_layers);
    v_buffers_.resize(n_layers);
    for (std::int32_t il = 0; il < n_layers; ++il) {
        k_buffers_[il].assign(per_buf, 0);
        v_buffers_[il].assign(per_buf, 0);
    }
}

std::uint8_t* HostSwapPool::k_slot(std::int32_t layer, std::int32_t page) noexcept {
    return k_buffers_[layer].data() + static_cast<std::size_t>(page) * page_bytes_;
}

std::uint8_t* HostSwapPool::v_slot(std::int32_t layer, std::int32_t page) noexcept {
    return v_buffers_[layer].data() + static_cast<std::size_t>(page) * page_bytes_;
}

}  // namespace pie_portable_driver
