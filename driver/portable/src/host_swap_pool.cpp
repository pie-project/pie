#include "host_swap_pool.hpp"

namespace pie_portable_driver {

HostSwapPool::HostSwapPool(std::vector<std::size_t> per_layer_page_bytes,
                           std::int32_t cpu_pages,
                           std::int32_t page_size)
    : n_layers_(static_cast<std::int32_t>(per_layer_page_bytes.size())),
      cpu_pages_(cpu_pages),
      page_size_(page_size),
      per_layer_page_bytes_(std::move(per_layer_page_bytes)) {
    if (cpu_pages <= 0) return;
    k_buffers_.resize(n_layers_);
    v_buffers_.resize(n_layers_);
    for (std::int32_t il = 0; il < n_layers_; ++il) {
        const std::size_t per_buf =
            static_cast<std::size_t>(cpu_pages) * per_layer_page_bytes_[il];
        k_buffers_[il].assign(per_buf, 0);
        v_buffers_[il].assign(per_buf, 0);
        total_bytes_ += 2 * per_buf;
    }
}

std::uint8_t* HostSwapPool::k_slot(std::int32_t layer, std::int32_t page) noexcept {
    return k_buffers_[layer].data()
           + static_cast<std::size_t>(page) * per_layer_page_bytes_[layer];
}

std::uint8_t* HostSwapPool::v_slot(std::int32_t layer, std::int32_t page) noexcept {
    return v_buffers_[layer].data()
           + static_cast<std::size_t>(page) * per_layer_page_bytes_[layer];
}

}  // namespace pie_portable_driver
