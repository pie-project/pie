#pragma once

// Host-side KV swap pool. Per-layer K and V buffers of
// `cpu_pages × page_size` rows each, used by the cold-path D2H/H2D
// page-copy ops to materialize device pages on the CPU side and back.
//
// Plain process memory; the binary owns it. Constructed at startup
// when `[batching].cpu_pages > 0`; otherwise the cold-path D2H/H2D ops
// return `Status::NoSwapPool`.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pie_portable_driver {

class HostSwapPool {
public:
    HostSwapPool(std::int32_t n_layers,
                 std::int32_t n_kv_heads,
                 std::int32_t head_dim,
                 std::int32_t cpu_pages,
                 std::int32_t page_size,
                 std::size_t  dtype_size);

    std::int32_t  cpu_pages()  const noexcept { return cpu_pages_; }
    std::int32_t  page_size()  const noexcept { return page_size_; }
    std::size_t   page_bytes() const noexcept { return page_bytes_; }
    std::int32_t  n_layers()   const noexcept { return n_layers_; }

    // Pointer to the start of the page-th `page_size`-row block of
    // layer `layer`'s K (or V) buffer.
    std::uint8_t* k_slot(std::int32_t layer, std::int32_t page) noexcept;
    std::uint8_t* v_slot(std::int32_t layer, std::int32_t page) noexcept;

private:
    std::int32_t  n_layers_;
    std::int32_t  cpu_pages_;
    std::int32_t  page_size_;
    std::size_t   page_bytes_;  // n_kv_heads * head_dim * page_size * dtype_size
    // Two flat host buffers (K and V) per layer, sized for cpu_pages.
    std::vector<std::vector<std::uint8_t>> k_buffers_;
    std::vector<std::vector<std::uint8_t>> v_buffers_;
};

}  // namespace pie_portable_driver
