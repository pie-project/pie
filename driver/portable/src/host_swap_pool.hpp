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
    // `per_layer_page_bytes[il]` is the byte size of one page of layer il's K
    // (or V) buffer — taken from the device KV cache (KvCachePaged::page_bytes)
    // so the host mirror matches the device geometry exactly. Gemma 4's
    // sliding and full layers differ, so this must be per layer; a uniform
    // value mis-sizes the smaller layers and the D2H/H2D copy aborts.
    HostSwapPool(std::vector<std::size_t> per_layer_page_bytes,
                 std::int32_t cpu_pages,
                 std::int32_t page_size);

    std::int32_t  cpu_pages()  const noexcept { return cpu_pages_; }
    std::int32_t  page_size()  const noexcept { return page_size_; }
    std::size_t   page_bytes(std::int32_t layer) const noexcept {
        return per_layer_page_bytes_[layer];
    }
    std::size_t   total_bytes() const noexcept { return total_bytes_; }
    std::int32_t  n_layers()   const noexcept { return n_layers_; }

    // Pointer to the start of the page-th `page_size`-row block of
    // layer `layer`'s K (or V) buffer.
    std::uint8_t* k_slot(std::int32_t layer, std::int32_t page) noexcept;
    std::uint8_t* v_slot(std::int32_t layer, std::int32_t page) noexcept;

private:
    std::int32_t  n_layers_;
    std::int32_t  cpu_pages_;
    std::int32_t  page_size_;
    std::vector<std::size_t> per_layer_page_bytes_;
    std::size_t   total_bytes_ = 0;  // sum over layers of cpu_pages * page_bytes * 2
    // Two flat host buffers (K and V) per layer, sized for cpu_pages.
    std::vector<std::vector<std::uint8_t>> k_buffers_;
    std::vector<std::vector<std::uint8_t>> v_buffers_;
};

}  // namespace pie_portable_driver
