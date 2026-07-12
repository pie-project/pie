#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "kernels/kv_cache_view.hpp"
#include "../tensor.hpp"

namespace pie_cuda_driver {

enum class KvCacheScaleLayout : std::uint8_t {
    None,
    PerTokenHead,
    PerTokenHeadBlock,
};

struct KvCacheFormat {
    std::string name = "bf16";
    KvCacheScheme scheme = KvCacheScheme::Native;
    KvCacheScaleLayout scale_layout = KvCacheScaleLayout::None;
    DType storage_dtype = DType::BF16;
    int block_size = 0;

    bool is_native_bf16() const noexcept {
        return scheme == KvCacheScheme::Native && storage_dtype == DType::BF16;
    }
    bool has_side_scales() const noexcept {
        return scale_layout != KvCacheScaleLayout::None;
    }

    // Number of scalar storage elements for one token/head row. For FP4 this
    // is packed two logical values per byte, so it is ceil(head_dim / 2).
    std::int64_t storage_head_dim(int head_dim) const noexcept;

    // Bytes for one K or V page.
    std::size_t kv_bytes_per_page(int page_size,
                                  int num_kv_heads,
                                  int head_dim) const noexcept;

    // Bytes for one K-scale or V-scale page. Zero when the format has no side
    // scale buffer. Side scales are fp32 in this first milestone.
    std::size_t scale_bytes_per_page(int page_size,
                                     int num_kv_heads,
                                     int head_dim) const noexcept;

    std::size_t total_bytes_per_page(int page_size,
                                     int num_kv_heads,
                                     int head_dim) const noexcept {
        return 2 * kv_bytes_per_page(page_size, num_kv_heads, head_dim) +
               2 * scale_bytes_per_page(page_size, num_kv_heads, head_dim);
    }
};

KvCacheFormat kv_cache_format_from_string(const std::string& value,
                                          const std::string& activation_dtype = "bfloat16");

bool is_valid_kv_cache_dtype(const std::string& value) noexcept;

std::string valid_kv_cache_dtype_values();

}  // namespace pie_cuda_driver
