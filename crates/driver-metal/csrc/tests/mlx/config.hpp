#pragma once

#include <cstdint>
#include <string>

namespace pie_metal_driver {

struct BatchingConfig {
    std::uint32_t kv_page_size = 32;
    std::uint32_t total_pages = 1024;
    std::uint32_t max_forward_tokens = 10240;
    std::uint32_t max_forward_requests = 512;
    std::uint32_t cpu_pages = 0;
    std::string kv_cache_dtype = "auto";
};

}  // namespace pie_metal_driver
