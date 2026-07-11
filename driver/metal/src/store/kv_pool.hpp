#pragma once

#include <cstdint>
#include <vector>

#include "mtl4_context.hpp"

namespace pie::metal {

struct KvPagePool {
    struct LayerPages {
        SlotHandle k_pages;
        SlotHandle v_pages;
    };

    std::vector<LayerPages> layers;
    std::uint32_t total_pages = 0;
    std::uint32_t page_size = 0;
    bool enabled = false;
};

struct KvMoveCell {
    std::uint32_t dst_page_id;
    std::uint32_t dst_token_offset;
    std::uint32_t src_page_id;
    std::uint32_t src_token_offset;
};

}  // namespace pie::metal
