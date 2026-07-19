#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pie_cuda_driver::pipeline {

inline bool translate_resolved_page_ids(
    std::vector<std::uint32_t>& read_pages,
    std::vector<std::uint32_t>& write_pages,
    std::span<const std::uint32_t> translation,
    bool masked_reads,
    std::string* error = nullptr) {
    for (std::uint32_t& value : read_pages) {
        if (value < translation.size()) {
            value = translation[value];
        } else if (masked_reads) {
            value = 0;
        } else {
            if (error) *error = "ptir: untranslatable unmasked read page";
            return false;
        }
    }
    for (std::uint32_t& value : write_pages) {
        if (value >= translation.size()) {
            if (error) *error = "ptir: untranslatable WSlot write target";
            return false;
        }
        value = translation[value];
    }
    return true;
}

}  // namespace pie_cuda_driver::pipeline
