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

// A buffer page that is reserved but has no physical slot behind it. Mirrors
// `RS_TRANSLATION_UNMAPPED` on the engine side.
inline constexpr std::uint32_t kRsTranslationUnmapped = 0xFFFFFFFFu;

// The RS twin of `translate_resolved_page_ids`, for ONE request row.
//
// Unlike KV there is no masked-read escape: an out-of-range or unmaterialized
// buffer page is always an error. A KV page the mask discards costs nothing to
// misread, but a buffered activation is folded into the recurrent state, where
// a wrong value is indistinguishable from a right one and never recoverable.
inline bool translate_resolved_rs_slot_ids(
    std::span<std::uint32_t> slot_ids,
    std::span<const std::uint32_t> translation,
    std::string* error = nullptr) {
    for (std::uint32_t& value : slot_ids) {
        if (value >= translation.size()) {
            if (error) *error = "ptir: rs buffer page outside the working set";
            return false;
        }
        const std::uint32_t physical = translation[value];
        if (physical == kRsTranslationUnmapped) {
            if (error) {
                *error = "ptir: rs buffer page is reserved but not materialized";
            }
            return false;
        }
        value = physical;
    }
    return true;
}

}  // namespace pie_cuda_driver::pipeline
