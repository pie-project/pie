#pragma once

#include <cstddef>
#include <cstdint>

namespace pie_cuda_driver {

enum class RsExecutionMode : std::int32_t {
    None = 0,
    Forward = 1,
    BufferWrite = 2,
    BufferFold = 3,
};

inline bool valid_rs_execution_mode(std::int32_t value) noexcept {
    return value >= static_cast<std::int32_t>(RsExecutionMode::None) &&
        value <= static_cast<std::int32_t>(RsExecutionMode::BufferFold);
}

inline bool rs_launch_requires_readiness_settlement(
    std::size_t slot_ids,
    std::size_t fold_lens,
    std::size_t buffer_ids,
    std::size_t /*buffer_indptr*/) noexcept {
    return slot_ids != 0 || fold_lens != 0 || buffer_ids != 0;
}

inline bool tp_rs_metadata_shape_valid(
    RsExecutionMode mode,
    std::size_t requests,
    std::size_t slot_ids,
    std::size_t slot_flags,
    std::size_t fold_lens,
    std::size_t buffer_ids,
    std::size_t buffer_indptr) noexcept {
    if (mode == RsExecutionMode::None) {
        return slot_ids == 0 && slot_flags == 0 && fold_lens == 0 &&
            buffer_ids == 0 && buffer_indptr == 0;
    }
    if (slot_ids != requests || slot_flags != requests) return false;
    if (mode == RsExecutionMode::Forward) {
        return (fold_lens == 0 || fold_lens == requests) &&
            buffer_ids == 0 && buffer_indptr == 0;
    }
    if (buffer_ids == 0 || buffer_indptr != requests + 1) return false;
    if (mode == RsExecutionMode::BufferWrite) {
        return fold_lens == 0 || fold_lens == requests;
    }
    return mode == RsExecutionMode::BufferFold && fold_lens == requests;
}

}  // namespace pie_cuda_driver
