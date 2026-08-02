#pragma once

#include <cstdint>

namespace pie_cuda_driver::pipeline {

#if defined(__CUDACC__)
#define PIE_GROUPED_HOST_DEVICE __host__ __device__
#else
#define PIE_GROUPED_HOST_DEVICE
#endif

PIE_GROUPED_HOST_DEVICE inline constexpr std::uint64_t
grouped_row_strided_source_index(
    std::uint64_t logical_index,
    std::uint32_t logical_row_width,
    std::uint32_t physical_row_stride) noexcept {
    return logical_row_width == 0
        ? logical_index
        : (logical_index / logical_row_width) * physical_row_stride +
            logical_index % logical_row_width;
}

#undef PIE_GROUPED_HOST_DEVICE

}  // namespace pie_cuda_driver::pipeline
