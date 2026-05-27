#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif
#endif

namespace pie_cuda_driver {

struct SlabScatterPlacement {
    std::uint64_t src_offset;
    std::uint64_t dest_offset;
    std::uint64_t bytes;
};

void launch_slab_scatter(
    const std::uint8_t* src,
    std::uint8_t* dst,
    const SlabScatterPlacement* placements,
    std::size_t placement_count,
    cudaStream_t stream);

}  // namespace pie_cuda_driver
