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

void launch_zero_slots_if_fresh(
    std::uint8_t* base,
    std::size_t slot_bytes,
    std::size_t layer_stride_bytes,
    std::size_t layer_count,
    const std::int32_t* slot_ids,
    const std::uint8_t* is_fresh,
    std::size_t request_count,
    cudaStream_t stream);

void launch_copy_if_valid_slot(
    const std::uint8_t* src,
    std::uint8_t* dst,
    std::size_t bytes,
    const std::int32_t* slot_ids,
    std::size_t request,
    cudaStream_t stream);

}  // namespace pie_cuda_driver
