#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif
#endif

namespace pie_cuda_driver::kernels::layout {

void zero_slots_if_fresh(
    std::uint8_t* base,
    std::size_t slot_bytes,
    std::size_t layer_stride_bytes,
    std::size_t layer_count,
    const std::int32_t* slot_ids,
    const std::uint8_t* is_fresh,
    std::size_t request_count,
    cudaStream_t stream);

void copy_if_valid_slot(
    const std::uint8_t* src,
    std::uint8_t* dst,
    std::size_t bytes,
    const std::int32_t* slot_ids,
    std::size_t request,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::layout
