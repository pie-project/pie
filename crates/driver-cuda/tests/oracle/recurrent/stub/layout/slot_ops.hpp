#pragma once
// Stub for kernels-cuda/csrc/src/layout/slot_ops.hpp.
//
// `zero_slots_if_fresh` is a device-predicated scatter: the slot ids live in
// device memory, so which rows it touches is not knowable on the host. What IS
// knowable, and what a porting slip would get wrong, is the geometry it is
// handed -- slot stride, layer stride, layer count, request count. That is
// what the recorder writes down.
#include <cstddef>
#include <cstdint>
#include "cuda_runtime.h"

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

}  // namespace pie_cuda_driver::kernels::layout
