#include "slot_ops.hpp"

#include <string>

#include "../cuda_runtime.h"

namespace pie_cuda_driver::kernels::layout {

void zero_slots_if_fresh(
    std::uint8_t* base,
    std::size_t slot_bytes,
    std::size_t layer_stride_bytes,
    std::size_t layer_count,
    const std::int32_t*,
    const std::uint8_t*,
    std::size_t request_count,
    cudaStream_t)
{
    oracle_cuda::note(
        "zerofresh " + oracle_cuda::where(base) +
        " slot=" + std::to_string(slot_bytes) +
        " pitch=" + std::to_string(layer_stride_bytes) +
        " rows=" + std::to_string(layer_count) +
        " reqs=" + std::to_string(request_count));
}

}  // namespace pie_cuda_driver::kernels::layout
