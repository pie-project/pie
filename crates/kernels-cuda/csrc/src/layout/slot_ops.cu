//===-- slot_ops.cu - the two slot-conditional launchers -------------===//
//
// Two host launchers and not one `__global__`: the device text is in
// `layout/slot_ops.cuh`, which this file includes so the archive and the JIT
// header set hold the SAME definition rather than two that drift.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"
#include "layout/slot_ops.hpp"

// The `__global__`s these launchers fire. ONE definition of each.
#include "layout/slot_ops.cuh"

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels::layout {

void zero_slots_if_fresh(
    device::u8* base,
    device::usize slot_bytes,
    device::usize layer_stride_bytes,
    device::usize layer_count,
    const device::i32* slot_ids,
    const device::u8* is_fresh,
    device::usize request_count,
    cudaStream_t stream)
{
    if (base == nullptr || slot_bytes == 0 || layer_count == 0 ||
        request_count == 0) {
        return;
    }
    constexpr int kThreads = 256;
    device::zero_slots_if_fresh<<<
        dim3(
            static_cast<unsigned int>(request_count),
            static_cast<unsigned int>(layer_count)),
        kThreads, 0, stream>>>(
        base,
        slot_bytes,
        layer_stride_bytes,
        slot_ids,
        is_fresh,
        request_count);
    CUDA_CHECK(cudaGetLastError());
}

void copy_if_valid_slot(
    const device::u8* src,
    device::u8* dst,
    device::usize bytes,
    const device::i32* slot_ids,
    device::usize request,
    cudaStream_t stream)
{
    if (bytes == 0) return;
    constexpr int kThreads = 256;
    device::copy_if_valid_slot<<<1, kThreads, 0, stream>>>(
        src, dst, bytes, slot_ids, request);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels::layout
