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

// `zero_slots_if_fresh` was deleted here by §43. It is the clearest
// mentions-versus-fires case in the tree: `layout/slot_ops.cuh` still holds
// the `__global__`, `families::layout` documents its grid at length, and
// `driver-cuda`'s `StateOp::ZeroSlotsIfFresh` is built twice in
// `pools/recurrent_state_cache.rs` -- but `serve/state.rs` matches that
// variant with `=> continue` and resets from the host instead. Four channels
// name it and none fires it, so the launcher reached nothing.

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
