//===-- slot_ops.cuh - the two slot-conditional memory ops -----------===//
//
// Two `__global__`s. `slot_ops.cu` IS DELETED (`cd5cebd3d`) and took both of
// its launchers with it, so this file is the only text either kernel has.
//
// # Why neither is a row yet
//
// `zero_slots_if_fresh` launches `dim3(request_count, layer_count)` -- a
// second grid axis over LAYERS, which no `LaunchRule` spells; a rule invented
// for it would be a grid nothing checks.
//
// `copy_if_valid_slot` launches `<<<1, 256>>>` -- exactly one block, whatever
// the fire's row count, because `request` selects a single slot and the loop
// strides the whole byte span. `LaunchRule::RouteRows` is one block PER ROW
// and reads `dims.rows` from the fire, so a fire with sixteen rows would run
// the same copy sixteen times. Correct by accident (the copy is idempotent)
// and wrong as a statement, so the row is not written. There is no rule for
// "one block, whatever the rectangle".
//
// **THE SECOND HALF IS RETIRED.** `LaunchRule::Single` is that rule -- one
// block of 256, no `Dims` field read -- written from this launcher and
// `attn/kv_paged.cu:516` together, and `families/layout.rs`'s `SLOT_OPS` unit
// carries the row. Every sentence above survives as the ARGUMENT for it: the
// `1` is a literal the host wrote and `RowsFlat`'s `ceil(rows / 256)` equals
// it only up to 256 rows. `zero_slots_if_fresh` is refused still.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::layout::device {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels below read as they always did, so a row may keep spelling its
// element type `device::bf16`, and so the launchers in the enclosing
// namespace -- which write `device::` meaning the prelude's -- go on
// resolving to the same types through these declarations.
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u8;
using ::pie_cuda_driver::kernels::device::usize;

__global__ void zero_slots_if_fresh(
    u8* base,
    usize slot_bytes,
    usize layer_stride_bytes,
    const i32* slot_ids,
    const u8* is_fresh,
    usize request_count)
{
    const usize request = blockIdx.x;
    const usize layer = blockIdx.y;
    if (request >= request_count || is_fresh[request] == 0) return;
    const i32 slot = slot_ids[request];
    if (slot < 0) return;
    u8* out =
        base + layer * layer_stride_bytes +
        static_cast<usize>(slot) * slot_bytes;
    for (usize i = threadIdx.x; i < slot_bytes; i += blockDim.x) {
        out[i] = 0;
    }
}

__global__ void copy_if_valid_slot(
    const u8* src,
    u8* dst,
    usize bytes,
    const i32* slot_ids,
    usize request)
{
    if (slot_ids[request] < 0) return;
    for (usize i = threadIdx.x; i < bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace pie_cuda_driver::kernels::layout::device
