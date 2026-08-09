//===-- geometry.cu - the two CSR launchers --------------------------===//
//
// Two host launchers and not one `__global__`: the device text is in
// `layout/geometry.cuh`, which this file includes so the archive and the JIT
// header set hold the SAME definition rather than two that drift.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"
#include "geometry.hpp"

// The `__global__`s these launchers fire. ONE definition of each.
#include "layout/geometry.cuh"

namespace pie_cuda_driver::kernels::layout {

void launch_derive_kv_len(
    const device::u32* kv_page_indptr,
    const device::u32* kv_last_page_lens,
    device::u32 page_size,
    device::u32 num_requests,
    device::u32* kv_len,
    cudaStream_t stream) {
  if (num_requests == 0) {
    return;
  }
  constexpr device::u32 kThreads = 256;
  const device::u32 blocks = (num_requests + kThreads - 1) / kThreads;
  device::derive_kv_len<<<blocks, kThreads, 0, stream>>>(
      kv_page_indptr, kv_last_page_lens, page_size, num_requests, kv_len);
}

void launch_resolve_slot_to_block(
    const device::u32* pages,
    const device::u32* slot_to_block,
    device::u32 num_slots,
    device::u32 count,
    device::u32* page_indices,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  constexpr device::u32 kThreads = 256;
  const device::u32 blocks = (count + kThreads - 1) / kThreads;
  device::resolve_slot_to_block<<<blocks, kThreads, 0, stream>>>(
      pages, slot_to_block, num_slots, count, page_indices);
}

}  // namespace pie_cuda_driver::kernels::layout
