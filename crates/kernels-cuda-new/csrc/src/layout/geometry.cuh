//===-- geometry.cuh - the CSR arithmetic two launchers do on device --===//
//
// Two `__global__`s and nothing else. `geometry.cu` includes this file and
// keeps its two launchers, so there is exactly ONE definition of each kernel
// -- a split, not a copy, because two copies agree on the day they are
// written and each stays right for whichever half of the tests exercises it.
//
// Neither kernel has a row yet, and the reason is not geometry: both launch
// `<<<ceil(n/256), 256>>>` with their own bound check, which is exactly
// `LaunchRule::Elementwise`. It is that neither is reachable from a model
// text. `derive_kv_len` and `resolve_slot_to_block` are called by the DRIVER
// while it composes a wave, not by a statement, so there is no fire whose
// operands a `Source` could name and inventing one would be a contract
// nothing checks. The device text moves anyway: `new-horizon.md` §10.10 puts
// the extraction first precisely so a row is a later, separable decision.
//
// Both are handed to NVRTC through the carried header set rather than through
// an include path, so nothing here may reach for the C++ standard library --
// `device::u32` is `pie_device.cuh`'s, which is what `<cstdint>` used to be.
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
using ::pie_cuda_driver::kernels::device::u32;

// One thread per request. Derives `kv_len[r]` from the CSR page descriptors,
// bit-identical to the host formula in request.rs (append_request_with_options):
//   page_count = kv_page_indptr[r+1] - kv_page_indptr[r]
//   kv_len[r]  = page_count == 0 ? 0
//                                : (page_count - 1) * page_size + last_page_len
// All arithmetic is u32 (matches the host's Vec<u32> column) so the device and
// host results are byte-for-byte equal — the M5 C1-FINAL handshake invariant.
__global__ void derive_kv_len(
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    u32 page_size,
    u32 num_requests,
    u32* __restrict__ kv_len) {
  const u32 r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_requests) {
    return;
  }
  const u32 page_count = kv_page_indptr[r + 1] - kv_page_indptr[r];
  kv_len[r] =
      page_count == 0u ? 0u : (page_count - 1u) * page_size + kv_last_page_lens[r];
}

// One thread per flattened page slot. Resolves a working-set slot id to its
// physical page-pool BlockId via the runtime-uploaded dictionary:
//   page_indices[i] = slot_to_block[pages[i]]
// An out-of-range slot id (>= num_slots) is a loud sentinel (0xFFFFFFFF), never
// a silent wrap — a corrupt/padding slot must fail visibly, not gather a wrong
// page. Slot id 0 is valid and resolved like any other.
__global__ void resolve_slot_to_block(
    const u32* __restrict__ pages,
    const u32* __restrict__ slot_to_block,
    u32 num_slots,
    u32 count,
    u32* __restrict__ page_indices) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const u32 slot = pages[i];
  page_indices[i] = slot < num_slots ? slot_to_block[slot] : 0xFFFFFFFFu;
}

}  // namespace pie_cuda_driver::kernels::layout::device
