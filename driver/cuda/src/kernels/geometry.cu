#include "geometry.hpp"

namespace pie_cuda_driver::kernels {

namespace {

// One thread per request. Derives `kv_len[r]` from the CSR page descriptors,
// bit-identical to the host formula in request.rs (append_request_with_options):
//   page_count = kv_page_indptr[r+1] - kv_page_indptr[r]
//   kv_len[r]  = page_count == 0 ? 0
//                                : (page_count - 1) * page_size + last_page_len
// All arithmetic is u32 (matches the host's Vec<u32> column) so the device and
// host results are byte-for-byte equal — the M5 C1-FINAL handshake invariant.
__global__ void derive_kv_len_kernel(
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    std::uint32_t page_size,
    std::uint32_t num_requests,
    std::uint32_t* __restrict__ kv_len) {
  const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_requests) {
    return;
  }
  const std::uint32_t page_count = kv_page_indptr[r + 1] - kv_page_indptr[r];
  kv_len[r] =
      page_count == 0u ? 0u : (page_count - 1u) * page_size + kv_last_page_lens[r];
}

}  // namespace

void launch_derive_kv_len(
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    std::uint32_t page_size,
    std::uint32_t num_requests,
    std::uint32_t* kv_len,
    cudaStream_t stream) {
  if (num_requests == 0) {
    return;
  }
  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t blocks = (num_requests + kThreads - 1) / kThreads;
  derive_kv_len_kernel<<<blocks, kThreads, 0, stream>>>(
      kv_page_indptr, kv_last_page_lens, page_size, num_requests, kv_len);
}

namespace {

// One thread per flattened page slot. Resolves a working-set slot id to its
// physical page-pool BlockId via the runtime-uploaded dictionary:
//   page_indices[i] = slot_to_block[pages[i]]
// An out-of-range slot id (>= num_slots) is a loud sentinel (0xFFFFFFFF), never
// a silent wrap — a corrupt/padding slot must fail visibly, not gather a wrong
// page. Slot id 0 is valid and resolved like any other.
__global__ void resolve_slot_to_block_kernel(
    const std::uint32_t* __restrict__ pages,
    const std::uint32_t* __restrict__ slot_to_block,
    std::uint32_t num_slots,
    std::uint32_t count,
    std::uint32_t* __restrict__ page_indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::uint32_t slot = pages[i];
  page_indices[i] = slot < num_slots ? slot_to_block[slot] : 0xFFFFFFFFu;
}

}  // namespace

void launch_resolve_slot_to_block(
    const std::uint32_t* pages,
    const std::uint32_t* slot_to_block,
    std::uint32_t num_slots,
    std::uint32_t count,
    std::uint32_t* page_indices,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t blocks = (count + kThreads - 1) / kThreads;
  resolve_slot_to_block_kernel<<<blocks, kThreads, 0, stream>>>(
      pages, slot_to_block, num_slots, count, page_indices);
}

}  // namespace pie_cuda_driver::kernels
