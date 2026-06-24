// Raw-Metal PAGED KV-append (scatter) for multi-batch (M>1) decode/prefill.
//
// M>1 variant of kv_append.metal. Instead of a contiguous per-head ring indexed by
// a single absolute position, this scatters each of N batched tokens into the runtime's
// PAGED KV pool via the locked physical-slot formula. Token-major, head_dim LAST.
//
// Page layout (LOCKED, NHD — driver/cuda/src/kv_cache.hpp:3-4):
//   k/v pages per layer = [num_pages, page_size, n_kv_heads, head_dim]
//   → flattened: row `s` of [num_pages*page_size, n_kv_heads, head_dim].
//
// Physical-slot formula (the delta↔beta seam — this WRITES, beta's batched paged-attention
//   READS the same; verbatim MLX compute_write_indices, executor.cpp:44-53). For token i in
//   request r (r = req_of_token[i]; token i ∈ [qo_indptr[r], qo_indptr[r+1])), absolute pos p:
//     phys_slot = kv_page_indices[ kv_page_indptr[r] + p/page_size ] * page_size + (p % page_size)
//   No contiguity assumption — always walk the page table.
//
// Buffer order MATCHES alpha's published decode_abi.hpp KvAppend ordinals (f34e0eff0):
// M=1 binds 0-7 preserved byte-identical; paged binds appended at 8-12. The M=1 ring
// strides (6=KHeadStride, 7=KSeqStride) are bound by the host but UNUSED by this paged
// path, so they're omitted from the signature (Metal ignores bound-but-unreferenced
// buffers). See note `mac-paged-kv-bridge`. At N=1 with a one-page table this reduces
// to kv_append.metal.
//
// Launch: dispatchThreads grid=(head_dim, n_kv_heads, N). bfloat native.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void kv_append_paged(
    const device T* k_new   [[buffer(0)]],   // K:  [N, n_kv_heads, head_dim]
    const device T* v_new   [[buffer(1)]],   // V
    device T* k_pages       [[buffer(2)]],   // KPages: [num_pages*page_size, n_kv_heads, head_dim]
    device T* v_pages       [[buffer(3)]],   // VPages
    const device int* position_ids       [[buffer(4)]],   // PositionPtr: [N] absolute positions (IO)
    const constant int& head_dim         [[buffer(5)]],   // HeadDim
    // buffers 6 (KHeadStride) + 7 (KSeqStride) = M=1 ring strides — bound, unused here.
    const device int* kv_page_indices    [[buffer(8)]],   // KvPageIndices: [total_pages_in_batch]
    const device int* kv_page_indptr     [[buffer(9)]],   // KvPageIndptr:  [n_requests+1]
    const constant int& page_size        [[buffer(10)]],  // PageSize
    const device int* req_of_token       [[buffer(11)]],  // ReqOfToken: [N] request id per token (IO)
    const constant int& n_kv_heads       [[buffer(12)]],  // NKvHeads
    uint3 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);   // channel within head_dim
  const int h = int(tid.y);   // kv head
  const int i = int(tid.z);   // token within the batch [0, N)
  if (d >= head_dim) return;

  const int r = req_of_token[i];
  const int p = position_ids[i];
  const int page = kv_page_indices[kv_page_indptr[r] + p / page_size];
  const size_t slot = size_t(page) * size_t(page_size) + size_t(p % page_size);

  const size_t row_stride = size_t(n_kv_heads) * size_t(head_dim);  // NHD page row
  const size_t dst = slot * row_stride + size_t(h) * size_t(head_dim) + size_t(d);
  const size_t src = size_t(i) * row_stride + size_t(h) * size_t(head_dim) + size_t(d);

  k_pages[dst] = k_new[src];
  v_pages[dst] = v_new[src];
}

#define instantiate_kv_append_paged(name, itype)                  \
  template [[host_name("kv_append_paged_" #name)]]                \
  [[kernel]] void kv_append_paged<itype>(                         \
      const device itype*, const device itype*, device itype*,    \
      device itype*, const device int*, const constant int&,      \
      const device int*, const device int*, const constant int&,  \
      const device int*, const constant int&, uint3);

instantiate_kv_append_paged(float32, float)
instantiate_kv_append_paged(float16, half)
instantiate_kv_append_paged(bfloat16, bfloat)
