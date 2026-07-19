#pragma once

// `gather_tokens`: the driver-side op behind `compact`. Packs
// live token runs densely into fresh page slots per a host-given plan, so the
// working set reclaims token-space waste (frozen fork tails, H2O-style
// eviction). A STANDALONE streaming-copy kernel, NOT an attention-kernel
// modification (open question 5).
//
// The default KV page layout is NHD `[num_pages, page_size, num_kv_heads,
// head_dim]`, so a run of `len` consecutive tokens WITHIN a page is a single
// CONTIGUOUS span (`len · num_kv_heads · head_dim` elements). `compact` already
// splits any run that would straddle a destination page boundary, so every op
// here copies a contiguous src span to a contiguous dst span — the op is a
// batched device-to-device memcpy, targeting `cudaMemcpy` bandwidth.

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// One entry of the gather plan (mirrors the runtime `GatherOp`, but over
// PHYSICAL page ids — the host resolves slot id → page id before launch).
// Copies `len` tokens from token offset `src_off` of page `src_page` to offset
// `dst_off` of page `dst_page`. `src_off + len <= page_size` and the dst span
// stays within one page (guaranteed by `compact`'s per-page split).
struct GatherTokenOp {
    std::uint32_t src_page;
    std::uint32_t src_off;
    std::uint32_t dst_page;
    std::uint32_t dst_off;
    std::uint32_t len;
};

// Pack the plan `ops` densely for ONE layer's paged K/V (bf16, NHD). `k_pages`
// and `v_pages` are `[num_pages, page_size, num_kv_heads, head_dim]` bf16 (as
// `std::uint16_t`). Both K and V are copied per op. Safe to run on the copy
// stream off the decode path (the copies ride behind the grace period). A
// per-layer call; the caller loops layers (or batches them via `num_layers` +
// `layer_stride_elems` below).
void launch_gather_tokens_bf16(
    std::uint16_t* k_pages,
    std::uint16_t* v_pages,
    const GatherTokenOp* ops,
    int num_ops,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

// Multi-layer variant: `k_pages`/`v_pages` point at layer 0; layer L starts at
// `layer_stride_elems * L` elements (typically `num_pages · page_size ·
// num_kv_heads · head_dim`). One launch copies every op for every layer.
void launch_gather_tokens_bf16_layers(
    std::uint16_t* k_pages,
    std::uint16_t* v_pages,
    const GatherTokenOp* ops,
    int num_ops,
    int num_layers,
    std::int64_t layer_stride_elems,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
