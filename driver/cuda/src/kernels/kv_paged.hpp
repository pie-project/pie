#pragma once

// Write current-step K/V into the paged KV pool.
//
// Per-token destination resolved as (described in the wire format):
//   pre_kv_len_r   = total_kv_after_r - num_new_tokens_r
//   abs_kv_pos     = pre_kv_len_r + offset_in_new_tokens
//   page_idx_in_r  = abs_kv_pos / page_size
//   offset_in_page = abs_kv_pos % page_size
//   actual_page    = kv_page_indices[kv_page_indptr[r] + page_idx_in_r]

#include <cstdint>
#include <cuda_runtime.h>

#include "kernels/kv_cache_view.hpp"

namespace pie_cuda_driver::kernels {

void launch_write_kv_to_pages_bf16(
    void* k_pages,                                 // NHD: [pages, page_size, h_kv, d]; HND: [pages, h_kv, page_size, d]
    void* v_pages,
    const void* k_curr,                            // [total_tokens, h_kv, d]
    const void* v_curr,
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    const std::uint32_t* kv_last_page_lens,        // [R]
    int total_tokens,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    cudaStream_t stream,
    const std::uint8_t* row_valid = nullptr);

void launch_write_kv_to_pages(
    KvCacheLayerView layer,
    const void* k_curr,                            // [total_tokens, h_kv, d]
    const void* v_curr,
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    const std::uint32_t* kv_last_page_lens,        // [R]
    int total_tokens,
    int num_requests,
    cudaStream_t stream,
    const std::uint8_t* row_valid = nullptr);

void launch_write_kv_to_pages_at_positions_bf16(
    KvCacheLayerView layer,
    const void* k_curr,                            // [total_tokens, h_kv, d]
    const void* v_curr,
    const std::int32_t* positions,                 // [total_tokens], absolute positions
    int position_delta,
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    int total_tokens,
    int num_requests,
    cudaStream_t stream);

void launch_dequant_kv_cache_layer_to_bf16_active(
    KvCacheLayerView layer,
    const std::uint32_t* kv_page_indices,
    int num_pages_in_batch,
    cudaStream_t stream);

// Explicit-descriptor KV write (the general WSlot/WOff lowering; formerly
// write_kv_beam): each lane writes its ONE new-token K/V into an EXPLICIT
// (physical page id `w_page[lane]`, offset `w_off[lane]`) target, consuming a
// program's WSlot/WOff (write-offset separated from KvLen) rather than
// re-deriving the position from the page-table + last_page_len. Single-cell per
// lane → shared-page-safe (a sibling's mask hides this cell). Requires a
// native-bf16 KV cache. `w_page` must already be PHYSICAL page ids (resolve
// slot→physical before the call).
void launch_write_kv_explicit_bf16(
    KvCacheLayerView layer,
    const void* k_curr,                 // [LANES, h_kv, d]
    const void* v_curr,
    const std::uint32_t* w_page,        // [LANES] physical page id per lane
    const std::uint32_t* w_off,         // [LANES] offset-in-page per lane
    int B,
    cudaStream_t stream,
    const std::uint8_t* row_valid = nullptr);

// Compaction primitive (Design-B lazy GC): move N token KV cells (single layer)
// from explicit (src physical page, src offset) → (dst physical page, dst offset)
// targets, for both K and V. Raw element copy — correct because the KV cache is
// stored POST-RoPE (slot = pure storage; positions live in the per-beam mask).
// Caller guarantees DISJOINT src/dst spans (in-place two-pointer) so one pass
// needs no scratch. Invoke per layer to move all layers. Native-bf16 KV.
void launch_copy_kv_cells_bf16(
    KvCacheLayerView layer,
    const std::uint32_t* dst_page,      // [N] physical page id per cell
    const std::uint32_t* dst_off,       // [N] offset-in-page per cell
    const std::uint32_t* src_page,      // [N] physical page id per cell
    const std::uint32_t* src_off,       // [N] offset-in-page per cell
    int N,
    cudaStream_t stream);

// ── Sliding-window page trim ───────────────────────────────────────────
// A paged decode kernel handed a `window_left` still WALKS every page it is
// given and masks what falls outside; the window buys arithmetic, not traffic.
// Measured on an H100 at gpt-oss's decode shape (8 kv heads, gqa 8, head_dim
// 64, window 128), cost is perfectly linear in context -- 10.2us at 256, 136us
// at 4096 -- for a window that never grows. The fix is to hand the kernel a
// shorter page list, which is a PLAN-level change: a decode query sits at the
// END of its range, so dropping whole pages off the FRONT leaves the last
// `window+1` tokens exactly where they were and `window_left` keeps masking
// correctly against the same absolute positions.
//
// Gathers, per request, the last `dst_indptr[r+1] - dst_indptr[r]` page ids of
// that request's slice of `src_indices`. The caller decides how many to keep
// (see `trimmed_window_pages`), which is why the count is read off the
// destination indptr rather than recomputed here.
void launch_gather_trailing_pages(
    const std::uint32_t* src_indices,   // [src_indptr[R]] physical page ids
    const std::uint32_t* src_indptr,    // [R+1] device
    const std::uint32_t* dst_indptr,    // [R+1] device
    std::uint32_t* dst_indices,         // [dst_indptr[R]] out
    int R,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
