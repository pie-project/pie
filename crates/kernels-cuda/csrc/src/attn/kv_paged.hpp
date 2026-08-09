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

#include "attn/kv_cache_view.hpp"

namespace pie_cuda_driver::kernels::attn {

// `write_kv_to_pages_bf16` and `write_kv_to_pages` are DECLARED NOWHERE now,
// because they are defined nowhere: both are Rust, in
// `driver-cuda/src/fire/kv_paged.rs`. `attn::write_kv_to_pages` is still a
// live `table::attn` row reached from model text -- what changed is that
// `execution::RUST_SERVED` routes its generated arm to
// `bind::service::attn_write_kv_to_pages` instead of to a `pie_k_` shim body
// that called the launcher this declaration used to name.

// `write_kv_to_pages_bf16_devwin` and `write_kv_to_pages_at_positions_bf16`
// WERE declared here. Both launchers are deleted: nothing reachable called
// either, and the kernels they fired are `attn/kv_paged.cuh`'s and are still
// compiled. The `.cu` carries the evidence for each separately.

void dequant_kv_cache_layer_to_bf16_active(
    KvCacheLayerView layer,
    const std::uint32_t* kv_page_indices,
    int num_pages_in_batch,
    cudaStream_t stream);

// `write_kv_explicit_bf16_devwin`'s declaration is GONE with its definition.
// The Rust is `fire::kv_paged::write_kv_explicit_bf16_devwin`, reached
// through `bind::service::attn_write_kv_explicit_bf16_devwin`; the DEVICE
// rows it fires are `attn::write_kv_explicit_bf16_devwin_dev#hnd`/`#nhd`,
// renamed by §60.6 for exactly the reason the sibling above was.
//
// The contract paragraph that stood here is not lost -- it is the doc comment
// on the Rust, which reproduces both `throw` messages verbatim and both
// `<<<>>>` lines by number. What it said: each lane writes its ONE new-token
// K/V into an EXPLICIT (physical page id `w_page[lane]`, offset
// `w_off[lane]`) target, consuming a program's WSlot/WOff rather than
// re-deriving the position from the page table; single-cell per lane, so
// shared-page-safe; requires a native-bf16 KV cache; `w_page` must already be
// PHYSICAL page ids. The `_devwin` part: the `{start, len}` row window rides
// in DEVICE memory so a captured launch replays across row splits, the grid
// is the full lane count, and out-of-window rows early-out. Envelope (quest)
// maintenance is not wired on this variant.

// `write_kv_explicit_bf16`'s declaration is GONE with its definition, for
// the reason above and one more: its envelope merge was one of the two call
// sites holding `layout/envelope.hpp` in `kv_paged.cu`. The Rust is
// `fire::kv_paged::write_kv_explicit_bf16`; the DEVICE rows it fires are
// `attn::write_kv_explicit_bf16_dev#hnd`/`#nhd`, renamed by §60.6 so the
// ahead-of-time symbol a trace records could become a `Walk`.

// `copy_kv_cells_bf16`'s declaration is GONE with its definition; the
// contract paragraph that stood here -- Design-B lazy GC, raw element copy
// correct because the cache is stored POST-RoPE, caller guarantees DISJOINT
// src/dst spans so one in-place two-pointer pass needs no scratch, invoke per
// layer -- is carried verbatim on `fire::kv_paged::copy_kv_cells_bf16`, which
// is what fires the two device rows now. `kv_paged.cu` has the consumer-set
// evidence.

// `build_window_page_view` was declared here and IS DELETED — the host program is
// `driver-cuda/src/fire/kv_paged.rs::build_window_page_view`.

// `build_full_split_view` was declared here and IS DELETED — the host program is
// `driver-cuda/src/fire/kv_paged.rs::build_full_split_view`.

}  // namespace pie_cuda_driver::kernels::attn
