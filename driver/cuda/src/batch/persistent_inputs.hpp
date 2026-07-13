#pragma once

// Pre-allocated input device buffers, sized once at startup for the
// configured worst case and reused across `fire_batch` invocations.
//
// Why: the previous design called `DeviceBuffer<T>::from_host(...)` per
// fire — six fresh `cudaMalloc`s each time the wire control arrays were
// uploaded. That cost ~50 µs/fire on its own (small), but more
// importantly the freshly-malloc'd device addresses changed every fire
// → kernels couldn't be CUDA-graph-captured because the captured graph
// hardcodes pointer arguments at capture time. Replay with new
// pointers would read stale memory and produce gibberish.
//
// With this struct, launch geometry and model scratch live at stable addresses.
// The executor refreshes their contents per fire and hands raw pointers to the
// forward and PTIR kernels.
//
// All sizes are upper bounds derived from the configured workspace
// limits. Allocation is one-shot at engine init; deallocation is
// scope-driven (`DeviceBuffer<T>` is RAII).

#include <cstddef>
#include <cstdint>

#include "device_buffer.hpp"
#include "kernels/pack_dense_mask.hpp"

namespace pie_cuda_driver {

struct PersistentInputs {
    // Per-token arrays (capacity = max_workspace_tokens).
    DeviceBuffer<std::uint32_t> tokens;
    DeviceBuffer<std::uint32_t> positions;
    DeviceBuffer<std::int32_t>  sampled;

    // Per-request arrays (capacity = max_requests + 1 for indptrs,
    // max_requests for last_page_lens).
    DeviceBuffer<std::uint32_t> qo_indptr;        // R+1
    DeviceBuffer<std::uint32_t> kv_page_indptr;   // R+1
    DeviceBuffer<std::uint32_t> kv_last_page_lens;// R

    // Page-id list (capacity = planner's max_page_refs, independent of
    // the total KV-cache page count).
    DeviceBuffer<std::uint32_t> kv_page_indices;

    // Custom mask. `mask` is a packed bitmap; capacity is the worst
    // case ceil(max_qo * max_kv / 8). `indptr` has R+1 byte offsets.
    DeviceBuffer<std::uint8_t>  custom_mask;
    DeviceBuffer<std::int32_t>  custom_mask_indptr;
    DeviceBuffer<std::uint32_t> structured_mask_klen;
    DeviceBuffer<kernels::StructuredMaskParams> structured_masks;

    // Explicit KV-write descriptor (device-geometry WSlot/WOff lowering, B2).
    // Per-lane physical page id + offset-in-page for the single new-token K/V
    // write, consumed by `launch_write_kv_explicit_bf16` when a device-geometry
    // program binds the WSlot/WOff ports. Capacity = max_workspace_tokens
    // because prefill descriptors carry one target per token row.
    DeviceBuffer<std::uint32_t> w_page;
    DeviceBuffer<std::uint32_t> w_off;

    // Per-request linear-attention rs_cache slot ids (Qwen3.5 / 3.6).
    // Capacity = max_requests. Inert (zero-length writes) on archs that
    // don't use rs_cache. Lives in PersistentInputs so the TP
    // broadcast from rank 0 reaches followers via the same NCCL op
    // pattern as the rest of the per-fire payload.
    DeviceBuffer<std::int32_t>  slot_ids;
    DeviceBuffer<std::uint8_t>  is_fresh;
    DeviceBuffer<std::uint8_t>  rs_slot_flags;
    DeviceBuffer<std::uint32_t> rs_fold_lens;
    DeviceBuffer<std::uint32_t> rs_buffer_slot_ids;
    DeviceBuffer<std::uint32_t> rs_buffer_slot_indptr;
    PinnedHostBuffer<std::uint8_t>  rs_slot_flags_host;
    PinnedHostBuffer<std::uint32_t> rs_fold_lens_host;
    PinnedHostBuffer<std::uint32_t> rs_buffer_slot_ids_host;
    PinnedHostBuffer<std::uint32_t> rs_buffer_slot_indptr_host;
    DeviceBuffer<std::int32_t>  mtp_request_ids;
    PinnedHostBuffer<std::int32_t> mtp_positions_host;
    PinnedHostBuffer<std::int32_t> mtp_hidden_rows_host;
    PinnedHostBuffer<std::int32_t> mtp_request_ids_host;

    // Absolute model rows requested by the launched PTIR instances.
    DeviceBuffer<std::int32_t>   sample_idx;

    static PersistentInputs allocate(
        int max_workspace_tokens,
        int max_requests,
        int max_kv_pages,
        std::size_t max_custom_mask_bytes,
        int max_mtp_draft_rows);
};

// Memory-planner helper. Returns the byte budget for one PersistentInputs
// arena at the given (N tokens, R requests, max_page_refs, custom-mask)
// shape. Stays separate from `allocate` because the planner uses it
// repeatedly while sweeping bucket candidates.
std::size_t persistent_input_bytes(int N,
                                   int R,
                                   int max_page_refs,
                                   int max_custom_mask_bytes);

}  // namespace pie_cuda_driver
