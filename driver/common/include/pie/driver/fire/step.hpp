#pragma once

// The expanded per-step launch (Project Venus, ABI v14).
//
// A `PieFrameDesc` carries frame-invariant tables (lane roster, WorkingSet
// page translation, admission demand) once, plus per-step sections that
// reference the roster by index. Driver internals, however, operate on the
// v13 batch shape: one launch with materialized instance ids and a per-member
// translation CSR. `StepLaunch` IS that shape — each step of an admitted
// frame expands into one (or, with multiple sub-batches, several) of these
// before entering the per-batch pipeline. The expansion owns the
// materialized arrays; every other field borrows the frame descriptor for
// the duration of the launch call, exactly like the old ABI struct did.

#include <cstdint>

#include <pie_driver_abi.h>

namespace pie::driver::fire {

struct StepLaunch {
    PieU64Slice instance_ids;
    PieTerminalCellPtrSlice terminal_cells;
    PieU32Slice token_ids;
    PieU32Slice position_ids;
    PieU32Slice kv_page_indices;
    PieU32Slice kv_page_indptr;
    PieU32Slice kv_last_page_lens;
    PieU32Slice qo_indptr;
    PieU32Slice rs_slot_ids;
    PieU8Slice rs_slot_flags;
    PieU32Slice rs_fold_lens;
    PieU32Slice rs_buffer_slot_ids;
    PieU32Slice rs_buffer_slot_indptr;
    PieU32Slice rs_buffer_read_slot_ids;
    PieU32Slice rs_buffer_read_indptr;
    PieU32Slice rs_buffer_read_lens;
    PieU32Slice rs_buffer_heads;
    // WorkingSet-relative buffer page -> physical slot for channel-resolved
    // `rs-geometry`, concatenated over request rows with `rs_translation_indptr`
    // as the row CSR. Per ROW, unlike `kv_translation`: a pass binds one RS
    // working set per request, not one per pass. 0xFFFFFFFF marks a page that
    // is reserved but not yet materialized -- naming one is an error, not
    // readable garbage, because unmaterialized activations fold silently.
    PieU32Slice rs_translation;
    PieU32Slice rs_translation_indptr;
    PieMaskWordsDesc masks;
    PieU32Slice sampling_indices;
    PieU32Slice sampling_indptr;
    PieU64Slice context_ids;
    std::uint8_t single_token_mode = 0;
    std::uint8_t has_user_mask = 0;
    /// Frame-union admission demand (the frame's `required_kv_pages`),
    /// mirrored onto every step so per-batch accounting sees the high-water.
    std::uint32_t required_kv_pages = 0;
    PieU32Slice image_indptr;
    PieU32Slice image_grids;
    PieU32Slice image_anchor_positions;
    PieBytes image_pixels;
    PieU32Slice image_pixel_indptr;
    PieU32Slice image_mrope_positions;
    PieU32Slice image_mrope_indptr;
    PieU32Slice image_patch_positions;
    PieU32Slice image_anchor_rows;
    PieBytes audio_features;
    PieU32Slice audio_feature_indptr;
    PieU32Slice audio_anchor_rows;
    PieU32Slice audio_indptr;
    PieBytes embed_rows;
    PieU32Slice embed_indptr;
    PieU32Slice embed_shapes;
    PieU8Slice embed_dtypes;
    PieU32Slice embed_anchor_rows;
    PieU32Slice embed_block_indptr;
    PieU32Slice kv_len;
    PieU64Slice kv_len_device;
    /// Per-member WorkingSet page translation, sliced from the frame table
    /// for THIS step's members (CSR per `instance_ids` entry).
    PieU32Slice kv_translation;
    PieU32Slice kv_translation_indptr;
    PieU32Slice ptir_program_row_indptr;
    PieU64Slice ptir_kv_write_lower_bounds;
    PieU64Slice ptir_kv_write_upper_bounds;
    PieU64Slice logical_fire_ids;
    PieU64Slice channel_expected_head;
    PieU64Slice channel_expected_tail;
    PieU32Slice channel_ticket_indptr;
    // V2 rung 3c-ii: the region table (north-star-dsl.md "RUNG 3
    // SPEC") — the seriation's output, and since the four planned
    // words died, the plans' SINGLE source: the driver derives them at
    // view assembly (region_plans.hpp). Empty = no table.
    PieU32Slice region_row_indptr;
    PieU32Slice region_sig;
    PieU32Slice region_k;
};

}  // namespace pie::driver::fire
