#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <pie_driver_abi.h>

namespace pie_native {

template <typename T>
struct Slice {
    const T* ptr = nullptr;
    std::size_t len = 0;

    constexpr std::size_t size() const noexcept { return len; }
    constexpr bool empty() const noexcept { return len == 0; }
    constexpr const T* data() const noexcept { return ptr; }
    template <typename U>
    std::span<const U> as() const noexcept {
        return std::span<const U>(reinterpret_cast<const U*>(ptr), len);
    }
};

struct ByteSlice {
    const void* ptr = nullptr;
    std::size_t len = 0;

    constexpr std::size_t size() const noexcept { return len; }
    constexpr bool empty() const noexcept { return len == 0; }
    template <typename U>
    std::span<const U> as() const noexcept {
        return std::span<const U>(reinterpret_cast<const U*>(ptr), len);
    }
};

template <typename T>
constexpr Slice<T> slice_from(const T* ptr, std::size_t len) noexcept {
    return Slice<T>{ptr, len};
}
inline Slice<std::uint32_t> slice_from_u32(const std::uint32_t* p, std::size_t n) noexcept {
    return Slice<std::uint32_t>{p, n};
}
inline Slice<std::uint64_t> slice_from_u64(const std::uint64_t* p, std::size_t n) noexcept {
    return Slice<std::uint64_t>{p, n};
}
inline Slice<std::uint8_t> slice_from_u8(const std::uint8_t* p, std::size_t n) noexcept {
    return Slice<std::uint8_t>{p, n};
}

struct LaunchView {
    Slice<PieTerminalCell*> terminal_cells;
    Slice<std::uint32_t> token_ids;
    Slice<std::uint32_t> position_ids;

    Slice<std::uint32_t> kv_page_indices;
    Slice<std::uint32_t> kv_page_indptr;
    Slice<std::uint32_t> kv_last_page_lens;
    Slice<std::uint32_t> qo_indptr;
    Slice<std::uint32_t> rs_slot_ids;
    Slice<std::uint8_t> rs_slot_flags;
    Slice<std::uint32_t> rs_fold_lens;
    Slice<std::uint32_t> rs_buffer_slot_ids;
    Slice<std::uint32_t> rs_buffer_slot_indptr;

    Slice<std::uint32_t> flattened_masks;
    Slice<std::uint32_t> mask_indptr;

    Slice<std::uint32_t> sampling_indices;
    Slice<std::uint32_t> sampling_indptr;

    Slice<std::uint64_t> ptir_program_hashes;
    Slice<std::uint64_t> ptir_program_instances;
    // Per-instance WorkingSet page translation (relative index -> physical
    // page id), CSR-partitioned per `ptir_program_instances` entry. Channel-
    // resolved `Pages`/`WSlot` values are mapped through it; an empty
    // segment passes values through untranslated (legacy physical geometry).
    Slice<std::uint32_t> kv_translation;
    Slice<std::uint32_t> kv_translation_indptr;
    // Program → wire-request attribution CSR (`n_prog + 1` entries when
    // present): program `p` owns wire request rows
    // `[row_indptr[p], row_indptr[p+1])` of qo/kv/sampling. A device-geometry
    // program's span is its empty wire placeholder; the composed batch
    // substitutes its channel-resolved geometry for that span.
    Slice<std::uint32_t> ptir_program_row_indptr;
    Slice<std::uint64_t> ptir_kv_write_lower_bounds;
    Slice<std::uint64_t> ptir_kv_write_upper_bounds;
    // Per-program sampled-logit placement in the composed workspace. Unlike a
    // CSR, starts need not be monotonic in original program order because
    // composition places wire programs before device-geometry programs.
    Slice<std::uint32_t> ptir_sample_starts;
    Slice<std::uint32_t> ptir_sample_counts;
    // Per-program runtime symbolic extents. UINT32_MAX denotes an extent that
    // cannot be represented by one scalar (for example, KV lengths across
    // multiple request rows) and must not be consumed by a grouped lane.
    Slice<std::uint32_t> ptir_row_counts;
    Slice<std::uint32_t> ptir_token_counts;
    Slice<std::uint32_t> ptir_kv_lens;
    Slice<std::uint32_t> ptir_page_counts;
    Slice<std::uint32_t> ptir_query_lens;
    Slice<std::uint32_t> ptir_key_lens;
    Slice<std::uint64_t> logical_fire_ids;
    Slice<std::uint64_t> channel_expected_head;
    Slice<std::uint64_t> channel_expected_tail;
    Slice<std::uint32_t> channel_ticket_indptr;
    // The batch carries a GUEST-supplied custom mask (vs engine-synthesized
    // causal BRLE rows, which accompany every wire prefill and are safely
    // dropped when a composed batch runs the standard causal path).
    bool has_user_mask = false;
    Slice<std::uint32_t> image_grids;
    Slice<std::uint8_t> image_pixels;
    Slice<std::uint32_t> image_pixel_indptr;
    Slice<std::uint32_t> image_mrope_positions;
    Slice<std::uint32_t> image_mrope_indptr;
    Slice<std::uint32_t> image_patch_positions;
    Slice<std::uint32_t> image_anchor_rows;

    Slice<std::uint8_t> audio_features;
    Slice<std::uint32_t> audio_feature_indptr;
    Slice<std::uint32_t> audio_anchor_rows;

    Slice<std::uint8_t> embed_rows;
    Slice<std::uint32_t> embed_indptr;
    Slice<std::uint32_t> embed_shapes;
    Slice<std::uint8_t> embed_dtypes;
    Slice<std::uint32_t> embed_anchor_rows;
    Slice<std::uint32_t> embed_block_indptr;

    constexpr std::size_t num_images() const noexcept { return image_grids.size() / 3; }
    constexpr std::size_t num_clips() const noexcept { return audio_anchor_rows.size(); }
    constexpr std::size_t size() const noexcept { return token_ids.size(); }
    constexpr bool empty() const noexcept { return token_ids.empty(); }
};

}  // namespace pie_native
