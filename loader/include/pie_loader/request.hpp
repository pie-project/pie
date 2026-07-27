#pragma once

// Assembling a compile request.
//
// `PieLoaderRequest` is a flat POD of borrowed byte spans, which is the right
// shape for an ABI and the wrong shape to fill in by hand: every string member
// needs the same `reinterpret_cast`, and a member left at zero is a silently
// different plan rather than a compile error. This header is the small typed
// front for it.
//
// Both drivers used to keep their own copy of this, differing only in which
// backend enum and tile-map mask they passed. The copies had already drifted —
// CUDA's grew an env-gated fused-transcode default that Metal's never
// mentioned — so the merged version takes those three as ordinary fields the
// caller states, and each driver keeps only its own constants.

#include <cstdint>
#include <string_view>

#include "pie_loader.h"

namespace pie_loader {

/// What the calling driver knows about the device it is about to load onto.
///
/// The driver is the only party that can measure these, so it is the party that
/// compiles the plan (`loader/architecture.md` §3). Everything here flows
/// straight into `PieLoaderTargetSpec`, and the loader refuses a target it
/// cannot satisfy rather than emitting a plan the caller must re-check (§9).
struct DeviceTarget {
    /// Which driver is asking, and which tile transforms its kernels implement.
    /// The loader has no opinion of its own and will not emit a transform
    /// outside the mask.
    PieLoaderBackendKind backend = PieLoaderBackendKind::Unknown;
    std::uint32_t tile_map_mask = 0;

    std::uint32_t tp_rank = 0;
    std::uint32_t tp_size = 1;
    std::uint64_t max_tile_bytes = 0;
    std::uint32_t preferred_alignment = 0;
    bool native_mxfp4_moe = false;
    /// Which fused transform chains this build has kernels for
    /// (`PIE_LOADER_FUSION_*`), ORed together.
    ///
    /// A compile input rather than an execution-time switch, so it reaches the
    /// plan and the artifact key: flipping it mid-process would otherwise make
    /// two different plans share one cache entry. The loader knows what each
    /// fusion *means*; this says only which ones exist here.
    std::uint32_t fusion_mask = 0;
    /// The dtype this device's encode kernels dequantize through, which is what
    /// decides how many rows of scratch fit in `max_tile_bytes`.
    PieLoaderDType encode_scratch_dtype = PieLoaderDType::BF16;
    /// Row granularity of the block scales this device's encode path consumes,
    /// or `0` if it has none. A block-scaled source is not tiled, because a tile
    /// boundary would cut a scale block in half.
    std::uint32_t block_scale_rows = 0;
};

inline PieLoaderBytes borrow(std::string_view text) {
    return PieLoaderBytes{
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size()};
}

/// Assemble the request for this device.
///
/// Marshal a `DeviceTarget` into the POD the loader reads.
///
/// One function, in one place. The two copies this replaced — one per driver —
/// had already disagreed once: a field added for the contract path was a compile
/// error on one side and a silently stale value on the other.
///
/// The `static_cast`s are the ABI rule, not sloppiness. Every enum-valued field
/// crosses as a `uint32_t` because these are *inputs*: the loader must be able
/// to reject a value C++ made up, and a Rust enum holding a value outside its
/// variants is undefined behaviour before any check can run. The plan's
/// `PieLoaderTargetView` reads back as real enums, because the loader wrote it.
inline PieLoaderTargetSpec target_spec(const DeviceTarget& target) {
    return PieLoaderTargetSpec{
        .backend = static_cast<std::uint32_t>(target.backend),
        .tp_rank = target.tp_rank,
        .tp_size = target.tp_size,
        .max_tile_bytes = target.max_tile_bytes,
        .preferred_alignment = target.preferred_alignment,
        .tile_map_mask = target.tile_map_mask,
        .native_mxfp4_moe = target.native_mxfp4_moe,
        .fusion_mask = target.fusion_mask,
        .encode_scratch_dtype = static_cast<std::uint32_t>(target.encode_scratch_dtype),
        .block_scale_rows = target.block_scale_rows,
    };
}

}  // namespace pie_loader
