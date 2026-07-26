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
// CUDA's grew an env-gated `fused_transcode` default that Metal's never
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
    bool fp8_native = false;
    bool native_mxfp4_moe = false;
    /// Whether this build wants fused transform kernels where they exist. A
    /// compile input rather than an execution-time switch, so it reaches the
    /// plan and the artifact key: flipping it mid-process would otherwise make
    /// two different plans share one cache entry.
    bool fused_transcode = false;
};

/// The model facts the storage compile keys off.
///
/// The driver has already parsed `config.json` to build its model; the loader
/// does not read it a second time (`loader/architecture.md` §10.4). The string
/// members are borrowed and must outlive the compile call.
struct ModelFacts {
    std::string_view model_type;
    std::string_view quant_method;
    std::uint32_t num_hidden_layers = 0;
    std::uint32_t num_experts = 0;
    std::uint32_t num_experts_per_tok = 0;
};

inline PieLoaderBytes borrow(std::string_view text) {
    return PieLoaderBytes{
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size()};
}

/// Assemble the request for this device.
///
/// Everything the request borrows — `snapshot_dir`, `runtime_quant`, and the
/// strings in `model` — must outlive the `pie_loader_compile` call that
/// consumes it.
inline PieLoaderRequest build_request(
    std::string_view snapshot_dir,
    const DeviceTarget& target,
    const ModelFacts& model,
    std::string_view runtime_quant,
    PieLoaderMxfp4MoeRequest mxfp4_moe,
    PieLoaderComponent component) {
    return PieLoaderRequest{
        .snapshot_dir = borrow(snapshot_dir),
        .target =
            {
                .backend = static_cast<std::uint32_t>(target.backend),
                .tp_rank = target.tp_rank,
                .tp_size = target.tp_size,
                .max_tile_bytes = target.max_tile_bytes,
                .preferred_alignment = target.preferred_alignment,
                .tile_map_mask = target.tile_map_mask,
                .fp8_native = target.fp8_native,
                .native_mxfp4_moe = target.native_mxfp4_moe,
                .fused_transcode = target.fused_transcode,
            },
        .model =
            {
                .model_type = borrow(model.model_type),
                .quant_method = borrow(model.quant_method),
                .num_hidden_layers = model.num_hidden_layers,
                .num_experts = model.num_experts,
                .num_experts_per_tok = model.num_experts_per_tok,
            },
        .runtime_quant = borrow(runtime_quant),
        .mxfp4_moe = static_cast<std::uint32_t>(mxfp4_moe),
        .component = static_cast<std::uint32_t>(component),
    };
}

}  // namespace pie_loader
