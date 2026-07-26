#pragma once

#include <cstdint>

#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"

#include "loader_config.hpp"

namespace pie_cuda_driver {

using LoadPlan = pie_loader::LoadPlan;

/// What this driver's kernels implement, and therefore what the loader is
/// allowed to emit. The loader has no opinion of its own: it refuses a transform
/// outside this set rather than emitting one the executor would then have to
/// reject (`loader/architecture.md` §9). Stated here, beside those kernels, so
/// the cross-check stays one-sided.
inline constexpr std::uint32_t kCudaTileMapMask =
    pie_loader::kTileMapCast | pie_loader::kTileMapEncode |
    pie_loader::kTileMapReblock | pie_loader::kTileMapReorder |
    pie_loader::kTileMapRepack;

/// This driver's storage capability. One definition, two readers: the device
/// facts JSON published at create time, and the target spec supplied with every
/// compile request.
inline constexpr std::uint32_t kCudaPreferredAlignment = 256;
inline constexpr std::uint64_t kCudaMaxTileBytes = 64ull * 1024ull * 1024ull;

/// Whether this build wants the fused FP8 -> MXFP4 kernels.
///
/// Read once per process, not per instruction as the executor used to. The value
/// is a compile input, so it reaches the plan hash and the artifact cache key; a
/// mid-process change would make two different plans share one cache entry,
/// which is exactly the confusion moving the decision was meant to end.
inline bool fused_transcode_default() {
    static const bool enabled =
        !loader_config::env_truthy("PIE_CUDA_DISABLE_FUSED_TRANSCODE");
    return enabled;
}

/// This device, with the constants above already filled in. The caller states
/// only what it had to measure: its rank, and what the GPU turned out to
/// support.
inline pie_loader::DeviceTarget cuda_device_target() {
    return {
        .backend = pie_loader::PieLoaderBackendKind::Cuda,
        .tile_map_mask = kCudaTileMapMask,
        .max_tile_bytes = kCudaMaxTileBytes,
        .preferred_alignment = kCudaPreferredAlignment,
        .fused_transcode = fused_transcode_default(),
    };
}

}  // namespace pie_cuda_driver
