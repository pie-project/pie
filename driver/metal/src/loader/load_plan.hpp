#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "model/contract.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

namespace pie::metal {

using LoadPlan = pie_loader::LoadPlan;

/// What this driver's load-time kernels implement, and therefore what the loader
/// is allowed to emit. The loader has no opinion of its own: it refuses a
/// transform outside this set rather than emitting one the executor would then
/// have to reject. Stated here, beside those kernels, so the cross-check stays
/// one-sided.
///
/// `Scale` decodes a block-scaled scheme to values and `Cast` re-encodes those
/// values as the affine-U4 the matvecs read; together they are what lets the
/// published MXFP4 gpt-oss checkpoint load without an offline conversion. The
/// three this driver does not implement -- `Reblock`, `Repack` and the fused
/// kinds -- are layouts no kernel here wants.
inline constexpr std::uint32_t kMetalTileMapMask =
    pie_loader::kTileMapCast | pie_loader::kTileMapEncode | pie_loader::kTileMapScale;

/// This driver's storage capability. One definition, two readers: the device
/// facts JSON published at create time, and the target spec supplied with every
/// compile request.
inline constexpr std::uint32_t kMetalPreferredAlignment = 256;
inline constexpr std::uint64_t kMetalMaxTileBytes = 64ull * 1024ull * 1024ull;

/// This device, with the constants above already filled in.
inline pie_loader::DeviceTarget metal_device_target() {
    return {
        .backend = pie_loader::PieLoaderBackendKind::Metal,
        .tile_map_mask = kMetalTileMapMask,
        .max_tile_bytes = kMetalMaxTileBytes,
        .preferred_alignment = kMetalPreferredAlignment,
    };
}

/// Compile a contract this driver authored, for this device.
///
/// Three inputs and no fourth: what the files contain, what this driver will
/// bind, and what the GPU can do. The loader is not told which model this is,
/// so a family it has never heard of loads exactly as well as one it has
/// (`loader/architecture.md` §12 row 12). `model_type` selects the schema on
/// *this* side of the call and never crosses it.
inline LoadPlan compile_load_plan(
    std::string_view snapshot_dir,
    const pie_loader::DeviceTarget& target,
    std::string_view model_type,
    const model::ContractFacts& facts = {}) {
    std::string open_error;
    pie_loader::Checkpoint checkpoint =
        pie_loader::Checkpoint::open(snapshot_dir, &open_error);
    if (!checkpoint) {
        throw std::runtime_error("load plan: " + open_error);
    }

    pie_loader::ModelContract contract;
    model::author_model_contract(checkpoint, model_type, target, contract, facts);

    const auto request =
        pie_loader::build_contract_request(checkpoint, target, contract.view());
    LoadPlan plan = LoadPlan::compile(request);
    // A second opinion, not a restatement: `verify` shares no code with the
    // compiler, and since §6 it also stats each file the plan declares.
    plan.verify(request);
    return plan;
}

}  // namespace pie::metal
