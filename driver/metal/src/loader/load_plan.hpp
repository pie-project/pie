#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "pie_driver/model_contracts.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

namespace pie::metal {

using LoadPlan = pie_loader::LoadPlan;

/// This driver implements no tile transforms, so the loader may emit none; it
/// must reach every layout with copies alone. `backend::metal` states the same
/// thing from the Rust side. See the CUDA header for why the mask lives beside
/// the kernels rather than in the loader's SDK.
inline constexpr std::uint32_t kMetalTileMapMask = 0;

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
/// (`loader/architecture.md` §12 row 12).
inline LoadPlan compile_load_plan(
    std::string_view snapshot_dir,
    const pie_loader::DeviceTarget& target,
    const pie_driver::ModelFacts& model,
    std::string_view runtime_quant,
    pie_driver::Mxfp4MoeRequest mxfp4_moe,
    pie_driver::Component component) {
    std::string open_error;
    pie_loader::Checkpoint checkpoint =
        pie_loader::Checkpoint::open(snapshot_dir, &open_error);
    if (!checkpoint) {
        throw std::runtime_error("load plan: " + open_error);
    }

    pie_loader::ModelContract contract;
    pie_driver::author_model_contract(
        checkpoint, model, target, runtime_quant, mxfp4_moe, component, contract);

    const auto request =
        pie_loader::build_contract_request(checkpoint, target, contract.view());
    LoadPlan plan = LoadPlan::compile(request);
    // A second opinion, not a restatement: `verify` shares no code with the
    // compiler, and since §6 it also stats each file the plan declares.
    plan.verify(request);
    return plan;
}

}  // namespace pie::metal
