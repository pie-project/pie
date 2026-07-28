#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pie_loader/source_checkpoint.hpp"

#include "loader/rust_quant_attachment.hpp"
#include "loader/load_plan.hpp"

#include "model/config.hpp"

namespace pie_cuda_driver {

struct LoadPlanResult {
    LoadPlan plan;
    std::vector<RustQuantAttachment> quant_attachments;
    /// Tensors the plan produces. The only count anything reads: it sizes the
    /// weight store's reservation.
    std::size_t planned_tensor_count = 0;
    std::string cache_key;
};

/// Resolve the plan's quant attachments into the name-keyed form the executor's
/// `WeightStore` is addressed by.
///
/// Only a translation. The pairing itself used to be *inferred* here by matching
/// name suffixes over the plan's tensor list, and then, for a while, by matching
/// them one layer earlier inside the loader. It is now recorded by whoever
/// declares the scale tensor: `plan/build.rs::quant_metadata_outputs` for scales
/// the loader writes, and the contract's `scales` field — see
/// `dsv4_block_scales_to_fp32` -- for scales the checkpoint shipped.
inline std::vector<RustQuantAttachment> resolve_quant_attachments(
    const pie_loader::LoadPlanView& view) {
    std::unordered_map<std::uint32_t, pie_loader::PieLoaderBytes> names;
    names.reserve(view.tensors.len);
    for (std::size_t i = 0; i < view.tensors.len; ++i) {
        names.emplace(view.tensors.ptr[i].id, view.tensors.ptr[i].name);
    }

    std::vector<RustQuantAttachment> out;
    out.reserve(view.attachments.len);
    for (std::size_t i = 0; i < view.attachments.len; ++i) {
        const auto& attachment = view.attachments.ptr[i];
        const auto tensor = names.find(attachment.tensor_id);
        const auto scale = names.find(attachment.scale_tensor_id);
        if (tensor == names.end() || scale == names.end()) {
            // Both ids index the same table the loader built this from, so a
            // miss is a marshalling fault rather than a model that lacks scales.
            throw std::runtime_error(
                "engine: Rust loader attached a quant scale to a tensor id the "
                "plan does not declare");
        }
        out.push_back({
            .tensor_name = pie_loader::bytes_to_string(tensor->second),
            .scale_tensor_name = pie_loader::bytes_to_string(scale->second),
            .granularity = attachment.granularity,
            .group_size = static_cast<int>(attachment.group_size),
            .channel_axis = static_cast<int>(attachment.channel_axis),
            .scale_form = attachment.scale_form,
        });
    }
    return out;
}

/// Compile the plan for this device and derive everything the load path needs
/// from it.
///
/// The checks this used to run after the fact — that the plan is for CUDA, that
/// its compiler version matches, that its tile-map transforms are ones we
/// implement — are gone. They re-derived facts the driver itself now states in
/// the request, and the loader refuses a target it cannot satisfy
/// (`loader/architecture.md` §9).
/// Compile the plan for this device from a contract this driver authored.
///
/// Three inputs and no fourth. `checkpoint` is what the files contain,
/// `contract` is what this driver will bind, `target` is what the GPU can do —
/// and the driver states all three, which is what makes the loader a compiler
/// rather than a registry of the models someone taught it (§12 row 12).
///
/// `checkpoint` is opened once and handed to the compile, so the tensor table
/// the contract was written against and the one the plan is built from are
/// provably the same parse; the previous arrangement passed a directory and
/// read it twice.
inline LoadPlanResult prepare_load_plan(
    const pie_loader::Checkpoint& checkpoint,
    const pie_loader::ModelContract& contract,
    const pie_loader::DeviceTarget& target) {
    const pie_loader::PieLoaderContractRequest request =
        pie_loader::build_contract_request(checkpoint, target, contract.view());
    LoadPlan plan = LoadPlan::compile(request);

    // Re-check the plan the loader just produced against the request it came
    // from. Compiling and verifying share no code, so this is a second opinion
    // rather than a restatement: it walks the plan's internal invariants,
    // stats each declared checkpoint file to catch one that changed between
    // compile and load, and compares the plan against the contract — which is
    // the only one of these the loader could not have made on its own, because
    // the contract is the one input it did not author.
    plan.verify(request);

    const auto view = plan.view();
    return {
        .plan = std::move(plan),
        .quant_attachments = resolve_quant_attachments(view),
        .planned_tensor_count = view.tensors.len,
        .cache_key = pie_loader::bytes_to_string(view.cache_key),
    };
}

}  // namespace pie_cuda_driver
