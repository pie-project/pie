#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pie_loader/tensor_contract.hpp"

#include "loader/rust_quant_attachment.hpp"
#include "loader/load_plan.hpp"

#include "model/config.hpp"

namespace pie_cuda_driver {

struct LoadPlanResult {
    LoadPlan plan;
    std::vector<RustQuantAttachment> quant_attachments;
    std::size_t source_tensor_count = 0;
    /// Demanded tensors the plan delivers, and demanded tensors in total.
    ///
    /// These used to be two names for `view.tensors.len`, which made the check
    /// downstream `if (x != x)`. They are now counted against the contract the
    /// *driver* declared (§12 row 7b-4b), so they differ exactly when the loader
    /// did not build something this model asked for. A family that has not
    /// adopted `TensorContract` yet declares nothing and both are zero.
    std::size_t covered_contract_count = 0;
    std::size_t runtime_tensor_count = 0;
    /// Tensors the plan produces, whether or not anything demanded them.
    std::size_t planned_tensor_count = 0;
    std::string cache_key;
};

/// Resolve the plan's quant attachments into the name-keyed form the executor's
/// `WeightStore` is addressed by.
///
/// Only a translation. The pairing itself used to be *inferred* here by matching
/// name suffixes over the plan's tensor list — guessing at something the loader
/// already knew, since it is what named the scale tensor in the first place. The
/// plan now states it (`loader/src/load_plan.rs`, `derive_quant_attachments`).
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
inline LoadPlanResult prepare_load_plan(
    const HfConfig& hf,
    std::string_view snapshot_dir,
    const pie_loader::DeviceTarget& target,
    std::string_view runtime_quant,
    pie_loader::PieLoaderMxfp4MoeRequest mxfp4_moe,
    pie_loader::PieLoaderComponent component,
    const pie_loader::TensorContract& contract = {}) {
    // The driver already parsed `config.json` into `hf`; the loader is told
    // what it needs instead of opening the file again (§10.4).
    const pie_loader::ModelFacts model{
        .model_type = hf.model_type,
        .quant_method = hf.quant_method,
        .num_hidden_layers = static_cast<std::uint32_t>(std::max(0, hf.num_hidden_layers)),
        .num_experts = static_cast<std::uint32_t>(std::max(0, hf.num_experts)),
        .num_experts_per_tok = static_cast<std::uint32_t>(std::max(0, hf.num_experts_per_tok)),
    };
    pie_loader::PieLoaderRequest request = pie_loader::build_request(
        snapshot_dir, target, model, runtime_quant, mxfp4_moe, component);
    request.demands = contract.slice();
    LoadPlan plan = LoadPlan::compile(request);

    // Re-check the plan the loader just produced against the request it came
    // from. Compiling and verifying share no code, so this is a second opinion
    // rather than a restatement: it walks the plan's internal invariants and —
    // since §6 gave the plan a file table — stats each declared checkpoint file
    // to catch one that changed between compile and load. When the caller
    // declared a contract, it also checks the plan against that, which is the
    // only one of these checks the loader could not have made on its own.
    plan.verify(request);

    const auto view = plan.view();
    return {
        .plan = std::move(plan),
        .quant_attachments = resolve_quant_attachments(view),
        .source_tensor_count = view.sources.len,
        // `verify` already rejected any shortfall, so reaching here means every
        // demand is met. Carrying the numbers rather than asserting keeps them
        // available for the diagnostic dump.
        .covered_contract_count = view.coverage.covered,
        .runtime_tensor_count = view.coverage.demanded,
        .planned_tensor_count = view.tensors.len,
        .cache_key = pie_loader::bytes_to_string(view.cache_key),
    };
}

}  // namespace pie_cuda_driver
