#pragma once

// The boot's author path: facts and policy in, plan out.
//
// `plan/model-in-rust.md` §6: the driver sends what only it can know — the
// facts it parsed and the policy it decided, a handful of scalars — and
// authoring happens on the loader's side, in `pie_model::contract`. The
// C++ authors this replaced were proven byte-equal by the §8-3
// differential (17 synthetic cases, ten real checkpoints, a dual boot on
// real hardware) and then harvested; `model/facts.hpp` is what remains of
// their vocabulary.
//
// What the request does NOT carry mirrors what the CUDA authors never
// read: tied-embeddings and the MLX quantization facts are Metal
// vocabulary and keep their defaults here.

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pie_loader.h"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

#include "loader/load_plan.hpp"
#include "loader/loader_config.hpp"
#include "loader/rust_quant_attachment.hpp"
#include "model/facts.hpp"

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

using pie_loader::PieLoaderFamilyKnobs;
using pie_loader::PieLoaderModelFactsView;
using pie_loader::PieLoaderModelRequest;

/// The per-family switches, read from the same environment the C++ authors
/// read them from, with the same defaults. Reading them HERE and sending
/// values keeps the Rust author environment-blind: equal requests author
/// equal contracts.
inline PieLoaderFamilyKnobs cuda_family_knobs() {
    const auto on_by_default = [](const char* name) {
        const char* v = std::getenv(name);
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    };
    const auto off_by_default = [](const char* name) {
        const char* v = std::getenv(name);
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    };
    return PieLoaderFamilyKnobs{
        .glm5_moe_gate_up_swapped = on_by_default("PIE_GLM5_MOE_FLASHINFER"),
        .qwen35_fused_gdn_projection = off_by_default("PIE_QWEN35_FUSED_GDN_PROJ"),
        .qwen35_mtp_int8_lm_head = off_by_default("PIE_QWEN35_MTP_INT8_LM_HEAD"),
        .qwen35_moe_gate_up_swapped = on_by_default("PIE_QWEN35_MOE_FLASHINFER"),
        .qwen35_fused_shared_scalar_gate =
            off_by_default("PIE_QWEN35_FUSED_SHARED_SCALAR_GATE"),
        .kimi_k3_moe_gate_up_swapped = off_by_default("PIE_KIMI_K3_MOE_FLASHINFER"),
        .kimi_moe_gate_up_swapped = off_by_default("PIE_KIMI_MOE_FLASHINFER"),
        .nemotron_tp_mamba_sharding =
            !loader_config::env_truthy("PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD"),
    };
}

/// `RuntimeQuant`'s wire value for a resolved request string.
///
/// The resolution itself (`fp8` without native FP8 is no request) happened
/// in `resolve_runtime_quant` before this is called, exactly as on the C++
/// author path.
inline std::uint32_t runtime_quant_wire(std::string_view resolved) {
    if (resolved == "fp8") return 1;
    if (resolved == "int8") return 2;
    if (resolved == "fp4" || resolved == "mxfp4") return 3;
    return 0;
}

/// Compile the plan through the Rust author. The drop-in counterpart of
/// `prepare_load_plan`, with the resolved MXFP4 policy handed back the way
/// `ContractBuilder::mxfp4_moe()` hands it back on the C++ path.
inline LoadPlanResult prepare_load_plan_rust_author(
    const pie_loader::Checkpoint& checkpoint,
    const model::ModelFacts& facts,
    const pie_loader::DeviceTarget& target,
    std::string_view resolved_runtime_quant,
    model::Mxfp4MoeRequest mxfp4_moe_request,
    model::Component component,
    bool stream_routed_experts,
    model::Mxfp4MoePolicy* out_policy) {
    const PieLoaderModelRequest request{
        .checkpoint = checkpoint.get(),
        .target = pie_loader::target_spec(target),
        .facts =
            PieLoaderModelFactsView{
                .model_type = pie_loader::borrow(facts.model_type),
                .quant_method = pie_loader::borrow(facts.quant_method),
                .num_hidden_layers = facts.num_hidden_layers,
                .num_experts = facts.num_experts,
                .head_dim = facts.head_dim,
                .mamba_groups = facts.mamba_groups,
                .tied_embeddings = true,
                .mlx_quant_bits = 0,
                .mlx_quant_group_size = 0,
                .num_kv_shared_layers = 0,
            },
        .projections = 0,  // Fused: the CUDA lowering.
        .naming = 0,       // HF names, which is what this bind path reads.
        .runtime_quant = runtime_quant_wire(resolved_runtime_quant),
        .moe_request = static_cast<std::uint32_t>(mxfp4_moe_request),
        .component = static_cast<std::uint32_t>(component),
        .stream_routed_experts = stream_routed_experts,
        .knobs = cuda_family_knobs(),
    };

    std::uint32_t resolved_moe = 0;
    LoadPlan plan = LoadPlan::compile_model(request, &resolved_moe);
    if (out_policy != nullptr) {
        *out_policy = static_cast<model::Mxfp4MoePolicy>(resolved_moe);
    }

    // The same second opinion the default path gets from `plan.verify`: the
    // verifier re-authors from this request on the far side and holds the
    // plan to it, marshalling and author determinism both in scope.
    plan.verify_model(request);

    const auto view = plan.view();
    return {
        .plan = std::move(plan),
        .quant_attachments = resolve_quant_attachments(view),
        .planned_tensor_count = view.tensors.len,
        .cache_key = pie_loader::bytes_to_string(view.cache_key),
    };
}

}  // namespace pie_cuda_driver
