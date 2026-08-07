#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "model/facts.hpp"
#include "pie_loader.h"
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

/// Did the contract tie the embedding and the head, or ship two tensors?
///
/// Decided ONCE, by the contract, and read back rather than decided a second
/// time. The rule is that a shipped `lm_head` beats whatever the config says,
/// and the config can be wrong in both directions: Qwen3.5-35B-A3B is a
/// multimodal wrapper spelling `tie_word_embeddings` at the TOP level, outside
/// the `text_config` its family parses, so the facts default to tied; Qwen3-
/// 0.6B says `tie_word_embeddings: true` and then ships an `lm_head.weight`
/// anyway. Either way the contract staged `embed_tokens` and `lm_head` while
/// the DAG asked for `shared_embedding`, and the load stopped on "unstaged
/// weight shared_embedding.weight" — two opinions about one fact.
///
/// The plan's own tensor list is the only opinion that cannot be wrong in a
/// way the binding survives, so it is the one every family follows.
inline bool plan_ties_embeddings(const LoadPlan& plan) {
    const auto view = plan.view();
    for (std::size_t i = 0; i < view.tensors.len; ++i) {
        if (pie_loader::bytes_to_string(view.tensors.ptr[i].name) == "shared_embedding.weight") {
            return true;
        }
    }
    return false;
}

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

/// Build a `pie.model/1` document from facts assembled by hand.
///
/// **Test and tool scaffolding.** A serving boot forwards the descriptor it
/// was handed (`SetupConfig::descriptor_json`); this exists for the probes and
/// numerics tests that point at a checkpoint directory with no descriptor
/// beside it and state the two or three facts their family needs.
///
/// It writes the schema's own field names, which is the only reason it is
/// tolerable to spell a descriptor here at all: a name that drifts from
/// `pie-model-config`'s schema stops being read rather than being read wrong,
/// because `ModelFacts::from_descriptor` takes what it recognizes and leaves
/// the rest at its default.
inline std::string descriptor_for_testing(std::string_view model_type,
                                          const model::ContractFacts& facts = {}) {
    nlohmann::json j{
        {"version", "pie.model/1"},
        {"model_type", std::string(model_type)},
        {"num_hidden_layers", facts.num_hidden_layers},
        {"num_kv_shared_layers", facts.num_kv_shared_layers},
        {"tie_word_embeddings", facts.tied_embeddings},
        {"quant_bits", facts.quant_bits},
        {"quant_group_size", facts.quant_group_size},
    };
    return j.dump();
}

/// Compile the plan through the Rust author: the descriptor and policy in,
/// plan out.
///
/// `plan/model-in-rust.md` §6 — the driver sends what only it can know (the
/// compiled model descriptor it was handed, and that this device wants MLX
/// names with in-place projections) and authoring happens on the loader's
/// side, in the same `pie_model::contract` registry the CUDA boot goes
/// through. `model_type` reaches the author registry as a field of the
/// descriptor; an unknown one comes back as a diagnostic naming it rather
/// than a plan.
///
/// `descriptor_json` used to be a `ContractFacts` distilled here — five
/// fields, three of them sent as zero because the Metal authors never read
/// them. The document travels now, so which fields matter is decided beside
/// the authors that read them rather than at each call site.
///
/// What the request does NOT carry is a policy this driver has no operator
/// knob for: no MXFP4 MoE lowering choice, no component split, no expert
/// streaming, and none of the CUDA per-family environment knobs — zeros and
/// defaults, stated here rather than defaulted there, so equal requests author
/// equal contracts.
///
/// `runtime_quant` is zero for a reason of its own. The MLX authors DO read it
/// now (`RuntimeQuant::Int4` encodes a float weight to affine-U4), but this
/// driver binds what the checkpoint holds: a requantization is a decision about
/// an artifact, made once by `pie model build --quant int4` and written
/// down, not one to re-run over every weight on each boot.
inline LoadPlan compile_load_plan(
    std::string_view snapshot_dir,
    const pie_loader::DeviceTarget& target,
    std::string_view descriptor_json) {
    std::string open_error;
    pie_loader::Checkpoint checkpoint =
        pie_loader::Checkpoint::open(snapshot_dir, &open_error);
    if (!checkpoint) {
        throw std::runtime_error("load plan: " + open_error);
    }

    const pie_loader::PieLoaderModelRequest request{
        .checkpoint = checkpoint.get(),
        .target = pie_loader::target_spec(target),
        .descriptor = pie_loader::borrow(descriptor_json),
        .projections = 1,  // InPlace: the MLX lowering.
        .naming = 1,       // MLX names, which is what this bind path reads.
        .runtime_quant = 0,
        .moe_request = 0,
        .component = 0,
        .stream_routed_experts = false,
        .knobs = pie_loader::PieLoaderFamilyKnobs{},
    };

    LoadPlan plan = LoadPlan::compile_model(request, nullptr);
    // A second opinion, not a restatement: the verifier re-authors from this
    // request on the far side and holds the plan to it — marshalling and
    // author determinism both in scope — and since §6 it also stats each
    // file the plan declares.
    plan.verify_model(request);
    return plan;
}

}  // namespace pie::metal
