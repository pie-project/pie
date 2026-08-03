#pragma once

/// What the Metal driver binds for the Gemma 4 family.
///
/// Same shape of object as `qwen3_5_contract.hpp` — a contract is the *program*
/// half of `plan = compile(source_facts, program, target)` — and the mechanics
/// are shared through `../contract_detail.hpp`. What is family-specific, and all
/// that lives here, is the tensor NAMES.
///
/// Three things are worth knowing about this checkpoint before reading the map:
///
///  * **It is multimodal.** `google/gemma-4-E2B-it` ships 2011 tensors, of which
///    1411 are the audio and vision towers. This driver decodes text, so they
///    are skipped — the same job `qwen3_5_contract` does for `model.visual.`.
///
///  * **KV is shared over the tail of the stack.** `mlx_lm/models/gemma4_text.py`
///    computes `has_kv = layer_idx < num_hidden_layers - num_kv_shared_layers`,
///    so on E2B layers 0-14 own their KV and 15-34 attend what an earlier layer
///    already wrote. The checkpoint still *ships* `k_proj`/`v_proj`/`k_norm` for
///    all 35 layers; the shared ones are dead weight. They are not declared,
///    because a contract that declares a tensor nothing binds is a contract that
///    lies about what the driver needs.
///
///  * **Per-Layer Embeddings.** Gemma 4's "E" variants carry a second embedding
///    table (`embed_tokens_per_layer`) plus a projection and gate per layer.
///    Those are real bound tensors, not an artefact, and they get runtime names
///    here like everything else.
///
/// Weights arrive as MLX affine-U4 triplets (`mlx_lm convert -q --q-bits 4
/// --q-group-size 64 --q-mode affine`), exactly as for qwen3.5, so the tuned
/// `affine_qmv`/`affine_qmm` kernels bind them unchanged. The bring-up harness
/// in `tools/rawmetal/gemma4_abi.hpp` assumed dense bf16 GEMVs because it
/// predates that decision; the quantized path is both faster and already built.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pie_loader.h"
#include "pie_loader/model_contract.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

#include "../contract_detail.hpp"

namespace pie::metal::model::gemma4 {

using pie::metal::model::Checkpoint;
using pie::metal::model::ModelContract;
using pie::metal::model::SourceTensor;
using pie::metal::model::PieLoaderDType;

namespace contract_detail {

using namespace pie::metal::model::contract_detail;

/// Layers whose `k_proj`/`v_proj`/`k_norm` this driver does not bind.
///
/// Passed in rather than derived, because the split is a property of the config
/// (`num_hidden_layers - num_kv_shared_layers`) and the contract only sees
/// tensors.
struct KvShare {
    int first_shared_layer = -1;  // -1: every layer owns its KV
    bool shares(int layer) const {
        return first_shared_layer >= 0 && layer >= first_shared_layer;
    }
};

/// The runtime name for a checkpoint tensor, or none to skip it.
///
/// Every name is either mapped or explicitly skipped; an unrecognised one is an
/// error rather than a pass-through, because a tensor declared under its
/// checkpoint name would never be found by the binder.
inline std::optional<std::string> runtime_name(std::string_view raw_name,
                                               const KvShare& kv_share) {
    // The towers. Text decode binds none of it.
    for (std::string_view skip : {"audio_tower.", "vision_tower.", "embed_audio.",
                                  "embed_vision."}) {
        if (has_wrapper_member(raw_name, skip)) {
            return std::nullopt;
        }
    }

    // Gemma 4 ships tied embeddings: there is an `embed_tokens` and no
    // `lm_head`. `EmbedGather` and the logits matvec both bind
    // `shared_embedding.*`, which is the one slot they share. A checkpoint
    // carrying both would declare it twice and be rejected as a duplicate,
    // which is the truthful outcome.
    if (raw_name.rfind("lm_head.", 0) == 0) {
        return "shared_embedding." +
               std::string(raw_name.substr(std::string_view("lm_head.").size()));
    }

    const std::optional<std::string_view> decoder = decoder_member(raw_name);
    if (!decoder) {
        contract_detail::fail("Metal Gemma4 schema has no declared mapping or skip for '" +
                              std::string(raw_name) + "'");
    }
    const std::string_view rest = *decoder;

    constexpr std::string_view kEmbed = "embed_tokens.";
    if (rest.rfind(kEmbed, 0) == 0) {
        return "shared_embedding." + std::string(rest.substr(kEmbed.size()));
    }
    // The PLE table and its projection are layer-less and keep their own names.
    for (std::string_view direct : {"embed_tokens_per_layer.", "per_layer_model_projection.",
                                    "per_layer_projection_norm."}) {
        if (rest.rfind(direct, 0) == 0) {
            return std::string(rest);
        }
    }
    if (rest == "norm.weight") {
        return std::string("final_norm.weight");
    }

    constexpr std::string_view kLayers = "layers.";
    if (rest.rfind(kLayers, 0) != 0) {
        contract_detail::fail("Metal Gemma4 schema has no declared mapping or skip for '" +
                              std::string(raw_name) + "'");
    }
    const std::string_view tail = rest.substr(kLayers.size());
    const std::size_t dot = tail.find('.');
    if (dot == std::string_view::npos) {
        contract_detail::fail("Metal Gemma4 layer tensor '" + std::string(raw_name) +
                              "' is malformed");
    }
    const std::string_view layer = tail.substr(0, dot);
    if (layer.empty() || !std::all_of(layer.begin(), layer.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        })) {
        contract_detail::fail("Metal Gemma4 layer tensor '" + std::string(raw_name) +
                              "' has an invalid layer index");
    }
    const std::string_view member = tail.substr(dot + 1);

    // A KV-shared layer attends the KV an earlier layer wrote, so its own k/v
    // projections and k-norm are never bound. mlx-lm does not even construct
    // them (`gemma4_text.py`: `has_kv = layer_idx < n_layers - n_kv_shared`).
    if (kv_share.shares(std::stoi(std::string(layer)))) {
        for (std::string_view unused : {"self_attn.k_proj.", "self_attn.v_proj.",
                                        "self_attn.k_norm."}) {
            if (member.rfind(unused, 0) == 0) {
                return std::nullopt;
            }
        }
    }
    return "layers." + std::string(layer) + "." + std::string(member);
}

}  // namespace contract_detail

/// The `model_type` strings whose text decoder this schema describes.
inline bool is_supported_model_type(std::string_view model_type) {
    for (std::string_view name : {"gemma4", "gemma4_text"}) {
        if (model_type == name) {
            return true;
        }
    }
    return false;
}

/// Author the contract this driver will bind, against this checkpoint.
///
/// `first_kv_shared_layer` comes from the config
/// (`num_hidden_layers - num_kv_shared_layers`); pass -1 when every layer owns
/// its KV.
inline void author_model_contract(const Checkpoint& checkpoint, std::string_view model_type,
                                  const pie_loader::DeviceTarget& target, ModelContract& out,
                                  int first_kv_shared_layer, int quant_bits,
                                  int quant_group_size) {
    using namespace pie::metal::model::contract_detail;
    using contract_detail::KvShare;
    using contract_detail::runtime_name;

    if (!is_supported_model_type(model_type)) {
        fail("Metal Gemma4 storage schema does not support model_type='" +
             std::string(model_type) + "'");
    }
    out.align(std::max<std::uint32_t>(1, target.preferred_alignment));

    const KvShare kv_share{first_kv_shared_layer};
    const std::vector<SourceTensor> tensors = checkpoint.tensors();
    const auto find = [&tensors](const std::string& name) -> const SourceTensor* {
        for (const SourceTensor& raw : tensors) {
            if (raw.name == name) {
                return &raw;
            }
        }
        return nullptr;
    };

    std::size_t declared = 0;
    for (const SourceTensor& raw : tensors) {
        std::optional<std::string> output = runtime_name(raw.name, kv_share);
        if (!output.has_value()) {
            continue;
        }
        if (ends_with(raw.name, ".weight") && is_raw(raw.encoding, PieLoaderDType::U32)) {
            const std::string base = std::string(
                raw.name.substr(0, raw.name.size() - std::string_view(".weight").size()));
            const SourceTensor* scales = find(base + ".scales");
            const SourceTensor* biases = find(base + ".biases");
            if (scales == nullptr || biases == nullptr) {
                fail("Metal affine-U4 weight '" + std::string(raw.name) +
                     "' is missing scales or biases");
            }
            push_mlx_affine_declared(out, raw, *scales, *biases, quant_bits, quant_group_size,
                                     std::move(*output));
        } else {
            push_direct(out, raw, std::move(*output));
        }
        ++declared;
    }
    if (declared == 0) {
        fail("Metal Gemma4 schema found no text-decoder tensors");
    }
}

}  // namespace pie::metal::model::gemma4
