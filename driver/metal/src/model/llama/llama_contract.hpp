#pragma once

/// What the Metal driver binds for the llama-shaped families.
///
/// Same shape of object as `gemma4_contract.hpp` -- a contract is the *program*
/// half of `plan = compile(source_facts, program, target)`, and the mechanics
/// are shared through `../contract_detail.hpp`. What is family-specific, and
/// all that lives here, is the tensor NAMES.
///
/// There is remarkably little to say, and that is the point: the driver's
/// shared weight map (`loader/heap_bind.cpp`) already spells the HF llama names
/// verbatim -- `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
/// `input_layernorm`, `post_attention_layernorm`, `self_attn.{q,k}_norm` --
/// because qwen3.5 and gemma4 use them too. So this maps `model.layers.N.X` to
/// `layers.N.X` and is otherwise almost the identity.
///
/// Two things are worth knowing before reading the map:
///
///  * **The head may or may not be tied.** Llama 3 ships `lm_head.weight`
///    alongside `model.embed_tokens.weight`; Qwen3's smaller sizes ship only
///    the latter. A tied checkpoint maps both onto `shared_embedding`, as
///    gemma4 does. An untied one keeps them apart, because mapping them
///    together would declare `shared_embedding` twice and be rejected as a
///    duplicate -- correctly, since they are different tensors.
///  * **A routed FFN has its experts on axis 0.** `mlx_lm` stacks
///    `mlp.experts.N.gate_proj` into one `mlp.gate_proj` with the expert
///    leading, which is exactly the layout `affine_qmv_routed` indexes. A
///    checkpoint that ships them unstacked is refused rather than gathered at
///    load time.
///
/// Weights arrive as MLX affine-U4 triplets (`mlx_lm convert -q --q-bits 4
/// --q-group-size 64 --q-mode affine`), so the tuned `affine_qmv`/`affine_qmm`
/// kernels bind them unchanged.

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

namespace pie::metal::model::llama {

using pie::metal::model::Checkpoint;
using pie::metal::model::ModelContract;
using pie::metal::model::PieLoaderDType;
using pie::metal::model::SourceTensor;

namespace contract_detail {

using namespace pie::metal::model::contract_detail;

/// Whether this checkpoint ships its own `lm_head`.
///
/// Passed in rather than guessed, because it decides whether `lm_head.` is a
/// second name for the embedding table or a tensor of its own, and getting it
/// wrong is either a duplicate declaration or a head bound to the wrong matrix.
struct HeadTying {
    bool tied = true;
};

/// The runtime name for a checkpoint tensor, or none to skip it.
///
/// Every name is either mapped or explicitly skipped; an unrecognised one is an
/// error rather than a pass-through, because a tensor declared under its
/// checkpoint name would never be found by the binder.
inline std::optional<std::string> runtime_name(std::string_view raw_name,
                                               const HeadTying& head) {
    // A multimodal release wraps the decoder. Text decode binds none of the
    // towers, the same job `gemma4_contract` does for `model.vision_tower.`.
    for (std::string_view skip : {"model.visual.", "model.vision_tower.",
                                  "model.audio_tower.", "visual."}) {
        if (raw_name.rfind(skip, 0) == 0) return std::nullopt;
    }
    // Rotary inverse frequencies are recomputed on the GPU from `rope_theta`,
    // so a checkpoint that persists them is shipping a derived tensor.
    if (raw_name.find("rotary_emb.inv_freq") != std::string_view::npos) {
        return std::nullopt;
    }

    constexpr std::string_view kLmHead = "lm_head.";
    if (raw_name.rfind(kLmHead, 0) == 0) {
        const std::string tail(raw_name.substr(kLmHead.size()));
        return head.tied ? "shared_embedding." + tail : "lm_head." + tail;
    }

    // Some releases nest the decoder one level deeper. Accept either.
    std::string_view rest = raw_name;
    for (std::string_view root : {"model.language_model.", "model."}) {
        if (rest.rfind(root, 0) == 0) {
            rest = rest.substr(root.size());
            break;
        }
    }
    if (rest.size() == raw_name.size()) {
        fail("Metal llama schema has no declared mapping or skip for '" +
             std::string(raw_name) + "'");
    }

    constexpr std::string_view kEmbed = "embed_tokens.";
    if (rest.rfind(kEmbed, 0) == 0) {
        const std::string tail(rest.substr(kEmbed.size()));
        return head.tied ? "shared_embedding." + tail : "embed_tokens." + tail;
    }
    if (rest == "norm.weight") return std::string("final_norm.weight");

    constexpr std::string_view kLayers = "layers.";
    if (rest.rfind(kLayers, 0) != 0) {
        fail("Metal llama schema has no declared mapping or skip for '" +
             std::string(raw_name) + "'");
    }
    const std::string_view tail = rest.substr(kLayers.size());
    const std::size_t dot = tail.find('.');
    if (dot == std::string_view::npos) {
        fail("Metal llama layer tensor '" + std::string(raw_name) + "' is malformed");
    }
    const std::string_view layer = tail.substr(0, dot);
    if (layer.empty() || !std::all_of(layer.begin(), layer.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        })) {
        fail("Metal llama layer tensor '" + std::string(raw_name) +
             "' has an invalid layer index");
    }
    const std::string_view member = tail.substr(dot + 1);

    // The mixture's naming is the same rule in every routed family, so it is
    // asked of one place rather than restated here: stacked banks under either
    // spelling, and a refusal for the unstacked and shared forms.
    if (auto renamed = routed_expert_member(raw_name, member, "llama")) {
        return "layers." + std::string(layer) + "." + *renamed;
    }

    return "layers." + std::string(layer) + "." + std::string(member);
}

}  // namespace contract_detail

/// The `model_type` strings whose decoder this schema describes.
inline bool is_supported_model_type(std::string_view model_type) {
    for (std::string_view name : {"llama", "llama3", "mistral", "qwen2", "qwen3",
                                  "qwen3_moe", "qwen2_moe"}) {
        if (model_type == name) return true;
    }
    return false;
}

/// Whether this `model_type`'s FFN is routed.
inline bool is_moe_model_type(std::string_view model_type) {
    return model_type == "qwen3_moe" || model_type == "qwen2_moe";
}

/// Author the contract this driver will bind, against this checkpoint.
///
/// `tied_embeddings` comes from the config (`tie_word_embeddings`, defaulting
/// to true when absent); pass false when the checkpoint ships its own
/// `lm_head`.
inline void author_model_contract(const Checkpoint& checkpoint, std::string_view model_type,
                                  const pie_loader::DeviceTarget& target, ModelContract& out,
                                  bool tied_embeddings, int quant_bits, int quant_group_size) {
    using namespace pie::metal::model::contract_detail;
    using contract_detail::HeadTying;
    using contract_detail::runtime_name;

    if (!is_supported_model_type(model_type)) {
        fail("Metal llama storage schema does not support model_type='" +
             std::string(model_type) + "'");
    }
    out.align(std::max<std::uint32_t>(1, target.preferred_alignment));

    const std::vector<SourceTensor> tensors = checkpoint.tensors();
    // `tie_word_embeddings` is what the config SAYS; a shipped `lm_head` is
    // what the checkpoint DOES. When they disagree the tensors win, because
    // mapping a real `lm_head` onto `shared_embedding` declares that name twice
    // and the load fails with a duplicate rather than with the disagreement.
    const bool has_lm_head =
        std::any_of(tensors.begin(), tensors.end(), [](const SourceTensor& raw) {
            return raw.name.rfind("lm_head.", 0) == 0;
        });
    const HeadTying head{tied_embeddings && !has_lm_head};

    const auto find = [&tensors](const std::string& name) -> const SourceTensor* {
        for (const SourceTensor& raw : tensors) {
            if (raw.name == name) return &raw;
        }
        return nullptr;
    };

    std::size_t declared = 0;
    for (const SourceTensor& raw : tensors) {
        std::optional<std::string> output = runtime_name(raw.name, head);
        if (!output.has_value()) continue;
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
    if (declared == 0) fail("Metal llama schema found no decoder tensors");
}

}  // namespace pie::metal::model::llama
