#pragma once

/// What the Metal driver binds for the Qwen3.5 / GDN-hybrid family.
///
/// A contract is the *program* half of `plan = compile(source_facts, program,
/// target)`: it names every tensor this driver wants and gives, for each, an
/// expression over the checkpoint's tensors that produces it. The loader
/// type-checks the expression against what the files actually contain and
/// lowers it to storage instructions; it does not decide *what* to build, which
/// is why a family it has never heard of loads exactly as well as one it has
/// (`loader/architecture.md` §12 row 12).
///
/// This is Metal's, and only Metal's. It used to share a header with CUDA on
/// the theory that "Qwen3-MoE binds q/k/v separately" is a fact about the model
/// rather than about a GPU. That theory does not survive contact with what the
/// two drivers actually declare: CUDA fuses projections for its GEMMs, shards
/// across ranks, and re-quantizes at load time, and this driver does none of
/// those — it renames every tensor for MLX's binder and binds what the file
/// holds. Sharing meant Metal carried CUDA's vocabulary (a `Component`, an
/// MXFP4 policy, a runtime-quant request) through three call layers to an
/// author that read none of them.
///
/// The whole family is one storage schema, so the schema is the contract.

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

namespace pie::metal::model::qwen3_5 {

using pie::metal::model::Checkpoint;
using pie::metal::model::ModelContract;
using pie::metal::model::SourceTensor;
using pie::metal::model::PieLoaderDType;

namespace contract_detail {

using namespace pie::metal::model::contract_detail;

/// The runtime name for a checkpoint tensor, or none to skip it.
///
/// Every name is either mapped or explicitly skipped; an unrecognised one is an
/// error rather than a pass-through, because a tensor this driver silently
/// declared under its checkpoint name would never be found by the binder.
/// Whether this checkpoint's `lm_head` IS its embedding table.
///
/// A struct rather than a bare bool so a call site cannot pass it positionally
/// to the wrong parameter -- the same reason the llama schema has one.
struct HeadTying {
    bool tied = true;
};

inline std::optional<std::string> runtime_name(std::string_view raw_name,
                                               const HeadTying& head_tying = {}) {
    // Not the text decoder. The vision tower has the same two spellings as the
    // decoder does; `mtp.` is the multi-token-prediction head, which this
    // driver does not run. Skipping is a declaration too -- an unlisted tensor
    // is an error below, not a pass-through.
    for (std::string_view skip : {"visual.", "vision_tower.", "mtp."}) {
        if (has_wrapper_member(raw_name, skip)) {
            return std::nullopt;
        }
    }
    // The output projection. Spelled bare by the HF release, which puts it
    // outside `model`, and under the wrapper by the mlx repack -- the same
    // two-tools-one-tensor split as the decoder prefix. The dense members of
    // this family tie: 0.8B has no `lm_head` at all. Every ROUTED member is
    // untied and ships one, which is why this is not a detail.
    //
    // Untied it keeps its own name and `LmHeadUntied` binds it. Tied it lands
    // on `shared_embedding` beside the table it IS -- and if a checkpoint that
    // says it ties nonetheless ships the tensor, both would land there and the
    // load would fail as a duplicate. That is why the caller decides tying from
    // the TENSORS and not from the config alone.
    for (std::string_view head : {"lm_head.", "language_model.lm_head."}) {
        if (raw_name.rfind(head, 0) == 0) {
            const std::string tail(raw_name.substr(head.size()));
            return head_tying.tied ? "shared_embedding." + tail : "lm_head." + tail;
        }
    }
    const std::optional<std::string_view> decoder = decoder_member(raw_name);
    if (!decoder) {
        fail("Metal Qwen3.5 schema has no declared mapping or skip for '" + std::string(raw_name) +
             "'");
    }
    // The tied half of the same slot. `EmbedGather` and `QmvLmHead` both bind
    // `shared_embedding.*`, and this family ships tied — so its checkpoint has
    // an `embed_tokens` and no `lm_head` at all. Without this the driver could
    // not name the embedding table of the one model it targets.
    //
    // A checkpoint carrying both would declare `shared_embedding` twice and be
    // rejected as a duplicate, which is the truthful outcome: there is one
    // shared slot here, not two.
    constexpr std::string_view kEmbed = "embed_tokens.";
    if (decoder->rfind(kEmbed, 0) == 0) {
        const std::string tail(decoder->substr(kEmbed.size()));
        return head_tying.tied ? "shared_embedding." + tail : "embed_tokens." + tail;
    }
    if (*decoder == "norm.weight") {
        return std::string("final_norm.weight");
    }
    constexpr std::string_view kLayers = "layers.";
    if (decoder->rfind(kLayers, 0) != 0) {
        fail("Metal Qwen3.5 schema has no declared mapping or skip for '" + std::string(raw_name) +
             "'");
    }
    const std::string_view rest = decoder->substr(kLayers.size());
    const std::size_t dot = rest.find('.');
    if (dot == std::string_view::npos) {
        fail("Metal Qwen3.5 layer tensor '" + std::string(raw_name) + "' is malformed");
    }
    const std::string_view layer = rest.substr(0, dot);
    if (layer.empty() || !std::all_of(layer.begin(), layer.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        })) {
        fail("Metal Qwen3.5 layer tensor '" + std::string(raw_name) +
             "' has an invalid layer index");
    }
    const std::string_view member = rest.substr(dot + 1);
    // The mixture's naming is the same rule in every routed family, so it is
    // asked of one place rather than restated here. Until this commit the
    // pass-through below answered for it, which meant a Qwen3-Next MoE
    // checkpoint's experts were declared under names no dispatch reads: the
    // driver loaded a mixture and ran nothing with it.
    if (auto renamed = routed_expert_member(raw_name, member, "Qwen3.5",
                                            /*has_shared_expert=*/true)) {
        return "layers." + std::string(layer) + "." + *renamed;
    }
    return "layers." + std::string(layer) + "." + std::string(member);
}

}  // namespace contract_detail

/// The `model_type` strings whose text decoder this schema describes.
///
/// One list, and it is this one: there is no second table keyed by the same
/// strings anywhere in this driver.
inline bool is_supported_model_type(std::string_view model_type) {
    for (std::string_view name : {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
                                  "qwen3_next", "qwen3_next_text", "qwen3_6"}) {
        if (model_type == name) {
            return true;
        }
    }
    return false;
}

/// Author the contract this driver will bind, against this checkpoint.
///
/// Throws `std::runtime_error` with a message naming the tensor when the
/// checkpoint does not match what the schema expects. The contract is left in
/// `out`, which must outlive the compile call that consumes its `view()`.
inline void author_model_contract(const Checkpoint& checkpoint, std::string_view model_type,
                                  const pie_loader::DeviceTarget& target, ModelContract& out,
                                  bool tied_embeddings, int quant_bits, int quant_group_size) {
    using namespace pie::metal::model::contract_detail;
    using namespace contract_detail;
    if (!is_supported_model_type(model_type)) {
        fail("Metal storage schema does not support model_type='" + std::string(model_type) + "'");
    }
    out.align(std::max<std::uint32_t>(1, target.preferred_alignment));

    const std::vector<SourceTensor> tensors = checkpoint.tensors();
    const auto find = [&tensors](const std::string& name) -> const SourceTensor* {
        for (const SourceTensor& raw : tensors) {
            if (raw.name == name) {
                return &raw;
            }
        }
        return nullptr;
    };

    // `tie_word_embeddings` is what the config SAYS; a shipped `lm_head` is
    // what the checkpoint DOES. When they disagree the tensors win, because
    // mapping a real `lm_head` onto `shared_embedding` beside the table it is
    // supposed to BE declares that name twice and fails the load with a
    // duplicate rather than with the disagreement. Both spellings are checked,
    // for the same reason both are mapped.
    const bool has_lm_head =
        std::any_of(tensors.begin(), tensors.end(), [](const SourceTensor& raw) {
            return raw.name.rfind("lm_head.", 0) == 0 ||
                   raw.name.rfind("language_model.lm_head.", 0) == 0;
        });
    const HeadTying head{tied_embeddings && !has_lm_head};

    std::size_t declared = 0;
    for (const SourceTensor& raw : tensors) {
        std::optional<std::string> output = runtime_name(raw.name, head);
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
        fail("Metal Qwen3.5 schema found no text-decoder tensors");
    }
}

}  // namespace pie::metal::model::qwen3_5
