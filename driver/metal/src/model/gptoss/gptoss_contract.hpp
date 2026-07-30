#pragma once

/// What the Metal driver binds for the GPT-OSS family.
///
/// Same shape of object as `gemma4_contract.hpp` — a contract is the *program*
/// half of `plan = compile(source_facts, program, target)`, and the mechanics
/// are shared through `../contract_detail.hpp`. What is family-specific, and all
/// that lives here, is the tensor NAMES.
///
/// Four things are worth knowing before reading the map:
///
///  * **`.bias` and `.biases` are different tensors, on the same base.**
///    `q_proj.bias` is the Linear's additive bias — gpt-oss sets
///    `attention_bias: true`, so every attention projection has one. `.biases`
///    is the affine-U4 zero point that rides with `.scales`. They differ by one
///    character and mean nothing alike, so the quantized triplet is keyed off
///    `.weight` and looks its partners up by name.
///
///  * **Every layer is a sparse MoE.** 32 experts on the 20B, top-4 per token,
///    plus a router with its own bias. `mlx_lm convert` splits the checkpoint's
///    fused `gate_up_proj` into `gate_proj`/`up_proj` and stores each expert
///    stack as one 3-D tensor `[n_experts, out, in/8]`, so a layer's expert
///    weights are three tensors, not `3 * n_experts`.
///
///  * **Per-head attention sinks.** `self_attn.sinks` is one learned scalar per
///    query head that extends the softmax denominator. A real bound tensor with
///    no counterpart in either family already here.
///
///  * **The head is not tied.** Unlike qwen3.5 and gemma4, gpt-oss ships both
///    `model.embed_tokens` and `lm_head`, and both are quantized. They are
///    declared separately; folding them the way the tied families do would bind
///    the embedding as the head.
///
/// Weights arrive as MLX affine-U4 triplets (`mlx_lm convert -q --q-bits 4
/// --q-group-size 64 --q-mode affine`), as for the other two families. The
/// checkpoint openai ships is MXFP4, which no kernel here reads; converting is
/// what makes it loadable, and is the same step the other two took.

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

namespace pie::metal::model::gptoss {

using pie::metal::model::Checkpoint;
using pie::metal::model::ModelContract;
using pie::metal::model::SourceTensor;
using pie::metal::model::PieLoaderDType;

namespace contract_detail {

using namespace pie::metal::model::contract_detail;

/// The runtime name for a checkpoint tensor, or none to skip it.
///
/// Every name is either mapped or explicitly skipped; an unrecognised one is an
/// error rather than a pass-through, because a tensor declared under its
/// checkpoint name would never be found by the binder.
inline std::optional<std::string> runtime_name(std::string_view raw_name) {
    // The head is its own tensor here -- NOT the embedding under another name.
    if (raw_name.rfind("lm_head.", 0) == 0) {
        return std::string(raw_name);
    }

    constexpr std::string_view kModel = "model.";
    if (raw_name.rfind(kModel, 0) != 0) {
        contract_detail::fail("Metal GptOss schema has no declared mapping or skip for '" +
                              std::string(raw_name) + "'");
    }
    const std::string_view rest = raw_name.substr(kModel.size());

    if (rest.rfind("embed_tokens.", 0) == 0) {
        return std::string(rest);
    }
    if (rest == "norm.weight") {
        return std::string("final_norm.weight");
    }

    constexpr std::string_view kLayers = "layers.";
    if (rest.rfind(kLayers, 0) != 0) {
        contract_detail::fail("Metal GptOss schema has no declared mapping or skip for '" +
                              std::string(raw_name) + "'");
    }
    const std::string_view tail = rest.substr(kLayers.size());
    const std::size_t dot = tail.find('.');
    if (dot == std::string_view::npos) {
        contract_detail::fail("Metal GptOss layer tensor '" + std::string(raw_name) +
                              "' is malformed");
    }
    const std::string_view layer = tail.substr(0, dot);
    if (layer.empty() || !std::all_of(layer.begin(), layer.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        })) {
        contract_detail::fail("Metal GptOss layer tensor '" + std::string(raw_name) +
                              "' has an invalid layer index");
    }
    return "layers." + std::string(layer) + "." + std::string(tail.substr(dot + 1));
}

}  // namespace contract_detail

/// The `model_type` strings whose decoder this schema describes.
inline bool is_supported_model_type(std::string_view model_type) {
    return model_type == "gpt_oss";
}

/// Author the contract this driver will bind, against this checkpoint.
inline void author_model_contract(const Checkpoint& checkpoint, std::string_view model_type,
                                  const pie_loader::DeviceTarget& target, ModelContract& out) {
    using namespace pie::metal::model::contract_detail;
    using contract_detail::runtime_name;

    if (!is_supported_model_type(model_type)) {
        fail("Metal GptOss storage schema does not support model_type='" +
             std::string(model_type) + "'");
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

    std::size_t declared = 0;
    for (const SourceTensor& raw : tensors) {
        std::optional<std::string> output = runtime_name(raw.name);
        if (!output.has_value()) {
            continue;
        }
        // `.scales` and `.biases` are declared in their own right, under their
        // own runtime names, and fall through to `push_direct` below -- the
        // matvec binds all three slots, so all three have to be findable. The
        // `.weight` declaration below describes the PACKING; it does not consume
        // its partners.
        if (ends_with(raw.name, ".weight") && is_raw(raw.encoding, PieLoaderDType::U32)) {
            const std::string base = std::string(
                raw.name.substr(0, raw.name.size() - std::string_view(".weight").size()));
            const SourceTensor* scales = find(base + ".scales");
            const SourceTensor* biases = find(base + ".biases");
            // Three quantization modes can come out of `mlx_lm convert` on this
            // family, and this driver's storage vocabulary is one of them.
            // Refusals name the mode rather than the symptom: "missing biases"
            // is what MXFP4 looks like from here, and it is not what is wrong.
            if (scales == nullptr) {
                fail("Metal GptOss: '" + std::string(raw.name) +
                     "' is a packed weight with no scales, which no scheme here describes");
            }
            if (biases == nullptr) {
                fail("Metal GptOss: '" + std::string(raw.name) +
                     "' has scales but no zero points, which is MXFP4 (E2M1 nibbles + E8M0 "
                     "block scales). `mlx_lm convert -q` leaves the MoE experts in the mode "
                     "the checkpoint shipped them in; convert with --dequantize first, then "
                     "quantize, so every module is affine-U4 -- the one scheme this driver's "
                     "kernels read");
            }
            // Width comes from the tensors, not from the config: `mlx_lm`'s
            // quantization predicate leaves the router at 8 bits while
            // everything around it goes to 4, and nothing but the packing tells
            // you which is which.
            //
            // Solved rather than guessed. This driver's kernels are group-64, so
            // `logical_cols = groups * 64`, and a word holds `32/bits` values:
            //
            //     bits = 32 * packed_cols / (groups * 64) = packed_cols / (2*groups)
            //
            // A checkpoint quantized at another group size lands on a non-integer
            // or an out-of-range width and is refused here, which is better than
            // being refused later as "g128" -- the number that comes out of
            // assuming the width and solving for the group instead.
            const std::int64_t packed_cols = raw.shape.back();
            const std::int64_t groups = scales->shape.back();
            if (groups <= 0 || packed_cols % (2 * groups) != 0) {
                fail("Metal GptOss: '" + std::string(raw.name) +
                     "' is not quantized in groups of 64, which is what these kernels read");
            }
            const int bits = int(packed_cols / (2 * groups));
            if (bits != 4 && bits != 8) {
                fail("Metal GptOss: '" + std::string(raw.name) + "' is " +
                     std::to_string(bits) + "-bit, and only 4 and 8 are described here");
            }
            push_mlx_affine_stacked(out, raw, *scales, *biases, bits, std::move(*output));
        } else {
            push_direct(out, raw, std::move(*output));
        }
        ++declared;
    }
    if (declared == 0) {
        fail("Metal GptOss schema found no decoder tensors");
    }
}

}  // namespace pie::metal::model::gptoss
