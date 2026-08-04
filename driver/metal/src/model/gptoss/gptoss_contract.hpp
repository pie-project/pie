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

/// Declare the experts the way the PUBLISHED checkpoint stores them.
///
/// Three things separate this from `mlx_lm`'s conversion of the same weights,
/// and all three are why the experts cannot go through the generic loop:
///
///  * **The gate and up projections are one fused tensor, interleaved.**
///    `gate_up_proj` holds gate at the even rows and up at the odd ones (HF's
///    `gate_up[..., ::2]` and `[..., 1::2]`), so each half is a strided read
///    rather than a slice, and one source tensor becomes two runtime ones.
///  * **They are MXFP4**, stored as `_blocks` of E2M1 nibbles under `_scales`
///    of E8M0 exponents, which no kernel here reads. They are decoded to values
///    and re-encoded as the affine-U4 the matvecs do read -- the same two steps
///    `mlx_lm convert --dequantize` then `-q` performs, done at load time
///    instead of offline.
///  * **The biases follow the same fusion**, so they split the same way.
///
/// `declared` counts up so the caller's "found no decoder tensors" check still
/// sees these.
inline void declare_mxfp4_experts(ModelContract& out, const std::vector<SourceTensor>& tensors,
                                  std::size_t& declared) {
    using namespace pie::metal::model::contract_detail;
    constexpr std::string_view kBlocks = "_blocks";

    const auto find = [&tensors](const std::string& name) -> const SourceTensor* {
        for (const SourceTensor& raw : tensors) {
            if (raw.name == name) return &raw;
        }
        return nullptr;
    };

    for (const SourceTensor& blocks : tensors) {
        if (!ends_with(blocks.name, kBlocks)) continue;
        const std::string base =
            std::string(blocks.name.substr(0, blocks.name.size() - kBlocks.size()));
        const SourceTensor* scales = find(base + "_scales");
        if (scales == nullptr) {
            fail("Metal GptOss: '" + std::string(blocks.name) +
                 "' is an MXFP4 block tensor with no '_scales' beside it");
        }
        // `[experts, rows, groups, 16]` of nibbles against `[experts, rows,
        // groups]` of exponents. The 16 is the block's bytes and is the one axis
        // the two do not share.
        if (blocks.shape.size() != 4 || scales->shape.size() != 3 || blocks.shape[3] != 16 ||
            blocks.shape[0] != scales->shape[0] || blocks.shape[1] != scales->shape[1] ||
            blocks.shape[2] != scales->shape[2]) {
            fail("Metal GptOss: MXFP4 tensor '" + std::string(blocks.name) +
                 "' is not shaped [experts, rows, groups, 16] against its scales");
        }
        const std::int64_t experts = blocks.shape[0];
        const std::int64_t stored_rows = blocks.shape[1];
        const std::int64_t cols = blocks.shape[2] * 32;

        const std::optional<std::string> mapped = runtime_name(base);
        if (!mapped.has_value()) continue;
        const bool fused = ends_with(base, "gate_up_proj");
        if (!fused && !ends_with(base, "down_proj")) {
            fail("Metal GptOss: MXFP4 tensor '" + std::string(blocks.name) +
                 "' is neither the fused gate/up projection nor the down projection");
        }
        // `layers.N.mlp.experts.gate_up_proj` -> `layers.N.mlp.experts.`
        const std::string prefix =
            mapped->substr(0, mapped->size() - (fused ? 12u : 9u));
        const SourceTensor* bias = find(base + "_bias");

        struct Half {
            const char* name;
            std::int64_t start;
        };
        const std::vector<Half> halves =
            fused ? std::vector<Half>{{"gate_proj", 0}, {"up_proj", 1}}
                  : std::vector<Half>{{"down_proj", 0}};
        const std::int64_t rows = fused ? stored_rows / 2 : stored_rows;

        // Decoded whole and split afterwards is not available: a strided read of
        // a computed buffer is not lowered. So the split happens where a stride
        // IS lowered -- on the source extent -- and each half is published as
        // the plain bytes it is before anything reinterprets it, because a
        // transmute that changes the element width may only rename a whole
        // tensor. Both restrictions are satisfied by the same two steps, in this
        // order: select, publish, then reinterpret the published name.
        for (const Half& half : halves) {
            const std::string name = prefix + half.name;
            const auto select = [&](const std::string& source, std::int64_t axis_len,
                                    std::vector<std::int64_t> shape,
                                    const std::string& as) -> pie_loader::Node {
                out.define(as,
                           fused ? out.stride(out.src(source), /*axis=*/1, half.start, axis_len,
                                              /*step=*/2)
                                 : out.src(source),
                           pie_loader::raw(PieLoaderDType::U8))
                    .expect(std::move(shape))
                    .internal();
                return out.out(as);
            };
            const std::int64_t groups = blocks.shape[2];
            const pie_loader::Node half_blocks =
                select(std::string(blocks.name), rows, {experts, rows, groups, 16},
                       name + ".mxfp4_blocks");
            const pie_loader::Node half_scales =
                select(std::string(scales->name), rows, {experts, rows, groups},
                       name + ".mxfp4_exponents");

            const std::vector<std::int64_t> shape{experts * rows, cols};
            out.define(name + ".dequantized",
                       mxfp4_values(out, half_blocks, half_scales, experts * rows, cols,
                                    name + ".mxfp4_scales"),
                       pie_loader::raw(PieLoaderDType::BF16))
                .expect(shape)
                .internal();
            push_encoded_affine(out, out.out(name + ".dequantized"), experts * rows, cols,
                                name + ".weight");
            if (bias != nullptr) {
                if (bias->shape.size() != 2 || bias->shape[0] != experts ||
                    bias->shape[1] != stored_rows) {
                    fail("Metal GptOss: '" + std::string(bias->name) +
                         "' does not match the projection it biases");
                }
                out.define(name + ".bias",
                           fused ? out.stride(out.src(std::string(bias->name)), /*axis=*/1,
                                              half.start, rows, /*step=*/2)
                                 : out.src(std::string(bias->name)),
                           bias->encoding)
                    .expect(std::vector<std::int64_t>{experts, rows});
            }
            ++declared;
        }
    }
}

/// Declare an MXFP4 projection the way `mlx_lm` REPACKS it.
///
/// Same weights, same bytes, a different container. Where the published
/// checkpoint writes `X_blocks` as `[experts, rows, groups, 16]` of U8 beside
/// `X_scales`, `mlx_lm`'s conversion writes `X.weight` as
/// `[experts, rows, groups*4]` of U32 beside `X.scales` -- eight nibbles to a
/// word instead of two to a byte, which on a little-endian device is the same
/// byte stream with a different element width declared over it.
///
/// It is also already SPLIT: `gate_proj` and `up_proj` are separate tensors
/// rather than one interleaved `gate_up_proj`. So none of the striding the
/// published layout needs applies, and neither do its internal intermediates:
/// each source is a whole tensor, which is the case a width-changing transmute
/// is allowed in. This is the simpler half of the same job, which is why it is
/// a few lines against sixty.
///
/// Both spellings end in the same place -- decode to values, re-encode as the
/// affine-U4 the matvecs read -- because `mxfp4_values` takes expressions and
/// does not care where they came from.
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
        // The published checkpoint's MXFP4 experts are consumed as a pair and
        // declared from the `_blocks` half, so the `_scales` half is not a
        // tensor of its own here. `runtime_name` is not asked about either: the
        // names they produce are the split projections', which no per-tensor
        // mapping can state.
        if (ends_with(raw.name, "_scales") || ends_with(raw.name, "_blocks") ||
            ends_with(raw.name, "_bias")) {
            continue;
        }
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
            // Scales with no zero points is MXFP4, which `mlx_lm convert -q`
            // leaves the MoE experts in because that is the mode openai shipped
            // them in. Every published mlx conversion of this family is like
            // this, so refusing it would refuse the family -- and the loader
            // decodes MXFP4 already, for exactly this reason (`load_plan.hpp`).
            // The refusal that used to be here told the user to convert offline
            // for a transform the driver performs at load.
            if (biases == nullptr) {
                push_mlx_mxfp4_stacked(out, raw, *scales, std::move(*output));
                ++declared;
                continue;
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
            push_mlx_affine_stacked(out, raw, *scales, *biases, bits, /*declared_group_size=*/64,
                                    std::move(*output));
        } else if (ends_with(raw.name, ".weight") && raw.shape.size() == 2 &&
                   is_raw(raw.encoding, PieLoaderDType::BF16)) {
            // A projection the published checkpoint left in BF16. gpt-oss ships
            // only its experts quantized, so the attention projections, the
            // router, the embedding and the head arrive as values -- and every
            // one of them is read by a quantized matvec here, so the loader
            // quantizes them on the way in.
            //
            // Rank is what separates these from the norms, which are also
            // `.weight` and also BF16 and must stay values: a norm is a vector,
            // and there is no such thing as quantizing one under this scheme.
            push_encoded_affine(out, out.src(std::string(raw.name)), raw.shape[0], raw.shape[1],
                                std::move(*output));
        } else {
            push_direct(out, raw, std::move(*output));
        }
        ++declared;
    }
    contract_detail::declare_mxfp4_experts(out, tensors, declared);
    if (declared == 0) {
        fail("Metal GptOss schema found no decoder tensors");
    }
}

}  // namespace pie::metal::model::gptoss
