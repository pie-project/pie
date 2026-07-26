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

namespace pie::metal::model {

using pie_loader::Checkpoint;
using pie_loader::ModelContract;
using pie_loader::SourceTensor;
using pie_loader::PieLoaderDType;
using pie_loader::PieLoaderEncodingKind;
using pie_loader::PieLoaderEncodingSpec;
using pie_loader::PieLoaderQuantScheme;
using pie_loader::PieLoaderQuantSpecView;

namespace contract_detail {

[[noreturn]] inline void fail(const std::string& what) { throw std::runtime_error(what); }

inline bool ends_with(std::string_view value, std::string_view tail) {
    return value.size() >= tail.size() &&
           value.compare(value.size() - tail.size(), tail.size(), tail) == 0;
}

inline PieLoaderEncodingKind kind_of(const PieLoaderEncodingSpec& e) {
    return static_cast<PieLoaderEncodingKind>(e.kind);
}

inline bool is_raw(const PieLoaderEncodingSpec& e, PieLoaderDType dtype) {
    return kind_of(e) == PieLoaderEncodingKind::Raw &&
           static_cast<PieLoaderDType>(e.dtype) == dtype;
}

inline std::vector<std::int64_t> shape_of(const SourceTensor& raw) {
    return std::vector<std::int64_t>(raw.shape.begin(), raw.shape.end());
}

inline std::uint32_t u32_dim(std::int64_t value, std::string_view context) {
    if (value < 0 || value > static_cast<std::int64_t>(UINT32_MAX)) {
        fail(std::string(context) + ": dimension " + std::to_string(value) + " does not fit u32");
    }
    return static_cast<std::uint32_t>(value);
}

/// The runtime name for a checkpoint tensor, or none to skip it.
///
/// Every name is either mapped or explicitly skipped; an unrecognised one is an
/// error rather than a pass-through, because a tensor this driver silently
/// declared under its checkpoint name would never be found by the binder.
inline std::optional<std::string> runtime_name(std::string_view raw_name) {
    if (raw_name.rfind("model.visual.", 0) == 0 || raw_name.rfind("mtp.", 0) == 0) {
        return std::nullopt;
    }
    if (raw_name.rfind("lm_head.", 0) == 0) {
        return "shared_embedding." +
               std::string(raw_name.substr(std::string_view("lm_head.").size()));
    }
    // The tied half of the same slot. `EmbedGather` and `QmvLmHead` both bind
    // `shared_embedding.*`, and this family ships tied — so its checkpoint has
    // an `embed_tokens` and no `lm_head` at all. Without this the driver could
    // not name the embedding table of the one model it targets.
    //
    // A checkpoint carrying both would declare `shared_embedding` twice and be
    // rejected as a duplicate, which is the truthful outcome: there is one
    // shared slot here, not two.
    constexpr std::string_view kEmbed = "model.language_model.embed_tokens.";
    if (raw_name.rfind(kEmbed, 0) == 0) {
        return "shared_embedding." + std::string(raw_name.substr(kEmbed.size()));
    }
    if (raw_name == "model.language_model.norm.weight") {
        return std::string("final_norm.weight");
    }
    constexpr std::string_view kLayers = "model.language_model.layers.";
    if (raw_name.rfind(kLayers, 0) != 0) {
        fail("Metal Qwen3.5 schema has no declared mapping or skip for '" + std::string(raw_name) +
             "'");
    }
    const std::string_view rest = raw_name.substr(kLayers.size());
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
    return "layers." + std::string(layer) + "." + std::string(rest.substr(dot + 1));
}

/// Declare a tensor as it sits on disk, under its runtime name.
inline void push_direct(ModelContract& out, const SourceTensor& raw, std::string output) {
    out.define(std::move(output), out.src(std::string(raw.name)), raw.encoding)
        .expect(shape_of(raw));
}

/// Declare an MLX affine-U4 weight from its `weight`/`scales`/`biases` triplet.
///
/// The checkpoint stores these 4-bit weights eight to a u32 word. The contract
/// names them for what they are; no byte moves.
inline void push_mlx_affine_u4(ModelContract& out, const SourceTensor& raw,
                               const SourceTensor& scales, const SourceTensor& biases,
                               std::string output) {
    if (raw.shape.size() != 2 || scales.shape.size() != 2 ||
        biases.shape.size() != scales.shape.size() ||
        !std::equal(biases.shape.begin(), biases.shape.end(), scales.shape.begin())) {
        fail("MLX affine-U4 triplet '" + std::string(raw.name) + "' has incompatible shapes");
    }
    const std::int64_t rows = raw.shape[0];
    const std::int64_t logical_cols = raw.shape[1] * 8;
    if (rows != scales.shape[0] || scales.shape[1] <= 0 || logical_cols % scales.shape[1] != 0) {
        fail("MLX affine-U4 triplet '" + std::string(raw.name) + "' cannot derive a group size");
    }
    const std::uint32_t group_size =
        u32_dim(logical_cols / scales.shape[1], "MLX affine-U4 group size");

    PieLoaderQuantSpecView quant =
        pie_loader::quant_spec(PieLoaderQuantScheme::MlxAffineU4, PieLoaderDType::BF16);
    quant.bits_per_element = 4;
    quant.group_size = group_size;
    quant.channel_axis = 1;
    const PieLoaderEncodingSpec encoding = pie_loader::quantized(quant);

    out.define(std::move(output),
               out.transmute(out.src(std::string(raw.name)), {rows, logical_cols}, encoding),
               encoding)
        .expect(std::vector<std::int64_t>{rows, logical_cols});
}

}  // namespace contract_detail

/// The `model_type` strings whose text decoder this schema describes.
///
/// One list, and it is this one: there is no second table keyed by the same
/// strings anywhere in this driver.
inline bool is_supported_model_type(std::string_view model_type) {
    for (std::string_view name : {"qwen3_5", "qwen3_5_text", "qwen3_next", "qwen3_next_text",
                                  "qwen3_6"}) {
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
                                  const pie_loader::DeviceTarget& target, ModelContract& out) {
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

    std::size_t declared = 0;
    for (const SourceTensor& raw : tensors) {
        std::optional<std::string> output = runtime_name(raw.name);
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
            push_mlx_affine_u4(out, raw, *scales, *biases, std::move(*output));
        } else {
            push_direct(out, raw, std::move(*output));
        }
        ++declared;
    }
    if (declared == 0) {
        fail("Metal Qwen3.5 schema found no text-decoder tensors");
    }
}

}  // namespace pie::metal::model
