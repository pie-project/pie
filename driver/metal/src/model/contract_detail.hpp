#pragma once

/// Helpers every Metal storage schema shares.
///
/// A *contract* is per family — the tensor names are the family — but the
/// mechanics of declaring one are not: reading a shape, recognising an MLX
/// affine-U4 triplet, and defining a tensor under a runtime name are the same
/// work whatever the model is. They live here so a second family does not
/// arrive with a second copy that drifts.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <cctype>
#include <optional>
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

/// Declare a tensor as it sits on disk, under its runtime name.
inline void push_direct(ModelContract& out, const SourceTensor& raw, std::string output) {
    out.define(std::move(output), out.src(std::string(raw.name)), raw.encoding)
        .expect(shape_of(raw));
}

/// Declare an MLX affine weight whose leading axes are a STACK.
///
/// A sparse-MoE checkpoint stores one tensor per projection with the expert on
/// axis 0 -- `[n_experts, out, in/pack]` -- rather than `n_experts` matrices.
/// The quantization is per row of the last two axes exactly as in the 2-D case,
/// so the only thing that changes is that the row count is the product of every
/// axis but the last.
///
/// `bits` is 4 or 8: `mlx_lm`'s quantization predicate leaves a small, sensitive
/// projection (gpt-oss's router) at 8 bits while quantizing everything around it
/// to 4, so a family that declared one width for the whole checkpoint would be
/// describing a checkpoint that does not exist.
inline void push_mlx_affine_stacked(ModelContract& out, const SourceTensor& raw,
                                    const SourceTensor& scales, const SourceTensor& biases,
                                    int bits, std::string output) {
    if (raw.shape.size() < 2 || scales.shape.size() != raw.shape.size() ||
        biases.shape.size() != scales.shape.size() ||
        !std::equal(biases.shape.begin(), biases.shape.end(), scales.shape.begin())) {
        fail("MLX affine triplet '" + std::string(raw.name) + "' has incompatible shapes");
    }
    if (bits != 4 && bits != 8) {
        fail("MLX affine triplet '" + std::string(raw.name) + "' has an unsupported width");
    }
    const int per_word = 32 / bits;
    std::int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < raw.shape.size(); ++i) {
        if (raw.shape[i] != scales.shape[i]) {
            fail("MLX affine triplet '" + std::string(raw.name) +
                 "' disagrees with its scales on the stacked axes");
        }
        rows *= raw.shape[i];
    }
    const std::int64_t logical_cols = raw.shape.back() * per_word;
    const std::int64_t groups = scales.shape.back();
    if (groups <= 0 || logical_cols % groups != 0) {
        fail("MLX affine triplet '" + std::string(raw.name) + "' cannot derive a group size");
    }
    const std::uint32_t group_size =
        u32_dim(logical_cols / groups, "MLX affine group size");

    PieLoaderQuantSpecView quant = pie_loader::quant_spec(
        bits == 4 ? PieLoaderQuantScheme::MlxAffineU4 : PieLoaderQuantScheme::Int8Asymmetric,
        PieLoaderDType::BF16);
    quant.bits_per_element = static_cast<std::uint32_t>(bits);
    quant.group_size = group_size;
    quant.channel_axis = 1;
    const PieLoaderEncodingSpec encoding = pie_loader::quantized(quant);

    out.define(std::move(output),
               out.transmute(out.src(std::string(raw.name)), {rows, logical_cols}, encoding),
               encoding)
        .expect(std::vector<std::int64_t>{rows, logical_cols});
}

/// The 4-bit case, under the name the families already call.
///
/// This used to be a second implementation that accepted rank 2 only, which is
/// why a routed llama checkpoint -- whose experts arrive as `[n_experts, out,
/// in/8]` -- was refused by a driver that had supported stacked weights for
/// gpt-oss all along. Rank 2 IS the stacked case with an empty stack, so there
/// is one implementation and the special case is gone rather than doubled.
inline void push_mlx_affine_u4(ModelContract& out, const SourceTensor& raw,
                               const SourceTensor& scales, const SourceTensor& biases,
                               std::string output) {
    push_mlx_affine_stacked(out, raw, scales, biases, 4, std::move(output));
}

/// The encoding this driver's quantized matvecs read.
///
/// `bits` is 4 or 8: `mlx_lm`'s quantization predicate leaves a small, sensitive
/// projection (gpt-oss's router) at 8 bits while quantizing everything around it
/// to 4, so a family that declared one width for the whole checkpoint would be
/// describing a checkpoint that does not exist.
inline PieLoaderEncodingSpec affine_encoding(int bits, std::uint32_t group_size) {
    PieLoaderQuantSpecView quant = pie_loader::quant_spec(
        bits == 4 ? PieLoaderQuantScheme::MlxAffineU4 : PieLoaderQuantScheme::Int8Asymmetric,
        PieLoaderDType::BF16);
    quant.bits_per_element = static_cast<std::uint32_t>(bits);
    quant.group_size = group_size;
    quant.channel_axis = 1;
    return pie_loader::quantized(quant);
}

/// The columns this driver's kernels group under one scale.
inline constexpr std::int64_t kAffineGroup = 64;

/// Declare a weight the LOADER quantizes, rather than one the checkpoint
/// shipped quantized.
///
/// The distinction is the whole of the difference between a checkpoint someone
/// converted offline and the one its authors published: `cast` to a quantized
/// encoding is an encode, and an encode writes `<stem>.scales` and
/// `<stem>.biases` beside its output as part of the same pass. `output` must
/// therefore end in `.weight` -- that is the component the metadata names
/// replace -- and the driver binds all three under exactly the names a
/// converted checkpoint would have shipped.
///
/// `rows` is the product of every axis but the last, because the encode kernel
/// walks `[rows, cols]` tiles and a stacked expert tensor is a taller matrix as
/// far as quantization is concerned: the groups run along the last axis either
/// way.
inline void push_encoded_affine(ModelContract& out, pie_loader::Node value,
                                std::int64_t rows, std::int64_t cols, std::string output) {
    if (cols % kAffineGroup != 0) {
        fail("Metal: '" + output + "' has " + std::to_string(cols) +
             " columns, which these group-64 kernels cannot quantize");
    }
    const PieLoaderEncodingSpec encoding =
        affine_encoding(4, static_cast<std::uint32_t>(kAffineGroup));
    out.define(std::move(output), out.cast(value, encoding), encoding)
        .expect(std::vector<std::int64_t>{rows, cols});
}

/// The BF16 values behind an MXFP4 `_blocks`/`_scales` pair.
///
/// Two nodes and no kernel of this driver's own: the contract says the packed
/// bytes are E2M1 nibbles under E8M0 block scales, and the loader's dequantizer
/// is what turns that declaration into values. The scales have to be *declared*
/// before they can be scaled by -- `scale_per_block` takes a published tensor,
/// not a fresh `src` -- so this leaves an internal tensor behind under
/// `scales_name`.
///
/// `blocks` and `scales` are expressions rather than names so a caller can
/// select the half of a fused tensor it wants before anything is decoded.
inline pie_loader::Node mxfp4_values(ModelContract& out, pie_loader::Node blocks,
                                    pie_loader::Node scales, std::int64_t rows,
                                    std::int64_t cols, const std::string& scales_tensor) {
    if (cols % 32 != 0) {
        fail("MXFP4 tensor '" + scales_tensor + "' has " + std::to_string(cols) +
             " columns, which is not a whole number of 32-element blocks");
    }
    const std::vector<std::int64_t> groups{rows, cols / 32};
    const PieLoaderEncodingSpec e8m0 = pie_loader::raw(PieLoaderDType::E8M0);
    out.define(scales_tensor, out.transmute(scales, groups, e8m0), e8m0)
        .expect(groups)
        .internal();

    PieLoaderQuantSpecView quant = pie_loader::quant_spec(PieLoaderQuantScheme::Mxfp4E2M1E8M0,
                                                         PieLoaderDType::BF16);
    quant.bits_per_element = 4;
    quant.group_size = 32;
    quant.channel_axis = 1;
    return out.scale_per_block(
        out.transmute(blocks, {rows, cols}, pie_loader::quantized(quant)),
        out.out(scales_tensor));
}

/// The one rule every routed family's mixture is named by.
///
/// A routed FFN must arrive with its experts STACKED on axis 0, which is what
/// `affine_qmv_routed` indexes: one tensor per layer per projection,
/// expert-major. Two spellings of that exist and both are accepted, because
/// the two toolchains that produce it disagree -- `mlx_lm` wraps the mixture
/// in a `SwitchGLU` and emits `mlp.switch_mlp.gate_proj`, the fused HF export
/// emits `mlp.experts.gate_proj`. They are the same bytes in the same layout.
///
/// Two forms are refused rather than skipped, and both for the same reason:
/// skipping is what silently produces the wrong model.
///
///  - The UNSTACKED bank, `mlp.experts.0.gate_proj`, which a stock HF
///    checkpoint ships. Binding it would need a load-time gather this driver
///    does not do, and the failure mode of guessing is expert 0's weights used
///    for all of them -- fluent, and wrong.
///  - A SHARED expert, a dense SwiGLU every token runs beside the routed ones
///    under its own sigmoid gate. `qwen2_moe` and Qwen3-Next both ship one and
///    this driver computes no such thing. Left to a family's pass-through, its
///    weights are declared, loaded, and then read by no dispatch: the model
///    runs the routed mixture alone. Nothing catches that -- a bind test asks
///    whether every slot a dispatch NEEDS was filled, which is the opposite
///    direction, and a tensor nobody asks for is invisible to it.
///
/// Returns the rewritten member when the name is a stacked bank, and
/// `std::nullopt` when the name is not about a mixture at all -- in which case
/// the caller's own mapping continues. It never returns for a refusal.
///
/// `schema` names the family in the message, because the refusal is what the
/// user sees and "which driver said no" is the first thing they need.
inline std::optional<std::string> routed_expert_member(std::string_view raw_name,
                                                       std::string_view member,
                                                       std::string_view schema) {
    constexpr std::string_view kSwitch = "mlp.switch_mlp.";
    if (member.rfind(kSwitch, 0) == 0) {
        return "mlp.experts." + std::string(member.substr(kSwitch.size()));
    }
    for (std::string_view shared : {"mlp.shared_expert.", "mlp.shared_expert_gate.",
                                    "mlp.shared_experts."}) {
        if (member.rfind(shared, 0) == 0) {
            fail("Metal " + std::string(schema) + " schema has no shared expert, but '" +
                 std::string(raw_name) +
                 "' is one: this driver would load it and never read it, running the "
                 "routed mixture alone");
        }
    }
    constexpr std::string_view kExperts = "mlp.experts.";
    if (member.rfind(kExperts, 0) == 0) {
        const std::string_view rest = member.substr(kExperts.size());
        if (!rest.empty() && std::isdigit(static_cast<unsigned char>(rest.front())) != 0) {
            fail("Metal " + std::string(schema) +
                 " schema needs the routed experts stacked on axis 0 "
                 "(one `mlp.experts.gate_proj` per layer, expert-major), but '" +
                 std::string(raw_name) + "' is per-expert");
        }
    }
    return std::nullopt;
}

}  // namespace contract_detail

}  // namespace pie::metal::model
