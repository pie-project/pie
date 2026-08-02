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

/// Whether `raw_name` starts with a wrapper member, under either spelling.
///
/// A multimodal checkpoint nests its parts under a wrapper, and the nesting has
/// two real spellings. The HF release writes the wrapper's own `model.` first:
/// `model.audio_tower.`, `model.language_model.`. `mlx_lm`'s repack of the very
/// same checkpoint drops it, leaving the member bare. Neither is a variant of
/// the model -- they are two tools' names for one tensor.
///
/// So a schema names the member (`"audio_tower."`) and this reads both. It is
/// here and not in a family because it is a fact about `mlx_lm`, not about any
/// model: every multimodal family meets it, and the first two each met it
/// separately.
inline bool has_wrapper_member(std::string_view raw_name, std::string_view member) {
    if (raw_name.rfind(member, 0) == 0) return true;
    return raw_name.rfind("model.", 0) == 0 &&
           raw_name.substr(std::string_view("model.").size()).rfind(member, 0) == 0;
}

/// The text decoder's member, with whichever wrapper prefix spelled it stripped.
///
/// `model.language_model.*` (HF) and `language_model.model.*` (`mlx_lm`) are the
/// two spellings; note that they SWAP the two words rather than one merely
/// adding a prefix, so this cannot be expressed as `has_wrapper_member`. Only
/// the prefix differs -- everything downstream sees the same member string
/// either way, which is why this is a strip rather than a second name table.
inline std::optional<std::string_view> decoder_member(std::string_view raw_name) {
    for (std::string_view prefix : {"model.language_model.", "language_model.model."}) {
        if (raw_name.rfind(prefix, 0) == 0) return raw_name.substr(prefix.size());
    }
    return std::nullopt;
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

/// Declare a tensor under its runtime name, in the float format the kernels read.
///
/// Every kernel in this driver reads BF16 -- norm weights, affine scales and
/// biases alike -- so a checkpoint that ships F16 or F32 is CAST here rather
/// than transmuted. This is not a nicety. `mlx-community/Llama-3.2-1B-Instruct-4bit`
/// ships all 259 of its unpacked tensors as F16, and reinterpreting an F16 bit
/// pattern as BF16 is not an approximation -- the exponent field is a different
/// width and in a different place, so `0.0385` reads as `1.6e-12`. It does not
/// crash and it does not warn: the model loads, runs at full speed, and emits
/// the same token forever.
///
/// A cast is a load-time kernel and the heap then holds BF16, so nothing
/// downstream -- no bind, no PSO, no dispatch -- learns that the checkpoint was
/// ever anything else. One conversion, in the one place every family declares
/// its unpacked tensors.
inline void push_direct(ModelContract& out, const SourceTensor& raw, std::string output) {
    if (is_raw(raw.encoding, PieLoaderDType::F16) || is_raw(raw.encoding, PieLoaderDType::F32)) {
        const PieLoaderEncodingSpec bf16 = pie_loader::raw(PieLoaderDType::BF16);
        out.define(std::move(output), out.cast(out.src(std::string(raw.name)), bf16), bf16)
            .expect(shape_of(raw));
        return;
    }
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
                                    int declared_bits_hint, int declared_group_size,
                                    std::string output) {
    if (raw.shape.size() < 2 || scales.shape.size() != raw.shape.size() ||
        biases.shape.size() != scales.shape.size() ||
        !std::equal(biases.shape.begin(), biases.shape.end(), scales.shape.begin())) {
        fail("MLX affine triplet '" + std::string(raw.name) + "' has incompatible shapes");
    }
    std::int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < raw.shape.size(); ++i) {
        if (raw.shape[i] != scales.shape[i]) {
            fail("MLX affine triplet '" + std::string(raw.name) +
                 "' disagrees with its scales on the stacked axes");
        }
        rows *= raw.shape[i];
    }
    const std::int64_t groups = scales.shape.back();
    if (groups <= 0) {
        fail("MLX affine triplet '" + std::string(raw.name) + "' has no groups");
    }

    // Three numbers -- width, group, packed columns -- and the shapes pin only
    // their product: the same bytes are 8-bit g64 or 4-bit g128. So exactly one
    // has to be told to us, and it should be the group, because the group is
    // the one `config.json` states for the whole file while the WIDTH varies
    // per tensor. mlx_lm leaves a small sensitive projection at 8 bits inside
    // an otherwise 4-bit file (gpt-oss's router; gemma-4's dense MLP beside its
    // 4-bit experts), and it records that as a per-tensor override we would
    // otherwise have to parse.
    //
    // Deriving the width instead of assuming it reads those overrides for free,
    // and it removes a refusal rather than adding a parser. It used to be the
    // other way round, which is how a checkpoint that config.json describes
    // exactly got told that config.json was wrong.
    std::int64_t logical_cols = 0;
    int bits = declared_bits_hint;
    if (declared_group_size > 0) {
        logical_cols = groups * declared_group_size;
        const std::int64_t packed_bits = raw.shape.back() * 32;
        if (logical_cols <= 0 || packed_bits % logical_cols != 0) {
            fail("MLX affine triplet '" + std::string(raw.name) +
                 "' cannot derive a width from groups of " +
                 std::to_string(declared_group_size));
        }
        bits = static_cast<int>(packed_bits / logical_cols);
    }
    if (bits != 4 && bits != 8) {
        fail("MLX affine triplet '" + std::string(raw.name) + "' has an unsupported width (" +
             std::to_string(bits) + " bits)");
    }
    if (declared_group_size <= 0) {
        // gpt-oss states no quantization at all, so here the width is the told
        // number and the group is the derived one -- the same equation solved
        // for the other unknown.
        logical_cols = raw.shape.back() * (32 / bits);
        if (logical_cols % groups != 0) {
            fail("MLX affine triplet '" + std::string(raw.name) + "' cannot derive a group size");
        }
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

/// Declare an MXFP4 weight the checkpoint SHIPPED, without decoding it.
///
/// The counterpart of `push_mlx_affine_stacked` for the other 4-bit format mlx
/// publishes. `mlx_lm` writes MXFP4 as a `.weight` of U32 -- eight nibbles to a
/// little-endian word -- beside a U8 `.scales` of E8M0 block exponents, and no
/// `.biases`, because a block's values are a table lookup times a power of two
/// and there is no zero point to subtract.
///
/// This is a transmute and not a decode: the bytes staged into the heap are the
/// checkpoint's own. The alternative -- dequantize to BF16 and re-quantize
/// affine -- is what the loader did, and it is the one lossy step in a
/// checkpoint that is otherwise read verbatim.
///
/// That argument used to lean on a second one: that the re-quantized weights
/// could not be compared against mlx-lm because the driver's quantizer and
/// MLX's disagreed on 8.2% of codes. They no longer disagree -- the cause was a
/// rounding mode, and `transcode.metal` now reproduces `mx.quantize` bit for
/// bit. The transmute is still right, on its own merits: sixteen E2M1 levels
/// times a power of two do not survive a trip through a 15-step affine grid,
/// and the sixteen bits per group that grid costs are not bits this format
/// needs.
inline void push_mlx_mxfp4_stacked(ModelContract& out, const SourceTensor& raw,
                                   const SourceTensor& scales, std::string output) {
    if (raw.shape.size() < 2 || scales.shape.size() != raw.shape.size()) {
        fail("MXFP4 pair '" + std::string(raw.name) + "' and its scales differ in rank");
    }
    if (!is_raw(scales.encoding, PieLoaderDType::U8)) {
        fail("MXFP4 pair '" + std::string(raw.name) +
             "' has scales that are not the U8 E8M0 block exponents this format stores");
    }
    std::int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < raw.shape.size(); ++i) {
        if (raw.shape[i] != scales.shape[i]) {
            fail("MXFP4 pair '" + std::string(raw.name) +
                 "' disagrees with its scales on the stacked axes");
        }
        rows *= raw.shape[i];
    }
    // A U32 word holds eight nibbles; a block is 32 elements under one
    // exponent. Both counts must agree on the column width, and the shapes are
    // the only thing that says so.
    const std::int64_t groups = scales.shape.back();
    if (groups <= 0 || raw.shape.back() != groups * 4) {
        fail("MXFP4 pair '" + std::string(raw.name) + "' packs " +
             std::to_string(raw.shape.back()) + " words against " + std::to_string(groups) +
             " blocks, and eight nibbles to a word over 32-element blocks needs " +
             std::to_string(groups * 4));
    }
    const std::int64_t cols = groups * 32;

    PieLoaderQuantSpecView quant =
        pie_loader::quant_spec(PieLoaderQuantScheme::Mxfp4E2M1E8M0, PieLoaderDType::BF16);
    quant.bits_per_element = 4;
    quant.group_size = 32;
    quant.channel_axis = 1;
    const PieLoaderEncodingSpec encoding = pie_loader::quantized(quant);
    out.define(std::move(output),
               out.transmute(out.src(std::string(raw.name)), {rows, cols}, encoding), encoding)
        .expect(std::vector<std::int64_t>{rows, cols});
}

/// The case where `config.json` states the quantization, which is every family
/// but gpt-oss: one `quantization` block covers the whole file, so the width is
/// read rather than assumed and the group it implies is checked against it.
///
/// This used to be a second implementation that accepted rank 2 only, which is
/// why a routed llama checkpoint -- whose experts arrive as `[n_experts, out,
/// in/8]` -- was refused by a driver that had supported stacked weights for
/// gpt-oss all along. Rank 2 IS the stacked case with an empty stack, so there
/// is one implementation and the special case is gone rather than doubled.
inline void push_mlx_affine_declared(ModelContract& out, const SourceTensor& raw,
                                     const SourceTensor& scales, const SourceTensor& biases,
                                     int declared_bits, int declared_group_size,
                                     std::string output) {
    // A config that declares nothing is a checkpoint whose tensors are dense,
    // and reaching here at all means it is not -- so 4 is the historical
    // default kept for exactly that case, and the group check is skipped
    // because there is nothing to check it against.
    const int bits = declared_bits > 0 ? declared_bits : 4;
    push_mlx_affine_stacked(out, raw, scales, biases, bits, declared_group_size,
                            std::move(output));
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
                                                       std::string_view schema,
                                                       bool has_shared_expert = false) {
    constexpr std::string_view kSwitch = "mlp.switch_mlp.";
    if (member.rfind(kSwitch, 0) == 0) {
        return "mlp.experts." + std::string(member.substr(kSwitch.size()));
    }
    // A family that DISPATCHES a shared expert takes its two blocks through
    // unchanged -- `mlp.shared_expert.*` and `mlp.shared_expert_gate.*` are
    // already the names `weights_for_kind` asks for. The plural spelling is
    // refused either way: `mlp.shared_experts.` is DeepSeek's, where several
    // shared experts are stacked, and this is one dense FFN.
    if (has_shared_expert) {
        for (std::string_view ok : {"mlp.shared_expert.", "mlp.shared_expert_gate."}) {
            if (member.rfind(ok, 0) == 0) return std::string(member);
        }
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
