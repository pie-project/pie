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

}  // namespace pie::metal::model
