#pragma once

/// What DeepSeek-V4 binds (`model/registry.cpp` row).
///
/// The only family with its own tensor-parallel shard-axis rule: its experts
/// are named `.ffn.experts.w1/w2/w3` rather than `.mlp.experts.gate/up/down`,
/// and the intermediate dim is split within each expert so every rank computes
/// a partial expert output that an all-reduce combines.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

inline ShardAxis dsv4_shard_axis(std::string_view name) {
    // Routed experts: shard the intermediate dim within each expert. w1/w3 on
    // axis 0 (gate/up out dim), w2 on axis 1 (down in dim). Each rank computes
    // a partial expert output; the results are combined by an all-reduce.
    if (contains(name, ".ffn.experts.")) {
        if (ends_with_any(name, {".w1.weight", ".w1.scale", ".w3.weight", ".w3.scale"})) {
            return std::uint8_t{0};
        }
        if (ends_with_any(name, {".w2.weight", ".w2.scale"})) {
            return std::uint8_t{1};
        }
    }
    if (ends_with_any(name, {".shared_experts.w1.weight", ".shared_experts.w1.scale",
                             ".shared_experts.w3.weight", ".shared_experts.w3.scale"})) {
        return std::uint8_t{0};
    }
    if (ends_with_any(name, {".shared_experts.w2.weight", ".shared_experts.w2.scale"})) {
        return std::uint8_t{1};
    }
    // Everything else replicated, which avoids TP communication in the main path.
    return std::nullopt;
}

}  // namespace contract_detail

/// Decode the block scales that ride beside DeepSeek-V4's FP8 weights.
///
/// The checkpoint stores one byte per tile of the weight, and that byte is OCP
/// Microscaling's E8M0: it denotes `2^(b - 127)`. The FP8 GEMM wants fp32.
/// Both halves of that are sayable -- `Bitcast` names how the bytes are to be
/// read, and the declaration names the type wanted -- so the planner inserts
/// the cast itself.
///
/// `QuantScheme::Mxfp4E2M1E8M0` cannot name this pairing: that symbol bundles
/// the element format together with the scale format, and E8M0 beside
/// FP8-E4M3 is a combination it has no name for. Which is why the driver used
/// to copy every scale to the host, run `ldexpf` over it and upload the result
/// during bind -- arithmetic the algebra has no vocabulary for, done outside
/// the loader entirely.
inline void dsv4_block_scales_to_fp32(ContractBuilder& b) {
    constexpr std::string_view kSuffix = ".scale";
    for (const SourceTensor& raw : b.tensors()) {
        if (!contract_detail::ends_with(raw.name, kSuffix) ||
            !contract_detail::is_raw(raw.encoding, PieLoaderDType::U8)) {
            continue;
        }
        // Only a companion to a **block-FP8** weight is an fp32-bound E8M0
        // exponent. A `.scale` beside anything else is some other convention,
        // and guessing is how a scale tensor gets silently reinterpreted.
        //
        // DeepSeek-V4 ships exactly two quantizations, and only one of them
        // belongs here. The dense and shared paths store F8E4M3 weights with
        // 128x128 block scales, which the FP8 GEMM wants as fp32 -- 50 tensors
        // on both minis. The routed experts store **packed MXFP4**: an `I8`
        // tensor holding two E2M1 nibbles per byte, whose E8M0 scales
        // (`[rows, cols/32]`) `launch_dequant_mxfp4_to_bf16` consumes as raw
        // bytes -- 144 tensors. Widening this guard to I8 hands that kernel
        // fp32 words to read as exponents, and the routed experts come out
        // four orders of magnitude too large.
        const std::string weight =
            std::string(raw.name.substr(0, raw.name.size() - kSuffix.size())) + ".weight";
        const SourceTensor* companion = b.find(weight);
        if (companion == nullptr ||
            !contract_detail::is_raw(companion->encoding, PieLoaderDType::F8E4M3)) {
            continue;
        }
        std::vector<std::int64_t> shape = contract_detail::shape_of(raw);
        auto [expr, local] = b.shard(
            b.contract().bitcast(b.contract().src(std::string(raw.name)), shape,
                                 pie_loader::raw(PieLoaderDType::E8M0)),
            shape, b.shard_axis(raw.name));
        b.define(b.output_name(raw.name), expr, pie_loader::raw(PieLoaderDType::F32),
                 std::move(local));
        b.consume(raw.id);
    }
}

inline void author_deepseek_v4_contract(ContractBuilder& b) {
    b.shard_axis_fn(contract_detail::dsv4_shard_axis);
    dsv4_block_scales_to_fp32(b);
    author_dense_contract(b);
}
}  // namespace pie_cuda_driver::model
