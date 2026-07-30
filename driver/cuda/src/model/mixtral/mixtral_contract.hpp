#pragma once

/// What the Mixtral-family binders bind (`model/registry.cpp` rows).
///
/// Plain Mixtral's checkpoint needs nothing special. GPT-OSS is the whole file:
/// its experts ship as an MXFP4 `_blocks`/`_scales`/`_bias` triplet, and the
/// layout the contract asks for depends on whether this device has a native
/// MXFP4 GEMM.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

/// State that an MXFP4 scale tensor holds the block scales for `weight`.
///
/// The loader used to work this out for itself, by looking for `{weight}_scale`
/// beside every `Encoding::Quant(Mxfp4E2M1E8M0)` tensor — which happened to
/// match the names below, and would have stopped matching the moment either side
/// was renamed.
///
/// `channel_axis` is 1 for both halves even though `down_proj` is declared with
/// `mxfp4_encoding(.., 2)`. That is what the name matching produced — it read
/// the axis off the scheme, not off the encoding — and the value is live: it
/// reaches `QuantMeta` and is serialized into the weight-store cache. Changing
/// it is a separate question from moving where it is stated.
inline void state_mxfp4_block_scales(std::optional<pie_loader::ModelContract::Defined>& scales,
                                     std::string weight) {
    if (!scales.has_value()) {
        return;
    }
    scales->scaling(std::move(weight), PieLoaderQuantGranularity::PerGroup, 32, 1,
                    PieLoaderScaleForm::RawE8M0);
}

inline void gpt_oss_native_gate_up(ContractBuilder& b, const SourceTensor& block, const SourceTensor& scale,
                                const SourceTensor& bias, const std::string& base) {
    if (block.shape.size() != 4 || scale.shape.size() != 3 || bias.shape.size() != 2) {
        fail("GPT-OSS native gate/up '" + base + "' has an unsupported block/scale/bias rank");
    }
    const std::int64_t experts = block.shape[0];
    const std::int64_t fused_rows = block.shape[1];
    const std::int64_t groups = block.shape[2];
    if (fused_rows % 2 != 0 || block.shape[3] != 16) {
        fail("GPT-OSS native gate/up '" + base + "' expected [E, 2I, H/32, 16]");
    }
    if (scale.shape.size() != 3 || scale.shape[0] != experts || scale.shape[1] != fused_rows ||
        scale.shape[2] != groups || bias.shape[0] != experts || bias.shape[1] != fused_rows) {
        fail("GPT-OSS native gate/up '" + base + "' scale/bias shape mismatch");
    }
    const std::int64_t full_intermediate = fused_rows / 2;
    const std::int64_t hidden = groups * 32;
    const auto [local_start, local_intermediate] =
        b.local_range(full_intermediate, "the intermediate size of '" + base + "'");
    const std::int64_t intermediate_native = align_up(local_intermediate, 128);
    const std::string prefix =
        base.substr(0, base.size() - std::string_view("gate_up_proj").size());

    const struct {
        std::string_view name;
        PieLoaderRowMap row_map;
    } halves[] = {{"gate_proj", PieLoaderRowMap::Even}, {"up_proj", PieLoaderRowMap::Odd}};
    for (const auto& half : halves) {
        const std::string out_base = prefix + std::string(half.name);

        PieLoaderRepackSpecView weight =
            repack_spec(PieLoaderRepackLayout::MarlinMxfp4Weight, half.row_map);
        weight.batch = u32_dim(experts, "GPT-OSS experts");
        weight.source_rows = u32_dim(fused_rows, "GPT-OSS gate/up source rows");
        weight.source_row_offset = u32_dim(local_start, "GPT-OSS gate/up source row offset");
        weight.target_rows = u32_dim(intermediate_native, "GPT-OSS gate/up target rows");
        weight.valid_rows = u32_dim(local_intermediate, "GPT-OSS gate/up valid rows");
        weight.source_stride_cols = u32_dim(hidden, "GPT-OSS hidden stride");
        weight.source_col_offset = 0;
        weight.source_cols = u32_dim(hidden, "GPT-OSS hidden size");
        weight.target_cols = u32_dim(hidden, "GPT-OSS hidden size");
        b.push_repack(out_base + ".weight", block, mxfp4_encoding(b.contract(), 1),
                    {experts, intermediate_native, hidden}, weight);

        PieLoaderRepackSpecView scale_spec =
            repack_spec(PieLoaderRepackLayout::MarlinMxfp4Scale, half.row_map);
        scale_spec.batch = u32_dim(experts, "GPT-OSS experts");
        scale_spec.source_rows = u32_dim(fused_rows, "GPT-OSS gate/up source rows");
        scale_spec.source_row_offset =
            u32_dim(local_start, "GPT-OSS gate/up source row offset");
        scale_spec.target_rows = u32_dim(intermediate_native, "GPT-OSS gate/up target rows");
        scale_spec.valid_rows = u32_dim(local_intermediate, "GPT-OSS gate/up valid rows");
        scale_spec.source_stride_cols = u32_dim(groups, "GPT-OSS hidden group stride");
        scale_spec.source_col_offset = 0;
        scale_spec.source_cols = u32_dim(groups, "GPT-OSS hidden groups");
        scale_spec.target_cols = u32_dim(groups, "GPT-OSS hidden groups");
        auto scales = b.push_repack(out_base + ".weight_scale", scale,
                    pie_loader::raw(PieLoaderDType::U8),
                    {experts, intermediate_native, groups}, scale_spec);
        state_mxfp4_block_scales(scales, out_base + ".weight");

        PieLoaderRepackSpecView bias_spec =
            repack_spec(PieLoaderRepackLayout::DenseRowGather, half.row_map);
        bias_spec.batch = u32_dim(experts, "GPT-OSS experts");
        bias_spec.source_rows = u32_dim(fused_rows, "GPT-OSS gate/up bias rows");
        bias_spec.source_row_offset =
            u32_dim(local_start, "GPT-OSS gate/up bias source row offset");
        bias_spec.target_rows = u32_dim(local_intermediate, "GPT-OSS gate/up bias target rows");
        bias_spec.valid_rows = u32_dim(local_intermediate, "GPT-OSS gate/up bias valid rows");
        bias_spec.source_stride_cols = 1;
        bias_spec.source_col_offset = 0;
        bias_spec.source_cols = 1;
        bias_spec.target_cols = 1;
        b.push_repack(out_base + ".bias", bias, pie_loader::raw(PieLoaderDType::BF16),
                    {experts, local_intermediate}, bias_spec);
    }
}

inline void gpt_oss_native_down(ContractBuilder& b, const SourceTensor& block, const SourceTensor& scale,
                             const SourceTensor& bias, const std::string& base) {
    if (block.shape.size() != 4 || scale.shape.size() != 3 || bias.shape.size() != 2) {
        fail("GPT-OSS native down '" + base + "' has an unsupported block/scale/bias rank");
    }
    const std::int64_t experts = block.shape[0];
    const std::int64_t hidden = block.shape[1];
    const std::int64_t groups = block.shape[2];
    if (block.shape[3] != 16) {
        fail("GPT-OSS native down '" + base + "' expected [E, H, I/32, 16]");
    }
    if (scale.shape[0] != experts || scale.shape[1] != hidden || scale.shape[2] != groups ||
        bias.shape[0] != experts || bias.shape[1] != hidden) {
        fail("GPT-OSS native down '" + base + "' scale/bias shape mismatch");
    }
    const std::int64_t full_intermediate = groups * 32;
    const auto [local_start, local_intermediate] =
        b.local_range(full_intermediate, "the intermediate size of '" + base + "'");
    if (local_start % 32 != 0 || local_intermediate % 32 != 0) {
        fail("GPT-OSS native down '" + base +
             "' TP shard must align to MXFP4 32-wide groups");
    }
    const std::int64_t local_groups = local_intermediate / 32;
    const std::int64_t source_group_offset = local_start / 32;
    const std::int64_t intermediate_native = align_up(local_intermediate, 128);

    PieLoaderRepackSpecView weight =
        repack_spec(PieLoaderRepackLayout::MarlinMxfp4Weight, PieLoaderRowMap::Identity);
    weight.batch = u32_dim(experts, "GPT-OSS experts");
    weight.source_rows = u32_dim(hidden, "GPT-OSS down source rows");
    weight.source_row_offset = 0;
    weight.target_rows = u32_dim(hidden, "GPT-OSS down target rows");
    weight.valid_rows = u32_dim(hidden, "GPT-OSS down valid rows");
    weight.source_stride_cols = u32_dim(full_intermediate, "GPT-OSS down source stride");
    weight.source_col_offset = u32_dim(local_start, "GPT-OSS down source column offset");
    weight.source_cols = u32_dim(local_intermediate, "GPT-OSS intermediate size");
    weight.target_cols = u32_dim(intermediate_native, "GPT-OSS padded intermediate size");
    b.push_repack(base + ".weight", block, mxfp4_encoding(b.contract(), 2),
                {experts, hidden, intermediate_native}, weight);

    PieLoaderRepackSpecView scale_spec =
        repack_spec(PieLoaderRepackLayout::MarlinMxfp4Scale, PieLoaderRowMap::Identity);
    scale_spec.batch = u32_dim(experts, "GPT-OSS experts");
    scale_spec.source_rows = u32_dim(hidden, "GPT-OSS down source rows");
    scale_spec.source_row_offset = 0;
    scale_spec.target_rows = u32_dim(hidden, "GPT-OSS down target rows");
    scale_spec.valid_rows = u32_dim(hidden, "GPT-OSS down valid rows");
    scale_spec.source_stride_cols = u32_dim(groups, "GPT-OSS down source group stride");
    scale_spec.source_col_offset = u32_dim(source_group_offset, "GPT-OSS down source group offset");
    scale_spec.source_cols = u32_dim(local_groups, "GPT-OSS down source groups");
    scale_spec.target_cols = u32_dim(intermediate_native / 32, "GPT-OSS down target groups");
    auto scales = b.push_repack(base + ".weight_scale", scale, pie_loader::raw(PieLoaderDType::U8),
                {experts, hidden, intermediate_native / 32}, scale_spec);
    state_mxfp4_block_scales(scales, base + ".weight");

    b.push_direct(bias, base + ".bias", std::nullopt);
}

inline void gpt_oss_native_group(ContractBuilder& b, const SourceTensor& block, const SourceTensor& scale,
                              const SourceTensor& bias, const std::string& base) {
    if (ends_with(base, "gate_up_proj")) {
        gpt_oss_native_gate_up(b, block, scale, bias, base);
    } else if (ends_with(base, "down_proj")) {
        gpt_oss_native_down(b, block, scale, bias, base);
    } else {
        fail("GPT-OSS MXFP4 tensor '" + std::string(block.name) +
             "' is not gate_up_proj or down_proj");
    }
}

/// The same MXFP4 experts, declared as a group instead of a bank.
///
/// This is the other shape a checkpoint stores experts in, and the reason
/// `select` exists. Qwen's are one tensor per expert, so an instance is named
/// by an `index_src` template; GPT-OSS's are one tensor per *layer* with the
/// experts stacked along axis 0, so an instance is a band of a bank and the
/// index decides only where the band starts. Everything else about it -- shape,
/// dtype, destination -- is the same for every expert, which is exactly the
/// interchangeability a group claims, and here it is true by construction
/// rather than by inspection.
///
/// Weights and scales only. The biases are `E x 2I` and `E x H` of bf16, a few
/// hundred kilobytes a layer against tens of megabytes of weights, and the
/// bind de-interleaves the gate/up bias with a kernel -- which is host work a
/// group plan has no node for. Streaming them would buy nothing and cost a
/// transform the contract cannot express, so they stay resident.
///
/// Packed only. The native path Marlin-repacks into a layout whose rows are
/// permuted across the whole bank, so a single expert's repacked bytes are not
/// a contiguous band of the repacked bank; that needs its own instance-wise
/// repack spec and is a separate question from this one.
inline void gpt_oss_streamed_expert_groups(ContractBuilder& b) {
    const std::int64_t experts = b.facts().num_experts;
    if (experts <= 0) return;
    auto& c = b.contract();

    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string bound =
            "model.layers." + std::to_string(layer) + ".mlp.experts.";
        const std::string prefix = b.source_name(bound);

        auto group = c.group(bound.substr(0, bound.size() - 1),
                             static_cast<std::uint32_t>(experts));
        bool declared = false;
        for (const char* half : {"gate_up_proj", "down_proj"}) {
            const SourceTensor* block = b.find(prefix + half + "_blocks");
            const SourceTensor* scale = b.find(prefix + half + "_scales");
            if (block == nullptr || scale == nullptr) continue;
            if (block->shape.empty() || block->shape[0] != experts ||
                scale->shape.empty() || scale->shape[0] != experts) {
                fail(std::string("GPT-OSS expert group '") + half +
                     "' is not stacked over " + std::to_string(experts) +
                     " experts");
            }
            // One expert's band: `len` 1 along the expert axis, starting at
            // `index * 1`. The leading 1 stays, because a `Select` is a slice
            // and a slice keeps its rank -- and the bind reads a slot through
            // a view anyway, exactly as it read the bank through one.
            const auto band = [&](const SourceTensor& t) {
                std::vector<std::int64_t> shape(t.shape.begin(), t.shape.end());
                shape[0] = 1;
                return std::pair{c.select(c.src(std::string(t.name)), 0, 1, 1),
                                 shape};
            };
            const auto [block_node, block_shape] = band(*block);
            const auto [scale_node, scale_shape] = band(*scale);

            group.define(std::string(half) + ".weight", block_node,
                         pie_loader::raw(PieLoaderDType::U8))
                .expect(block_shape);
            std::optional<pie_loader::ModelContract::Defined> scales =
                group.define(std::string(half) + ".weight_scale", scale_node,
                             pie_loader::raw(PieLoaderDType::U8))
                    .expect(scale_shape);
            // The same pairing the resident path states, for the same reason:
            // the routed-dequant kernel reads the factors through
            // `quant_meta`, and a plain tensor leaves that empty.
            state_mxfp4_block_scales(scales, std::string(half) + ".weight");

            b.consume(block->id);
            b.consume(scale->id);
            declared = true;
        }
        if (!declared) continue;

        // The biases stay resident, under the names the bind already reads.
        for (const char* half : {"gate_up_proj", "down_proj"}) {
            if (const SourceTensor* bias = b.find(prefix + half + "_bias")) {
                b.push_direct(*bias, bound + half + ".bias", std::nullopt);
                b.consume(bias->id);
            }
        }
    }
}

/// Declare GPT-OSS's MXFP4 expert triplets the way this device wants them.
///
/// `_blocks`/`_scales`/`_bias` either pass through as three plain tensors for
/// the routed-decode path, or get Marlin-repacked into a native MXFP4 GEMM
/// layout. Which one it is is the driver's `Mxfp4MoePolicy`, resolved against
/// what the device measured — not a property of the checkpoint.
inline void gpt_oss_mxfp4_groups(ContractBuilder& b) {
    const bool native = b.mxfp4_moe() == Mxfp4MoePolicy::NativeGemm;
    if (native && !b.target().native_mxfp4_moe) {
        fail("GPT-OSS native MXFP4 requested, but target does not support native MXFP4 MoE");
    }
    if (!native && b.stream_routed_experts() && b.target().tp_size == 1) {
        gpt_oss_streamed_expert_groups(b);
        return;
    }
    for (const SourceTensor& raw : b.tensors()) {
        if (!ends_with(raw.name, "_blocks")) {
            continue;
        }
        const SourceTensor& block = raw;
        const std::string base =
            std::string(block.name.substr(0, block.name.size() - std::string_view("_blocks").size()));
        const SourceTensor* scale = b.find(base + "_scales");
        const SourceTensor* bias = b.find(base + "_bias");
        if (scale == nullptr || bias == nullptr) {
            continue;
        }
        if (native) {
            gpt_oss_native_group(b, block, *scale, *bias, base);
        } else {
            b.push_direct(block, base + ".weight", std::nullopt);
            auto scales = b.push_direct(*scale, base + ".weight_scale", std::nullopt);
            // The routed-dequant path reads these bytes through
            // `engine.quant_meta(...)->scale`, exactly as the native path does,
            // so it needs the same pairing stated. Publishing the scale as a
            // plain tensor leaves `quant_meta` empty and the bind fails with
            // "packed MXFP4 expert tensors are missing quant metadata".
            state_mxfp4_block_scales(scales, base + ".weight");
            b.push_direct(*bias, base + ".bias", std::nullopt);
        }
        b.consume(block.id);
        b.consume(scale->id);
        b.consume(bias->id);
    }
}

}  // namespace contract_detail

/// GPT-OSS. The dense QKV join is deliberately absent: this bind path reads
/// `q_proj`/`k_proj`/`v_proj` individually, so fusing them would consume the
/// three and leave the bind with a missing weight.
inline void author_gpt_oss_contract(ContractBuilder& b) {
    contract_detail::gpt_oss_mxfp4_groups(b);
    b.fused_moe_gate_up_tp_slices();
    b.publish_remaining();
}
}  // namespace pie_cuda_driver::model
