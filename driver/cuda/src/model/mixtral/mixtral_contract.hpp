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
    const std::int64_t local_intermediate = b.local_extent(full_intermediate);
    const std::int64_t intermediate_native = align_up(local_intermediate, 128);
    const std::string prefix =
        base.substr(0, base.size() - std::string_view("gate_up_proj").size());

    // gate and up are interleaved row by row, so this rank's share of the
    // *logical* intermediate axis is `split` on the fused axis -- which lands on
    // `[2 * local_start, 2 * (local_start + local_intermediate))` -- followed by
    // the even or odd rows of that band. Both are ordinary nodes, so the rank
    // stays where every other family puts it: in the target, resolved by the
    // loader, never written into the contract.
    const struct {
        std::string_view name;
        std::int64_t first_row;
    } halves[] = {{"gate_proj", 0}, {"up_proj", 1}};
    for (const auto& half : halves) {
        const std::string out_base = prefix + std::string(half.name);
        const auto rows = [&](const SourceTensor& raw) {
            return b.contract().stride(b.split(b.contract().src(std::string(raw.name)), 1), 1,
                                       half.first_row, local_intermediate, 2);
        };

        b.push_repack(out_base + ".weight", rows(block), PieLoaderRepackLayout::MarlinMxfp4Weight,
                      mxfp4_encoding(b.contract(), 1),
                      {experts, intermediate_native, hidden});

        auto scales = b.push_repack(out_base + ".weight_scale", rows(scale),
                                    PieLoaderRepackLayout::MarlinMxfp4Scale,
                                    pie_loader::raw(PieLoaderDType::U8),
                                    {experts, intermediate_native, groups});
        state_mxfp4_block_scales(scales, out_base + ".weight");

        // The bias needed a `DenseRowGather` kernel only because the algebra
        // could not say "every other row of this rank's band". It can, so this
        // is a copy.
        b.push_expr(out_base + ".bias", bias, {experts, local_intermediate}, rows(bias));
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
    const std::int64_t local_intermediate = b.local_extent(groups * 32);
    if (local_intermediate % 32 != 0) {
        fail("GPT-OSS native down '" + base +
             "' TP shard must align to MXFP4 32-wide groups");
    }
    const std::int64_t intermediate_native = align_up(local_intermediate, 128);

    // Down is sharded along K, which is the packed axis: one group is 32
    // elements, so this rank's column band is a `split` of the *group* axis and
    // the offset never has to be spelled in elements.
    b.push_repack(base + ".weight", b.split(b.contract().src(std::string(block.name)), 2),
                  PieLoaderRepackLayout::MarlinMxfp4Weight, mxfp4_encoding(b.contract(), 2),
                  {experts, hidden, intermediate_native});

    auto scales = b.push_repack(base + ".weight_scale",
                                b.split(b.contract().src(std::string(scale.name)), 2),
                                PieLoaderRepackLayout::MarlinMxfp4Scale,
                                pie_loader::raw(PieLoaderDType::U8),
                                {experts, hidden, intermediate_native / 32});
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
