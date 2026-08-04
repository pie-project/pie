#pragma once

/// What GLM-5.1 binds (`model/registry.cpp` row).
///
/// MLA attention plus a DSA indexer plus routed and shared MoE. Nothing about
/// the checkpoint's layout is unusual; what is unusual is that this is the one
/// family whose routed experts this driver will re-quantize at load time, and
/// the one that ships FP8 experts a runtime FP4 request is allowed to consume.

#include "model/contract.hpp"
#include "model/glm5/glm5.hpp"

namespace pie_cuda_driver::model {

namespace contract_detail {

/// Publish `kv_b_proj` in BF16, dequantizing the shipped FP8 in the loader.
///
/// The kimi_mla kernels this family shares (`launch_kimi_q_nope_to_latent_bf16`,
/// `launch_kimi_latent_to_v_bf16`) read `kv_b_proj` as BF16 and there is no
/// FP8 variant, so *something* has to dequantize it. `bind_glm5` used to, into
/// a `std::unique_ptr<DeviceTensor>` beside the loaded weight -- which meant
/// the FP8 original stayed resident forever holding bytes nothing reads, and
/// on a BF16 checkpoint the bind copied the tensor to itself so the pointer
/// type would be uniform. Stated here, the packed source is consumed, the
/// arena accounts for exactly one tensor, and a BF16 checkpoint needs no pass
/// at all because the generic dense path already publishes it.
///
/// **The scale's shape is what says how the weight is blocked.** GLM-5.1 ships
/// a DeepSeek-style square block (one factor per 128x128 tile); other exports
/// of the same architecture ship one factor per output channel. Both are the
/// same term here, because `scale_per_block` reads the blocking off the ratio
/// of the two shapes rather than taking a group size on the side. A rank-1
/// per-channel vector is transmuted to `[rows, 1]` first: that is the same
/// bytes under the rank the pairing needs, and it makes "one factor per row"
/// the block `[1, cols]` instead of a second case.
inline void glm5_bf16_kv_b_proj(ContractBuilder& b) {
    for (const SourceTensor& raw : b.tensors()) {
        if (!ends_with(raw.name, ".self_attn.kv_b_proj.weight")) {
            continue;
        }
        // A BF16 checkpoint is already what the kernel wants; leaving it to the
        // generic path is what deletes the copy the bind used to make.
        if (!is_raw(raw.encoding, PieLoaderDType::F8E4M3)) {
            continue;
        }
        const std::string weight_name(raw.name);
        const SourceTensor* factors = nullptr;
        for (const char* suffix : {"_scale_inv", "_scale"}) {
            factors = b.find(weight_name + suffix);
            if (factors != nullptr) {
                break;
            }
        }
        if (factors == nullptr) {
            // A bare `.scale` shares the base rather than hanging off
            // `.weight`, so the six characters of ".weight" come off first.
            factors = b.find(weight_name.substr(0, weight_name.size() - 6) + "scale");
        }
        const std::vector<std::int64_t> weight_shape = shape_of(raw);
        if (factors == nullptr || weight_shape.size() != 2) {
            continue;
        }
        std::vector<std::int64_t> factor_shape = shape_of(*factors);
        const PieLoaderEncodingSpec f32 = pie_loader::raw(PieLoaderDType::F32);
        if (!is_raw(factors->encoding, PieLoaderDType::F32) || factor_shape.empty()) {
            continue;
        }

        const ShardAxis axis = b.shard_axis(raw.name);
        Node factor_expr = b.contract().src(std::string(factors->name));
        if (factor_shape.size() == 1) {
            factor_shape = {factor_shape[0], 1};
            factor_expr = b.contract().transmute(factor_expr, factor_shape, f32);
        }
        if (factor_shape.size() != 2) {
            continue;
        }
        // Both sides shard on the same axis, which for a row-parallel weight is
        // the rows. The loader is what checks the two shards still line up: a
        // rank whose row band is not a whole number of blocks makes the pairing
        // fail with both shapes named, rather than silently reading a factor
        // that describes another rank's rows.
        auto [scale_local, scale_shape] = b.shard(factor_expr, factor_shape, axis);
        const std::string scale_name = b.output_name(factors->name);
        if (auto declared = b.define(scale_name, scale_local, f32, scale_shape)) {
            declared->internal();
        }

        auto [packed, local_shape] = b.shard(
            b.contract().transmute(
                b.contract().src(weight_name), weight_shape,
                pie_loader::quantized(pie_loader::quant_spec(PieLoaderQuantScheme::Fp8E4M3,
                                                             PieLoaderDType::BF16))),
            weight_shape, axis);
        b.define(b.output_name(weight_name),
                 b.contract().scale_per_block(packed, b.contract().out(scale_name)),
                 pie_loader::raw(PieLoaderDType::BF16), local_shape);
        b.consume(raw.id);
        b.consume(factors->id);
    }
}

}  // namespace contract_detail

/// glm_moe_dsa. `embed_tokens` is sharded on axis 0 under TP to save per-rank
/// memory; the FP4 path touches routed and shared experts only, because there
/// is no FP4 GEMM for the attention projections on this hardware.
inline void author_glm5_contract(ContractBuilder& b) {
    b.shard_embed_tokens();
    // Before the runtime-quant flags: this consumes the FP8 pair outright, so
    // there is no `kv_b_proj` left for a bf16 re-quant request to claim.
    contract_detail::glm5_bf16_kv_b_proj(b);
    b.allow_bf16_runtime_quant();
    b.allow_mxfp4_runtime_quant();
    // GLM-5.2 ships routed experts one tensor per expert; glm5_forward reads
    // the fused 3-D slabs. Float only: this family's quantised checkpoints
    // keep the per-expert layout and take the per-expert forward path.
    //
    // `gate_second` publishes each expert's halves as `[up | gate]`, which is
    // what flashinfer's CUTLASS grouped GEMM reads fc1 as. Stating it here is
    // the whole point: the alternative is a driver-side block swap over the
    // largest tensor in the model, done after the loader has already placed it.
    contract_detail::hf_moe_expert_stacks(b, glm5_moe_gate_up_swapped(),
                                          /*float_only=*/true);
    author_dense_contract(b);
}

}  // namespace pie_cuda_driver::model
