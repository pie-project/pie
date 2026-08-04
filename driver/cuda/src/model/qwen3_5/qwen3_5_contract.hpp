#pragma once

/// What the Qwen3.5 hybrid families bind (`model/registry.cpp` rows).
///
/// The dense hybrid needs nothing beyond the generic rules. The MoE hybrid
/// needs one thing: plain Qwen3-MoE checkpoints ship experts one tensor per
/// expert, and `qwen3_5_moe_forward` reads a single fused 3-D slab.

#include "model/contract.hpp"

#include <string_view>

#include "model/qwen3_5/qwen3_5_moe.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {


/// This rank's `[K/T | K/T | V/T]` view of a `[2K + V, ...]` tensor.
///
/// Returned with its declared shape, because the extent depends on whether the
/// world divides each block and only [`ContractBuilder::local_extent`] knows.
inline std::pair<Node, std::vector<std::int64_t>> gdn_kkv_blocked(
    ContractBuilder& b, const SourceTensor& raw, std::int64_t k_dim,
    std::int64_t v_dim) {
    const Node src = b.contract().src(std::string(raw.name));
    const auto key_lo = b.band(src, 0, 0, k_dim);
    const auto key_hi = b.band(src, 0, k_dim, k_dim);
    const auto value = b.band(src, 0, 2 * k_dim, v_dim);
    std::vector<std::int64_t> shape = contract_detail::shape_of(raw);
    shape[0] = 2 * key_lo.second + value.second;
    return {b.contract().concat({key_lo.first, key_hi.first, value.first}, 0),
            std::move(shape)};
}

/// Shard the Gated DeltaNet tensors whose leading axis stacks `[K | K | V]`.
///
/// `linear_attn.in_proj_qkv.weight`, `conv1d.weight` and `conv1d.bias` all
/// stack two key blocks and one value block on axis 0, so a uniform row shard
/// cuts across the block boundaries and hands a rank part of K where it needs
/// V. Take each block's band and join them: every rank gets its own
/// `[K/T | K/T | V/T]`, which is what the GDN kernels address.
///
/// Without this the loader has no shard axis for these names and leaves them
/// replicated, so *every rank loads the whole tensor* and the driver slices it
/// afterwards with device-to-device copies.
///
/// `K` and `V` come from the checkpoint, not from a config field: `in_proj_z`
/// is `[V, hidden]`, and `in_proj_qkv` is `[2K + V, hidden]`, so the pair
/// determines both.
inline void gdn_kkv_blocked_shards(ContractBuilder& b) {
    if (b.target().tp_size <= 1) {
        return;
    }
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string la = std::string(b.decoder_layer_prefix()) +
                               std::to_string(layer) + ".linear_attn.";
        const SourceTensor* qkv = b.find(b.source_name(la + "in_proj_qkv.weight"));
        const SourceTensor* z = b.find(b.source_name(la + "in_proj_z.weight"));
        if (qkv == nullptr || z == nullptr || qkv->shape.empty() || z->shape.empty()) {
            continue;
        }
        const std::int64_t v_dim = z->shape[0];
        const std::int64_t conv_dim = qkv->shape[0];
        if (conv_dim <= v_dim || (conv_dim - v_dim) % 2 != 0) {
            continue;
        }
        const std::int64_t k_dim = (conv_dim - v_dim) / 2;
        for (const char* leaf :
             {"in_proj_qkv.weight", "conv1d.weight", "conv1d.bias"}) {
            // The fused layout publishes qkv as the first leg of `in_proj_qkvz`
            // instead; conv1d still wants it under its own name either way.
            if (qwen35_fused_gdn_projection_enabled() &&
                std::string_view(leaf) == "in_proj_qkv.weight") {
                continue;
            }
            const SourceTensor* raw = b.find(b.source_name(la + leaf));
            if (raw == nullptr || raw->shape.empty() || raw->shape[0] != conv_dim) {
                continue;
            }
            auto blocked = gdn_kkv_blocked(b, *raw, k_dim, v_dim);
            b.define(b.output_name(raw->name), blocked.first, raw->encoding,
                     std::move(blocked.second));
            b.consume(raw->id);
        }
    }
}

/// Join the Gated DeltaNet input projections the forward reads pre-fused.
///
/// `qwen3_5_forward` branches on `la_in_proj_qkvz`: when the fused layout is
/// selected it wants `[2K | V | V]` (qkv then z) and `[H | H]` (b then a) as
/// single tensors. Stating that here rather than concatenating after the load
/// is what makes the layout free — the earlier bind-time version had to hold
/// the sources *and* the join, which on Qwen3.6-35B-A3B meant 1.4 GB of
/// duplicate weights, since arena-backed sources reclaim nothing when erased.
///
/// The qkv leg keeps the per-block shard from [`gdn_kkv_blocked_shards`]; z, b
/// and a shard uniformly on axis 0, so a plain [`ContractBuilder::split`] is
/// right for them.
inline void gdn_fused_in_proj_joins(ContractBuilder& b) {
    if (!qwen35_fused_gdn_projection_enabled()) {
        return;
    }
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string la = std::string(b.decoder_layer_prefix()) +
                               std::to_string(layer) + ".linear_attn.";
        const SourceTensor* qkv = b.find(b.source_name(la + "in_proj_qkv.weight"));
        const SourceTensor* z = b.find(b.source_name(la + "in_proj_z.weight"));
        const SourceTensor* bb = b.find(b.source_name(la + "in_proj_b.weight"));
        const SourceTensor* aa = b.find(b.source_name(la + "in_proj_a.weight"));
        if (qkv == nullptr || z == nullptr || bb == nullptr || aa == nullptr) {
            continue;
        }
        if (qkv->shape.empty() || z->shape.empty() || bb->shape.empty() ||
            aa->shape.empty()) {
            continue;
        }
        const std::int64_t v_dim = z->shape[0];
        const std::int64_t conv_dim = qkv->shape[0];
        if (conv_dim <= v_dim || (conv_dim - v_dim) % 2 != 0) {
            continue;
        }
        const std::int64_t k_dim = (conv_dim - v_dim) / 2;

        auto blocked = gdn_kkv_blocked(b, *qkv, k_dim, v_dim);
        const Node z_local = b.split(b.contract().src(std::string(z->name)), 0);
        std::vector<std::int64_t> qkvz_shape = blocked.second;
        qkvz_shape[0] += b.local_extent(v_dim);
        b.define(b.output_name(la + "in_proj_qkvz.weight"),
                 b.contract().concat({blocked.first, z_local}, 0), qkv->encoding,
                 std::move(qkvz_shape));
        b.consume(qkv->id);
        b.consume(z->id);

        const Node b_local = b.split(b.contract().src(std::string(bb->name)), 0);
        const Node a_local = b.split(b.contract().src(std::string(aa->name)), 0);
        std::vector<std::int64_t> ba_shape = contract_detail::shape_of(*bb);
        ba_shape[0] = b.local_extent(bb->shape[0]) + b.local_extent(aa->shape[0]);
        b.define(b.output_name(la + "in_proj_ba.weight"),
                 b.contract().concat({b_local, a_local}, 0), bb->encoding,
                 std::move(ba_shape));
        b.consume(bb->id);
        b.consume(aa->id);
    }
}

/// Widen the two gated-delta-net parameters the kernels read as fp32.
///
/// `A_log` and the gated RMSNorm's weight enter `launch_*_gated_delta_net` as
/// `const float*`, but HF ships them fp32 on Qwen3.5-4B and **bf16** on
/// Qwen3.6-35B-A3B. The driver used to reconcile that at bind, by copying each
/// one to a fresh `DeviceBuffer<float>` held in `owned_fp32_buffers` -- a
/// conversion the contract never heard about, so the bf16 original stayed
/// resident beside its fp32 copy for the life of the model.
///
/// Only these two. `dt_bias` sits beside them in the same module and is read as
/// bf16, so a suffix match any looser than this list would silently widen it.
///
/// The `already fp32` branch is not an optimization; it is required. A `Cast`
/// to the encoding its operand already has is refused (a node may not denote
/// exactly its operand), which is what makes the two checkpoint conventions
/// impossible to paper over with one unconditional cast.
inline void gdn_fp32_parameters(ContractBuilder& b) {
    for (const SourceTensor& raw : b.tensors()) {
        if (!ends_with_any(raw.name, {".linear_attn.A_log", ".linear_attn.norm.weight"})) {
            continue;
        }
        const bool bf16 = is_raw(raw.encoding, PieLoaderDType::BF16);
        if (!bf16 && !is_raw(raw.encoding, PieLoaderDType::F32)) {
            continue;
        }
        auto [expr, local] = b.shard(b.contract().src(std::string(raw.name)),
                                     shape_of(raw), b.shard_axis(raw.name));
        const PieLoaderEncodingSpec f32 = pie_loader::raw(PieLoaderDType::F32);
        b.define(b.output_name(raw.name), bf16 ? b.contract().cast(expr, f32) : expr, f32,
                 std::move(local));
        b.consume(raw.id);
    }
}

}  // namespace contract_detail

/// Publish an int8 view of `lm_head` for the speculative head to read.
///
/// The draft step and the main path read the *same* head, at different
/// precisions -- `bind_qwen3_5` sets `mtp.lm_head = w.lm_head`. So this is not
/// a re-encode: both views are published, and `quantized_view` leaves the bf16
/// original alone.
///
/// The bind used to do it, by allocating an `[V, H]` INT8 tensor plus a scale
/// vector into `owned_int8_buffers`/`owned_scale_buffers` and running
/// `quantize_bf16_to_int8_per_channel` over the loaded weight. Stating it here
/// gets the same bytes from the loader's transcode path, which already has an
/// `Int8Symmetric` encoder, and puts the second copy in the arena's accounting
/// instead of outside it.
///
/// A tied checkpoint has no `lm_head.weight`; the head is `embed_tokens`, which
/// is what `bind_qwen3_5` resolves to and therefore what gets quantized.
inline void mtp_int8_lm_head(ContractBuilder& b) {
    if (!qwen35_mtp_int8_lm_head_enabled() || b.find("mtp.fc.weight") == nullptr) {
        return;
    }
    // Tied checkpoints omit `lm_head.weight` and alias the head to
    // `embed_tokens`, which is what `bind_qwen3_5` resolves to; the decoder
    // prefix varies (the VL checkpoints nest it), so match the suffix.
    const SourceTensor* head = b.find("lm_head.weight");
    if (head == nullptr) {
        for (const SourceTensor& raw : b.tensors()) {
            if (contract_detail::ends_with(raw.name, ".embed_tokens.weight")) {
                head = &raw;
                break;
            }
        }
    }
    // Only a bf16 source. `quantize_bf16_to_int8_per_channel` was the only
    // converter the bind had, and a checkpoint that already ships a quantized
    // head wants that head, not a second encoding of it.
    if (head == nullptr || !contract_detail::is_raw(head->encoding, PieLoaderDType::BF16)) {
        return;
    }
    b.quantized_view(std::string(head->name), "mtp.lm_head", PieLoaderQuantScheme::Int8Symmetric);
}

namespace contract_detail {

/// Join the shared expert's gate and up projections the MoE forward reads
/// pre-fused, and optionally the scalar gate row after them.
///
/// `qwen3_5_moe_forward` picks one of three shapes for the shared expert:
/// `[2*Is+1, H]` when the scalar gate is folded in, `[2*Is, H]` when it is not,
/// and two separate GEMMs when neither fused tensor was bound. The bind used to
/// build whichever of the first two applied with a pair of device-to-device
/// copies; it is a `concat` on axis 0, which is what this says.
///
/// The sources are **not** consumed. Unlike the Gated DeltaNet join, both
/// unfused projections stay live: the fold-into-routed path
/// (`try_fold_shared_into_routed`) reads them separately, and which path runs
/// is a per-step decision. So this slab is additive, exactly like the Kimi and
/// DSv4 expert stacks.
///
/// Only bf16 sources, because the bind had exactly one converter and a
/// checkpoint that ships this pair quantized wants the quantized kernels rather
/// than a second dequantized copy. The scalar gate is replicated, not sharded,
/// so it is named directly while gate and up take the column-parallel split.
inline void shared_expert_gate_up_join(ContractBuilder& b, const std::string& layer_prefix) {
    const std::string lp = layer_prefix + "mlp.shared_expert";
    const SourceTensor* gate = b.find(b.source_name(lp + ".gate_proj.weight"));
    const SourceTensor* up = b.find(b.source_name(lp + ".up_proj.weight"));
    if (gate == nullptr || up == nullptr) return;
    if (!contract_detail::is_raw(gate->encoding, PieLoaderDType::BF16) ||
        !contract_detail::is_raw(up->encoding, PieLoaderDType::BF16)) {
        return;
    }
    if (gate->shape.size() != 2 || up->shape.size() != 2 ||
        gate->shape[1] != up->shape[1]) {
        return;
    }

    const Node gate_local = b.split(b.contract().src(std::string(gate->name)), 0);
    const Node up_local = b.split(b.contract().src(std::string(up->name)), 0);
    const std::int64_t rows =
        b.local_extent(gate->shape[0]) + b.local_extent(up->shape[0]);

    const SourceTensor* scalar =
        qwen35_fused_shared_scalar_gate_enabled()
            ? b.find(b.source_name(lp + "_gate.weight"))
            : nullptr;
    if (scalar != nullptr &&
        (!contract_detail::is_raw(scalar->encoding, PieLoaderDType::BF16) ||
         scalar->shape.size() != 2 || scalar->shape[1] != gate->shape[1])) {
        scalar = nullptr;
    }

    if (scalar != nullptr) {
        b.define(b.output_name(lp + ".gate_up_gate_proj.weight"),
                 b.contract().concat({gate_local, up_local,
                                      b.contract().src(std::string(scalar->name))},
                                     0),
                 gate->encoding,
                 std::vector<std::int64_t>{rows + scalar->shape[0], gate->shape[1]});
    } else {
        b.define(b.output_name(lp + ".gate_up_proj.weight"),
                 b.contract().concat({gate_local, up_local}, 0), gate->encoding,
                 std::vector<std::int64_t>{rows, gate->shape[1]});
    }

}

/// Every module that carries a shared expert: the decoder layers and, when the
/// checkpoint ships one, the speculative-decoding block. The MTP layer is not
/// under `decoder_layer_prefix()`, and its bind runs the same fusion.
inline void shared_expert_gate_up_joins(ContractBuilder& b) {
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        shared_expert_gate_up_join(
            b, std::string(b.decoder_layer_prefix()) + std::to_string(layer) + ".");
    }
    shared_expert_gate_up_join(b, "mtp.layers.0.");
}

}  // namespace contract_detail

/// qwen3_5, qwen3_5_text: a dense hybrid decoder under the usual names.
inline void author_qwen3_5_contract(ContractBuilder& b) {
    b.allow_bf16_runtime_quant();
    // The vision-language checkpoints nest the decoder; the text-only ones do
    // not. Both are this row, so the prefix is asked for rather than declared.
    b.decoder_layer_prefix_any_of({"model.language_model.layers.", "model.layers."});
    contract_detail::gdn_kkv_blocked_shards(b);
    contract_detail::gdn_fused_in_proj_joins(b);
    contract_detail::gdn_fp32_parameters(b);
    // The speculative-decoding head is a full-attention layer with the same
    // projection names, so it wants the same join. Checkpoints without one make
    // this a no-op.
    b.also_join_module("mtp.layers.0.");
    mtp_int8_lm_head(b);
    author_dense_contract(b);
}

/// qwen3_moe, qwen3_5_moe, qwen3_5_moe_text.
///
/// Deliberately no `dense_fused_projection_joins`: this bind path reads q/k/v
/// separately, and the MLP lives in the experts, so there is no layer-level
/// `gate_proj`/`up_proj` pair to join.
inline void author_qwen3_5_moe_contract(ContractBuilder& b) {
    b.allow_bf16_runtime_quant();
    b.decoder_layer_prefix_any_of({"model.language_model.layers.", "model.layers."});
    contract_detail::gdn_kkv_blocked_shards(b);
    contract_detail::gdn_fused_in_proj_joins(b);
    contract_detail::gdn_fp32_parameters(b);
    // The MoE decode runs through flashinfer's CUTLASS grouped GEMM, which
    // reads fc1's output as [linear|gate]; the checkpoint stores [gate|up].
    // Both the pre-fused and the per-expert stacking paths publish in the
    // order the bound driver expects.
    const bool gate_second = qwen35_moe_gate_up_swapped();
    b.fused_moe_gate_up_tp_slices(gate_second);
    contract_detail::shared_expert_gate_up_joins(b);
    contract_detail::hf_moe_expert_stacks(b, gate_second);
    b.publish_remaining();
}
}  // namespace pie_cuda_driver::model
