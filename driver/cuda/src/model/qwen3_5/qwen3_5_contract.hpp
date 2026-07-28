#pragma once

/// What the Qwen3.5 hybrid families bind (`model/registry.cpp` rows).
///
/// The dense hybrid needs nothing beyond the generic rules. The MoE hybrid
/// needs one thing: plain Qwen3-MoE checkpoints ship experts one tensor per
/// expert, and `qwen3_5_moe_forward` reads a single fused 3-D slab.

#include "model/contract.hpp"

#include "model/qwen3_5/qwen3_5_moe.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {


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
        const std::string la =
            "model.layers." + std::to_string(layer) + ".linear_attn.";
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
            const SourceTensor* raw = b.find(b.source_name(la + leaf));
            if (raw == nullptr || raw->shape.empty() || raw->shape[0] != conv_dim) {
                continue;
            }
            const Node src = b.contract().src(std::string(raw->name));
            const auto key_lo = b.band(src, 0, 0, k_dim);
            const auto key_hi = b.band(src, 0, k_dim, k_dim);
            const auto value = b.band(src, 0, 2 * k_dim, v_dim);
            std::vector<std::int64_t> shape = contract_detail::shape_of(*raw);
            shape[0] = 2 * key_lo.second + value.second;
            b.define(b.output_name(raw->name),
                     b.contract().cat({key_lo.first, key_hi.first, value.first}, 0),
                     raw->encoding, std::move(shape));
            b.consume(raw->id);
        }
    }
}

}  // namespace contract_detail

/// qwen3_5, qwen3_5_text: a dense hybrid decoder under the usual names.
inline void author_qwen3_5_contract(ContractBuilder& b) {
    b.allow_bf16_runtime_quant();
    contract_detail::gdn_kkv_blocked_shards(b);
    author_dense_contract(b);
}

/// qwen3_moe, qwen3_5_moe, qwen3_5_moe_text. No dense QKV join: this bind path
/// reads q/k/v separately.
inline void author_qwen3_5_moe_contract(ContractBuilder& b) {
    b.allow_bf16_runtime_quant();
    contract_detail::gdn_kkv_blocked_shards(b);
    // The MoE decode runs through flashinfer's CUTLASS grouped GEMM, which
    // reads fc1's output as [linear|gate]; the checkpoint stores [gate|up].
    // Both the pre-fused and the per-expert stacking paths publish in the
    // order the bound driver expects.
    const bool gate_second = qwen35_moe_gate_up_swapped();
    b.fused_moe_gate_up_tp_slices(gate_second);
    contract_detail::hf_moe_expert_stacks(b, gate_second);
    b.publish_remaining();
}
}  // namespace pie_cuda_driver::model
