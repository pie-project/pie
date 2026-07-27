#pragma once

/// What the Qwen3.5 hybrid families bind (`model/registry.cpp` rows).
///
/// The dense hybrid needs nothing beyond the generic rules. The MoE hybrid
/// needs one thing: plain Qwen3-MoE checkpoints ship experts one tensor per
/// expert, and `qwen3_5_moe_forward` reads a single fused 3-D slab.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

/// Stack per-expert MoE weights into the fused 3-D tensors the qwen3_5_moe
/// forward consumes. The HF Qwen3-MoE source layout is, per expert `e`,
/// `mlp.experts.{e}.gate_proj.weight` / `.up_proj.weight` as `[I, H]` and
/// `.down_proj.weight` as `[H, I]`. Output:
///   `mlp.experts.gate_up_proj` -> `[E, 2I, H]`, rows `[0,I)` gate, `[I,2I)` up
///   `mlp.experts.down_proj`    -> `[E, H, I]`
/// Built as an expression over the sources, so no GPU-side duplicate exists.
/// A no-op when the checkpoint already ships the fused tensors.
inline void qwen_moe_expert_stacks(ContractBuilder& b) {
    if (b.target().tp_size != 1) {
        // TP>1 per-expert sharding is not wired; the bind then fails loudly
        // on the missing fused tensor rather than loading silently wrong.
        return;
    }
    const std::int64_t num_experts = b.facts().num_experts;
    if (num_experts <= 0) {
        return;
    }
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string prefix =
            "model.layers." + std::to_string(layer) + ".mlp.experts.";
        if (b.find(prefix + "gate_up_proj") != nullptr) {
            continue;  // already pre-fused; the direct and TP-slice paths take it.
        }
        const SourceTensor* gate0 = b.find(prefix + "0.gate_proj.weight");
        if (gate0 == nullptr) {
            continue;  // not a per-expert checkpoint at this layer.
        }
        if (gate0->shape.size() != 2) {
            fail("qwen moe expert stack: '" + std::string(gate0->name) + "' expected 2-D");
        }
        const std::int64_t inter = gate0->shape[0];
        const std::int64_t hidden = gate0->shape[1];
        const PieLoaderDType dtype = logical_dtype(gate0->encoding);
        if (!is_dense_addressable(gate0->encoding)) {
            fail("qwen moe expert stack: '" + std::string(gate0->name) +
                 "' has a non-affine packed encoding");
        }

        std::vector<Node> gate_up_parts;
        std::vector<Node> down_parts;
        std::vector<std::uint32_t> consumed;
        gate_up_parts.reserve(static_cast<std::size_t>(num_experts));
        down_parts.reserve(static_cast<std::size_t>(num_experts));
        for (std::int64_t e = 0; e < num_experts; ++e) {
            const std::string tag = prefix + std::to_string(e) + ".";
            const SourceTensor* g = b.find(tag + "gate_proj.weight");
            const SourceTensor* u = b.find(tag + "up_proj.weight");
            const SourceTensor* d = b.find(tag + "down_proj.weight");
            if (g == nullptr || u == nullptr || d == nullptr) {
                fail("qwen moe expert stack: layer " + std::to_string(layer) + " expert " +
                     std::to_string(e) + " missing gate/up/down");
            }
            const bool shapes_ok = g->shape.size() == 2 && u->shape.size() == 2 &&
                                   d->shape.size() == 2 && g->shape[0] == inter &&
                                   g->shape[1] == hidden && u->shape[0] == inter &&
                                   u->shape[1] == hidden && d->shape[0] == hidden &&
                                   d->shape[1] == inter;
            if (!shapes_ok) {
                fail("qwen moe expert stack: layer " + std::to_string(layer) + " expert " +
                     std::to_string(e) + " shape mismatch");
            }
            if (logical_dtype(g->encoding) != dtype || logical_dtype(u->encoding) != dtype ||
                logical_dtype(d->encoding) != dtype) {
                fail("qwen moe expert stack: layer " + std::to_string(layer) + " expert " +
                     std::to_string(e) + " dtype mismatch");
            }
            // One expert slab is gate over up; the leading 1 makes the
            // per-expert concatenation a stack.
            gate_up_parts.push_back(b.contract().reshape(
                b.contract().cat({b.contract().src(std::string(g->name)),
                                b.contract().src(std::string(u->name))},
                               0),
                {1, 2 * inter, hidden}));
            down_parts.push_back(b.contract().reshape(
                b.contract().src(std::string(d->name)), {1, hidden, inter}));
            consumed.push_back(g->id);
            consumed.push_back(u->id);
            consumed.push_back(d->id);
        }

        b.define(prefix + "gate_up_proj", b.contract().cat(gate_up_parts, 0),
               pie_loader::raw(dtype),
               std::vector<std::int64_t>{num_experts, 2 * inter, hidden});
        b.define(prefix + "down_proj", b.contract().cat(down_parts, 0), pie_loader::raw(dtype),
               std::vector<std::int64_t>{num_experts, hidden, inter});
        for (std::uint32_t id : consumed) {
            b.consume(id);
        }
    }
}

}  // namespace contract_detail

/// qwen3_5, qwen3_5_text: a dense hybrid decoder under the usual names.
inline void author_qwen3_5_contract(ContractBuilder& b) {
    b.allow_bf16_runtime_quant();
    author_dense_contract(b);
}

/// qwen3_moe, qwen3_5_moe, qwen3_5_moe_text. No dense QKV join: this bind path
/// reads q/k/v separately.
inline void author_qwen3_5_moe_contract(ContractBuilder& b) {
    b.allow_bf16_runtime_quant();
    b.fused_moe_gate_up_tp_slices();
    contract_detail::qwen_moe_expert_stacks(b);
    b.publish_remaining();
}
}  // namespace pie_cuda_driver::model
