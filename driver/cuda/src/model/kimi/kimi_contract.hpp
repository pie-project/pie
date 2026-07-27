#pragma once

/// What the MLA families bind (`model/registry.cpp` rows).
///
/// DeepSeek-V2/V3 and Kimi-K2 share a binder and a forward. They differ in the
/// contract: Kimi hides the decoder under `language_model.` and wants
/// `embed_tokens` sharded and `lm_head` replicated, which is a memory trade
/// this driver makes and the checkpoint knows nothing about.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

/// Fuse the two MLA projection pairs that share an input.
inline void mla_fused_projection_joins(ContractBuilder& b) {
    std::vector<ContractBuilder::FusedCandidate> candidates;
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        // q_a_proj + kv_a_proj_with_mqa share an input (norm_x, unsharded).
        if (auto candidate = b.fused_join_candidate(
                p + "self_attn.q_kv_a_proj.fused.weight",
                {p + "self_attn.q_a_proj.weight", p + "self_attn.kv_a_proj_with_mqa.weight"})) {
            candidates.push_back(std::move(*candidate));
        }
        // Shared gate + up share an input (norm_y).
        if (auto candidate = b.fused_join_candidate(
                p + "mlp.shared_experts.gate_up_proj.fused.weight",
                {p + "mlp.shared_experts.gate_proj.weight",
                 p + "mlp.shared_experts.up_proj.weight"})) {
            candidates.push_back(std::move(*candidate));
        }
    }
    b.publish_fused(candidates);
}

}  // namespace contract_detail

/// deepseek_v2, deepseek_v3.
inline void author_deepseek_mla_contract(ContractBuilder& b) {
    b.fused_moe_gate_up_tp_slices();
    b.dense_fused_projection_joins();
    contract_detail::mla_fused_projection_joins(b);
    b.publish_remaining();
}

/// kimi_k2. Keeping `lm_head` whole costs ~1.7 GB a rank and buys not needing
/// a TP greedy argmax on the logits path.
inline void author_kimi_contract(ContractBuilder& b) {
    b.source_prefix("language_model.");
    b.shard_embed_tokens();
    b.replicate_lm_head();
    author_deepseek_mla_contract(b);
}
}  // namespace pie_cuda_driver::model
