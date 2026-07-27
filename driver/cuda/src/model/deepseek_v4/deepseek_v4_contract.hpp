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

inline void author_deepseek_v4_contract(ContractBuilder& b) {
    b.shard_axis_fn(contract_detail::dsv4_shard_axis);
    author_dense_contract(b);
}
}  // namespace pie_cuda_driver::model
