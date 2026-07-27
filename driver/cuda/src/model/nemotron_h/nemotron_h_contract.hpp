#pragma once

/// What Nemotron-H binds (`model/registry.cpp` row).
///
/// The Mamba2/attention/MoE hybrid keeps its decoder under
/// `language_model.backbone.`, and its MoE GEMM addresses all experts of a
/// layer as one contiguous slab. The contract declares the slab and then
/// declares each expert as a slice of it, so no byte is copied twice.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

inline void nemotron_h_layer_packed_experts(ContractBuilder& b, const std::string& base,
                                         const std::vector<const SourceTensor*>& up,
                                         const std::vector<const SourceTensor*>& down) {
    if (up.empty() || down.empty()) {
        return;
    }
    const SourceTensor& first_up = *up.front();
    const SourceTensor& first_down = *down.front();
    if (first_up.shape.size() != 2 || first_down.shape.size() != 2 ||
        !is_raw(first_up.encoding, PieLoaderDType::BF16) ||
        !is_raw(first_down.encoding, PieLoaderDType::BF16)) {
        return;
    }
    const std::int64_t full_intermediate = first_up.shape[0];
    const std::int64_t hidden = first_up.shape[1];
    if (first_down.shape[0] != hidden || first_down.shape[1] != full_intermediate) {
        return;
    }
    for (const SourceTensor* raw : up) {
        if (raw->shape.size() != 2 || raw->shape[0] != full_intermediate ||
            raw->shape[1] != hidden || !same_encoding(raw->encoding, first_up.encoding)) {
            return;
        }
    }
    for (const SourceTensor* raw : down) {
        if (raw->shape.size() != 2 || raw->shape[0] != hidden ||
            raw->shape[1] != full_intermediate ||
            !same_encoding(raw->encoding, first_down.encoding)) {
            return;
        }
    }

    const auto [local_start, local_intermediate] =
        b.local_range(full_intermediate, "the intermediate size of '" + base + "'");
    const std::int64_t expert_count = static_cast<std::int64_t>(up.size());

    // Each expert contributes its local row band; the pack is their
    // concatenation. The sharding is in the expression, not in a flag.
    const std::string up_name = base + ".up_proj.packed.weight";
    std::vector<Node> up_parts;
    up_parts.reserve(up.size());
    for (const SourceTensor* raw : up) {
        up_parts.push_back(b.contract().slice(b.contract().src(std::string(raw->name)), 0,
                                            local_start, local_intermediate));
    }
    b.define(up_name, b.contract().cat(up_parts, 0), pie_loader::raw(PieLoaderDType::BF16),
           std::vector<std::int64_t>{expert_count * local_intermediate, hidden});

    std::vector<Node> down_parts;
    down_parts.reserve(down.size());
    for (const SourceTensor* raw : down) {
        down_parts.push_back(b.contract().src(std::string(raw->name)));
    }
    auto [down_expr, down_shape] =
        b.shard(b.contract().cat(down_parts, 0),
              {expert_count * hidden, full_intermediate}, std::uint8_t{1});
    const std::string down_name = base + ".down_proj.packed.weight";
    b.define(down_name, down_expr, pie_loader::raw(PieLoaderDType::BF16), std::move(down_shape));

    for (std::size_t expert = 0; expert < up.size(); ++expert) {
        const std::int64_t index = static_cast<std::int64_t>(expert);
        b.define(std::string(up[expert]->name),
               b.contract().slice(b.contract().out(up_name), 0, index * local_intermediate,
                                local_intermediate),
               pie_loader::raw(PieLoaderDType::BF16),
               std::vector<std::int64_t>{local_intermediate, hidden});
        b.consume(up[expert]->id);
    }
    for (std::size_t expert = 0; expert < down.size(); ++expert) {
        const std::int64_t index = static_cast<std::int64_t>(expert);
        b.define(std::string(down[expert]->name),
               b.contract().slice(b.contract().out(down_name), 0, index * hidden, hidden),
               pie_loader::raw(PieLoaderDType::BF16),
               std::vector<std::int64_t>{hidden, local_intermediate});
        b.consume(down[expert]->id);
    }
}

/// Publish each layer's experts as one packed slab plus per-expert views into
/// it, which is what the Nemotron-H MoE GEMM addresses.
inline void nemotron_h_packed_expert_views(ContractBuilder& b) {
    if (b.facts().num_experts == 0) {
        return;
    }
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string base =
            "language_model.backbone.layers." + std::to_string(layer) + ".mixer.experts";
        if (b.find(base + ".up_proj.packed.weight") != nullptr ||
            b.find(base + ".down_proj.packed.weight") != nullptr) {
            continue;
        }
        std::vector<const SourceTensor*> up;
        std::vector<const SourceTensor*> down;
        bool complete = true;
        for (std::uint32_t expert = 0; expert < b.facts().num_experts; ++expert) {
            const std::string tag = base + "." + std::to_string(expert) + ".";
            const SourceTensor* u = b.find(tag + "up_proj.weight");
            const SourceTensor* d = b.find(tag + "down_proj.weight");
            if (u == nullptr || d == nullptr) {
                complete = false;
                break;
            }
            up.push_back(u);
            down.push_back(d);
        }
        if (complete) {
            nemotron_h_layer_packed_experts(b, base, up, down);
        }
    }
}

}  // namespace contract_detail

inline void author_nemotron_h_contract(ContractBuilder& b) {
    b.fused_moe_gate_up_tp_slices();
    contract_detail::nemotron_h_packed_expert_views(b);
    b.dense_fused_projection_joins();
    b.publish_remaining();
}
}  // namespace pie_cuda_driver::model
