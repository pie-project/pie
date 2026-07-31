#pragma once

/// What the MLA families bind (`model/registry.cpp` rows).
///
/// DeepSeek-V2/V3 and Kimi-K2 share a binder and a forward. They differ in the
/// contract: Kimi hides the decoder under `language_model.` and wants
/// `embed_tokens` sharded and `lm_head` replicated, which is a memory trade
/// this driver makes and the checkpoint knows nothing about.

#include "model/contract.hpp"
#include "model/kimi/kimi.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

/// Fuse the two MLA projection pairs that share an input.
inline void mla_fused_projection_joins(ContractBuilder& b) {
    std::vector<ContractBuilder::FusedCandidate> candidates;
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        const std::string s = b.source_name(p);
        // q_a_proj + kv_a_proj_with_mqa share an input (norm_x, unsharded).
        if (auto candidate = b.fused_join_candidate(
                p + "self_attn.q_kv_a_proj.fused.weight",
                {s + "self_attn.q_a_proj.weight", s + "self_attn.kv_a_proj_with_mqa.weight"})) {
            candidates.push_back(std::move(*candidate));
        }
        // Shared gate + up share an input (norm_y).
        if (auto candidate = b.fused_join_candidate(
                p + "mlp.shared_experts.gate_up_proj.fused.weight",
                {s + "mlp.shared_experts.gate_proj.weight",
                 s + "mlp.shared_experts.up_proj.weight"})) {
            candidates.push_back(std::move(*candidate));
        }
    }
    b.publish_fused(candidates);
}

/// Dequantize Kimi's routed experts and stack them, at load time.
///
/// The checkpoint stores each expert separately as W4A16: `weight_packed` is
/// `I32 [out, in/8]` holding eight 4-bit codes per word, low nibble first, and
/// `weight_scale` is `BF16 [out, in/32]`. An element is `code - 8` times its
/// group's factor, which is what `QuantScheme::Int4B8` names -- the bias is
/// the scheme's, not a zero-point tensor's.
///
/// The batched MoE path wants one dense bf16 slab per layer, `[E, 2I, H]` over
/// `[E, H, I]`, so a batched GEMM sees a base pointer and a stride. Building it
/// is `Bitcast` to say what the packed words are, `Concat` to stack, and one
/// `scale_per_block` per slab to dequantize -- with the sharding inside, so
/// each rank dequantizes only the slice it keeps.
///
/// Unlike DeepSeek-V4 this does **not** consume the packed originals. Kimi
/// picks between the two forms per step: the W4A16 GEMVs win below a token
/// count the batched path's 4x weight traffic cannot amortize, so both have to
/// be resident, and the stacks are additive. That is what `budget` is for.
///
/// This ran during bind until the algebra could state it: a loop calling
/// `launch_dequant_wna16_int4b8_to_bf16` 3E times per layer, into slabs
/// allocated outside the storage arena.
inline void kimi_bf16_expert_stacks(ContractBuilder& b, std::size_t budget) {
    using contract_detail::fail;
    using contract_detail::is_raw;
    using contract_detail::shape_of;
    constexpr std::int64_t kGroup = 32;
    constexpr std::int64_t kCodesPerWord = 8;

    // `[up | gate]` is what flashinfer's fused grouped GEMM reads fc1 as, and
    // choosing the order here costs nothing -- it is which `concat` leg goes
    // first. Undoing it later would cost a copy of the whole slab.
    const bool gate_second = kimi_moe_gate_up_swapped();

    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string mlp = "model.layers." + std::to_string(layer) + ".mlp.";
        // Kimi's leading layers are dense, and a dense layer simply has no
        // expert names -- which is why the loop probes rather than reading
        // `first_k_dense_replace`.
        if (b.find(b.source_name(mlp + "experts.0.gate_proj.weight_packed")) == nullptr) {
            continue;
        }
        std::vector<Node> gate_up, gate_up_scales, down, down_scales;
        std::int64_t local_inter = 0;
        std::int64_t hidden = 0;

        for (std::uint32_t expert = 0;; ++expert) {
            const std::string ep = mlp + "experts." + std::to_string(expert) + ".";
            if (b.find(b.source_name(ep + "gate_proj.weight_packed")) == nullptr) {
                break;
            }
            const SourceTensor* parts[6] = {
                b.find(b.source_name(ep + "gate_proj.weight_packed")),
                b.find(b.source_name(ep + "gate_proj.weight_scale")),
                b.find(b.source_name(ep + "up_proj.weight_packed")),
                b.find(b.source_name(ep + "up_proj.weight_scale")),
                b.find(b.source_name(ep + "down_proj.weight_packed")),
                b.find(b.source_name(ep + "down_proj.weight_scale")),
            };
            for (const SourceTensor* part : parts) {
                if (part == nullptr) {
                    fail("kimi expert stack: " + ep + " is missing a weight or scale");
                }
            }
            // Eight codes to a 32-bit word, and BF16 factors. A checkpoint
            // that stores its experts some other way is not this pass's to
            // rewrite, and stacking it anyway would hand the GEMM garbage --
            // so leave the whole checkpoint alone rather than half of it, and
            // let the W4A16 path run off the packed weights as before.
            for (const SourceTensor* packed_part : {parts[0], parts[2], parts[4]}) {
                if (!is_raw(packed_part->encoding, PieLoaderDType::I32)) {
                    return;
                }
            }
            for (const SourceTensor* scale_part : {parts[1], parts[3], parts[5]}) {
                if (!is_raw(scale_part->encoding, PieLoaderDType::BF16)) {
                    return;
                }
            }

            const std::vector<std::int64_t> up_raw = shape_of(*parts[0]);
            const std::vector<std::int64_t> down_raw = shape_of(*parts[4]);
            if (up_raw.size() != 2 || down_raw.size() != 2) {
                fail("kimi expert stack: " + ep + " expects rank-2 expert weights");
            }
            const std::int64_t inter_full = up_raw[0];
            const std::int64_t h = up_raw[1] * kCodesPerWord;
            const std::int64_t inter = b.local_extent(inter_full);
            if (h % kGroup != 0 || inter % kGroup != 0) {
                fail("kimi expert stack: " + ep +
                     " expects both expert dims to be a multiple of 32");
            }
            if (local_inter != 0 && (inter != local_inter || h != hidden)) {
                fail("kimi expert stack: " + ep + " disagrees with its siblings on shape");
            }
            local_inter = inter;
            hidden = h;

            // Every leg is declared rank 3 with a leading 1, so that the outer
            // concatenation over axis 0 is a stack. `Bitcast` carries the rank
            // lift as well as the unpacking, because reshaping a packed tensor
            // is not meaningful: its byte layout is a function of the shape it
            // was packed for.
            //
            // `gate`/`up` shard the out dim and `down` the in dim, the same
            // split the packed tensors take, applied to the logical shapes.
            auto packed = [&](const SourceTensor& raw, std::vector<std::int64_t> shape,
                              std::uint8_t axis) {
                return b.shard(b.contract().transmute(
                                   b.contract().src(std::string(raw.name)), shape,
                                   contract_detail::int4b8_encoding(static_cast<std::uint8_t>(2))),
                               shape, ShardAxis{axis})
                    .first;
            };
            auto factors = [&](const SourceTensor& raw, std::vector<std::int64_t> shape,
                               std::uint8_t axis) {
                return b.shard(b.contract().transmute(b.contract().src(std::string(raw.name)),
                                                    shape,
                                                    pie_loader::raw(PieLoaderDType::BF16)),
                               shape, ShardAxis{axis})
                    .first;
            };

            const Node gate = packed(*parts[0], {1, inter_full, h}, 1);
            const Node up = packed(*parts[2], {1, inter_full, h}, 1);
            const Node gate_s = factors(*parts[1], {1, inter_full, h / kGroup}, 1);
            const Node up_s = factors(*parts[3], {1, inter_full, h / kGroup}, 1);
            gate_up.push_back(b.contract().concat(
                gate_second ? std::vector<Node>{up, gate} : std::vector<Node>{gate, up}, 1));
            gate_up_scales.push_back(b.contract().concat(
                gate_second ? std::vector<Node>{up_s, gate_s} : std::vector<Node>{gate_s, up_s},
                1));
            down.push_back(packed(*parts[4], {1, down_raw[0], inter_full}, 2));
            down_scales.push_back(
                factors(*parts[5], {1, down_raw[0], inter_full / kGroup}, 2));
        }
        if (gate_up.empty()) {
            continue;
        }
        const auto experts = static_cast<std::int64_t>(gate_up.size());

        // Both forms stay resident, so a model whose slabs do not fit is
        // better off with only the packed one: the W4A16 GEMVs are slower per
        // token but they are not a fallback, they are the path that wins below
        // the crossover. Checked per layer against the whole model's cost, so
        // the answer cannot come out different for different layers.
        const std::size_t slab_bytes = static_cast<std::size_t>(experts) * 3 *
            static_cast<std::size_t>(local_inter) * static_cast<std::size_t>(hidden) *
            sizeof(std::uint16_t) * b.facts().num_hidden_layers;
        if (slab_bytes > budget) {
            return;
        }

        // The factors are published rather than kept internal because
        // `scale_per_block` takes them by output name: a scale is a tensor the
        // contract declared, not a companion the lowering guesses at from a
        // suffix. They cost two bytes per 32 weights.
        // Named but not bound. `scale_per_block` takes its factors by output
        // name -- a scale is a tensor the contract declared, not a companion
        // the lowering guesses at from a suffix -- and the stacked slab is
        // dequantized here, so no kernel ever reads these again. Left public
        // they would sit in the persistent arena for the process's lifetime,
        // two bf16 bytes per 32 weights, and appear in the driver's bind table
        // under a name nothing asks for.
        const std::string gu_scale = mlp + "experts.gate_up.scale";
        const std::string dn_scale = mlp + "experts.down.scale";
        if (auto gu = b.define(gu_scale, b.contract().concat(gate_up_scales, 0),
                               pie_loader::raw(PieLoaderDType::BF16),
                               std::vector<std::int64_t>{experts, 2 * local_inter,
                                                         hidden / kGroup})) {
            gu->internal();
        }
        if (auto dn = b.define(dn_scale, b.contract().concat(down_scales, 0),
                               pie_loader::raw(PieLoaderDType::BF16),
                               std::vector<std::int64_t>{experts, hidden,
                                                         local_inter / kGroup})) {
            dn->internal();
        }
        b.define(mlp + "experts.gate_up.weight",
                 b.contract().scale_per_block(b.contract().concat(gate_up, 0),
                                              b.contract().out(gu_scale)),
                 pie_loader::raw(PieLoaderDType::BF16),
                 std::vector<std::int64_t>{experts, 2 * local_inter, hidden});
        b.define(mlp + "experts.down.weight",
                 b.contract().scale_per_block(b.contract().concat(down, 0),
                                              b.contract().out(dn_scale)),
                 pie_loader::raw(PieLoaderDType::BF16),
                 std::vector<std::int64_t>{experts, hidden, local_inter});
    }
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
///
/// The bf16 expert slabs are additive on top of the packed W4A16 experts,
/// which the per-step GEMV path still reads, so they are only published when
/// the whole model's worth fits in this budget.
inline void author_kimi_contract(ContractBuilder& b) {
    b.source_prefix("language_model.");
    b.shard_embed_tokens();
    b.replicate_lm_head();
    contract_detail::kimi_bf16_expert_stacks(b, /*budget=*/4ull << 30);
    author_deepseek_mla_contract(b);
}
}  // namespace pie_cuda_driver::model
