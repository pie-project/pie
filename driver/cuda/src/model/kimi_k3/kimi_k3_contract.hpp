#pragma once

/// What Kimi-K3 binds (`model/registry.cpp` row).
///
/// Three things the generic name-pattern rules cannot state:
///
///   * K3's attention introduces six tensor names no other family has
///     (`g_proj`, `f_a_proj`, `f_b_proj`, `b_proj`, `dt_bias`, `A_log`,
///     `{q,k,v}_conv1d`), and getting their TP axis wrong is silent: the
///     model still loads and still emits tokens, just from the wrong heads.
///
///   * `A_log` ships as F32[128] while the layer has 96 KDA heads. A uniform
///     row shard would hand rank 1 entries [64:128) -- half real heads, half
///     storage padding. The band below takes [0:96) first and shards *that*,
///     which is what vLLM's `a_log_weight_loader` does by narrowing to the
///     parameter's own head count.
///
///   * The latent MoE's three projections (`routed_expert_down_proj`,
///     `routed_expert_norm`, `routed_expert_up_proj`) sit *outside* the
///     expert bank: the experts shard on their intermediate dim and all-reduce,
///     so the latent that enters and leaves them has to be full width on every
///     rank. They are replicated, and it only looks like a default -- HF's own
///     `..._down_proj.weight` spelling misses `.down_proj.weight` by one
///     character. State it rather than inherit it.

#include "model/contract.hpp"
#include "model/kimi_k3/kimi_k3.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

/// TP axis policy for Kimi-K3.
inline ShardAxis kimi_k3_shard_axis(std::string_view name) {
    // The latent MoE tail is replicated; see the header comment.
    if (ends_with_any(name, {".routed_expert_down_proj.weight",
                             ".routed_expert_up_proj.weight",
                             ".routed_expert_norm.weight"})) {
        return std::nullopt;
    }
    // Replicated by shape: `f_a_proj` is the rank-`head_dim` bottleneck every
    // head reads, `o_norm` is per-channel within a head, and the AttnRes
    // projections are a single score row over the full-width residual.
    if (ends_with_any(name, {".self_attn.f_a_proj.weight", ".self_attn.o_norm.weight",
                             ".self_attention_res_proj.weight", ".mlp_res_proj.weight",
                             ".self_attention_res_norm.weight", ".mlp_res_norm.weight"})) {
        return std::nullopt;
    }
    // Per-head / per-head-channel tensors that follow the head split.
    // `b_proj` is [num_heads, hidden] -- one beta row per head -- and must not
    // be confused with `q_b_proj`, which the generic list already claims.
    if (ends_with_any(name, {".self_attn.g_proj.weight", ".self_attn.f_b_proj.weight",
                             ".self_attn.b_proj.weight", ".self_attn.dt_bias",
                             ".self_attn.q_conv1d.weight", ".self_attn.k_conv1d.weight",
                             ".self_attn.v_conv1d.weight"})) {
        return std::uint8_t{0};
    }
    // Routed experts under K3's `block_sparse_moe` spelling: shard the
    // intermediate dim inside each expert, w1/w3 on their output axis and w2
    // on its input axis, and all-reduce the partial expert outputs.
    if (contains(name, ".block_sparse_moe.experts.")) {
        if (ends_with_any(name, {".w1.weight", ".w3.weight"})) return std::uint8_t{0};
        if (ends_with(name, ".w2.weight")) return std::uint8_t{1};
    }
    return hf_shard_axis(name);
}

/// Shard `A_log` past its storage padding.
///
/// `[0:num_heads)` is the real gate bank; the tail exists only because the
/// checkpoint rounded the allocation up to `head_dim`. Publishing the band
/// makes the loader hand each rank the heads it actually runs.
///
/// The head count comes from `b_proj.weight`, which is `[num_heads, hidden]` --
/// one beta row per head. `ModelFacts` does not carry it and `head_dim` there
/// is MLA's 192, not KDA's 128, so the checkpoint is both the only source and
/// the right one.
inline void kimi_k3_a_log_bands(ContractBuilder& b) {
    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string layer_prefix =
            std::string(b.decoder_layer_prefix()) + std::to_string(layer) + ".self_attn.";
        const SourceTensor* raw = b.find(b.source_name(layer_prefix + "A_log"));
        const SourceTensor* beta = b.find(b.source_name(layer_prefix + "b_proj.weight"));
        if (raw == nullptr || beta == nullptr) {
            continue;  // an MLA layer.
        }
        if (raw->shape.size() != 1 || beta->shape.empty()) {
            fail("kimi_k3 A_log band: layer " + std::to_string(layer) +
                 " has an unexpected A_log / b_proj rank");
        }
        const std::int64_t heads = beta->shape[0];
        if (raw->shape[0] < heads) {
            fail("kimi_k3 A_log band: layer " + std::to_string(layer) + " has " +
                 std::to_string(raw->shape[0]) + " gate entries for " +
                 std::to_string(heads) + " heads");
        }
        auto banded = b.band(b.contract().src(std::string(raw->name)), 0, 0, heads);
        b.define(b.output_name(raw->name), banded.first, raw->encoding,
                 std::vector<std::int64_t>{banded.second});
        b.consume(raw->id);
    }
}

/// Dequantize K3's routed experts and stack them, at load time.
///
/// Each expert ships MXFP4: `weight_packed` is `U8 [out, in/2]` holding two
/// E2M1 codes per byte, and `weight_scale` is `U8 [out, in/32]` holding one
/// E8M0 exponent per group of 32 along the input axis. The batched MoE path
/// wants one dense bf16 slab per layer -- `[E, 2I, L]` over `[E, L, I]` -- so
/// a grouped GEMM sees a base pointer and a stride.
///
/// The sharding lives *inside* the stack, so each rank dequantizes only the
/// intermediate slice it keeps rather than materialising the whole bank and
/// throwing half of it away.
inline void kimi_k3_bf16_expert_stacks(ContractBuilder& b, bool gate_second) {
    constexpr std::int64_t kGroup = 32;
    const std::int64_t experts = b.facts().num_experts;
    if (experts <= 0) {
        return;
    }

    for (std::uint32_t layer = 0; layer < b.facts().num_hidden_layers; ++layer) {
        const std::string moe = std::string(b.decoder_layer_prefix()) +
                                std::to_string(layer) + ".block_sparse_moe.";
        const std::string prefix = b.source_name(moe);
        // K3's leading layers are dense and simply have no expert names, which
        // is why this probes rather than reading `first_k_dense_replace`.
        if (b.find(prefix + "experts.0.w1.weight_packed") == nullptr) {
            continue;
        }

        std::vector<Node> gate_up, gate_up_scales, down, down_scales;
        std::vector<std::uint32_t> consumed;
        std::int64_t local_inter = 0;
        std::int64_t latent = 0;

        // A width-changing `Transmute` may only rename a whole tensor, so each
        // source is reinterpreted on its own -- U8 bytes as 4-bit MXFP4
        // elements, U8 bytes as E8M0 exponents -- and the stacking happens
        // afterwards, over expressions that already have their final encoding.
        auto packed = [&](const SourceTensor& raw, std::vector<std::int64_t> shape,
                          std::uint8_t axis) {
            return b.shard(b.contract().transmute(
                               b.contract().src(std::string(raw.name)), shape,
                               mxfp4_encoding(b.contract(), static_cast<std::uint8_t>(2))),
                           shape, ShardAxis{axis})
                .first;
        };
        auto factors = [&](const SourceTensor& raw, std::vector<std::int64_t> shape,
                           std::uint8_t axis) {
            return b.shard(b.contract().transmute(
                               b.contract().src(std::string(raw.name)), shape,
                               pie_loader::raw(PieLoaderDType::E8M0)),
                           shape, ShardAxis{axis})
                .first;
        };

        for (std::int64_t e = 0; e < experts; ++e) {
            const std::string ep = prefix + "experts." + std::to_string(e) + ".";
            const SourceTensor* parts[6] = {
                b.find(ep + "w1.weight_packed"), b.find(ep + "w1.weight_scale"),
                b.find(ep + "w3.weight_packed"), b.find(ep + "w3.weight_scale"),
                b.find(ep + "w2.weight_packed"), b.find(ep + "w2.weight_scale"),
            };
            for (const SourceTensor* part : parts) {
                if (part == nullptr) {
                    fail("kimi_k3 expert stack: layer " + std::to_string(layer) +
                         " expert " + std::to_string(e) + " is missing a weight or scale");
                }
            }
            // A checkpoint that packs its experts some other way is not this
            // pass's to rewrite, and stacking it anyway would hand the GEMM
            // garbage -- leave the whole model alone rather than half of it.
            for (const SourceTensor* part : parts) {
                if (!is_raw(part->encoding, PieLoaderDType::U8) || part->shape.size() != 2) {
                    return;
                }
            }

            // w1/w3 are [I, L/2] packed with [I, L/32] scales; w2 is [L, I/2]
            // with [L, I/32]. The declared shapes below are in *elements*, not
            // bytes -- that is what the MXFP4 encoding's 4 bits per element
            // and the E8M0 group of 32 are for.
            const std::int64_t inter = parts[0]->shape[0];
            const std::int64_t latent_here = parts[0]->shape[1] * 2;
            if (parts[4]->shape[0] != latent_here ||
                parts[4]->shape[1] * 2 != inter ||
                parts[1]->shape[1] != latent_here / kGroup ||
                parts[5]->shape[1] != inter / kGroup) {
                fail("kimi_k3 expert stack: layer " + std::to_string(layer) +
                     " expert " + std::to_string(e) + " has inconsistent MXFP4 shapes");
            }
            if (e == 0) {
                latent = latent_here;
                local_inter = b.local_extent(inter);
            } else if (latent_here != latent) {
                fail("kimi_k3 expert stack: layer " + std::to_string(layer) +
                     " expert " + std::to_string(e) + " changes the latent width");
            }

            // `[up | gate]` is what flashinfer's grouped GEMM reads fc1 as, and
            // choosing the order here costs nothing -- it is which leg of the
            // concat goes first. Undoing it later would copy the whole slab.
            const Node w1 = packed(*parts[0], {1, inter, latent}, 1);
            const Node w3 = packed(*parts[2], {1, inter, latent}, 1);
            const Node w1s = factors(*parts[1], {1, inter, latent / kGroup}, 1);
            const Node w3s = factors(*parts[3], {1, inter, latent / kGroup}, 1);
            // Raw-byte views of the same four sources for the packed
            // republish below; the nodes above carry the MXFP4/E8M0 encodings
            // the dequantizing stack needs.
            //
            // The decode GEMV reads gate and up as *adjacent* rows -- row 2i is
            // gate row i, row 2i+1 is up row i -- because that is how gpt-oss
            // ships them, and it is what lets one warp pull a slab of both
            // with no gather. K3 ships them as two tensors, so the
            // interleaving is built here: reshape each to `[I, 1, L/2]` and
            // join on the new middle axis, which is a rename plus a
            // concatenation rather than a permutation.
            auto pair = [&](const SourceTensor& a, const SourceTensor& c,
                            std::int64_t cols) {
                const auto u8 = pie_loader::raw(PieLoaderDType::U8);
                const Node an = b.contract().transmute(
                    b.split(b.contract().src(std::string(a.name)), 0),
                    {local_inter, 1, cols}, u8);
                const Node cn = b.contract().transmute(
                    b.split(b.contract().src(std::string(c.name)), 0),
                    {local_inter, 1, cols}, u8);
                return b.contract().concat({an, cn}, 1);
            };
            gate_up.push_back(b.contract().concat(
                gate_second ? std::vector<Node>{w3, w1} : std::vector<Node>{w1, w3}, 1));
            gate_up_scales.push_back(b.contract().concat(
                gate_second ? std::vector<Node>{w3s, w1s} : std::vector<Node>{w1s, w3s}, 1));
            down.push_back(packed(*parts[4], {1, latent, inter}, 2));
            down_scales.push_back(
                factors(*parts[5], {1, latent, inter / kGroup}, 2));
            // The same four sources, republished in the shape the decode
            // GEMV addresses: one `[2I, L/2]` slab per expert with gate over
            // up. Decode is bandwidth-bound and reads the whole expert bank
            // for a single token, so the four-bit form is worth four times its
            // weight there; the dequantized stacks below are what the prefill
            // GEMMs want. Both stay resident and the forward picks per step,
            // the way Kimi-K2 does.
            //
            // Gate over up, always -- `gate_second` reorders the *bf16* stack
            // for flashinfer's fc1, and this path reads its two halves by
            // name into separate outputs.
            const std::string ep_out = moe + "experts." + std::to_string(e) + ".";
            b.define(ep_out + "gate_up.weight_packed",
                     pair(*parts[0], *parts[2], latent / 2),
                     pie_loader::raw(PieLoaderDType::U8),
                     std::vector<std::int64_t>{local_inter, 2, latent / 2});
            b.define(ep_out + "gate_up.weight_scale",
                     pair(*parts[1], *parts[3], latent / kGroup),
                     pie_loader::raw(PieLoaderDType::U8),
                     std::vector<std::int64_t>{local_inter, 2, latent / kGroup});
            b.define(ep_out + "down.weight_packed",
                     b.split(b.contract().src(std::string(parts[4]->name)), 1),
                     pie_loader::raw(PieLoaderDType::U8),
                     std::vector<std::int64_t>{latent, local_inter / 2});
            b.define(ep_out + "down.weight_scale",
                     b.split(b.contract().src(std::string(parts[5]->name)), 1),
                     pie_loader::raw(PieLoaderDType::U8),
                     std::vector<std::int64_t>{latent, local_inter / kGroup});
            for (const SourceTensor* part : parts) {
                consumed.push_back(part->id);
            }
        }

        if (gate_up.empty()) {
            continue;
        }
        // Named but not bound: `scale_per_block` takes its factors by output
        // name, and the stacked slab is dequantized here, so nothing reads
        // these again. Left public they would hold an E8M0 byte per 32 weights
        // in the persistent arena for the process's lifetime.
        const std::string gu_scale = moe + "experts.gate_up.scale";
        const std::string dn_scale = moe + "experts.down.scale";
        if (auto gu = b.define(gu_scale, b.contract().concat(gate_up_scales, 0),
                               pie_loader::raw(PieLoaderDType::E8M0),
                               std::vector<std::int64_t>{experts, 2 * local_inter,
                                                         latent / kGroup})) {
            gu->internal();
        }
        if (auto dn = b.define(dn_scale, b.contract().concat(down_scales, 0),
                               pie_loader::raw(PieLoaderDType::E8M0),
                               std::vector<std::int64_t>{experts, latent,
                                                         local_inter / kGroup})) {
            dn->internal();
        }
        b.define(moe + "experts.gate_up_proj",
                 b.contract().scale_per_block(b.contract().concat(gate_up, 0),
                                              b.contract().out(gu_scale)),
                 pie_loader::raw(PieLoaderDType::BF16),
                 std::vector<std::int64_t>{experts, 2 * local_inter, latent});
        b.define(moe + "experts.down_proj",
                 b.contract().scale_per_block(b.contract().concat(down, 0),
                                              b.contract().out(dn_scale)),
                 pie_loader::raw(PieLoaderDType::BF16),
                 std::vector<std::int64_t>{experts, latent, local_inter});
        for (std::uint32_t id : consumed) {
            b.consume(id);
        }
    }
}

}  // namespace contract_detail

/// kimi_k3. `embed_tokens` is sharded on axis 0 under TP to save per-rank
/// memory, matching the Kimi-K2 row; the decoder lives under
/// `language_model.model.layers.`.
inline void author_kimi_k3_contract(ContractBuilder& b) {
    // The multimodal checkpoints nest the decoder under `language_model.`,
    // the text-only ones do not. `source_prefix` asks the checkpoint rather
    // than declaring it, so one row covers both; without it every bind name
    // misses by exactly that prefix and authoring fails on the first tensor
    // it looks up by hand (`model.embed_tokens.weight`).
    b.source_prefix("language_model.");
    b.shard_axis_fn(contract_detail::kimi_k3_shard_axis);
    b.shard_embed_tokens();
    contract_detail::kimi_k3_a_log_bands(b);
    contract_detail::kimi_k3_bf16_expert_stacks(b, kimi_k3_moe_gate_up_swapped());
    // Deliberately *not* `author_dense_contract`: its
    // `dense_fused_projection_joins` would join `self_attn.{q,k,v}_proj` into
    // one QKV weight, which is right for a llama-like layer and wrong here --
    // KDA's q, k and v each go through their own short convolution before they
    // ever meet, so a fused projection would have to be split straight back
    // apart. The MoE slice still runs first, because a family can have both a
    // fused expert weight and fused dense projections.
    b.fused_moe_gate_up_tp_slices();
    b.publish_remaining();
}

}  // namespace pie_cuda_driver::model
