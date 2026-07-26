#pragma once

/// Per-family contract declarations — the driver's statement of what it will
/// bind, made *before* the load (architecture.md §10.2.1, §12 row 7b-4b).
///
/// Each family's binder already contains this knowledge; `bind_llama_like`'s
/// `must(engine, p + "self_attn.o_proj.weight")` and its
/// `Hq = num_attention_heads * head_dim` are exactly it. Stated only after the
/// load, it checks nothing: an `arch/` pass that computes a different shape
/// binds a wrongly-sized buffer, and an output name no binder asks for is never
/// noticed. Declared here, `pie_loader_verify` compares two independently
/// derived answers.
///
/// # Adoption is per family
///
/// `declare_model_contract` returns an empty contract for any family that has
/// not been written out yet, and an empty contract disables only the
/// contract-vs-plan comparison — every other check still runs. So a family can
/// be added without a flag day, and adding one is a pure strengthening.

#include <cstdint>
#include <string>

#include "pie_loader/tensor_contract.hpp"

#include "model/config.hpp"

namespace pie_cuda_driver {

namespace contract_detail {

/// Divide a dimension across the TP group.
///
/// Declaring the model-wide extent for a sharded weight would report a mismatch
/// on every one of them, since the plan is compiled for a single rank.
inline std::int64_t shard(std::int64_t dim, std::uint32_t tp_size) {
    return tp_size <= 1 ? dim : dim / static_cast<std::int64_t>(tp_size);
}

/// A shape stated only when it is the shape the checkpoint actually holds.
///
/// AWQ/GPTQ/compressed-tensors checkpoints store projections packed — and often
/// transposed — so the logical `[out, in]` the binder reasons about is not what
/// the plan declares. The driver genuinely does not know those extents, so for
/// such a checkpoint it declares the name alone (an empty shape is "unstated",
/// not "rank 0").
///
/// Passing the logical shape through this is what keeps each family's demand
/// list written *once*. Writing it twice — a shaped branch and a bare one — put
/// the two lists one edit apart from disagreeing about which tensors exist.
class LogicalShape {
  public:
    explicit LogicalShape(const HfConfig& hf) : exact_(hf.quant_method.empty()) {}

    std::vector<std::int64_t> operator()(std::vector<std::int64_t> logical) const {
        if (!exact_) return {};
        return logical;
    }

  private:
    bool exact_;
};

}  // namespace contract_detail

/// The llama-like family, bound by `model/llama_like/qwen3.cpp::bind_llama_like`.
///
/// "llama-like" is narrower than `Family::LlamaLike`: `registry.cpp` files
/// `mistral3`, `phi3`, `olmo2` and `olmo3` under the same family but binds each
/// with its own row (`bind_row_mistral3`, `bind_row_phi3`, `bind_row_olmo3`).
/// A declaration describes a *binder*, not a family, so those are excluded until
/// someone writes them out.
///
/// Only the tensors that binder demands *unconditionally* are declared with
/// shapes. `q/k/v/gate/up` are deliberately absent: `bind_projection_or_fused_view`
/// accepts either the separate weights or a view into a fused buffer, and which
/// one exists is the loader's fusion decision (`arch/fusion.rs`, budgeted). A
/// contract cannot demand one form without taking that choice away, and this
/// step is not where that is resolved — see §12 row 7b-4b on `ArchProfile`.
inline pie_loader::TensorContract declare_llama_like_contract(
    const HfConfig& hf,
    std::uint32_t tp_size) {
    using contract_detail::shard;
    using contract_detail::LogicalShape;

    pie_loader::TensorContract c;
    const LogicalShape shape(hf);
    const std::int64_t hidden = hf.hidden_size;
    const std::int64_t vocab = hf.vocab_size;
    const std::int64_t q_dim =
        static_cast<std::int64_t>(hf.num_attention_heads) * hf.head_dim;
    const std::int64_t inter = hf.intermediate_size;

    // `shard_embed_tokens` and `replicate_lm_head` are both off for this family
    // (`arch/arch.rs`), so neither table is divided.
    c.demand("model.embed_tokens.weight", shape({vocab, hidden}));
    c.demand("model.norm.weight", {hidden});
    // Dropped when `tie_word_embeddings`, where the binder reuses `embed`.
    c.demand_optional("lm_head.weight", shape({vocab, hidden}));

    for (int i = 0; i < hf.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        c.demand(p + "input_layernorm.weight", {hidden});
        c.demand(p + "post_attention_layernorm.weight", {hidden});
        // Both are row-parallel: sharded on axis 1 (`arch/policy.rs`).
        c.demand(p + "self_attn.o_proj.weight", shape({hidden, shard(q_dim, tp_size)}));
        c.demand(p + "mlp.down_proj.weight", shape({hidden, shard(inter, tp_size)}));
        if (hf.use_qk_norm) {
            // Per-head norms are replicated: the head dimension is not the
            // sharded one.
            c.demand(p + "self_attn.q_norm.weight", {hf.head_dim});
            c.demand(p + "self_attn.k_norm.weight", {hf.head_dim});
        }
    }
    return c;
}

/// gpt-oss, bound by `model/mixtral/gpt_oss.cpp::bind_gpt_oss`.
///
/// Everything outside the experts is declared with exact shapes even on an
/// mxfp4 checkpoint, because the packing is scoped: gpt-oss's
/// `quantization_config.modules_to_not_convert` lists `self_attn`,
/// `mlp.router`, `embed_tokens` and `lm_head`, and `bind_gpt_oss` already
/// asserts these very extents (`q_bias->numel() != Hq` and friends) whatever the
/// checkpoint's quantization. Declaring them restates a check the binder makes
/// anyway, only early enough to name the tensor that is wrong.
///
/// The experts are left undeclared: their names depend on which of the three
/// lowerings the driver picked (native mxfp4 / dequantized / bf16), and that is
/// a runtime decision, not a property of the checkpoint.
inline pie_loader::TensorContract declare_gpt_oss_contract(
    const HfConfig& hf,
    std::uint32_t tp_size) {
    using contract_detail::shard;

    pie_loader::TensorContract c;
    const std::int64_t hidden = hf.hidden_size;
    const std::int64_t vocab = hf.vocab_size;
    const std::int64_t q_dim =
        shard(static_cast<std::int64_t>(hf.num_attention_heads) * hf.head_dim, tp_size);
    const std::int64_t kv_dim =
        shard(static_cast<std::int64_t>(hf.num_key_value_heads) * hf.head_dim, tp_size);
    const std::int64_t experts = hf.num_experts;

    c.demand("model.embed_tokens.weight", {vocab, hidden});
    c.demand("model.norm.weight", {hidden});
    // `bind_gpt_oss` falls back to the embedding table when `tie_word_embeddings`.
    c.demand_optional("lm_head.weight", {vocab, hidden});

    for (int i = 0; i < hf.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        c.demand(p + "input_layernorm.weight", {hidden});
        c.demand(p + "post_attention_layernorm.weight", {hidden});
        // q/k/v are column-parallel (axis 0), o is row-parallel (axis 1).
        c.demand(p + "self_attn.q_proj.weight", {q_dim, hidden});
        c.demand(p + "self_attn.k_proj.weight", {kv_dim, hidden});
        c.demand(p + "self_attn.v_proj.weight", {kv_dim, hidden});
        c.demand(p + "self_attn.o_proj.weight", {hidden, q_dim});
        c.demand(p + "self_attn.q_proj.bias", {q_dim});
        c.demand(p + "self_attn.k_proj.bias", {kv_dim});
        c.demand(p + "self_attn.v_proj.bias", {kv_dim});
        // The output bias is added after the row-parallel reduction, so every
        // rank holds the whole thing.
        c.demand(p + "self_attn.o_proj.bias", {hidden});
        // One learned sink per local query head.
        c.demand(p + "self_attn.sinks",
                 {shard(hf.num_attention_heads, tp_size)});
        // The router runs on every rank over every expert: never sharded.
        c.demand(p + "mlp.router.weight", {experts, hidden});
        c.demand(p + "mlp.router.bias", {experts});
    }
    return c;
}

/// Qwen3-VL, bound by `model/qwen3_vl/qwen3_vl.cpp`.
///
/// Two towers. The text one is a Qwen3 under the `model.language_model.` prefix
/// and is TP-sharded; the vision one is replicated on every rank.
///
/// Unlike `bind_llama_like`, `bind_qwen3_vl_text` reaches for `q/k/v/gate/up`
/// with `must` rather than `bind_projection_or_fused_view`, so those names have
/// to exist however the loader decided to fuse. They are declared here without
/// shapes for exactly that reason: their presence is the driver's requirement,
/// their extent is not — a fused load republishes them as views.
inline pie_loader::TensorContract declare_qwen3_vl_contract(
    const HfConfig& hf,
    std::uint32_t tp_size) {
    using contract_detail::shard;
    using contract_detail::LogicalShape;

    pie_loader::TensorContract c;
    const LogicalShape shape(hf);
    const std::int64_t hidden = hf.hidden_size;
    const std::int64_t vocab = hf.vocab_size;
    const std::int64_t q_dim =
        shard(static_cast<std::int64_t>(hf.num_attention_heads) * hf.head_dim, tp_size);
    const std::int64_t inter = shard(hf.intermediate_size, tp_size);
    const std::string mp = "model.language_model.";

    c.demand(mp + "embed_tokens.weight", shape({vocab, hidden}));
    c.demand(mp + "norm.weight", {hidden});
    c.demand_optional("lm_head.weight", shape({vocab, hidden}));

    for (int i = 0; i < hf.num_hidden_layers; ++i) {
        const std::string p = mp + "layers." + std::to_string(i) + ".";
        c.demand(p + "input_layernorm.weight", {hidden});
        c.demand(p + "post_attention_layernorm.weight", {hidden});
        c.demand(p + "self_attn.o_proj.weight", shape({hidden, q_dim}));
        c.demand(p + "mlp.down_proj.weight", shape({hidden, inter}));
        c.demand(p + "self_attn.q_proj.weight");
        c.demand(p + "self_attn.k_proj.weight");
        c.demand(p + "self_attn.v_proj.weight");
        c.demand(p + "mlp.gate_proj.weight");
        c.demand(p + "mlp.up_proj.weight");
        if (hf.attention_bias) {
            c.demand(p + "self_attn.q_proj.bias");
            c.demand(p + "self_attn.k_proj.bias");
            c.demand(p + "self_attn.v_proj.bias");
        }
        if (hf.use_qk_norm) {
            c.demand(p + "self_attn.q_norm.weight", {hf.head_dim});
            c.demand(p + "self_attn.k_norm.weight", {hf.head_dim});
        }
    }

    if (!hf.qwen3_vl_vision.has_value()) return c;
    const Qwen3VLVisionConfig& vc = *hf.qwen3_vl_vision;
    const std::int64_t vh = vc.hidden_size;
    const std::string vp = "model.visual.";

    // The vision tower is not sharded, so its shapes are the config's outright.
    c.demand(vp + "patch_embed.proj.weight");
    c.demand(vp + "patch_embed.proj.bias", {vh});
    c.demand(vp + "pos_embed.weight", {vc.num_position_embeddings, vh});
    for (int i = 0; i < vc.depth; ++i) {
        const std::string lp = vp + "blocks." + std::to_string(i) + ".";
        c.demand(lp + "norm1.weight", {vh});
        c.demand(lp + "norm1.bias", {vh});
        c.demand(lp + "norm2.weight", {vh});
        c.demand(lp + "norm2.bias", {vh});
        c.demand(lp + "attn.qkv.weight", {3 * vh, vh});
        c.demand(lp + "attn.qkv.bias", {3 * vh});
        c.demand(lp + "attn.proj.weight", {vh, vh});
        c.demand(lp + "attn.proj.bias", {vh});
        c.demand(lp + "mlp.linear_fc1.weight", {vc.intermediate_size, vh});
        c.demand(lp + "mlp.linear_fc1.bias", {vc.intermediate_size});
        c.demand(lp + "mlp.linear_fc2.weight", {vh, vc.intermediate_size});
        c.demand(lp + "mlp.linear_fc2.bias", {vh});
    }
    // The main merger norms over `hidden` before the 2x2 shuffle; the DeepStack
    // mergers norm over `4 * hidden` after it (`use_postshuffle_norm`).
    const std::int64_t merged = static_cast<std::int64_t>(vc.spatial_merge_size) *
                                vc.spatial_merge_size * vh;
    c.demand(vp + "merger.norm.weight", {vh});
    c.demand(vp + "merger.norm.bias", {vh});
    c.demand(vp + "merger.linear_fc1.weight", {merged, merged});
    c.demand(vp + "merger.linear_fc1.bias", {merged});
    c.demand(vp + "merger.linear_fc2.weight", {vc.out_hidden_size, merged});
    c.demand(vp + "merger.linear_fc2.bias", {vc.out_hidden_size});
    for (std::size_t d = 0; d < vc.deepstack_visual_indexes.size(); ++d) {
        const std::string dp = vp + "deepstack_merger_list." + std::to_string(d) + ".";
        c.demand(dp + "norm.weight", {merged});
        c.demand(dp + "norm.bias", {merged});
        c.demand(dp + "linear_fc1.weight", {merged, merged});
        c.demand(dp + "linear_fc1.bias", {merged});
        c.demand(dp + "linear_fc2.weight", {vc.out_hidden_size, merged});
        c.demand(dp + "linear_fc2.bias", {vc.out_hidden_size});
    }
    return c;
}

/// Qwen3.5 (dense), bound by `model/qwen3_5/qwen3_5.cpp::bind_qwen3_5`.
///
/// Two kinds of layer interleave under `HfConfig.layer_types`, and the binder
/// reaches for a different set of names in each — so the declaration has to walk
/// the same list. Getting that wrong is precisely the failure this step exists
/// to catch: a plan built for the wrong interleaving produces every tensor the
/// driver never asks for and none of the ones it does.
///
/// Shapes are declared only where the tensor is not the subject of a runtime
/// decision. The gated-delta-net block is sliced and re-fused by the binder
/// itself (`maybe_fuse_*`, `slice_stacked`), and `self_attn.q_proj` is twice the
/// usual width because query and gate are fused into it, so those are named
/// rather than measured.
inline pie_loader::TensorContract declare_qwen3_5_contract(
    const HfConfig& hf,
    std::uint32_t tp_size) {
    using contract_detail::shard;
    using contract_detail::LogicalShape;

    pie_loader::TensorContract c;
    // The binder requires `layer_types` to have exactly one entry per layer and
    // throws otherwise. Declaring against a disagreeing list would report the
    // wrong tensors, so leave the contract empty and let the binder say so.
    if (static_cast<int>(hf.layer_types.size()) != hf.num_hidden_layers) return {};

    const LogicalShape shape(hf);
    const std::int64_t hidden = hf.hidden_size;
    const std::int64_t vocab = hf.vocab_size;
    const std::string p = "model.language_model.";

    c.demand(p + "embed_tokens.weight", shape({vocab, hidden}));
    c.demand_optional("lm_head.weight", shape({vocab, hidden}));
    c.demand(p + "norm.weight", {hidden});

    for (int i = 0; i < hf.num_hidden_layers; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        c.demand(lp + "input_layernorm.weight", {hidden});
        c.demand(lp + "post_attention_layernorm.weight", {hidden});
        // Present on every layer whichever kind it is. `down_proj` is
        // row-parallel, so it is the one place this family's declaration is
        // sensitive to `tp_size`.
        c.demand(lp + "mlp.gate_proj.weight");
        c.demand(lp + "mlp.up_proj.weight");
        c.demand(lp + "mlp.down_proj.weight",
                 shape({hidden, shard(hf.intermediate_size, tp_size)}));

        if (hf.layer_types[i] == "linear_attention") {
            const std::string la = lp + "linear_attn.";
            c.demand(la + "in_proj_qkv.weight");
            c.demand(la + "in_proj_z.weight");
            c.demand(la + "in_proj_b.weight");
            c.demand(la + "in_proj_a.weight");
            c.demand(la + "conv1d.weight");
            c.demand_optional(la + "conv1d.bias");
            c.demand(la + "dt_bias");
            c.demand(la + "A_log");
            c.demand(la + "norm.weight");
            c.demand(la + "out_proj.weight");
        } else if (hf.layer_types[i] == "full_attention") {
            const std::string fa = lp + "self_attn.";
            c.demand(fa + "q_proj.weight");
            c.demand(fa + "k_proj.weight");
            c.demand(fa + "v_proj.weight");
            c.demand(fa + "o_proj.weight");
            c.demand(fa + "q_norm.weight", {hf.head_dim});
            c.demand(fa + "k_norm.weight", {hf.head_dim});
        }
    }
    return c;
}

/// Dispatch on `model_type`. Unknown or not-yet-declared families get an empty
/// contract, which is a no-op rather than a failure.
///
/// The `_text` spellings are not aliases to be tidied away: `parse_hf_config`
/// flattens a multimodal config down to its `text_config`, so a Qwen3-VL
/// checkpoint arrives here as `qwen3_vl_text`. `registry.cpp` registers both for
/// the same reason, and a mismatch here is silent — the family would simply
/// declare nothing.
inline pie_loader::TensorContract declare_model_contract(
    const HfConfig& hf,
    std::uint32_t tp_size) {
    const std::string& t = hf.model_type;
    // Exactly the set `registry.cpp` gives `bind_row_llama_like`.
    if (t == "qwen3" || t == "qwen2" || t == "llama" || t == "llama3" || t == "mistral") {
        return declare_llama_like_contract(hf, tp_size);
    }
    if (t == "gpt_oss") {
        return declare_gpt_oss_contract(hf, tp_size);
    }
    if (t == "qwen3_vl" || t == "qwen3_vl_text") {
        return declare_qwen3_vl_contract(hf, tp_size);
    }
    if (t == "qwen3_5" || t == "qwen3_5_text") {
        return declare_qwen3_5_contract(hf, tp_size);
    }
    return {};
}

}  // namespace pie_cuda_driver
