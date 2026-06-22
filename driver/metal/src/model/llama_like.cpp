#include "model/llama_like.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <mlx/ops.h>

#include "ops/ops.hpp"

namespace pie_metal_driver::model {

namespace {

// HF tensor-name helpers (canonical "model." namespace; multimodal prefix
// remapping is the loader's job before binding).
std::string layer_prefix(std::int32_t il) {
    return "model.layers." + std::to_string(il) + ".";
}

// Reshape a [n_tokens, n_heads*head_dim] projection into per-head
// [n_tokens, n_heads, head_dim] (zero-copy; head-major rows split cleanly).
Tensor to_heads(const Tensor& x, std::int32_t n_tokens,
                std::int32_t n_heads, std::int32_t head_dim) {
    return mlx::core::reshape(x, {n_tokens, n_heads, head_dim});
}

// Collapse per-head [n_tokens, n_heads, head_dim] back to [n_tokens, n_heads*head_dim].
Tensor from_heads(const Tensor& x, std::int32_t n_tokens, std::int32_t width) {
    return mlx::core::reshape(x, {n_tokens, width});
}

// Build RoPE parameters from config/spec. Plain theta-only RoPE for
// Llama-3.0 / Qwen 2/3; YaRN smooth-ramp for Ministral 3.
//
// NOTE: Llama-3.1 NTK-by-parts (freq_factors) is not yet expressible through
// beta's `RopeParams` (no per-dim factor channel). Tracked as a follow-up
// ops extension; plain theta RoPE is correct for Llama-3.0/Qwen/Mistral.
ops::RopeParams rope_params_for(const ModelConfig& c, const ArchSpec& s) {
    ops::RopeParams rp;
    rp.theta = c.rope_theta;
    if (s.yarn_n_ctx_orig > 0) {
        rp.yarn           = true;
        rp.scaling_factor = c.rope_scaling_factor;
        rp.yarn_orig_ctx  = static_cast<float>(s.yarn_n_ctx_orig);
        rp.yarn_beta_fast = s.yarn_beta_fast;
        rp.yarn_beta_slow = s.yarn_beta_slow;
        rp.yarn_mscale    = s.yarn_attn_factor;
    }
    return rp;
}

}  // namespace

LlamaLikeGraph::LlamaLikeGraph(ModelConfig cfg, ModelWeights weights)
    : cfg_(std::move(cfg)),
      w_(std::move(weights)),
      spec_(arch_spec_for(cfg_.arch, cfg_)) {}

Tensor LlamaLikeGraph::ffn(const LayerWeights& L, Tensor x) {
    if (spec_.n_experts > 0) {
        // MoE FFN (Qwen3-MoE / Mixtral). Wiring follows once the routed-expert
        // gather/scatter op lands; the dense path below is the first target.
        throw std::runtime_error(
            "LlamaLikeGraph: MoE FFN not yet implemented (dense archs only)");
    }
    // Dense gated MLP. gate/up project to intermediate, activation gates,
    // down projects back to hidden.
    Tensor gate = ops::linear(*L.gate_proj, x);
    Tensor up   = ops::linear(*L.up_proj,   x);
    Tensor act  = spec_.ffn_use_gelu ? ops::geglu(gate, up)
                                     : ops::swiglu(gate, up);
    return ops::linear(*L.down_proj, act);
}

Tensor LlamaLikeGraph::decoder_layer(std::int32_t il,
                                     Tensor hidden,
                                     const ForwardBatch& batch,
                                     KvCacheView& kv) {
    const auto& L = w_.layers[il];
    const std::int32_t n_total    = batch.n_total;
    const std::int32_t head_dim   = cfg_.head_dim;
    const std::int32_t n_q_heads  = cfg_.num_attention_heads;
    const std::int32_t n_kv_heads = cfg_.num_key_value_heads;

    // ── Attention block ──
    Tensor residual = hidden;
    Tensor cur = L.attn_norm
        ? ops::rms_norm(hidden, *L.attn_norm, cfg_.rms_norm_eps,
                        spec_.norm_weight_plus_one)
        : hidden;  // post-norm archs feed the residual straight in

    Tensor Q = ops::linear(L.q_proj, cur);
    Tensor K = ops::linear(L.k_proj, cur);
    Tensor V = ops::linear(L.v_proj, cur);

    if (spec_.has_qkv_bias) {
        if (L.q_bias) Q = ops::add_bias(Q, *L.q_bias);
        if (L.k_bias) K = ops::add_bias(K, *L.k_bias);
        if (L.v_bias) V = ops::add_bias(V, *L.v_bias);
    }

    Q = to_heads(Q, n_total, n_q_heads,  head_dim);
    K = to_heads(K, n_total, n_kv_heads, head_dim);
    V = to_heads(V, n_total, n_kv_heads, head_dim);

    // Per-head Q/K RMSNorm (Qwen3). Weight length head_dim; normalizes over
    // the last axis, broadcasting across heads/tokens.
    if (spec_.has_qk_norm && L.q_norm && L.k_norm) {
        Q = ops::rms_norm(Q, *L.q_norm, cfg_.rms_norm_eps,
                          spec_.norm_weight_plus_one);
        K = ops::rms_norm(K, *L.k_norm, cfg_.rms_norm_eps,
                          spec_.norm_weight_plus_one);
    }

    const ops::RopeParams rp = rope_params_for(cfg_, spec_);
    Q = ops::rope(Q, batch.positions, head_dim, rp);
    K = ops::rope(K, batch.positions, head_dim, rp);

    // Write new K/V into the paged cache, then attend over the full cache.
    kv.append(il, K, V, batch.kv_write_indices);

    ops::AttnParams ap;
    ap.scale = (spec_.query_pre_attn_scalar > 0.0f)
        ? 1.0f / std::sqrt(spec_.query_pre_attn_scalar)
        : 1.0f / std::sqrt(static_cast<float>(head_dim));
    ap.sliding_window = spec_.is_sliding_layer(il) ? spec_.sliding_window : 0;
    ap.softcap    = spec_.attn_softcap;
    ap.n_heads    = n_q_heads;
    ap.n_kv_heads = n_kv_heads;
    ap.head_dim   = head_dim;

    Tensor attn = ops::paged_attention(
        Q, kv.k_pages(il), kv.v_pages(il),
        batch.kv_page_indices, batch.qo_indptr, batch.kv_page_indptr,
        batch.kv_last_page_lens, kv.page_size(), ap);

    attn = from_heads(attn, n_total, n_q_heads * head_dim);
    Tensor attn_out = ops::linear(L.o_proj, attn);
    hidden = ops::residual_add(attn_out, residual);

    // ── FFN block ──
    Tensor ffn_residual = hidden;
    Tensor normed = L.ffn_norm
        ? ops::rms_norm(hidden, *L.ffn_norm, cfg_.rms_norm_eps,
                        spec_.norm_weight_plus_one)
        : hidden;
    Tensor ffn_out = ffn(L, normed);
    return ops::residual_add(ffn_out, ffn_residual);
}

Tensor LlamaLikeGraph::forward(const ForwardBatch& batch, KvCacheView& kv) {
    // Embed tokens -> [n_total, hidden].
    Tensor hidden = ops::embedding(w_.embed, batch.token_ids);

    for (std::int32_t il = 0; il < cfg_.num_hidden_layers; ++il) {
        hidden = decoder_layer(il, hidden, batch, kv);
    }

    // Final norm, then gather only the rows that produce logits.
    hidden = ops::rms_norm(hidden, w_.final_norm, cfg_.rms_norm_eps,
                           spec_.norm_weight_plus_one);
    Tensor sampled = ops::gather_rows(hidden, batch.logit_rows);

    const Tensor& lm_head = cfg_.tie_word_embeddings ? w_.embed : *w_.lm_head;
    Tensor logits = ops::linear(lm_head, sampled);  // [n_slots, vocab]

    if (spec_.final_softcap > 0.0f) {
        logits = ops::softcap(logits, spec_.final_softcap);
    }
    return logits;
}

// ── Weight binding (Llama-like family) ──
ModelWeights bind_llama_like(const WeightSource& src, const ModelConfig& cfg) {
    ModelWeights w;
    w.embed      = src.get("model.embed_tokens.weight");
    w.final_norm = src.get("model.norm.weight");
    if (!cfg.tie_word_embeddings) {
        w.lm_head = src.try_get("lm_head.weight");
        if (!w.lm_head) w.lm_head = w.embed;  // defensive fallback
    }

    w.layers.resize(cfg.num_hidden_layers);
    for (std::int32_t il = 0; il < cfg.num_hidden_layers; ++il) {
        const std::string p = layer_prefix(il);
        LayerWeights& L = w.layers[il];

        L.attn_norm = src.try_get(p + "input_layernorm.weight");
        L.ffn_norm  = src.try_get(p + "post_attention_layernorm.weight");

        L.q_proj = src.get(p + "self_attn.q_proj.weight");
        L.k_proj = src.get(p + "self_attn.k_proj.weight");
        L.v_proj = src.get(p + "self_attn.v_proj.weight");
        L.o_proj = src.get(p + "self_attn.o_proj.weight");

        // Optional Qwen2 QKV bias.
        L.q_bias = src.try_get(p + "self_attn.q_proj.bias");
        L.k_bias = src.try_get(p + "self_attn.k_proj.bias");
        L.v_bias = src.try_get(p + "self_attn.v_proj.bias");

        // Optional Qwen3 per-head Q/K norm.
        L.q_norm = src.try_get(p + "self_attn.q_norm.weight");
        L.k_norm = src.try_get(p + "self_attn.k_norm.weight");

        // Dense MLP.
        L.gate_proj = src.try_get(p + "mlp.gate_proj.weight");
        L.up_proj   = src.try_get(p + "mlp.up_proj.weight");
        L.down_proj = src.try_get(p + "mlp.down_proj.weight");

        // MoE router (experts bound by the MoE path once implemented).
        L.moe_router = src.try_get(p + "mlp.gate.weight");
    }
    return w;
}

}  // namespace pie_metal_driver::model
