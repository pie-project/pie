#include "model/gemma.hpp"

#include <cmath>
#include <string>

#include <mlx/ops.h>

#include "ops/ops.hpp"

namespace pie_metal_driver::model {

namespace {

std::string layer_prefix(std::int32_t il) {
    return "model.layers." + std::to_string(il) + ".";
}

Tensor to_heads(const Tensor& x, std::int32_t n_tokens,
                std::int32_t n_heads, std::int32_t head_dim) {
    return mlx::core::reshape(x, {n_tokens, n_heads, head_dim});
}

Tensor from_heads(const Tensor& x, std::int32_t n_tokens, std::int32_t width) {
    return mlx::core::reshape(x, {n_tokens, width});
}

}  // namespace

GemmaGraph::GemmaGraph(ModelConfig cfg, ModelWeights weights)
    : cfg_(std::move(cfg)),
      w_(std::move(weights)),
      spec_(arch_spec_for(cfg_.arch, cfg_)) {}

Tensor GemmaGraph::decoder_layer(std::int32_t il,
                                 Tensor hidden,
                                 const ForwardBatch& batch,
                                 KvCacheView& kv) {
    const auto& L = w_.layers[il];
    const std::int32_t n_total    = batch.n_total;
    const std::int32_t head_dim   = cfg_.head_dim;
    const std::int32_t n_q_heads  = cfg_.num_attention_heads;
    const std::int32_t n_kv_heads = cfg_.num_key_value_heads;
    const bool plus_one = spec_.norm_weight_plus_one;

    // ── Attention block (norm sandwich) ──
    Tensor residual = hidden;
    Tensor cur = ops::rms_norm(hidden, *L.attn_norm, cfg_.rms_norm_eps, plus_one);

    Tensor Q = apply_linear(L.q_proj, cur);
    Tensor K = apply_linear(L.k_proj, cur);
    Tensor V = apply_linear(L.v_proj, cur);

    Q = to_heads(Q, n_total, n_q_heads,  head_dim);
    K = to_heads(K, n_total, n_kv_heads, head_dim);
    V = to_heads(V, n_total, n_kv_heads, head_dim);

    // Gemma 3 per-head Q/K RMSNorm.
    if (spec_.has_qk_norm && L.q_norm && L.k_norm) {
        Q = ops::rms_norm(Q, *L.q_norm, cfg_.rms_norm_eps, plus_one);
        K = ops::rms_norm(K, *L.k_norm, cfg_.rms_norm_eps, plus_one);
    }

    // Gemma 3 uses a different RoPE base on sliding-window layers
    // (rope_local_base_freq) than on global layers (rope_theta).
    ops::RopeParams rp;
    rp.theta = (spec_.is_sliding_layer(il) && cfg_.rope_local_base_freq > 0.0f)
        ? cfg_.rope_local_base_freq
        : cfg_.rope_theta;
    Q = ops::rope(Q, batch.positions, head_dim, rp);
    K = ops::rope(K, batch.positions, head_dim, rp);

    kv.append(il, K, V, batch.kv_write_indices);

    ops::AttnParams ap;
    ap.scale = (spec_.query_pre_attn_scalar > 0.0f)
        ? 1.0f / std::sqrt(spec_.query_pre_attn_scalar)
        : 1.0f / std::sqrt(static_cast<float>(head_dim));
    ap.sliding_window = spec_.is_sliding_layer(il) ? spec_.sliding_window : 0;
    ap.softcap    = spec_.attn_softcap;   // Gemma 2 attention soft-cap
    ap.n_heads    = n_q_heads;
    ap.n_kv_heads = n_kv_heads;
    ap.head_dim   = head_dim;

    Tensor attn = ops::paged_attention(
        Q, kv.k_pages(il), kv.v_pages(il),
        batch.kv_page_indices, batch.qo_indptr, batch.kv_page_indptr,
        batch.kv_last_page_lens, kv.page_size(), ap);

    attn = from_heads(attn, n_total, n_q_heads * head_dim);
    Tensor attn_out = apply_linear(L.o_proj, attn);
    // Post-attention norm applied to the sub-layer output before the residual.
    if (L.post_attn_norm) {
        attn_out = ops::rms_norm(attn_out, *L.post_attn_norm,
                                 cfg_.rms_norm_eps, plus_one);
    }
    hidden = ops::residual_add(attn_out, residual);

    // ── FFN block (norm sandwich) ──
    Tensor ffn_residual = hidden;
    Tensor normed = ops::rms_norm(hidden, *L.ffn_norm, cfg_.rms_norm_eps, plus_one);
    Tensor gate = apply_linear(*L.gate_proj, normed);
    Tensor up   = apply_linear(*L.up_proj,   normed);
    Tensor ffn_out = apply_linear(*L.down_proj,
                                  ops::geglu(gate, up, /*tanh_approx=*/true));
    if (L.post_ffn_norm) {
        ffn_out = ops::rms_norm(ffn_out, *L.post_ffn_norm,
                                cfg_.rms_norm_eps, plus_one);
    }
    return ops::residual_add(ffn_out, ffn_residual);
}

Tensor GemmaGraph::forward(const ForwardBatch& batch, KvCacheView& kv) {
    Tensor hidden = ops::embedding(w_.embed, batch.token_ids);
    // Gemma scales embeddings by sqrt(hidden_size).
    hidden = ops::scale(hidden, std::sqrt(static_cast<float>(cfg_.hidden_size)));

    for (std::int32_t il = 0; il < cfg_.num_hidden_layers; ++il) {
        hidden = decoder_layer(il, hidden, batch, kv);
    }

    hidden = ops::rms_norm(hidden, w_.final_norm, cfg_.rms_norm_eps,
                           spec_.norm_weight_plus_one);
    Tensor sampled = ops::gather_rows(hidden, batch.logit_rows);

    // Gemma ties word embeddings.
    Tensor logits = w_.lm_head
        ? apply_linear(*w_.lm_head, sampled)   // [n_slots, vocab]
        : ops::linear(w_.embed, sampled);      // tied, dense embed

    // Gemma 2 final logit soft-cap.
    if (spec_.final_softcap > 0.0f) {
        logits = ops::softcap(logits, spec_.final_softcap);
    }
    return logits;
}

// ── Weight binding (Gemma 2 / 3) ──
ModelWeights bind_gemma(const WeightSource& src, const ModelConfig& cfg) {
    ModelWeights w;
    w.embed      = src.get("model.embed_tokens.weight");
    w.final_norm = src.get("model.norm.weight");
    // Prefer an explicit lm_head bundle when present; absent -> dense embed.
    w.lm_head = try_bind_linear(src, "lm_head", cfg);

    w.layers.resize(cfg.num_hidden_layers);
    for (std::int32_t il = 0; il < cfg.num_hidden_layers; ++il) {
        const std::string p = layer_prefix(il);
        LayerWeights& L = w.layers[il];

        // The four-norm sandwich.
        L.attn_norm      = src.get(p + "input_layernorm.weight");
        L.post_attn_norm = src.get(p + "post_attention_layernorm.weight");
        L.ffn_norm       = src.get(p + "pre_feedforward_layernorm.weight");
        L.post_ffn_norm  = src.get(p + "post_feedforward_layernorm.weight");

        L.q_proj = bind_linear(src, p + "self_attn.q_proj", cfg);
        L.k_proj = bind_linear(src, p + "self_attn.k_proj", cfg);
        L.v_proj = bind_linear(src, p + "self_attn.v_proj", cfg);
        L.o_proj = bind_linear(src, p + "self_attn.o_proj", cfg);

        // Gemma 3 per-head Q/K norm (absent on Gemma 2).
        L.q_norm = src.try_get(p + "self_attn.q_norm.weight");
        L.k_norm = src.try_get(p + "self_attn.k_norm.weight");

        L.gate_proj = bind_linear(src, p + "mlp.gate_proj", cfg);
        L.up_proj   = bind_linear(src, p + "mlp.up_proj", cfg);
        L.down_proj = bind_linear(src, p + "mlp.down_proj", cfg);
    }
    return w;
}

}  // namespace pie_metal_driver::model
