#include "model/qwen36.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/ops.h>
#include <mlx/io.h>

#include "ops/ops.hpp"
#include "linear_state_cache.hpp"  // delta-owned; complete type for gated_delta_net

namespace pie_metal_driver::model {

namespace mx = mlx::core;

namespace {

std::string layer_prefix(const std::string& root, std::int32_t il) {
    return root + "layers." + std::to_string(il) + ".";
}

Tensor to_heads(const Tensor& x, std::int32_t n_tokens,
                std::int32_t n_heads, std::int32_t head_dim) {
    return mx::reshape(x, {n_tokens, n_heads, head_dim});
}

Tensor from_heads(const Tensor& x, std::int32_t n_tokens, std::int32_t width) {
    return mx::reshape(x, {n_tokens, width});
}

// Env-gated per-decoder-layer hidden-state dump for parity bisection.
// When PIE_METAL_DUMP_LAYERS is set to a directory, writes the residual-stream
// hidden state ([n_tokens, hidden], fp32) after each decoder layer as
// `<dir>/qwen36_layer_<NN>.npy`, mirroring HF `output_hidden_states`:
//   layer_00 = token embeddings (HF index 0)
//   layer_<il+1> = output of decoder layer `il` (HF index il+1)
//   layer_final = post-final-norm hidden (pre-LM-head)
void maybe_dump_hidden(std::int32_t slot, const Tensor& h) {
    static const char* dir = std::getenv("PIE_METAL_DUMP_LAYERS");
    if (dir == nullptr) return;
    Tensor hf = mx::astype(h, mx::float32);
    mx::eval(hf);
    char name[32];
    if (slot < 0) {
        std::snprintf(name, sizeof(name), "qwen36_layer_final.npy");
    } else {
        std::snprintf(name, sizeof(name), "qwen36_layer_%02d.npy", slot);
    }
    mx::save(std::string(dir) + "/" + name, hf);
}

}  // namespace

Qwen36Graph::Qwen36Graph(ModelConfig cfg, ModelWeights weights)
    : cfg_(std::move(cfg)),
      w_(std::move(weights)),
      spec_(arch_spec_for(cfg_.arch, cfg_)) {}

// ── Full-attention layer: Qwen3 paged attention + output gate + partial RoPE ──
Tensor Qwen36Graph::full_attn_layer(std::int32_t il, Tensor hidden,
                                    const ForwardBatch& batch, KvCacheView& kv) {
    const auto& L = w_.layers[il];
    const std::int32_t N = batch.n_total;
    const float eps = cfg_.rms_norm_eps;
    const std::int32_t n_q  = cfg_.num_attention_heads;
    const std::int32_t n_kv = cfg_.num_key_value_heads;
    const std::int32_t d    = cfg_.head_dim;

    Tensor residual = hidden;
    Tensor cur = ops::rms_norm(hidden, *L.attn_norm, eps, /*plus_one=*/true);

    // q_proj is 2x wide: per head the layout is [query(d) | gate(d)].
    Tensor qg = ops::linear(L.q_proj, cur);                // [N, n_q*2*d]
    qg = mx::reshape(qg, {N, n_q, 2, d});
    Tensor Q    = mx::reshape(mx::slice(qg, {0, 0, 0, 0}, {N, n_q, 1, d}),
                              {N, n_q, d});
    Tensor gate = mx::reshape(mx::slice(qg, {0, 0, 1, 0}, {N, n_q, 2, d}),
                              {N, n_q, d});

    Tensor K = to_heads(ops::linear(L.k_proj, cur), N, n_kv, d);
    Tensor V = to_heads(ops::linear(L.v_proj, cur), N, n_kv, d);

    // Gemma-style (1+w) per-head Q/K RMSNorm, then partial RoPE.
    Q = ops::rms_norm(Q, *L.q_norm, eps, /*plus_one=*/true);
    K = ops::rms_norm(K, *L.k_norm, eps, /*plus_one=*/true);

    const float prf = cfg_.partial_rotary_factor;
    const std::int32_t rope_dims =
        (prf < 1.0f)
            ? std::max(2, 2 * static_cast<int>(std::floor(0.5f * prf * d)))
            : d;
    ops::RopeParams rp;
    rp.theta = cfg_.rope_theta;
    Q = ops::rope(Q, batch.positions, rope_dims, rp);
    K = ops::rope(K, batch.positions, rope_dims, rp);

    kv.append(il, K, V, batch.kv_write_indices);

    ops::AttnParams ap;
    ap.scale      = 0.0f;  // default 1/sqrt(head_dim)
    ap.n_heads    = n_q;
    ap.n_kv_heads = n_kv;
    ap.head_dim   = d;

    Tensor attn = ops::paged_attention(
        Q, kv.k_pages(il), kv.v_pages(il),
        batch.kv_page_indices, batch.qo_indptr, batch.kv_page_indptr,
        batch.kv_last_page_lens, kv.page_size(), ap);

    attn = from_heads(attn, N, n_q * d);
    // Output gate: attn *= sigmoid(gate) before o_proj.
    Tensor gate_flat = from_heads(gate, N, n_q * d);
    attn = ops::mul(attn, mx::sigmoid(gate_flat));

    Tensor attn_out = ops::linear(L.o_proj, attn);
    return ops::residual_add(attn_out, residual);
}

// ── Linear-attention layer: Gated DeltaNet (beta's op owns the core) ──
Tensor Qwen36Graph::linear_attn_layer(std::int32_t il, std::int32_t lin_ordinal,
                                      Tensor hidden, const ForwardBatch& batch) {
    const auto& L = w_.layers[il];
    const std::int32_t N = batch.n_total;
    const float eps = cfg_.rms_norm_eps;

    if (batch.lin_cache == nullptr || !batch.slot_ids) {
        throw std::runtime_error(
            "qwen3.6: linear-attention layer reached without lin_cache/slot_ids "
            "(executor must stage them for hybrid models)");
    }

    Tensor residual = hidden;
    Tensor cur = ops::rms_norm(hidden, *L.attn_norm, eps, /*plus_one=*/true);

    Tensor mixed_qkv = ops::linear(*L.la_in_proj_qkv, cur);  // [N, conv_dim] = [q|k|v]
    Tensor z = ops::linear(*L.la_in_proj_z, cur);            // [N, V_dim]
    Tensor a = ops::linear(*L.la_in_proj_a, cur);            // [N, V_h]
    Tensor b = ops::linear(*L.la_in_proj_b, cur);            // [N, V_h]

    ops::GdnParams p;
    p.n_heads_k   = spec_.linear_num_key_heads;
    p.n_heads_v   = spec_.linear_num_value_heads;
    p.head_k      = spec_.linear_key_head_dim;
    p.head_v      = spec_.linear_value_head_dim;
    p.conv_kernel = spec_.linear_conv_kernel_dim;
    p.norm_eps    = eps;

    Tensor out = ops::empty_tensor();
    if (batch.pure_decode) {
        // One token per request: batched decode (gather -> step -> scatter).
        out = ops::gated_delta_net(
            mixed_qkv, z, a, b, *L.la_conv1d_w, L.la_conv1d_b,
            *L.la_A_log, *L.la_dt_bias, *L.la_gate_norm,
            *batch.lin_cache, lin_ordinal, *batch.slot_ids, p);
    } else {
        // Prefill / mixed: ragged scan over per-request token spans.
        std::vector<int> qo = batch.qo_indptr_host;
        if (qo.empty()) qo = {0, N};  // single-request fallback (parity path)
        out = ops::gated_delta_net_varlen(
            mixed_qkv, z, a, b, *L.la_conv1d_w, L.la_conv1d_b,
            *L.la_A_log, *L.la_dt_bias, *L.la_gate_norm,
            *batch.lin_cache, lin_ordinal, *batch.slot_ids, qo, p);
    }

    Tensor attn_out = ops::linear(*L.la_out_proj, out);  // [N, hidden]
    return ops::residual_add(attn_out, residual);
}

// ── Shared dense SwiGLU MLP block ──
Tensor Qwen36Graph::mlp_block(std::int32_t il, Tensor hidden) {
    const auto& L = w_.layers[il];
    const float eps = cfg_.rms_norm_eps;

    Tensor residual = hidden;
    Tensor normed = ops::rms_norm(hidden, *L.ffn_norm, eps, /*plus_one=*/true);
    Tensor gate = ops::linear(*L.gate_proj, normed);
    Tensor up   = ops::linear(*L.up_proj,   normed);
    Tensor ffn  = ops::linear(*L.down_proj, ops::swiglu(gate, up));
    return ops::residual_add(ffn, residual);
}

Tensor Qwen36Graph::forward(const ForwardBatch& batch, KvCacheView& kv) {
    const float eps = cfg_.rms_norm_eps;
    const std::int32_t Lc = cfg_.num_hidden_layers;

    // Qwen does NOT scale embeddings (unlike Gemma).
    Tensor hidden = ops::embedding(w_.embed, batch.token_ids);
    maybe_dump_hidden(0, hidden);  // HF output_hidden_states[0] = embeddings

    std::int32_t lin_ordinal = 0;
    for (std::int32_t il = 0; il < Lc; ++il) {
        if (spec_.is_linear_attn_layer(il)) {
            hidden = linear_attn_layer(il, lin_ordinal++, hidden, batch);
        } else {
            hidden = full_attn_layer(il, hidden, batch, kv);
        }
        hidden = mlp_block(il, hidden);
        maybe_dump_hidden(il + 1, hidden);  // HF output_hidden_states[il+1]
    }

    hidden = ops::rms_norm(hidden, w_.final_norm, eps, /*plus_one=*/true);
    maybe_dump_hidden(-1, hidden);  // post-final-norm (pre-LM-head)
    Tensor sampled = ops::gather_rows(hidden, batch.logit_rows);

    const Tensor& lm_head = cfg_.tie_word_embeddings ? w_.embed : *w_.lm_head;
    return ops::linear(lm_head, sampled);  // [n_slots, vocab]
}

// ── Weight binding (Qwen3.6 hybrid) ──
ModelWeights bind_qwen36(const WeightSource& src, const ModelConfig& cfg) {
    ModelWeights w;

    // Qwen3.5 nests the text decoder under `model.language_model.` in the
    // multimodal checkpoint; some dumps drop the `language_model.` segment.
    std::string root = "model.language_model.";
    if (!src.has(root + "embed_tokens.weight")) {
        root = "model.";
    }

    w.embed      = src.get(root + "embed_tokens.weight");
    w.final_norm = src.get(root + "norm.weight");
    if (!cfg.tie_word_embeddings) {
        w.lm_head = src.try_get("lm_head.weight");
        if (!w.lm_head) w.lm_head = w.embed;
    }

    // conv1d weight is stored [conv_dim, 1, conv_K]; flatten to [conv_dim, K]
    // for beta's gated_delta_net (which expects [conv_dim, conv_K]).
    const std::int32_t conv_dim =
        2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const std::int32_t conv_k = cfg.linear_conv_kernel_dim;

    w.layers.resize(cfg.num_hidden_layers);
    for (std::int32_t il = 0; il < cfg.num_hidden_layers; ++il) {
        const std::string p = layer_prefix(root, il);
        LayerWeights& L = w.layers[il];

        // Pre-attn + pre-FFN norms (both layer kinds; (1+w) applied in graph).
        L.attn_norm = src.get(p + "input_layernorm.weight");
        L.ffn_norm  = src.get(p + "post_attention_layernorm.weight");

        const std::string la = p + "linear_attn.";
        if (src.has(la + "in_proj_qkv.weight")) {
            // Gated-DeltaNet linear-attention layer.
            L.la_in_proj_qkv = src.get(la + "in_proj_qkv.weight");  // [conv_dim, H]
            L.la_in_proj_z   = src.get(la + "in_proj_z.weight");    // [V_dim, H]
            L.la_in_proj_a   = src.get(la + "in_proj_a.weight");    // [V_h, H]
            L.la_in_proj_b   = src.get(la + "in_proj_b.weight");    // [V_h, H]
            L.la_conv1d_w    = mx::reshape(src.get(la + "conv1d.weight"),
                                           {conv_dim, conv_k});
            L.la_conv1d_b    = src.try_get(la + "conv1d.bias");
            L.la_A_log       = src.get(la + "A_log");
            L.la_dt_bias     = src.get(la + "dt_bias");
            L.la_gate_norm   = src.get(la + "norm.weight");         // [V_d]
            L.la_out_proj    = src.get(la + "out_proj.weight");     // [H, V_dim]
        } else {
            // Full-attention layer (Qwen3 + output gate; 2x-wide q_proj).
            const std::string sa = p + "self_attn.";
            L.q_proj = src.get(sa + "q_proj.weight");
            L.k_proj = src.get(sa + "k_proj.weight");
            L.v_proj = src.get(sa + "v_proj.weight");
            L.o_proj = src.get(sa + "o_proj.weight");
            L.q_norm = src.get(sa + "q_norm.weight");
            L.k_norm = src.get(sa + "k_norm.weight");
        }

        // Dense SwiGLU MLP (both layer kinds; MoE deferred — not on the 0.8B).
        L.gate_proj = src.get(p + "mlp.gate_proj.weight");
        L.up_proj   = src.get(p + "mlp.up_proj.weight");
        L.down_proj = src.get(p + "mlp.down_proj.weight");
    }
    return w;
}

}  // namespace pie_metal_driver::model
