#include "model/gemma4.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <mlx/ops.h>

#include "ops/compiled.hpp"
#include "ops/ops.hpp"

namespace pie_metal_driver::model {

namespace mx = mlx::core;

namespace {

std::string layer_prefix(const std::string& root, std::int32_t il) {
    return root + "layers." + std::to_string(il) + ".";
}

// ── Compiled FFN region (gemma4): pre-FFN norm -> gate/up -> GeGLU-tanh ->
// down -> post-FFN norm -> residual, fused via mx::compile into one kernel
// sequence. This collapses the loose-pointwise launch-storm (two sandwich
// norms + geglu + residual around the GEMVs) that dominates batch=1 decode.
// Must be a capture-less function pointer; eps is the gemma4 config literal
// (1e-6, plain RMSNorm). in = {hidden, ffn_norm_w, gate_w, up_w, down_w,
// post_ffn_norm_w}.
std::vector<Tensor> gemma4_ffn(const std::vector<Tensor>& in) {
    Tensor normed = ops::rms_norm(in[0], in[1], 1e-6f, /*plus_one=*/false);
    Tensor gate   = ops::linear(in[2], normed);
    Tensor up     = ops::linear(in[3], normed);
    Tensor down   = ops::linear(in[4], ops::geglu(gate, up, /*tanh_approx=*/true));
    Tensor post   = ops::rms_norm(down, in[5], 1e-6f, /*plus_one=*/false);
    return { ops::residual_add(post, in[0]) };
}

Tensor to_heads(const Tensor& x, std::int32_t n_tokens,
                std::int32_t n_heads, std::int32_t head_dim) {
    return mx::reshape(x, {n_tokens, n_heads, head_dim});
}

Tensor from_heads(const Tensor& x, std::int32_t n_tokens, std::int32_t width) {
    return mx::reshape(x, {n_tokens, width});
}

}  // namespace

Gemma4Graph::Gemma4Graph(ModelConfig cfg, ModelWeights weights)
    : cfg_(std::move(cfg)),
      w_(std::move(weights)),
      spec_(arch_spec_for(cfg_.arch, cfg_)) {}

Tensor Gemma4Graph::decoder_layer(std::int32_t il,
                                  Tensor hidden,
                                  const std::optional<Tensor>& ple_signal,
                                  const ForwardBatch& batch,
                                  KvCacheView& kv) {
    const auto& L = w_.layers[il];
    const std::int32_t n_total = batch.n_total;
    const float eps = cfg_.rms_norm_eps;
    const std::int32_t n_layers = cfg_.num_hidden_layers;

    // ── Per-layer geometry ──
    const bool is_full   = !spec_.is_sliding_layer(il);
    const bool is_shared = spec_.gemma4_is_kv_shared(il, n_layers);
    const std::int32_t head_dim =
        (is_full && cfg_.global_head_dim > 0) ? cfg_.global_head_dim
                                              : cfg_.head_dim;
    const std::int32_t n_q_heads  = cfg_.num_attention_heads;
    const std::int32_t n_kv_heads =
        (is_full && cfg_.num_global_kv_heads > 0) ? cfg_.num_global_kv_heads
                                                  : cfg_.num_key_value_heads;

    // Per-attention-type RoPE base + optional partial rotary.
    const float prf = cfg_.partial_rotary_factor;
    const std::int32_t rope_dims =
        (prf < 1.0f) ? static_cast<std::int32_t>(prf * head_dim) : head_dim;
    ops::RopeParams rp;
    rp.theta = (is_full || cfg_.rope_local_base_freq <= 0.0f)
                   ? cfg_.rope_theta
                   : cfg_.rope_local_base_freq;

    // ── Attention block (norm sandwich; PLAIN RMSNorm, no +1) ──
    Tensor residual = hidden;
    Tensor cur = ops::rms_norm(hidden, *L.attn_norm, eps, /*plus_one=*/false);

    Tensor Q = to_heads(ops::linear(L.q_proj, cur), n_total, n_q_heads, head_dim);
    Q = ops::rms_norm(Q, *L.q_norm, eps, /*plus_one=*/false);  // per-head Q-norm

    Tensor k_pages = w_.embed;  // placeholder; reassigned below
    Tensor v_pages = w_.embed;
    if (!is_shared) {
        Tensor K = to_heads(ops::linear(L.k_proj, cur), n_total, n_kv_heads, head_dim);
        Tensor V = to_heads(ops::linear(L.v_proj, cur), n_total, n_kv_heads, head_dim);
        K = ops::rms_norm(K, *L.k_norm, eps, /*plus_one=*/false);  // per-head K-norm
        V = ops::rms_norm(V, eps);  // weightless V-norm before the KV write
        // RoPE (Gemma-4 rotates after qk-norm). Shared layers only rotate Q.
        K = ops::rope(K, batch.positions, rope_dims, rp);
        Q = ops::rope(Q, batch.positions, rope_dims, rp);
        kv.append(il, K, V, batch.kv_write_indices);
        k_pages = kv.k_pages(il);
        v_pages = kv.v_pages(il);
    } else {
        Q = ops::rope(Q, batch.positions, rope_dims, rp);
        const std::int32_t src = spec_.gemma4_kv_source(il, n_layers);
        if (src < 0) {
            throw std::runtime_error(
                "gemma4: no KV source layer for shared layer " +
                std::to_string(il));
        }
        k_pages = kv.k_pages(src);
        v_pages = kv.v_pages(src);
    }

    ops::AttnParams ap;
    ap.scale          = 1.0f;  // Gemma-4: Q/K-norm absorbs 1/sqrt(d)
    ap.sliding_window = is_full ? 0 : spec_.sliding_window;
    ap.softcap        = spec_.attn_softcap;  // 0 on gemma4 (dropped attn softcap)
    ap.n_heads        = n_q_heads;
    ap.n_kv_heads     = n_kv_heads;
    ap.head_dim       = head_dim;

    Tensor attn = ops::paged_attention(
        Q, k_pages, v_pages,
        batch.kv_page_indices, batch.qo_indptr, batch.kv_page_indptr,
        batch.kv_last_page_lens, kv.page_size(), ap);

    attn = from_heads(attn, n_total, n_q_heads * head_dim);
    Tensor attn_out = ops::linear(L.o_proj, attn);
    attn_out = ops::rms_norm(attn_out, *L.post_attn_norm, eps, /*plus_one=*/false);
    hidden = ops::residual_add(attn_out, residual);

    // ── FFN block (norm sandwich; GeGLU-tanh) — fused via mx::compile to
    // collapse the loose pointwise glue (sandwich norms + geglu + residual)
    // into one traced kernel sequence; the dominant batch=1 decode overhead. ──
    hidden = ops::compiled(
        "gemma4.ffn",
        {hidden, *L.ffn_norm, *L.gate_proj, *L.up_proj, *L.down_proj,
         *L.post_ffn_norm},
        gemma4_ffn)[0];

    // ── PLE residual: GeGLU-gate the per-layer signal back into the stream ──
    // ple_gate = ple_input_gate(hidden); GeGLU_tanh(ple_gate, signal);
    // project to hidden; RMSNorm; add. Then the per-layer output scalar.
    if (ple_signal && L.ple_input_gate && L.ple_projection && L.ple_norm) {
        Tensor ple_gate  = ops::linear(*L.ple_input_gate, hidden);     // [N, ple_dim]
        Tensor ple_gated = ops::geglu(ple_gate, *ple_signal, /*tanh_approx=*/true);
        Tensor ple = ops::linear(*L.ple_projection, ple_gated);        // [N, hidden]
        ple = ops::rms_norm(ple, *L.ple_norm, eps, /*plus_one=*/false);
        hidden = ops::residual_add(ple, hidden);
    }
    if (L.layer_scalar) {
        hidden = ops::mul(hidden, *L.layer_scalar);  // broadcast [1] over [N,H]
    }
    return hidden;
}

Tensor Gemma4Graph::forward(const ForwardBatch& batch, KvCacheView& kv) {
    const std::int32_t N = batch.n_total;
    const std::int32_t H = cfg_.hidden_size;
    const std::int32_t Lc = cfg_.num_hidden_layers;
    const std::int32_t ple_dim = spec_.per_layer_emb_dim;
    const float eps = cfg_.rms_norm_eps;

    Tensor hidden = ops::embedding(w_.embed, batch.token_ids);
    hidden = ops::scale(hidden, std::sqrt(static_cast<float>(H)));

    // ── Per-Layer-Embedding (PLE) inputs, computed once up-front ──
    //   token = embed_per_layer[ids] * sqrt(ple_dim)            [N, L*ple_dim]
    //   proj  = (main_embed @ ple_model_proj.T) * 1/sqrt(H)     [N, L*ple_dim]
    //   proj  = rms_norm(proj, ple_model_norm)   (per ple_dim row)
    //   ple   = (proj + token) * 1/sqrt(2)                      [N, L*ple_dim]
    // Each layer consumes its `[N, ple_dim]` column slice.
    std::optional<Tensor> ple_inputs;  // [N, L*ple_dim]
    const bool ple_active =
        ple_dim > 0 && w_.embed_per_layer && w_.ple_model_proj && w_.ple_model_norm;
    if (ple_active) {
        Tensor token = ops::embedding(*w_.embed_per_layer, batch.token_ids);
        token = ops::scale(token, std::sqrt(static_cast<float>(ple_dim)));
        Tensor proj = ops::linear(*w_.ple_model_proj, hidden);
        proj = ops::scale(proj, 1.0f / std::sqrt(static_cast<float>(H)));
        // RMSNorm over each ple_dim row: reshape to [N*L, ple_dim].
        proj = mx::reshape(proj, {N * Lc, ple_dim});
        proj = ops::rms_norm(proj, *w_.ple_model_norm, eps, /*plus_one=*/false);
        proj = mx::reshape(proj, {N, Lc * ple_dim});
        ple_inputs = ops::scale(ops::add(proj, token), 1.0f / std::sqrt(2.0f));
    }

    for (std::int32_t il = 0; il < Lc; ++il) {
        std::optional<Tensor> signal;
        if (ple_inputs) {
            // Column slice [il*ple_dim, (il+1)*ple_dim) -> [N, ple_dim].
            signal = mx::slice(*ple_inputs, {0, il * ple_dim},
                               {N, (il + 1) * ple_dim});
        }
        hidden = decoder_layer(il, hidden, signal, batch, kv);
    }

    hidden = ops::rms_norm(hidden, w_.final_norm, eps, /*plus_one=*/false);
    Tensor sampled = ops::gather_rows(hidden, batch.logit_rows);

    const Tensor& lm_head = cfg_.tie_word_embeddings ? w_.embed : *w_.lm_head;
    Tensor logits = ops::linear(lm_head, sampled);  // [n_slots, vocab]

    if (spec_.final_softcap > 0.0f) {
        logits = ops::softcap(logits, spec_.final_softcap);
    }
    return logits;
}

// ── Weight binding (Gemma-4 dense E2B / E4B) ──
ModelWeights bind_gemma4(const WeightSource& src, const ModelConfig& cfg) {
    ModelWeights w;

    // Gemma-4 nests the text decoder under `model.language_model.` in the
    // multimodal checkpoint; some dumps drop the `language_model.` segment.
    // Detect the live prefix from the embedding table.
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

    // PLE model-level triple (absent on the 26B-A4B variant where ple_dim==0).
    w.embed_per_layer = src.try_get(root + "embed_tokens_per_layer.weight");
    w.ple_model_proj  = src.try_get(root + "per_layer_model_projection.weight");
    w.ple_model_norm  = src.try_get(root + "per_layer_projection_norm.weight");

    w.layers.resize(cfg.num_hidden_layers);
    for (std::int32_t il = 0; il < cfg.num_hidden_layers; ++il) {
        const std::string p = layer_prefix(root, il);
        LayerWeights& L = w.layers[il];

        // The four-norm sandwich (plain-RMSNorm gains).
        L.attn_norm      = src.get(p + "input_layernorm.weight");
        L.post_attn_norm = src.get(p + "post_attention_layernorm.weight");
        L.ffn_norm       = src.get(p + "pre_feedforward_layernorm.weight");
        L.post_ffn_norm  = src.get(p + "post_feedforward_layernorm.weight");

        // Q always present; per-head Q/K-norm always present on Gemma-4.
        L.q_proj = src.get(p + "self_attn.q_proj.weight");
        L.q_norm = src.get(p + "self_attn.q_norm.weight");
        // K/V/k_norm: present on non-shared layers (HF keeps them on shared
        // layers too in some dumps — bind when present, the graph only reads
        // them on non-shared layers).
        if (auto k = src.try_get(p + "self_attn.k_proj.weight")) L.k_proj = *k;
        if (auto v = src.try_get(p + "self_attn.v_proj.weight")) L.v_proj = *v;
        L.k_norm = src.try_get(p + "self_attn.k_norm.weight");
        L.o_proj = src.get(p + "self_attn.o_proj.weight");

        // Dense MLP (GeGLU).
        L.gate_proj = src.get(p + "mlp.gate_proj.weight");
        L.up_proj   = src.get(p + "mlp.up_proj.weight");
        L.down_proj = src.get(p + "mlp.down_proj.weight");

        // PLE per-layer triple + optional learnable output scalar.
        L.ple_input_gate = src.try_get(p + "per_layer_input_gate.weight");
        L.ple_projection = src.try_get(p + "per_layer_projection.weight");
        L.ple_norm       = src.try_get(p + "post_per_layer_input_norm.weight");
        L.layer_scalar   = src.try_get(p + "layer_scalar");
    }
    return w;
}

}  // namespace pie_metal_driver::model
