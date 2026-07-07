#include "model/gemma4.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <mlx/ops.h>
#include <mlx/io.h>

#include "ops/attention.hpp"
#include "ops/compiled.hpp"
#include "ops/ops.hpp"

namespace pie_metal_driver::model {

namespace mx = mlx::core;

namespace {

// ── Golden parity taps (sealed MLX reference) ──────────────────────────────
// When PIE_METAL_GOLDEN_DIR is set, write each tapped intermediate as
// `<dir>/<layer>.<kernel>.npy` (fp32); layer<0 -> `<kernel>.npy`. Mirrors the
// qwen3.6 harness (driver/metal/src/model/qwen36.cpp) so cosine_bisect.py
// localizes the FIRST diverging (kernel,layer). Compiled out of the bench build
// (PIE_METAL_NO_LAYER_DUMP) so it can never inject a decode-path sync. `dumping()`
// additionally swaps the fused FFN/PLE compiled regions for an eager,
// per-op-tapped path that is numerically identical to the compiled identity.
bool dumping() {
#ifndef PIE_METAL_NO_LAYER_DUMP
    static const bool on = std::getenv("PIE_METAL_GOLDEN_DIR") != nullptr;
    return on;
#else
    return false;
#endif
}

void dump_kernel(std::int32_t layer, const char* kernel, const Tensor& t) {
#ifndef PIE_METAL_NO_LAYER_DUMP
    static const char* dir = std::getenv("PIE_METAL_GOLDEN_DIR");
    if (dir == nullptr) return;
    Tensor f = mx::astype(t, mx::float32);
    mx::eval(f);
    std::string name = (layer < 0)
        ? std::string(kernel) + ".npy"
        : std::to_string(layer) + "." + kernel + ".npy";
    mx::save(std::string(dir) + "/" + name, f);
#else
    (void)layer; (void)kernel; (void)t;
#endif
}

// Route single-stream (R=1) pure_decode through the host-readback-free device
// decode attention (no per-token `to_host_i32` sync), unblocking async_eval
// pipelining and the whole-step compile trace. Default ON; opt out with
// PIE_DEVICE_DECODE=0 (falls back to the host-readback paged_attention path).
// The call site additionally guards on n_requests<=1 && softcap==0 so the
// fast path only fires where paged_attention_decode is bit-exact.
bool device_decode_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("PIE_DEVICE_DECODE");
        return !(e && e[0] == '0');  // default on; PIE_DEVICE_DECODE=0 disables
    }();
    return on;
}

// True when the host-readback-free decode fast path is valid for this batch.
bool use_device_decode(const model::ForwardBatch& batch, const ops::AttnParams& ap) {
    return device_decode_enabled() && batch.pure_decode &&
           batch.n_requests <= 1 && ap.softcap == 0.0f;
}

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

// ── Compiled PLE region (gemma4): GeGLU-gate the per-layer signal back into
// the residual stream — ple_input_gate -> geglu-tanh(·, signal) -> projection
// -> norm -> residual, fused into one compiled kernel sequence (the per-layer
// PLE glue, ~13% of decode). Capture-less; eps is the gemma4 config literal.
// in = {hidden, ple_input_gate_w, ple_signal, ple_projection_w, ple_norm_w}.
std::vector<Tensor> gemma4_ple(const std::vector<Tensor>& in) {
    Tensor gate  = ops::linear(in[1], in[0]);
    Tensor gated = ops::geglu(gate, in[2], /*tanh_approx=*/true);
    Tensor ple   = ops::linear(in[3], gated);
    ple = ops::rms_norm(ple, in[4], 1e-6f, /*plus_one=*/false);
    return { ops::residual_add(ple, in[0]) };
}

// ── Quantized (4-bit) variants of the FFN/PLE regions. Same op sequence with
// the GEMVs as fused dequant-in-GEMV `quantized_linear`. A separate fn = a
// separate compiled identity (the harness contract: distinct op sequence ->
// distinct cache instance). Quant params are the mlx-community gemma4 constants
// (4-bit / group_size 64), baked as literals into the function identity exactly
// like eps — these regions only run on a 4-bit checkpoint with those params.
// in = {hidden, ffn_norm_w, gate_w,gate_s,gate_b, up_w,up_s,up_b,
//       down_w,down_s,down_b, post_ffn_norm_w}.
std::vector<Tensor> gemma4_ffn_q4(const std::vector<Tensor>& in) {
    Tensor normed = ops::rms_norm(in[0], in[1], 1e-6f, /*plus_one=*/false);
    Tensor gate   = ops::quantized_linear(in[2], in[3], in[4], normed, 64, 4);
    Tensor up     = ops::quantized_linear(in[5], in[6], in[7], normed, 64, 4);
    Tensor down   = ops::quantized_linear(
        in[8], in[9], in[10], ops::geglu(gate, up, /*tanh_approx=*/true), 64, 4);
    Tensor post   = ops::rms_norm(down, in[11], 1e-6f, /*plus_one=*/false);
    return { ops::residual_add(post, in[0]) };
}

// in = {hidden, gate_w,gate_s,gate_b, ple_signal, proj_w,proj_s,proj_b,
//       ple_norm_w}.
std::vector<Tensor> gemma4_ple_q4(const std::vector<Tensor>& in) {
    Tensor gate  = ops::quantized_linear(in[1], in[2], in[3], in[0], 64, 4);
    Tensor gated = ops::geglu(gate, in[4], /*tanh_approx=*/true);
    Tensor ple   = ops::quantized_linear(in[5], in[6], in[7], gated, 64, 4);
    ple = ops::rms_norm(ple, in[8], 1e-6f, /*plus_one=*/false);
    return { ops::residual_add(ple, in[0]) };
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
    dump_kernel(il, "attn_norm", cur);

    Tensor q_lin = apply_linear(L.q_proj, cur);
    dump_kernel(il, "q_proj", q_lin);
    Tensor Q = to_heads(q_lin, n_total, n_q_heads, head_dim);
    Q = ops::rms_norm(Q, *L.q_norm, eps, /*plus_one=*/false);  // per-head Q-norm
    dump_kernel(il, "q_norm", Q);

    Tensor k_pages = ops::empty_tensor();  // placeholder; reassigned below
    Tensor v_pages = ops::empty_tensor();
    if (!is_shared) {
        Tensor k_lin = apply_linear(L.k_proj, cur);
        dump_kernel(il, "k_proj", k_lin);
        Tensor v_lin = apply_linear(L.v_proj, cur);
        dump_kernel(il, "v_proj", v_lin);
        Tensor K = to_heads(k_lin, n_total, n_kv_heads, head_dim);
        Tensor V = to_heads(v_lin, n_total, n_kv_heads, head_dim);
        K = ops::rms_norm(K, *L.k_norm, eps, /*plus_one=*/false);  // per-head K-norm
        dump_kernel(il, "k_norm", K);
        V = ops::rms_norm(V, eps);  // weightless V-norm before the KV write
        dump_kernel(il, "v_norm", V);
        // RoPE (Gemma-4 rotates after qk-norm). Shared layers only rotate Q.
        K = ops::rope(K, batch.positions, rope_dims, rp);
        dump_kernel(il, "rope_k", K);
        Q = ops::rope(Q, batch.positions, rope_dims, rp);
        dump_kernel(il, "rope_q", Q);
        kv.append(il, K, V, batch.kv_write_indices);
        k_pages = kv.k_pages(il);
        v_pages = kv.v_pages(il);
    } else {
        Q = ops::rope(Q, batch.positions, rope_dims, rp);
        dump_kernel(il, "rope_q", Q);
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

    Tensor attn = use_device_decode(batch, ap)
        ? ops::paged_attention_decode(
              Q, k_pages, v_pages,
              batch.kv_page_indices, batch.kv_last_page_lens,
              kv.page_size(), ap)
        : ops::paged_attention(
              Q, k_pages, v_pages,
              batch.kv_page_indices, batch.qo_indptr, batch.kv_page_indptr,
              batch.kv_last_page_lens, kv.page_size(), ap);

    attn = from_heads(attn, n_total, n_q_heads * head_dim);
    dump_kernel(il, "sdpa", attn);
    Tensor attn_out = apply_linear(L.o_proj, attn);
    dump_kernel(il, "o_proj", attn_out);
    attn_out = ops::rms_norm(attn_out, *L.post_attn_norm, eps, /*plus_one=*/false);
    dump_kernel(il, "post_attn_norm", attn_out);
    hidden = ops::residual_add(attn_out, residual);
    dump_kernel(il, "attn_resid", hidden);

    // ── FFN block (norm sandwich; GeGLU-tanh) — fused via mx::compile to
    // collapse the loose pointwise glue (sandwich norms + geglu + residual)
    // into one traced kernel sequence; the dominant batch=1 decode overhead.
    // On a 4-bit checkpoint the three GEMVs become fused dequant-in-GEMV via a
    // separate compiled identity (gemma4_ffn_q4). ──
    if (dumping()) {
        Tensor pre = hidden;
        Tensor normed = ops::rms_norm(hidden, *L.ffn_norm, eps, /*plus_one=*/false);
        dump_kernel(il, "ffn_norm", normed);
        Tensor gate = apply_linear(*L.gate_proj, normed);
        dump_kernel(il, "gate_proj", gate);
        Tensor up = apply_linear(*L.up_proj, normed);
        dump_kernel(il, "up_proj", up);
        Tensor g = ops::geglu(gate, up, /*tanh_approx=*/true);
        dump_kernel(il, "geglu", g);
        Tensor down = apply_linear(*L.down_proj, g);
        dump_kernel(il, "down_proj", down);
        Tensor post = ops::rms_norm(down, *L.post_ffn_norm, eps, /*plus_one=*/false);
        dump_kernel(il, "post_ffn_norm", post);
        hidden = ops::residual_add(post, pre);
        dump_kernel(il, "ffn_resid", hidden);
    } else if (L.gate_proj->quantized()) {
        hidden = ops::compiled(
            "gemma4.ffn.q4",
            {hidden, *L.ffn_norm,
             L.gate_proj->weight, *L.gate_proj->scales, *L.gate_proj->biases,
             L.up_proj->weight,   *L.up_proj->scales,   *L.up_proj->biases,
             L.down_proj->weight, *L.down_proj->scales, *L.down_proj->biases,
             *L.post_ffn_norm},
            gemma4_ffn_q4)[0];
    } else {
        hidden = ops::compiled(
            "gemma4.ffn",
            {hidden, *L.ffn_norm, L.gate_proj->weight, L.up_proj->weight,
             L.down_proj->weight, *L.post_ffn_norm},
            gemma4_ffn)[0];
    }

    // ── PLE residual: GeGLU-gate the per-layer signal back into the stream,
    // fused via mx::compile (ple_input_gate -> geglu -> projection -> norm ->
    // residual). The layer-output scalar stays separate (distinct guard). ──
    if (ple_signal && L.ple_input_gate && L.ple_projection && L.ple_norm) {
        if (dumping()) {
            Tensor pre = hidden;
            Tensor gate = apply_linear(*L.ple_input_gate, hidden);
            dump_kernel(il, "ple_gate", gate);
            Tensor gated = ops::geglu(gate, *ple_signal, /*tanh_approx=*/true);
            dump_kernel(il, "ple_gated", gated);
            Tensor ple = apply_linear(*L.ple_projection, gated);
            dump_kernel(il, "ple_proj", ple);
            ple = ops::rms_norm(ple, *L.ple_norm, eps, /*plus_one=*/false);
            dump_kernel(il, "ple_norm", ple);
            hidden = ops::residual_add(ple, pre);
            dump_kernel(il, "ple_resid", hidden);
        } else if (L.ple_input_gate->quantized()) {
            hidden = ops::compiled(
                "gemma4.ple.q4",
                {hidden,
                 L.ple_input_gate->weight, *L.ple_input_gate->scales,
                 *L.ple_input_gate->biases, *ple_signal,
                 L.ple_projection->weight, *L.ple_projection->scales,
                 *L.ple_projection->biases, *L.ple_norm},
                gemma4_ple_q4)[0];
        } else {
            hidden = ops::compiled(
                "gemma4.ple",
                {hidden, L.ple_input_gate->weight, *ple_signal,
                 L.ple_projection->weight, *L.ple_norm},
                gemma4_ple)[0];
        }
    }
    if (L.layer_scalar) {
        hidden = ops::mul(hidden, *L.layer_scalar);  // broadcast [1] over [N,H]
    }
    dump_kernel(il, "layer_out", hidden);
    return hidden;
}

Tensor Gemma4Graph::forward(const ForwardBatch& batch, KvCacheView& kv) {
    const std::int32_t N = batch.n_total;
    const std::int32_t H = cfg_.hidden_size;
    const std::int32_t Lc = cfg_.num_hidden_layers;
    const std::int32_t ple_dim = spec_.per_layer_emb_dim;
    const float eps = cfg_.rms_norm_eps;

    Tensor hidden = apply_embedding(w_.embed, batch.token_ids);
    hidden = ops::scale(hidden, std::sqrt(static_cast<float>(H)));
    dump_kernel(-1, "embed", hidden);

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
        Tensor token = apply_embedding(*w_.embed_per_layer, batch.token_ids);
        token = ops::scale(token, std::sqrt(static_cast<float>(ple_dim)));
        Tensor proj = apply_linear(*w_.ple_model_proj, hidden);
        proj = ops::scale(proj, 1.0f / std::sqrt(static_cast<float>(H)));
        // RMSNorm over each ple_dim row: reshape to [N*L, ple_dim].
        proj = mx::reshape(proj, {N * Lc, ple_dim});
        proj = ops::rms_norm(proj, *w_.ple_model_norm, eps, /*plus_one=*/false);
        proj = mx::reshape(proj, {N, Lc * ple_dim});
        ple_inputs = ops::scale(ops::add(proj, token), 1.0f / std::sqrt(2.0f));
        dump_kernel(-1, "ple_input", *ple_inputs);
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
    dump_kernel(-1, "final_norm", hidden);
    Tensor sampled = ops::gather_rows(hidden, batch.logit_rows);

    // lm_head: an explicit (possibly quantized) bundle wins — delta synthesizes
    // a quantized lm_head from the tied embed for the 262k-vocab BW win. Absent
    // (v1 tied) -> the dense embed GEMV.
    Tensor logits = w_.lm_head
        ? apply_linear(*w_.lm_head, sampled)      // [n_slots, vocab]
        : apply_linear(w_.embed, sampled);        // tied embed (dense or quant)
    dump_kernel(-1, "logits", logits);

    if (spec_.final_softcap > 0.0f) {
        logits = ops::softcap(logits, spec_.final_softcap);
        dump_kernel(-1, "logits_softcap", logits);
    }
    return logits;
}

// ── Weight binding (Gemma-4 dense E2B / E4B) ──
ModelWeights bind_gemma4(const WeightSource& src, const ModelConfig& cfg) {
    ModelWeights w;

    // Gemma-4 nests the text decoder under `model.language_model.` in the
    // multimodal checkpoint; some dumps drop the `language_model.` segment.
    // Detect the live prefix from the final norm (always present — unlike the
    // embed table, which the embed-drop optimization may omit entirely).
    std::string root = "model.language_model.";
    if (!src.has(root + "norm.weight")) {
        root = "model.";
    }

    w.final_norm = src.get(root + "norm.weight");
    // lm_head: prefer an explicit bundle when present — delta synthesizes a
    // (quantized) lm_head from the tied embed for the 262k-vocab BW win, so
    // check `lm_head.*` even when tie_word_embeddings=true. Absent -> the graph
    // falls back to the dense embed (tied, v1 checkpoints).
    w.lm_head = try_bind_linear(src, "lm_head", cfg);
    // Input embed: dense bf16 table, or — when dropped for true-4-bit memory
    // parity — the tied quantized lm_head, dequant-gathered per token.
    w.embed = bind_embedding(src, root + "embed_tokens", w.lm_head, cfg);

    // PLE model-level triple (absent on the 26B-A4B variant where ple_dim==0).
    // The per-layer token table is a gather — quant-aware bind so it can be
    // 4-bit dequant-gathered (apply_embedding) when the checkpoint quantizes it.
    w.embed_per_layer = try_bind_linear(src, root + "embed_tokens_per_layer", cfg);
    w.ple_model_proj  = try_bind_linear(src, root + "per_layer_model_projection", cfg);
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
        L.q_proj = bind_linear(src, p + "self_attn.q_proj", cfg);
        L.q_norm = src.get(p + "self_attn.q_norm.weight");
        // K/V/k_norm: present on non-shared layers (HF keeps them on shared
        // layers too in some dumps — bind when present, the graph only reads
        // them on non-shared layers).
        if (src.has(p + "self_attn.k_proj.weight"))
            L.k_proj = bind_linear(src, p + "self_attn.k_proj", cfg);
        if (src.has(p + "self_attn.v_proj.weight"))
            L.v_proj = bind_linear(src, p + "self_attn.v_proj", cfg);
        L.k_norm = src.try_get(p + "self_attn.k_norm.weight");
        L.o_proj = bind_linear(src, p + "self_attn.o_proj", cfg);

        // Dense MLP (GeGLU).
        L.gate_proj = bind_linear(src, p + "mlp.gate_proj", cfg);
        L.up_proj   = bind_linear(src, p + "mlp.up_proj", cfg);
        L.down_proj = bind_linear(src, p + "mlp.down_proj", cfg);

        // PLE per-layer triple + optional learnable output scalar.
        L.ple_input_gate = try_bind_linear(src, p + "per_layer_input_gate", cfg);
        L.ple_projection = try_bind_linear(src, p + "per_layer_projection", cfg);
        L.ple_norm       = src.try_get(p + "post_per_layer_input_norm.weight");
        L.layer_scalar   = src.try_get(p + "layer_scalar");
    }
    return w;
}

}  // namespace pie_metal_driver::model
