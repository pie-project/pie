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

#include "ops/attention.hpp"
#include "ops/ops.hpp"
#include "linear_state_cache.hpp"  // delta-owned; complete type for gated_delta_net

namespace pie_metal_driver::model {

namespace mx = mlx::core;

namespace {

// Route single-stream (R=1) pure_decode full-attn layers through the host-
// readback-free device decode attention (no per-token to_host_i32 sync),
// unblocking async_eval pipelining. Default ON; PIE_DEVICE_DECODE=0 opts out.
// The call site additionally guards on n_requests<=1 && softcap==0 so the fast
// path only fires where paged_attention_decode is bit-exact.
bool device_decode_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("PIE_DEVICE_DECODE");
        return !(e && e[0] == '0');  // default on; PIE_DEVICE_DECODE=0 disables
    }();
    return on;
}

bool use_device_decode(const model::ForwardBatch& batch, const ops::AttnParams& ap) {
    return device_decode_enabled() && batch.pure_decode &&
           batch.n_requests <= 1 && ap.softcap == 0.0f;
}

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
#ifndef PIE_METAL_NO_LAYER_DUMP
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
#else
    // Pipelined/bench build (-DPIE_METAL_NO_LAYER_DUMP): the parity dump — the
    // only mx::eval in the decode path — is compiled out, so a stray
    // PIE_METAL_DUMP_LAYERS can't inject a per-token sync barrier and pollute
    // the async_eval pipeline / ceiling-confirmation numbers.
    (void)slot;
    (void)h;
#endif
}

// Env-gated PER-KERNEL golden dump for the raw-Metal cosine-bisection gate.
// When PIE_METAL_GOLDEN_DIR is a directory, writes each tapped intermediate as
// `<dir>/<layer>.<kernel>.npy` (fp32), e.g. `7.gdn_core.npy`, `3.rope_q.npy`.
// Layer-less tensors (embed/final_norm/logits) pass layer < 0 → `<kernel>.npy`.
// This is the sealed MLX-path golden; delta/beta emit identically-named
// raw-Metal dumps so cosine_bisect.py localizes the FIRST diverging
// (kernel,layer) below 0.99999 — "gdn_core layer 7", not just "logits wrong".
// Compiled out of the bench/pipelined build (PIE_METAL_NO_LAYER_DUMP) with the
// same discipline as maybe_dump_hidden so it can never inject a decode-path sync.
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
    (void)layer;
    (void)kernel;
    (void)t;
#endif
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
    dump_kernel(il, "attn_norm", cur);

    // q_proj is 2x wide: per head the layout is [query(d) | gate(d)].
    Tensor qg = apply_linear(L.q_proj, cur);                // [N, n_q*2*d]
    dump_kernel(il, "q_proj", qg);
    qg = mx::reshape(qg, {N, n_q, 2, d});
    Tensor Q    = mx::reshape(mx::slice(qg, {0, 0, 0, 0}, {N, n_q, 1, d}),
                              {N, n_q, d});
    Tensor gate = mx::reshape(mx::slice(qg, {0, 0, 1, 0}, {N, n_q, 2, d}),
                              {N, n_q, d});

    Tensor K = to_heads(apply_linear(L.k_proj, cur), N, n_kv, d);
    Tensor V = to_heads(apply_linear(L.v_proj, cur), N, n_kv, d);
    dump_kernel(il, "k_proj", K);
    dump_kernel(il, "v_proj", V);

    // Gemma-style (1+w) per-head Q/K RMSNorm, then partial RoPE.
    Q = ops::rms_norm(Q, *L.q_norm, eps, /*plus_one=*/true);
    K = ops::rms_norm(K, *L.k_norm, eps, /*plus_one=*/true);
    dump_kernel(il, "q_norm", Q);
    dump_kernel(il, "k_norm", K);

    const float prf = cfg_.partial_rotary_factor;
    const std::int32_t rope_dims =
        (prf < 1.0f)
            ? std::max(2, 2 * static_cast<int>(std::floor(0.5f * prf * d)))
            : d;
    ops::RopeParams rp;
    rp.theta = cfg_.rope_theta;
    Q = ops::rope(Q, batch.positions, rope_dims, rp);
    K = ops::rope(K, batch.positions, rope_dims, rp);
    dump_kernel(il, "rope_q", Q);
    dump_kernel(il, "rope_k", K);

    kv.append(il, K, V, batch.kv_write_indices);

    ops::AttnParams ap;
    ap.scale      = 0.0f;  // default 1/sqrt(head_dim)
    ap.n_heads    = n_q;
    ap.n_kv_heads = n_kv;
    ap.head_dim   = d;

    Tensor attn = use_device_decode(batch, ap)
        ? ops::paged_attention_decode(
              Q, kv.k_pages(il), kv.v_pages(il),
              batch.kv_page_indices, batch.kv_last_page_lens,
              kv.page_size(), ap)
        : ops::paged_attention(
              Q, kv.k_pages(il), kv.v_pages(il),
              batch.kv_page_indices, batch.qo_indptr, batch.kv_page_indptr,
              batch.kv_last_page_lens, kv.page_size(), ap);
    dump_kernel(il, "sdpa", attn);

    attn = from_heads(attn, N, n_q * d);
    // Output gate: attn *= sigmoid(gate) before o_proj.
    Tensor gate_flat = from_heads(gate, N, n_q * d);
    attn = ops::mul(attn, mx::sigmoid(gate_flat));
    dump_kernel(il, "attn_gated", attn);

    Tensor attn_out = apply_linear(L.o_proj, attn);
    dump_kernel(il, "o_proj", attn_out);
    Tensor out = ops::residual_add(attn_out, residual);
    dump_kernel(il, "attn_resid", out);
    return out;
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
    dump_kernel(il, "attn_norm", cur);

    Tensor mixed_qkv = apply_linear(*L.la_in_proj_qkv, cur);  // [N, conv_dim] = [q|k|v]
    Tensor z = apply_linear(*L.la_in_proj_z, cur);            // [N, V_dim]
    Tensor a = apply_linear(*L.la_in_proj_a, cur);            // [N, V_h]
    Tensor b = apply_linear(*L.la_in_proj_b, cur);            // [N, V_h]
    dump_kernel(il, "gdn_in_qkv", mixed_qkv);
    dump_kernel(il, "gdn_in_z", z);
    dump_kernel(il, "gdn_in_a", a);
    dump_kernel(il, "gdn_in_b", b);

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
    dump_kernel(il, "gdn_core", out);

    Tensor attn_out = apply_linear(*L.la_out_proj, out);  // [N, hidden]
    dump_kernel(il, "gdn_out", attn_out);
    Tensor lout = ops::residual_add(attn_out, residual);
    dump_kernel(il, "attn_resid", lout);
    return lout;
}

// ── Shared dense SwiGLU MLP block ──
Tensor Qwen36Graph::mlp_block(std::int32_t il, Tensor hidden) {
    const auto& L = w_.layers[il];
    const float eps = cfg_.rms_norm_eps;

    Tensor residual = hidden;
    Tensor normed = ops::rms_norm(hidden, *L.ffn_norm, eps, /*plus_one=*/true);
    dump_kernel(il, "ffn_norm", normed);
    Tensor gate = apply_linear(*L.gate_proj, normed);
    Tensor up   = apply_linear(*L.up_proj,   normed);
    dump_kernel(il, "gate_proj", gate);
    dump_kernel(il, "up_proj", up);
    Tensor act  = ops::swiglu(gate, up);
    dump_kernel(il, "swiglu", act);
    Tensor ffn  = apply_linear(*L.down_proj, act);
    dump_kernel(il, "down_proj", ffn);
    Tensor mout = ops::residual_add(ffn, residual);
    dump_kernel(il, "layer_out", mout);
    return mout;
}

Tensor Qwen36Graph::forward(const ForwardBatch& batch, KvCacheView& kv) {
    const float eps = cfg_.rms_norm_eps;
    const std::int32_t Lc = cfg_.num_hidden_layers;

    // Qwen does NOT scale embeddings (unlike Gemma). Dequant-gathers from the
    // tied quant lm_head when the dense embed is dropped (embed-drop).
    Tensor hidden = apply_embedding(w_.embed, batch.token_ids);
    dump_kernel(-1, "embed", hidden);
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
    dump_kernel(-1, "final_norm", hidden);
    maybe_dump_hidden(-1, hidden);  // post-final-norm (pre-LM-head)
    Tensor sampled = ops::gather_rows(hidden, batch.logit_rows);

    Tensor logits = w_.lm_head
        ? apply_linear(*w_.lm_head, sampled)   // [n_slots, vocab]
        : apply_linear(w_.embed, sampled);     // tied embed (dense or quant)
    dump_kernel(-1, "logits", logits);
    return logits;
}

// ── Weight binding (Qwen3.6 hybrid) ──
ModelWeights bind_qwen36(const WeightSource& src, const ModelConfig& cfg) {
    ModelWeights w;

    // Qwen3.5 nests the text decoder under `model.language_model.` in the
    // multimodal checkpoint; some dumps drop the `language_model.` segment.
    // Key off the final norm (always present, unlike the droppable embed).
    std::string root = "model.language_model.";
    if (!src.has(root + "norm.weight")) {
        root = "model.";
    }

    w.final_norm = src.get(root + "norm.weight");
    // Prefer an explicit lm_head bundle when present (incl. a synthesized
    // quantized lm_head over a tied embed); absent -> dense embed GEMV.
    w.lm_head = try_bind_linear(src, "lm_head", cfg);
    // Input embed: dense bf16 table, or the tied quantized lm_head (dequant-
    // gathered) when the dense embed is dropped for true-4-bit memory parity.
    w.embed = bind_embedding(src, root + "embed_tokens", w.lm_head, cfg);

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
            // Gated-DeltaNet linear-attention layer. The in/out projections are
            // plain matmuls (bind_linear -> 4-bit transparently when the
            // checkpoint quantizes them); conv1d + scalars stay dense.
            L.la_in_proj_qkv = bind_linear(src, la + "in_proj_qkv", cfg);  // [conv_dim, H]
            L.la_in_proj_z   = bind_linear(src, la + "in_proj_z", cfg);    // [V_dim, H]
            L.la_in_proj_a   = bind_linear(src, la + "in_proj_a", cfg);    // [V_h, H]
            L.la_in_proj_b   = bind_linear(src, la + "in_proj_b", cfg);    // [V_h, H]
            L.la_conv1d_w    = mx::reshape(src.get(la + "conv1d.weight"),
                                           {conv_dim, conv_k});
            L.la_conv1d_b    = src.try_get(la + "conv1d.bias");
            L.la_A_log       = src.get(la + "A_log");
            L.la_dt_bias     = src.get(la + "dt_bias");
            L.la_gate_norm   = src.get(la + "norm.weight");         // [V_d]
            L.la_out_proj    = bind_linear(src, la + "out_proj", cfg);     // [H, V_dim]
        } else {
            // Full-attention layer (Qwen3 + output gate; 2x-wide q_proj).
            const std::string sa = p + "self_attn.";
            L.q_proj = bind_linear(src, sa + "q_proj", cfg);
            L.k_proj = bind_linear(src, sa + "k_proj", cfg);
            L.v_proj = bind_linear(src, sa + "v_proj", cfg);
            L.o_proj = bind_linear(src, sa + "o_proj", cfg);
            L.q_norm = src.get(sa + "q_norm.weight");
            L.k_norm = src.get(sa + "k_norm.weight");
        }

        // Dense SwiGLU MLP (both layer kinds; MoE deferred — not on the 0.8B).
        L.gate_proj = bind_linear(src, p + "mlp.gate_proj", cfg);
        L.up_proj   = bind_linear(src, p + "mlp.up_proj", cfg);
        L.down_proj = bind_linear(src, p + "mlp.down_proj", cfg);
    }
    return w;
}

}  // namespace pie_metal_driver::model
