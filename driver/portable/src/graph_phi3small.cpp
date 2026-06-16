// graph_phi3small.cpp — Phi-3-small graph builder.
//
// Phi-3-small differs from the Phi-3 / Qwen3 family in three structural ways
// that make it incompatible with the shared `build_qwen3_graph`:
//
//   1. LayerNorm with bias (NOT RMSNorm). We use ggml_norm + an explicit
//      per-channel scale + bias add for input/post-attn/final norms.
//   2. Fused tensor names: query_key_value (vs qkv_proj), self_attn.dense
//      (vs o_proj), mlp.up_proj (packed gate||up like phi3's gate_up_proj).
//   3. mup parameterization: Q is divided by mup_attn_multiplier *
//      sqrt(head_dim) (NOT just sqrt(head_dim)), embeddings are multiplied
//      by mup_embedding_multiplier, and final logits are divided by
//      mup_width_multiplier.
//
// Blocksparse attention (block_size=64, vert_stride=8, num_local_blocks=16,
// dense_attention_every_n_layers=2) is NOT implemented in v1. For prompts
// that fit in num_local_blocks * block_size = 1024 tokens, blocksparse
// degenerates to standard causal attention (every query position can see
// every prior token in the local window), so v1 is correct up to ~1024
// tokens. Long-context use needs the per-layer custom mask.

#include "graph_phi3small.hpp"

#include <cmath>

#include "graph_common.hpp"

namespace pie_portable_driver {

namespace {

// Phi-3-small LayerNorm: y = scale * (x - mean) / sqrt(var + eps) + bias.
// ggml_norm does the (x - mean) / sqrt(var + eps) part; we apply the
// per-channel scale via mul + add for the bias.
ggml_tensor* layernorm_with_bias(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* w, ggml_tensor* b,
                                 float eps) {
    auto* n = ggml_norm(ctx, x, eps);
    auto* w_f32 = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
    auto* b_f32 = (b->type == GGML_TYPE_F32) ? b : ggml_cast(ctx, b, GGML_TYPE_F32);
    auto* scaled = ggml_mul(ctx, n, w_f32);
    return ggml_add(ctx, scaled, b_f32);
}

}  // namespace

GraphResult build_phi3small_graph(ggml_context* ctx,
                                  const Model& model,
                                  KvCachePaged& kv,
                                  const Executor::BatchPlan& plan) {
    const auto& h = model.hparams();
    const auto& w = model.weights();
    const std::int32_t n_q_heads  = h.num_attention_heads;
    const std::int32_t n_kv_heads = h.num_key_value_heads;
    const std::int32_t head_dim   = h.head_dim;
    const std::int32_t n_total    = plan.total_n_tokens;
    const std::int32_t n_req      = static_cast<std::int32_t>(plan.reqs.size());
    const std::int32_t n_embd_gqa = n_kv_heads * head_dim;

    auto* gf = ggml_new_graph_custom(
        ctx, static_cast<int>(GRAPH_MAX_NODES), /*grads=*/false);

    GraphInputs in = declare_graph_inputs(ctx, plan, n_total, n_req,
                                          model.supports_paged_attn_ext());

    std::vector<ggml_tensor*> live_k(h.num_hidden_layers, nullptr);
    std::vector<ggml_tensor*> live_v(h.num_hidden_layers, nullptr);

    // ---- Token embedding + mup embedding scale ----
    auto* inpL = ggml_get_rows(ctx, w.tok_embd, in.tok_input);
    if (h.mup_embedding_multiplier > 0.0f && h.mup_embedding_multiplier != 1.0f) {
        inpL = ggml_scale(ctx, inpL, h.mup_embedding_multiplier);
    }

    // mup attention scale: 1 / (mup_attn_multiplier * sqrt(head_dim))
    const float attn_scale =
        (h.mup_attn_multiplier > 0.0f
            ? 1.0f / (h.mup_attn_multiplier * std::sqrt(static_cast<float>(head_dim)))
            : 1.0f / std::sqrt(static_cast<float>(head_dim)));

    for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
        const auto& L = w.layers[il];
        auto* inpSA = inpL;

        // ---- Pre-attention LayerNorm (with bias) ----
        auto* cur = layernorm_with_bias(
            ctx, inpSA, L.attn_norm, L.attn_norm_b, h.rms_norm_eps);

        // ---- Q / K / V projections (split from fused query_key_value) ----
        auto* Q = ggml_mul_mat(ctx, L.q_proj, cur);
        if (L.q_proj_b) Q = ggml_add(ctx, Q, ggml_cast(ctx, L.q_proj_b, Q->type));
        auto* K = ggml_mul_mat(ctx, L.k_proj, cur);
        if (L.k_proj_b) K = ggml_add(ctx, K, ggml_cast(ctx, L.k_proj_b, K->type));
        auto* V = ggml_mul_mat(ctx, L.v_proj, cur);
        if (L.v_proj_b) V = ggml_add(ctx, V, ggml_cast(ctx, L.v_proj_b, V->type));

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_q_heads, n_total);
        K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, n_total);
        V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, n_total);

        // ---- RoPE (Phi-3-small uses standard θ-only RoPE) ----
        Q = ggml_rope_ext(ctx, Q, in.pos_input, /*c=*/nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          h.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx, K, in.pos_input, /*c=*/nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          h.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // ---- KV cache write ----
        auto* k_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, K), n_embd_gqa, n_total);
        auto* v_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, V), n_embd_gqa, n_total);
        k_2d = kv.qdq_for_append(ctx, il, k_2d);
        v_2d = kv.qdq_for_append(ctx, il, v_2d);
        live_k[il] = ggml_set_rows(ctx, kv.k(il), k_2d, in.kv_idxs);
        live_v[il] = ggml_set_rows(ctx, kv.v(il), v_2d, in.kv_idxs);
        auto* k_cached = live_k[il];
        auto* v_cached = live_v[il];

        // ---- Attention (causal; v1 skips blocksparse) ----
        ggml_tensor* attn_2d = nullptr;
        if (plan.pure_decode) {
            auto* Q_4d = ggml_reshape_4d(ctx, Q, head_dim, 1, n_q_heads, n_req);
            if (in.page_indices != nullptr) {
                auto* Q_bf16 = (Q_4d->type == GGML_TYPE_BF16)
                    ? Q_4d : ggml_cast(ctx, Q_4d, GGML_TYPE_BF16);
                auto* attn = ggml_paged_attn_ext(
                    ctx, Q_bf16, k_cached, v_cached,
                    in.page_indices, in.page_indptr, in.last_page_lens,
                    kv.page_size(), head_dim, n_kv_heads,
                    /*sliding_window=*/-1,
                    attn_scale, /*softcap=*/0.0f);
                auto* attn_f32 = ggml_cast(ctx, attn, GGML_TYPE_F32);
                attn_2d = ggml_reshape_2d(
                    ctx, attn_f32, head_dim * n_q_heads, n_total);
            } else {
                // Materialize: gather + permute + flash_attn_ext.
                auto* K_g = ggml_get_rows(ctx, k_cached, in.packed_gather);
                auto* V_g = ggml_get_rows(ctx, v_cached, in.packed_gather);
                auto* K_4d = ggml_reshape_4d(
                    ctx, K_g, head_dim, n_kv_heads, plan.max_n_kv, n_req);
                auto* V_4d = ggml_reshape_4d(
                    ctx, V_g, head_dim, n_kv_heads, plan.max_n_kv, n_req);
                auto* K_p = ggml_permute(ctx, K_4d, 0, 2, 1, 3);
                auto* V_p = ggml_permute(ctx, V_4d, 0, 2, 1, 3);
                auto* attn = ggml_flash_attn_ext(
                    ctx, Q_4d, K_p, V_p, in.packed_mask,
                    attn_scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
                attn_2d = ggml_reshape_2d(
                    ctx, ggml_cont(ctx, attn), head_dim * n_q_heads, n_total);
            }
        } else {
            // Per-request prefill / mixed batch path. Use the shared
            // build_request_flash_attn helper.
            std::vector<ggml_tensor*> attn_per_req;
            attn_per_req.reserve(n_req);
            for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
                const auto& R = plan.reqs[r];
                auto* attn = build_request_flash_attn(
                    ctx, Q, k_cached, v_cached,
                    in.gather_idxs[r], in.masks[r],
                    R.qo_start, R.n_tokens, R.n_kv,
                    head_dim, n_kv_heads, n_q_heads,
                    attn_scale, /*attn_softcap=*/0.0f,
                    /*sinks=*/nullptr);
                attn_per_req.push_back(attn);
            }
            attn_2d = concat_per_request_attn(
                ctx, attn_per_req, head_dim, n_q_heads, n_total);
        }

        // ---- Output projection (self_attn.dense, with bias) ----
        auto* attn_out = ggml_mul_mat(ctx, L.o_proj, attn_2d);
        if (L.o_proj_b) {
            attn_out = ggml_add(ctx, attn_out, ggml_cast(ctx, L.o_proj_b, attn_out->type));
        }
        auto* attn_residual = ggml_add(ctx, inpSA, attn_out);

        // ---- Post-attention LayerNorm (with bias) ----
        cur = layernorm_with_bias(
            ctx, attn_residual, L.ffn_norm, L.ffn_norm_b, h.rms_norm_eps);

        // ---- MLP: GeGLU on packed up_proj output [gate || up] ----
        auto* gate = ggml_mul_mat(ctx, L.gate_proj, cur);
        if (L.gate_proj_b) gate = ggml_add(ctx, gate, ggml_cast(ctx, L.gate_proj_b, gate->type));
        auto* up = ggml_mul_mat(ctx, L.up_proj, cur);
        if (L.up_proj_b) up = ggml_add(ctx, up, ggml_cast(ctx, L.up_proj_b, up->type));
        gate = ggml_gelu(ctx, gate);
        auto* gated = ggml_mul(ctx, gate, up);
        auto* ffn_out = ggml_mul_mat(ctx, L.down_proj, gated);
        if (L.down_proj_b) {
            ffn_out = ggml_add(ctx, ffn_out, ggml_cast(ctx, L.down_proj_b, ffn_out->type));
        }
        inpL = ggml_add(ctx, attn_residual, ffn_out);
    }

    // ---- Final LayerNorm + LM head with mup width scaling ----
    auto* cur = layernorm_with_bias(
        ctx, inpL, w.output_norm, w.output_norm_b, h.rms_norm_eps);
    auto* sampled = ggml_get_rows(ctx, cur, in.out_idx);
    ggml_tensor* lm_head_w =
        h.tie_word_embeddings ? w.tok_embd : w.output_head;
    auto* logits = ggml_mul_mat(ctx, lm_head_w, sampled);
    if (h.mup_width_multiplier > 0.0f && h.mup_width_multiplier != 1.0f) {
        logits = ggml_scale(ctx, logits, 1.0f / h.mup_width_multiplier);
    }

    GraphResult r;
    r.gf = gf;
    r.in = in;
    if (plan.all_greedy) {
        auto* tokens_out = ggml_argmax(ctx, logits);
        ggml_set_name(tokens_out, "tokens_out");
        ggml_set_output(tokens_out);
        ggml_build_forward_expand(gf, tokens_out);
        r.tokens_out = tokens_out;
    } else if (plan.uniform_top_sample) {
        const float inv_t = 1.0f / plan.reqs[0].sampler.temperature;
        auto* probs = ggml_soft_max_ext(ctx, logits, /*mask=*/nullptr,
                                        /*scale=*/inv_t, /*max_bias=*/0.0f);
        auto* top_k_idx = ggml_top_k(ctx, probs, plan.uniform_top_k);
        auto* probs_3d  = ggml_reshape_3d(ctx, probs, 1, h.vocab_size, n_req);
        auto* gathered  = ggml_get_rows(ctx, probs_3d, top_k_idx);
        auto* top_k_probs = ggml_reshape_2d(
            ctx, gathered, plan.uniform_top_k, n_req);
        ggml_set_name(top_k_idx, "top_k_idx");
        ggml_set_name(top_k_probs, "top_k_probs");
        ggml_set_output(top_k_idx);
        ggml_set_output(top_k_probs);
        ggml_build_forward_expand(gf, top_k_idx);
        ggml_build_forward_expand(gf, top_k_probs);
        r.top_k_idx = top_k_idx;
        r.top_k_probs = top_k_probs;
    } else {
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        r.logits = logits;
    }
    return r;
}

}  // namespace pie_portable_driver
