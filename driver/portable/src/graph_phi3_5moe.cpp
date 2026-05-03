// graph_phi3_5moe.cpp — Phi-3.5-MoE graph builder.
//
// Phi-3.5-MoE = Mixtral-style sparse MoE + LayerNorm-with-bias + biases on
// Q/K/V/O and lm_head. Reuses build_moe_ffn for the routing+expert dispatch.
//
// Key differences from build_qwen3_graph:
//   - LayerNorm with bias on input/post-attn/final norms (NOT RMSNorm).
//   - Standard split Q/K/V projections with per-projection biases.
//   - MoE FFN (block_sparse_moe pattern, Mixtral naming via build_moe_ffn).
//   - lm_head with bias.

#include "graph_phi3_5moe.hpp"

#include <cmath>

#include "graph_common.hpp"

namespace pie_portable_driver {

namespace {

// LayerNorm with bias: y = scale * (x - mean) / sqrt(var + eps) + bias.
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

GraphResult build_phi3_5moe_graph(ggml_context* ctx,
                               const Model& model,
                               KvCachePaged& kv,
                               const ForwardEngine::BatchPlan& plan) {
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

    GraphInputs in = declare_graph_inputs(ctx, plan, n_total, n_req);

    std::vector<ggml_tensor*> live_k(h.num_hidden_layers, nullptr);
    std::vector<ggml_tensor*> live_v(h.num_hidden_layers, nullptr);

    auto* inpL = ggml_get_rows(ctx, w.tok_embd, in.tok_input);

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
        const auto& L = w.layers[il];
        auto* inpSA = inpL;

        // Pre-attention LayerNorm.
        auto* cur = layernorm_with_bias(
            ctx, inpSA, L.attn_norm, L.attn_norm_b, h.rms_norm_eps);

        // Q / K / V (with biases).
        auto* Q = ggml_mul_mat(ctx, L.q_proj, cur);
        if (L.q_proj_b) Q = ggml_add(ctx, Q, ggml_cast(ctx, L.q_proj_b, Q->type));
        auto* K = ggml_mul_mat(ctx, L.k_proj, cur);
        if (L.k_proj_b) K = ggml_add(ctx, K, ggml_cast(ctx, L.k_proj_b, K->type));
        auto* V = ggml_mul_mat(ctx, L.v_proj, cur);
        if (L.v_proj_b) V = ggml_add(ctx, V, ggml_cast(ctx, L.v_proj_b, V->type));

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_q_heads, n_total);
        K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, n_total);
        V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, n_total);

        // RoPE.
        Q = ggml_rope_ext(ctx, Q, in.pos_input, /*c=*/nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          h.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx, K, in.pos_input, /*c=*/nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          h.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // KV write.
        auto* k_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, K), n_embd_gqa, n_total);
        auto* v_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, V), n_embd_gqa, n_total);
        live_k[il] = ggml_set_rows(ctx, kv.k(il), k_2d, in.kv_idxs);
        live_v[il] = ggml_set_rows(ctx, kv.v(il), v_2d, in.kv_idxs);
        auto* k_cached = live_k[il];
        auto* v_cached = live_v[il];

        // Attention.
        ggml_tensor* attn_2d = nullptr;
        if (plan.pure_decode) {
            auto* Q_4d = ggml_reshape_4d(ctx, Q, head_dim, 1, n_q_heads, n_req);
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
        } else {
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

        // Output projection (with bias).
        auto* attn_out = ggml_mul_mat(ctx, L.o_proj, attn_2d);
        if (L.o_proj_b) {
            attn_out = ggml_add(ctx, attn_out, ggml_cast(ctx, L.o_proj_b, attn_out->type));
        }
        auto* attn_residual = ggml_add(ctx, inpSA, attn_out);

        // Post-attention LayerNorm.
        cur = layernorm_with_bias(
            ctx, attn_residual, L.ffn_norm, L.ffn_norm_b, h.rms_norm_eps);

        // MoE FFN — reuse build_moe_ffn (mixtral pattern: silu(w1)*w3 → w2).
        auto* moe_out = build_moe_ffn(
            ctx, cur,
            L.moe_router,
            L.moe_gate_exps, L.moe_up_exps, L.moe_down_exps,
            h.num_experts, h.num_experts_per_tok,
            MoeActivation::Silu, /*norm_topk=*/h.norm_topk_prob);

        inpL = ggml_add(ctx, attn_residual, moe_out);
    }

    // Final LayerNorm + LM head (with bias).
    auto* cur = layernorm_with_bias(
        ctx, inpL, w.output_norm, w.output_norm_b, h.rms_norm_eps);
    auto* sampled = ggml_get_rows(ctx, cur, in.out_idx);
    ggml_tensor* lm_head_w =
        h.tie_word_embeddings ? w.tok_embd : w.output_head;
    auto* logits = ggml_mul_mat(ctx, lm_head_w, sampled);
    if (w.output_head_b) {
        logits = ggml_add(ctx, logits, ggml_cast(ctx, w.output_head_b, logits->type));
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
