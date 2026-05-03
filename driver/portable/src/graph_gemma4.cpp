#include "graph_gemma4.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "arch_spec.hpp"
#include "plan.hpp"

namespace pie_portable_driver {

namespace {

// Resolve a shared gemma4 layer's KV-cache source. Mirrors python:
// the most recent earlier non-shared layer with the same attention type.
inline std::int32_t gemma4_kv_source(const std::string& pattern,
                                     std::int32_t il,
                                     std::int32_t first_shared) {
    if (il < first_shared) return il;
    const char target = pattern[il];
    for (std::int32_t i = first_shared - 1; i >= 0; --i) {
        if (pattern[i] == target) return i;
    }
    return il;  // unreachable on well-formed configs
}

}  // namespace

// Gemma 4 / 3n-style graph. Differs from build_qwen3_graph in:
//   - per-layer head_dim (sliding=head_dim, full=gemma4_head_dim_global),
//   - per-layer-type rope_theta + proportional rotary factor,
//   - V-norm (pure rms_norm, no learnable weight) before KV write,
//   - sm_scale=1.0 (Q/K-norm absorbs the head-dim scale),
//   - RMSNorm with `w` directly (NOT `(1+w)`),
//   - per-layer scalar applied to layer output,
//   - Per-Layer Embeddings (PLE) auxiliary residual injected after MLP,
//   - KV-cache sharing for the last gemma4_first_shared..n_layers-1 layers
//     (those layers skip K/V matmul + writes and read upstream's KV slot).
//
// Attention layout: when `plan.pure_decode`, the whole batch attention
// is one packed call per layer (M11 fast path). Sliding layers go
// through `ggml_flash_attn_ext` with `in.packed_mask` (SWA-clipped);
// full layers (head_dim=512, beyond flash_attn_ext's max) fall through
// to a manual SDPA + `in.packed_mask_full` (no-SWA companion). The
// existing per-request slow path is kept verbatim for prefill.
GraphResult build_gemma4_graph(ggml_context* ctx,
                               const Model& model,
                               KvCachePaged& kv,
                               const ForwardEngine::BatchPlan& plan) {
    const auto& h = model.hparams();
    const auto& w = model.weights();
    const std::int32_t n_q_heads  = h.num_attention_heads;
    const std::int32_t n_kv_heads = h.num_key_value_heads;
    const std::int32_t n_total    = plan.total_n_tokens;
    const std::int32_t n_req      = static_cast<std::int32_t>(plan.reqs.size());
    const ArchSpec spec = arch_spec_for(h.arch, h);

    auto* gf = ggml_new_graph_custom(
        ctx, static_cast<int>(GRAPH_MAX_NODES), /*grads=*/false);

    GraphInputs in = declare_graph_inputs(ctx, plan, n_total, n_req);

    // ---- Embed -----------------------------------------------------------
    auto* embd = ggml_get_rows(ctx, w.tok_embd, in.tok_input);
    auto* inpL = embd;
    if (spec.scale_embed_by_sqrt_d) {
        inpL = ggml_scale(ctx, inpL,
                          std::sqrt(static_cast<float>(h.hidden_size)));
    }

    // Per-layer post-write KV cache handles. Shared layers read from the
    // upstream non-shared layer's POST-set_rows handle, NOT the original
    // kv.k() tensor — otherwise ggml's graph topology won't enforce
    // "write-before-read" order across the layer-share boundary, so the
    // shared layer can race against the source layer's set_rows. We seed
    // with the original cache handles and overwrite as each non-shared
    // layer writes.
    std::vector<ggml_tensor*> live_k(h.num_hidden_layers);
    std::vector<ggml_tensor*> live_v(h.num_hidden_layers);
    for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
        live_k[il] = kv.k(il);
        live_v[il] = kv.v(il);
    }

    // ---- PLE setup ------------------------------------------------------
    // per_layer_inputs shape: [ple_dim, n_layers, n_total]
    ggml_tensor* per_layer_inputs = nullptr;
    if (spec.gemma4_ple_enabled && w.ple_token_embed) {
        const std::int32_t ple_dim   = spec.gemma4_ple_dim;
        const std::int32_t n_layers  = h.num_hidden_layers;
        const float ple_token_norm = std::sqrt(static_cast<float>(ple_dim));
        const float ple_proj_norm  = 1.0f / std::sqrt(static_cast<float>(h.hidden_size));
        const float ple_combine    = 1.0f / std::sqrt(2.0f);

        // Token-identity component.
        auto* tok_emb = ggml_get_rows(ctx, w.ple_token_embed, in.tok_input);
        // Shape: [n_layers*ple_dim, n_total] → [ple_dim, n_layers, n_total]
        tok_emb = ggml_reshape_3d(ctx, tok_emb, ple_dim, n_layers, n_total);
        tok_emb = ggml_scale(ctx, tok_emb, ple_token_norm);

        // Context component: ple_model_proj @ inpL.
        auto* ctx_proj = ggml_mul_mat(ctx, w.ple_model_proj, inpL);
        ctx_proj = ggml_reshape_3d(ctx, ctx_proj, ple_dim, n_layers, n_total);
        ctx_proj = ggml_scale(ctx, ctx_proj, ple_proj_norm);
        ctx_proj = ggml_rms_norm(ctx, ctx_proj, h.rms_norm_eps);
        ctx_proj = norm_scale(ctx, ctx_proj, w.ple_model_norm, /*plus_one=*/false);

        per_layer_inputs = ggml_add(ctx, ctx_proj, tok_emb);
        per_layer_inputs = ggml_scale(ctx, per_layer_inputs, ple_combine);
    }

    for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
        const auto& L = w.layers[il];
        auto* inpSA = inpL;

        // Per-layer head_dim & layer-type.
        const bool is_full = !spec.layer_pattern.empty()
            && static_cast<std::size_t>(il) < spec.layer_pattern.size()
            && spec.layer_pattern[il] == 'g';
        const std::int32_t head_dim_il =
            is_full ? spec.gemma4_head_dim_global : h.head_dim;
        // Per-layer kv_heads. Full_attention layers on Gemma 4 31B / 26B-A4B
        // use a smaller kv_heads count (`num_global_key_value_heads`) than
        // sliding layers. KV cache is sized accordingly per layer.
        const std::int32_t n_kv_heads_il =
            (is_full && h.num_global_key_value_heads > 0)
                ? h.num_global_key_value_heads
                : n_kv_heads;
        const std::int32_t n_embd_gqa_il = n_kv_heads_il * head_dim_il;
        const bool is_shared = il >= spec.gemma4_first_shared;
        const std::int32_t kv_layer = is_shared
            ? gemma4_kv_source(spec.layer_pattern, il, spec.gemma4_first_shared)
            : il;

        // Pre-attention norm (gemma4: w directly, NOT 1+w).
        auto* cur = ggml_rms_norm(ctx, inpSA, h.rms_norm_eps);
        cur = norm_scale(ctx, cur, L.attn_norm, /*plus_one=*/false);

        // Q always exists; K/V are skipped on shared layers.
        auto* Q = ggml_mul_mat(ctx, L.q_proj, cur);
        Q = ggml_reshape_3d(ctx, Q, head_dim_il, n_q_heads, n_total);
        // Per-head Q-norm with [head_dim_il] weight.
        Q = ggml_rms_norm(ctx, Q, h.rms_norm_eps);
        Q = norm_scale(ctx, Q, L.q_norm, /*plus_one=*/false);

        // RoPE. Sliding layers: standard RoPE over all head_dim with
        // theta_sliding. Full layers ("proportional"): rotate only the
        // first `partial * head_dim` dim-pairs, but PAIR them at the
        // model's head_dim/2 offset (NOT rotary_dim/2 as ggml's default
        // would). We achieve this by passing n_dims = head_dim and a
        // freq_factors tensor that zeroes inv_freq for the non-rotated
        // tail (1e30 makes inv_freq~0 → cos=1, sin=0, identity).
        const float rope_theta_il = is_full
            ? spec.gemma4_rope_theta_full : spec.gemma4_rope_theta_sliding;
        const std::int32_t rope_n_dims = head_dim_il;
        ggml_tensor* rope_factors_il = is_full
            ? w.gemma4_rope_full_factors : nullptr;
        Q = ggml_rope_ext(ctx, Q, in.pos_input, rope_factors_il,
                          rope_n_dims, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                          rope_theta_il, /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                          /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);

        if (!is_shared) {
            auto* K = ggml_mul_mat(ctx, L.k_proj, cur);
            // Alternative attention: when v_proj is absent (Gemma 4 large
            // variants on full_attention layers), V is the same projection
            // result as K. They diverge after this point — V skips K-norm
            // and RoPE.
            auto* V = (L.v_proj != nullptr)
                ? ggml_mul_mat(ctx, L.v_proj, cur)
                : K;
            K = ggml_reshape_3d(ctx, K, head_dim_il, n_kv_heads_il, n_total);
            V = ggml_reshape_3d(ctx, V, head_dim_il, n_kv_heads_il, n_total);
            // Per-head K-norm with [head_dim_il] weight.
            K = ggml_rms_norm(ctx, K, h.rms_norm_eps);
            K = norm_scale(ctx, K, L.k_norm, /*plus_one=*/false);
            // V-norm: pure rms_norm, no learnable scale.
            V = ggml_rms_norm(ctx, V, h.rms_norm_eps);

            K = ggml_rope_ext(ctx, K, in.pos_input, rope_factors_il,
                              rope_n_dims, GGML_ROPE_TYPE_NEOX, 0,
                              rope_theta_il, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

            auto* k_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, K),
                                         n_embd_gqa_il, n_total);
            auto* v_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, V),
                                         n_embd_gqa_il, n_total);
            live_k[kv_layer] = ggml_set_rows(ctx, kv.k(kv_layer), k_2d, in.kv_idxs);
            live_v[kv_layer] = ggml_set_rows(ctx, kv.v(kv_layer), v_2d, in.kv_idxs);
        }
        ggml_tensor* k_cached = live_k[kv_layer];
        ggml_tensor* v_cached = live_v[kv_layer];

        // Attention. Sliding layers (head_dim=256) go through the shared
        // flash_attn helper. Full layers (head_dim=512) fall back to
        // manual SDPA: ggml's CUDA flash_attn has no dkq=512 / GQA-8
        // kernel.
        const float layer_kq_scale = spec.gemma4_unit_sm_scale
            ? 1.0f
            : 1.0f / std::sqrt(static_cast<float>(head_dim_il));
        const std::size_t Q_stride_ne2 =
            static_cast<std::size_t>(head_dim_il) * n_q_heads *
            ggml_type_size(Q->type);

        ggml_tensor* attn_2d = nullptr;

        if (plan.pure_decode) {
            // M11 packed path: single attention call per layer. Uses the
            // sliding-clipped mask on 's' layers and the no-SWA mask on
            // 'g' layers (built by build_pure_decode_packing with
            // also_build_no_swa_mask=true).
            auto* mask = is_full ? in.packed_mask_full : in.packed_mask;
            // n_total == n_req in pure_decode (one token per request).
            auto* Q_4d = ggml_reshape_4d(ctx, Q,
                                          head_dim_il, 1, n_q_heads, n_req);
            auto* K_gather = ggml_get_rows(ctx, k_cached, in.packed_gather);
            auto* V_gather = ggml_get_rows(ctx, v_cached, in.packed_gather);
            auto* K_4d = ggml_reshape_4d(ctx, K_gather,
                                          head_dim_il, n_kv_heads_il,
                                          plan.max_n_kv, n_req);
            auto* V_4d = ggml_reshape_4d(ctx, V_gather,
                                          head_dim_il, n_kv_heads_il,
                                          plan.max_n_kv, n_req);

            ggml_tensor* attn = nullptr;
            if (head_dim_il <= kFlashAttnMaxHeadDim) {
                // Sliding layers (head_dim=256): single flash_attn_ext.
                auto* K_perm = ggml_permute(ctx, K_4d, 0, 2, 1, 3);
                auto* V_perm = ggml_permute(ctx, V_4d, 0, 2, 1, 3);
                attn = ggml_flash_attn_ext(ctx, Q_4d, K_perm, V_perm,
                                           mask, layer_kq_scale,
                                           /*max_bias=*/0.0f,
                                           /*logit_softcap=*/spec.attn_softcap);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            } else {
                // Full layers (head_dim=512): manual SDPA, batched over
                // n_req via ne[3]. GQA expands K/V from n_kv_heads_il to
                // n_q_heads via repeat_4d so mul_mat lines up per head.
                const std::int32_t gqa_factor = n_q_heads / n_kv_heads_il;
                ggml_tensor* K_h = K_4d;
                ggml_tensor* V_h = V_4d;
                if (gqa_factor > 1) {
                    K_h = ggml_repeat_4d(
                        ctx, K_4d, head_dim_il, n_q_heads,
                        plan.max_n_kv, n_req);
                    V_h = ggml_repeat_4d(
                        ctx, V_4d, head_dim_il, n_q_heads,
                        plan.max_n_kv, n_req);
                }
                // K_perm: [head_dim, max_n_kv, n_q_heads, n_req]
                // Q_4d  : [head_dim, 1,        n_q_heads, n_req]
                // KQ = mul_mat(K_perm, Q_4d) → [max_n_kv, 1, n_q_heads, n_req]
                auto* K_perm = ggml_cont(ctx, ggml_permute(ctx, K_h, 0, 2, 1, 3));
                auto* KQ = ggml_mul_mat(ctx, K_perm, Q_4d);
                // mask shape [max_n_kv, MASK_PAD, 1, n_req]; broadcast
                // ne[2]=1 onto KQ's ne[2]=n_q_heads.
                auto* KQ_soft = ggml_soft_max_ext(
                    ctx, KQ, mask,
                    layer_kq_scale, /*max_bias=*/0.0f);
                // V_T: transpose kv-axis and head-axis to feed mul_mat.
                // Want [max_n_kv, head_dim, n_q_heads, n_req] for V^T @ KQ_soft.
                auto* V_T = ggml_cont(
                    ctx, ggml_permute(ctx, V_h, 1, 2, 0, 3));
                // mul_mat: [head_dim, 1, n_q_heads, n_req]
                attn = ggml_mul_mat(ctx, V_T, KQ_soft);
                ggml_set_name(attn, "gemma4_full_attn_out");
            }
            // attn shape: [head_dim, n_q_heads, 1, n_req]. Repack to
            // [head_dim*n_q_heads, n_total] so the existing o_proj path
            // applies unchanged.
            attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn),
                                       head_dim_il * n_q_heads, n_total);
        } else {
            // Slow path (prefill, custom masks).
            std::vector<ggml_tensor*> attn_out_per_req;
            attn_out_per_req.reserve(n_req);
            for (std::int32_t r = 0; r < n_req; ++r) {
                const auto& R = plan.reqs[r];

                ggml_tensor* attn = nullptr;
                if (head_dim_il <= kFlashAttnMaxHeadDim) {
                    attn = build_request_flash_attn(
                        ctx, Q, k_cached, v_cached,
                        in.gather_idxs[r], in.masks[r],
                        R.qo_start, R.n_tokens, R.n_kv,
                        head_dim_il, n_kv_heads_il, n_q_heads,
                        layer_kq_scale, spec.attn_softcap,
                        /*sinks=*/nullptr);
                } else {
                    auto* Q_r = ggml_view_3d(ctx, Q,
                                             head_dim_il, n_q_heads, R.n_tokens,
                                             Q->nb[1], Q->nb[2],
                                             static_cast<std::size_t>(R.qo_start) * Q_stride_ne2);
                    auto* K_gather = ggml_get_rows(ctx, k_cached, in.gather_idxs[r]);
                    auto* V_gather = ggml_get_rows(ctx, v_cached, in.gather_idxs[r]);
                    auto* K_r = ggml_reshape_3d(ctx, K_gather,
                                                head_dim_il, n_kv_heads_il, R.n_kv);
                    auto* V_r = ggml_reshape_3d(ctx, V_gather,
                                                head_dim_il, n_kv_heads_il, R.n_kv);

                    auto* Q_perm = ggml_cont(ctx, ggml_permute(ctx, Q_r, 0, 2, 1, 3));
                    auto* K_perm = ggml_cont(ctx, ggml_permute(ctx, K_r, 0, 2, 1, 3));
                    auto* KQ = ggml_mul_mat(ctx, K_perm, Q_perm);

                    auto* mask_full_r = in.masks[r];
                    auto* mask_view = ggml_view_4d(
                        ctx, mask_full_r,
                        mask_full_r->ne[0], R.n_tokens,
                        mask_full_r->ne[2], mask_full_r->ne[3],
                        mask_full_r->nb[1], mask_full_r->nb[2], mask_full_r->nb[3],
                        /*offset=*/0);
                    auto* KQ_soft = ggml_soft_max_ext(ctx, KQ, mask_view,
                                                      layer_kq_scale, /*max_bias=*/0.0f);

                    auto* V_T = ggml_cont(ctx,
                        ggml_permute(ctx, V_r, /*ax0=*/1, /*ax1=*/2,
                                     /*ax2=*/0, /*ax3=*/3));
                    attn = ggml_mul_mat(ctx, V_T, KQ_soft);
                    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
                }
                attn_out_per_req.push_back(attn);
            }

            attn_2d = concat_per_request_attn(
                ctx, attn_out_per_req, head_dim_il, n_q_heads, n_total);
        }

        auto* attn_out = ggml_mul_mat(ctx, L.o_proj, attn_2d);
        // Post-attn norm.
        attn_out = ggml_rms_norm(ctx, attn_out, h.rms_norm_eps);
        attn_out = norm_scale(ctx, attn_out, L.post_attn_norm, /*plus_one=*/false);
        auto* ffn_in = ggml_add(ctx, attn_out, inpSA);

        // FFN — pre-norm, GeGLU. The dense path always runs; on
        // Gemma 4 26B-A4B (`gemma4_enable_moe`) the sparse-MoE block
        // runs in parallel and the two post-normed branches are summed
        // before the final post_feedforward_layernorm.
        cur = ggml_rms_norm(ctx, ffn_in, h.rms_norm_eps);
        cur = norm_scale(ctx, cur, L.ffn_norm, /*plus_one=*/false);
        auto* gate = ggml_mul_mat(ctx, L.gate_proj, cur);
        auto* up   = ggml_mul_mat(ctx, L.up_proj,   cur);
        gate = ggml_gelu(ctx, gate);
        auto* gated = ggml_mul(ctx, gate, up);
        auto* dense_out = ggml_mul_mat(ctx, L.down_proj, gated);

        ggml_tensor* combined = dense_out;
        if (h.gemma4_enable_moe && L.moe_router != nullptr) {
            // Branch 1: dense path's post-norm (post_feedforward_layernorm_1).
            auto* b1 = ggml_rms_norm(ctx, dense_out, h.rms_norm_eps);
            b1 = norm_scale(ctx, b1, L.gemma4_moe_post_ffn_norm_1,
                            /*plus_one=*/false);
            // Branch 2 has TWO inputs:
            //   - router_in = rmsnorm(ffn_in) * router.scale * 1/sqrt(H),
            //     used by the routing projection.
            //   - moe_in    = pre_feedforward_layernorm_2(ffn_in), used
            //     by the gate / up / down expert projections.
            auto* router_in = ggml_rms_norm(ctx, ffn_in, h.rms_norm_eps);
            router_in = norm_scale(ctx, router_in, L.moe_router_scale,
                                   /*plus_one=*/false);
            const float inv_sqrt_h =
                1.0f / std::sqrt(static_cast<float>(h.hidden_size));
            router_in = ggml_scale(ctx, router_in, inv_sqrt_h);

            auto* moe_in = ggml_rms_norm(ctx, ffn_in, h.rms_norm_eps);
            moe_in = norm_scale(ctx, moe_in, L.gemma4_moe_pre_ffn_norm_2,
                                /*plus_one=*/false);

            // Inline MoE block — same pattern as build_moe_ffn, but with
            // Gemma-4's per_expert_scale gain applied to the routing
            // weights after top-K renorm.
            auto* logits = ggml_mul_mat(ctx, L.moe_router, router_in);
            auto* probs  = ggml_soft_max(ctx, logits);
            auto* selected = ggml_top_k(ctx, probs, h.num_experts_per_tok);

            // Gather selected probs as [n_used, n_total] weights.
            auto* probs_3d = ggml_reshape_3d(
                ctx, probs, 1, h.num_experts, n_total);
            auto* w_gather = ggml_get_rows(ctx, probs_3d, selected);
            auto* weights = ggml_reshape_2d(
                ctx, w_gather, h.num_experts_per_tok, n_total);
            // Renormalise (Gemma 4 sets norm_topk_prob = true).
            auto* w_sum = ggml_sum_rows(ctx, weights);
            weights = ggml_div(ctx, weights, w_sum);

            // Per-expert scalar gain. Cast pes to F32, broadcast across
            // n_total so ggml_get_rows can index by `selected` (which has
            // batch dim n_total). Then multiply into weights.
            auto* pes_f32 = (L.moe_router_per_expert_scale->type == GGML_TYPE_F32)
                ? L.moe_router_per_expert_scale
                : ggml_cast(ctx, L.moe_router_per_expert_scale, GGML_TYPE_F32);
            auto* pes_2d_orig = ggml_reshape_2d(ctx, pes_f32, h.num_experts, 1);
            auto* pes_bc = ggml_repeat(ctx, pes_2d_orig, probs);
            auto* pes_3d = ggml_reshape_3d(
                ctx, pes_bc, 1, h.num_experts, n_total);
            auto* pes_gather = ggml_get_rows(ctx, pes_3d, selected);
            auto* pes_weights = ggml_reshape_2d(
                ctx, pes_gather, h.num_experts_per_tok, n_total);
            weights = ggml_mul(ctx, weights, pes_weights);

            // Per-token, per-selected expert: gate / up / down.
            auto* moe_in_3d = ggml_reshape_3d(
                ctx, moe_in, h.hidden_size, 1, n_total);
            auto* gate_e = ggml_mul_mat_id(ctx, L.moe_gate_exps, moe_in_3d, selected);
            auto* up_e   = ggml_mul_mat_id(ctx, L.moe_up_exps,   moe_in_3d, selected);
            auto* gated_e = ggml_mul(ctx, ggml_gelu(ctx, gate_e), up_e);
            auto* expert_out = ggml_mul_mat_id(
                ctx, L.moe_down_exps, gated_e, selected);
            // expert_out: [hidden, n_used, n_total]

            // Apply weights and sum across n_used.
            auto* w_3d = ggml_reshape_3d(
                ctx, weights, 1, h.num_experts_per_tok, n_total);
            auto* weighted = ggml_mul(ctx, expert_out, w_3d);
            // permute(1,0,2,3) puts n_used at ne[0] for sum_rows.
            auto* perm = ggml_permute(ctx, weighted, 1, 0, 2, 3);
            auto* perm_cont = ggml_cont(ctx, perm);
            auto* moe_out_sum = ggml_sum_rows(ctx, perm_cont);
            auto* moe_out = ggml_reshape_2d(
                ctx, ggml_cont(ctx, moe_out_sum), h.hidden_size, n_total);
            auto* b2 = ggml_rms_norm(ctx, moe_out, h.rms_norm_eps);
            b2 = norm_scale(ctx, b2, L.gemma4_moe_post_ffn_norm_2,
                            /*plus_one=*/false);
            combined = ggml_add(ctx, b1, b2);
        }

        auto* ffn_out = ggml_rms_norm(ctx, combined, h.rms_norm_eps);
        ffn_out = norm_scale(ctx, ffn_out, L.post_ffn_norm, /*plus_one=*/false);
        inpL = ggml_add(ctx, ffn_out, ffn_in);

        // PLE residual injection.
        if (per_layer_inputs && L.ple_gate && L.ple_proj) {
            const std::int32_t ple_dim = spec.gemma4_ple_dim;
            // per_layer_inputs is [ple_dim, n_layers, n_total]; slice
            // layer il via offset il*ple_dim (innermost dim varies fastest).
            const std::size_t ple_layer_offset =
                static_cast<std::size_t>(il) * ple_dim *
                ggml_type_size(per_layer_inputs->type);
            auto* ple_signal = ggml_view_2d(
                ctx, per_layer_inputs,
                ple_dim, n_total,
                /*nb1=*/per_layer_inputs->nb[2],
                /*offset=*/ple_layer_offset);

            auto* ple_g = ggml_mul_mat(ctx, L.ple_gate, inpL);   // [ple_dim, n_total]
            ple_g = ggml_gelu(ctx, ple_g);
            // ple_signal is BF16 (slice of per_layer_inputs); cast to F32
            // for the elementwise mul (CUDA's mul kernel rejects BF16 src1).
            auto* ple_signal_f32 = ggml_cast(ctx, ple_signal, GGML_TYPE_F32);
            auto* ple_gated = ggml_mul(ctx, ple_g, ple_signal_f32);
            auto* ple_out = ggml_mul_mat(ctx, L.ple_proj, ple_gated); // [hidden, n_total]
            ple_out = ggml_rms_norm(ctx, ple_out, h.rms_norm_eps);
            ple_out = norm_scale(ctx, ple_out, L.ple_norm, /*plus_one=*/false);
            inpL = ggml_add(ctx, inpL, ple_out);
        }

        // Per-layer scalar (shape [1] in shipped E2B/E4B). The model loader
        // hoists the BF16→F32 conversion to load time (declare_synth_f32_
        // from_bf16_) so the elementwise mul lands on an F32 weight
        // directly — keep the ternary as a safety net for any future
        // checkpoint that ships F32 already.
        if (spec.gemma4_layer_scalar && L.layer_scalar) {
            auto* sc = (L.layer_scalar->type == GGML_TYPE_F32)
                ? L.layer_scalar
                : ggml_cast(ctx, L.layer_scalar, GGML_TYPE_F32);
            inpL = ggml_mul(ctx, inpL, sc);
        }
    }

    // Final norm + LM head.
    auto* cur = ggml_rms_norm(ctx, inpL, h.rms_norm_eps);
    cur = norm_scale(ctx, cur, w.output_norm, /*plus_one=*/false);
    auto* sampled = ggml_get_rows(ctx, cur, in.out_idx);
    ggml_tensor* lm_head_w =
        h.tie_word_embeddings ? w.tok_embd : w.output_head;
    auto* logits = ggml_mul_mat(ctx, lm_head_w, sampled);

    if (spec.final_softcap > 0.0f) {
        logits = ggml_scale(ctx, logits, 1.0f / spec.final_softcap);
        logits = ggml_tanh(ctx, logits);
        logits = ggml_scale(ctx, logits, spec.final_softcap);
    }

    // GPU-side sampler fast paths (mirror graph_qwen3 / graph_qwen3_5).
    ggml_tensor* tokens_out  = nullptr;
    ggml_tensor* top_k_idx   = nullptr;
    ggml_tensor* top_k_probs = nullptr;
    if (plan.all_greedy) {
        tokens_out = ggml_argmax(ctx, logits);
        ggml_set_name(tokens_out, "tokens_out");
        ggml_set_output(tokens_out);
        ggml_build_forward_expand(gf, tokens_out);
    } else if (plan.uniform_top_sample) {
        const float inv_t = 1.0f / plan.reqs[0].sampler.temperature;
        ggml_tensor* probs = ggml_soft_max_ext(ctx, logits, /*mask=*/nullptr,
                                                /*scale=*/inv_t, /*max_bias=*/0.0f);
        top_k_idx = ggml_top_k(ctx, probs, plan.uniform_top_k);
        ggml_tensor* probs_3d = ggml_reshape_3d(ctx, probs, 1, h.vocab_size, n_req);
        ggml_tensor* gathered = ggml_get_rows(ctx, probs_3d, top_k_idx);
        top_k_probs = ggml_reshape_2d(ctx, gathered, plan.uniform_top_k, n_req);
        ggml_set_name(top_k_idx,   "top_k_idx");
        ggml_set_name(top_k_probs, "top_k_probs");
        ggml_set_output(top_k_idx);
        ggml_set_output(top_k_probs);
        ggml_build_forward_expand(gf, top_k_idx);
        ggml_build_forward_expand(gf, top_k_probs);
    } else {
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }

    GraphResult res{};
    res.gf = gf;
    if (plan.all_greedy)              res.tokens_out = tokens_out;
    else if (plan.uniform_top_sample) { res.top_k_idx = top_k_idx; res.top_k_probs = top_k_probs; }
    else                               res.logits = logits;
    res.in = in;
    return res;
}

}  // namespace pie_portable_driver
