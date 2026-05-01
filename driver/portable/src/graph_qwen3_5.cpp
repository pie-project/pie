#include "graph_qwen3_5.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "arch_spec.hpp"
#include "plan.hpp"

// Qwen 3.5 / 3.6 graph (hybrid: gated-delta-rule "linear attention" +
// GQA full attention with mrope/output-gate). Linear-attn math follows
// HF transformers `torch_recurrent_gated_delta_rule` per token, threading
// state through. Standard RMSNorms apply `(1.0 + weight)` (Gemma-style);
// only `linear_attn.norm` (the gated SSM norm) uses bare `weight`.
//
// Optional debug-tap: set PIE_QWEN35_DUMP=<name> to capture an
// intermediate tensor from layer 0 of request 0 — forward.cpp downloads
// it and prints first 16 floats + l2. Tap names: x_r_input, embed,
// qkv_proj, conv_out, q_post_l2, k_post_l2, v_post_conv, g_exp, beta,
// state_final, y_concat, y_normed, gated, lin_out.

namespace pie_portable_driver {

namespace {

// Per-token gated-delta-rule update. Builds one iteration of the
// recurrence directly with ggml primitives, mirroring llama.cpp's
// `build_delta_net_autoregressive` (pull/19468). Inputs are single-token
// 4D tensors; state is updated in-place. Returns (output_token, new_state).
//
// State convention here matches OUR state cache layout: ne[0] = key
// axis (S), ne[1] = value axis (S). (The reference uses the opposite
// labeling — ne[0]=v, ne[1]=k — but since head_k_dim == head_v_dim for
// Qwen 3.5 the byte layout is identical; we just align the per-token
// reshapes to our labeling.)
//
//   q_t, k_t : [S, H, 1, B] post-l2norm; q already scaled by 1/sqrt(S)
//   v_t      : [S, H, 1, B]
//   g_t      : [1, 1, H, B] already exp'd
//   beta_t   : [1, 1, H, B] already sigmoid'd
//   state_in : [S, S, H, B] ne[0]=k axis, ne[1]=v axis
//   B is the request batch (n_seqs).
struct DeltaStep { ggml_tensor* y; ggml_tensor* state_out; };

DeltaStep delta_step(ggml_context* ctx,
                     ggml_tensor* q_t, ggml_tensor* k_t, ggml_tensor* v_t,
                     ggml_tensor* g_t, ggml_tensor* beta_t,
                     ggml_tensor* state_in,
                     int S, int H) {
    const int B = state_in->ne[3];

    // 1. Decay: state *= g_t (broadcast over [S, S, H, B])
    auto* state = ggml_mul(ctx, state_in, g_t);

    auto* k_t_unsq = ggml_reshape_4d(ctx, k_t, S, 1, H, B);
    auto* v_t_unsq = ggml_reshape_4d(ctx, v_t, 1, S, H, B);

    // 2./3. delta = β·(v_t - kv_mem)
    auto* sk     = ggml_mul(ctx, state, k_t_unsq);
    auto* kv_mem = ggml_sum_rows(ctx, sk);             // [1, S, H, B]
    auto* v_diff = ggml_sub(ctx, v_t_unsq, kv_mem);
    auto* delta  = ggml_mul(ctx, v_diff, beta_t);      // [1, S, H, B]

    // 4. state[k, v] += k_t[k] * delta[v]
    auto* k_t_outer = ggml_repeat_4d(ctx, k_t_unsq, S, S, H, B);
    auto* k_outer   = ggml_mul(ctx, k_t_outer, delta);
    state = ggml_add(ctx, state, k_outer);

    // 5. out[v] = sum_k state[k, v] * q_t[k]
    auto* q_t_unsq = ggml_reshape_4d(ctx, q_t, S, 1, H, B);
    auto* sq       = ggml_mul(ctx, state, q_t_unsq);
    auto* out      = ggml_sum_rows(ctx, sq);
    // out shape: [1, S, H, B]
    return {out, state};
}

// Build the gated-delta-rule "linear attention" block, batched across
// `n_seqs` requests. Modeled on llama.cpp PR #19468 / HF's
// torch_recurrent_gated_delta_rule, autoregressive per token.
//   x_r          : [hidden, n_tokens, n_seqs]
//   ssm_state    : [S, S, H, n_seqs]   per-request slot view (contiguous slots)
//   conv_state   : [kernel-1, conv_dim, n_seqs]
// Returns [hidden, n_tokens, n_seqs].
ggml_tensor* build_qwen3_5_linear_layer(
        ggml_context* ctx,
        ggml_cgraph*  gf,
        ggml_tensor*  x_r,
        const LayerWeights& L,
        const Hparams& h,
        ggml_tensor*  ssm_state_r,
        ggml_tensor*  conv_state_r,
        const char*   dbg_tap_name,
        ggml_tensor** dbg_tap_out) {
    const std::int32_t n_kh    = h.qwen35_linear_num_k_heads;
    const std::int32_t n_vh    = h.qwen35_linear_num_v_heads;
    const std::int32_t hk      = h.qwen35_linear_k_head_dim;
    const std::int32_t hv      = h.qwen35_linear_v_head_dim;
    const std::int32_t key_dim = n_kh * hk;
    const std::int32_t val_dim = n_vh * hv;
    const std::int32_t conv_dim = 2 * key_dim + val_dim;
    const std::int32_t kernel  = h.qwen35_linear_conv_kernel;
    const std::int32_t n_tokens = x_r->ne[1];
    const std::int32_t n_seqs   = x_r->ne[2];

    // Reference assumes head_k_dim == head_v_dim and n_k_heads == n_v_heads
    // (the latter holds in Qwen3.5-0.8B; broader GQA via repeat — TBD).
    GGML_ASSERT(hk == hv);
    GGML_ASSERT(n_kh == n_vh);
    GGML_ASSERT(ssm_state_r->ne[3] == n_seqs);
    GGML_ASSERT(conv_state_r->ne[2] == n_seqs);
    const std::int32_t S = hv;       // head dim (== head_k == head_v)
    const std::int32_t H = n_vh;     // n_heads

    // ---- 1. Input projections (batched over n_seqs).
    auto* qkv   = ggml_mul_mat(ctx, L.lin_in_proj_qkv, x_r); // [conv_dim, n_tokens, n_seqs]
    auto* z     = ggml_mul_mat(ctx, L.lin_in_proj_z,   x_r); // [val_dim,  n_tokens, n_seqs]
    auto* alpha = ggml_mul_mat(ctx, L.lin_in_proj_a,   x_r); // [H, n_tokens, n_seqs]
    auto* b_inp = ggml_mul_mat(ctx, L.lin_in_proj_b,   x_r); // [H, n_tokens, n_seqs]

    // gate = -exp(A_log) * softplus(alpha + dt_bias).
    auto* dt_bias = (L.lin_dt_bias->type == GGML_TYPE_F32)
        ? L.lin_dt_bias
        : ggml_cast(ctx, L.lin_dt_bias, GGML_TYPE_F32);
    auto* alpha_dt = ggml_add(ctx, alpha, dt_bias);
    // softplus(x) = log(1 + exp(x)) — our ggml b6993 lacks ggml_softplus.
    auto* sp_alpha = ggml_log(ctx,
        ggml_scale_bias(ctx, ggml_exp(ctx, alpha_dt), 1.0f, 1.0f));
    auto* A_pos = ggml_exp(ctx, L.lin_A_log);                  // [H]
    auto* g = ggml_mul(ctx, sp_alpha, A_pos);                  // [H, n_tokens, n_seqs]
    g = ggml_scale(ctx, g, -1.0f);

    // ---- 2. Causal conv1d (parallel over tokens, batched over n_seqs).
    auto* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, qkv));    // [n_tokens, conv_dim, n_seqs]
    auto* conv_in = ggml_concat(ctx, conv_state_r, qkv_T, /*dim=*/0);
    // Save trailing kernel-1 tokens back to conv state.
    auto* new_conv_state = ggml_view_3d(
        ctx, conv_in,
        kernel - 1, conv_dim, n_seqs,
        conv_in->nb[1], conv_in->nb[2],
        /*offset=*/static_cast<std::size_t>(n_tokens) * conv_in->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_conv_state, conv_state_r));

    auto* qkv_conv = ggml_ssm_conv(ctx, conv_in, L.lin_conv1d); // [conv_dim, n_tokens, n_seqs]
    qkv_conv = ggml_silu(ctx, qkv_conv);

    // ---- 3. Split conv output into q, k, v (along ne[0]).
    auto* q_flat = ggml_cont(ctx, ggml_view_3d(ctx, qkv_conv,
        key_dim, n_tokens, n_seqs,
        qkv_conv->nb[1], qkv_conv->nb[2], 0));
    auto* k_flat = ggml_cont(ctx, ggml_view_3d(ctx, qkv_conv,
        key_dim, n_tokens, n_seqs,
        qkv_conv->nb[1], qkv_conv->nb[2],
        static_cast<std::size_t>(key_dim) * qkv_conv->nb[0]));
    auto* v_flat = ggml_cont(ctx, ggml_view_3d(ctx, qkv_conv,
        val_dim, n_tokens, n_seqs,
        qkv_conv->nb[1], qkv_conv->nb[2],
        static_cast<std::size_t>(2 * key_dim) * qkv_conv->nb[0]));

    auto* q = ggml_reshape_4d(ctx, q_flat, S, H, n_tokens, n_seqs);
    auto* k = ggml_reshape_4d(ctx, k_flat, S, H, n_tokens, n_seqs);
    auto* v = ggml_reshape_4d(ctx, v_flat, S, H, n_tokens, n_seqs);

    // ---- 4. L2-norm + Q scaling.
    q = ggml_l2_norm(ctx, q, h.rms_norm_eps);
    k = ggml_l2_norm(ctx, k, h.rms_norm_eps);
    q = ggml_scale(ctx, q, 1.0f / std::sqrt(static_cast<float>(S)));
    auto* beta = ggml_sigmoid(ctx, b_inp);                     // [H, n_tokens, n_seqs]
    auto* g_exp = ggml_exp(ctx, g);                            // [H, n_tokens, n_seqs]

    // Read state slot. ssm_state_r is already [S, S, H, n_seqs].
    ggml_tensor* state = ssm_state_r;

    // ---- 5. Per-token recurrence loop (autoregressive across n_tokens,
    //         batched across n_seqs). Each step processes one token per
    //         sequence and threads state.
    std::vector<ggml_tensor*> outs;
    outs.reserve(n_tokens);
    for (std::int32_t t = 0; t < n_tokens; ++t) {
        // q_t/k_t/v_t: [S, H, 1, n_seqs] — slice on token axis.
        auto* q_t = ggml_view_4d(ctx, q, S, H, 1, n_seqs,
            q->nb[1], q->nb[2], q->nb[3],
            static_cast<std::size_t>(t) * q->nb[2]);
        auto* k_t = ggml_view_4d(ctx, k, S, H, 1, n_seqs,
            k->nb[1], k->nb[2], k->nb[3],
            static_cast<std::size_t>(t) * k->nb[2]);
        auto* v_t = ggml_view_4d(ctx, v, S, H, 1, n_seqs,
            v->nb[1], v->nb[2], v->nb[3],
            static_cast<std::size_t>(t) * v->nb[2]);
        // g_t/beta_t: [1, 1, H, n_seqs] — broadcast over (S, S) state plane.
        auto* g_t_view = ggml_view_3d(ctx, g_exp, H, 1, n_seqs,
            g_exp->nb[1], g_exp->nb[2],
            static_cast<std::size_t>(t) * g_exp->nb[1]);
        auto* g_t = ggml_reshape_4d(ctx, ggml_cont(ctx, g_t_view),
                                     1, 1, H, n_seqs);
        auto* beta_t_view = ggml_view_3d(ctx, beta, H, 1, n_seqs,
            beta->nb[1], beta->nb[2],
            static_cast<std::size_t>(t) * beta->nb[1]);
        auto* beta_t = ggml_reshape_4d(ctx, ggml_cont(ctx, beta_t_view),
                                       1, 1, H, n_seqs);

        auto step = delta_step(ctx, q_t, k_t, v_t, g_t, beta_t, state, S, H);
        outs.push_back(step.y);
        state = step.state_out;
    }

    // Persist updated state slots.
    const std::int64_t state_n_elem =
        static_cast<std::int64_t>(S) * S * H * n_seqs;
    auto* state_flat_dst = ggml_view_1d(ctx, ssm_state_r, state_n_elem, 0);
    auto* state_flat_new = ggml_view_1d(ctx, state,       state_n_elem, 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, state_flat_new, state_flat_dst));

    // ---- 6. Concat per-token outputs along the token axis.
    //   Each `outs[t]` is [1, S, H, n_seqs]. Reshape to [S, 1, H, n_seqs]
    //   (no-op since ne[0]=1), then concat along ne[1].
    for (auto*& y : outs) {
        y = ggml_reshape_4d(ctx, ggml_cont(ctx, y), S, 1, H, n_seqs);
    }
    ggml_tensor* y_concat = outs[0];
    for (std::int32_t t = 1; t < n_tokens; ++t) {
        y_concat = ggml_concat(ctx, y_concat, outs[t], /*dim=*/1);
    }
    // y_concat: [S, n_tokens, H, n_seqs]. Permute to [S, H, n_tokens, n_seqs].
    y_concat = ggml_cont(ctx, ggml_permute(ctx, y_concat, 0, 2, 1, 3));
    auto* y_2d_for_norm = ggml_reshape_2d(ctx, y_concat, S, H * n_tokens * n_seqs);

    // ---- 7. z-gated norm: rmsnorm(y) * silu(z) * lin_norm.
    auto* y_normed = ggml_rms_norm(ctx, y_2d_for_norm, h.rms_norm_eps);
    y_normed = norm_scale(ctx, y_normed, L.lin_norm, /*plus_one=*/false);
    auto* z_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, z), S, H * n_tokens * n_seqs);
    auto* gated = ggml_mul(ctx, y_normed, ggml_silu(ctx, z_2d));
    auto* gated_3d = ggml_reshape_3d(ctx, gated, val_dim, n_tokens, n_seqs);

    // ---- 8. Output projection.
    auto* lin_out = ggml_mul_mat(ctx, L.lin_out_proj, gated_3d);  // [hidden, n_tokens, n_seqs]

    // Debug tap: route requested tensor to dbg_tap_out (one-shot).
    if (dbg_tap_name && dbg_tap_out && !*dbg_tap_out) {
        ggml_tensor* sel = nullptr;
        if      (std::strcmp(dbg_tap_name, "x_r_input")       == 0) sel = x_r;
        else if (std::strcmp(dbg_tap_name, "qkv_proj")        == 0) sel = qkv;
        else if (std::strcmp(dbg_tap_name, "conv_out")        == 0) sel = qkv_conv;
        else if (std::strcmp(dbg_tap_name, "q_post_l2")       == 0) sel = q;
        else if (std::strcmp(dbg_tap_name, "k_post_l2")       == 0) sel = k;
        else if (std::strcmp(dbg_tap_name, "v_post_conv")     == 0) sel = v;
        else if (std::strcmp(dbg_tap_name, "g_exp")           == 0) sel = g_exp;
        else if (std::strcmp(dbg_tap_name, "beta")            == 0) sel = beta;
        else if (std::strcmp(dbg_tap_name, "state_final")     == 0) sel = state;
        else if (std::strcmp(dbg_tap_name, "y_concat")        == 0) sel = y_concat;
        else if (std::strcmp(dbg_tap_name, "y_normed")        == 0) sel = y_normed;
        else if (std::strcmp(dbg_tap_name, "gated")           == 0) sel = gated;
        else if (std::strcmp(dbg_tap_name, "lin_out")         == 0) sel = lin_out;
        if (sel) {
            *dbg_tap_out = ggml_cont(ctx, sel);
            ggml_set_name(*dbg_tap_out, "qwen35_dbg_tap");
            ggml_set_output(*dbg_tap_out);
            ggml_build_forward_expand(gf, *dbg_tap_out);
        }
    }
    return lin_out;
}

// Build the standard full-attention block for one request (Qwen 3.5
// flavor: GQA + per-head q_norm/k_norm + mrope + attn_output_gate).
ggml_tensor* build_qwen3_5_full_layer(
        ggml_context* ctx,
        ggml_tensor*  x_r,
        const LayerWeights& L,
        const Hparams& h,
        ggml_tensor*  k_cache_layer,
        ggml_tensor*  v_cache_layer,
        ggml_tensor*  pos_input,
        ggml_tensor*  kv_idxs_full_batch,
        std::int32_t  qo_start_in_full_batch,
        ggml_tensor*  gather_idx_r,
        ggml_tensor*  mask_r,
        std::int32_t  n_tokens,
        std::int32_t  n_kv) {
    const std::int32_t hd      = h.head_dim;
    const std::int32_t n_q_h   = h.num_attention_heads;
    const std::int32_t n_kv_h  = h.num_key_value_heads;
    const std::int32_t n_embd_gqa = n_kv_h * hd;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // Q is double-wide when attn_output_gate=true.
    auto* QG = ggml_mul_mat(ctx, L.q_proj, x_r);   // [q_out (==2*hd*n_q_h when gated), n_tokens]
    auto* K  = ggml_mul_mat(ctx, L.k_proj, x_r);   // [n_kv_h*hd, n_tokens]
    auto* V  = ggml_mul_mat(ctx, L.v_proj, x_r);

    ggml_tensor* Q   = nullptr;
    ggml_tensor* gate = nullptr;
    if (h.qwen35_attn_output_gate) {
        // QG layout post-projection [q_out, n_tokens]. Reshape per python:
        //   self.q_proj(x).view(*input_shape, -1, head_dim*2) → split last dim.
        // In ggml: reshape to [hd*2, n_q_h, n_tokens] then split.
        auto* QG3 = ggml_reshape_3d(ctx, QG, hd * 2, n_q_h, n_tokens);
        Q = ggml_cont(ctx, ggml_view_3d(ctx, QG3, hd, n_q_h, n_tokens,
                                         QG3->nb[1], QG3->nb[2], 0));
        gate = ggml_cont(ctx, ggml_view_3d(ctx, QG3, hd, n_q_h, n_tokens,
                                            QG3->nb[1], QG3->nb[2],
                                            static_cast<std::size_t>(hd) * QG3->nb[0]));
    } else {
        Q = ggml_reshape_3d(ctx, QG, hd, n_q_h, n_tokens);
    }

    // Per-head q_norm, k_norm.
    K = ggml_reshape_3d(ctx, K, hd, n_kv_h, n_tokens);
    V = ggml_reshape_3d(ctx, V, hd, n_kv_h, n_tokens);
    Q = ggml_rms_norm(ctx, Q, h.rms_norm_eps);
    Q = norm_scale(ctx, Q, L.q_norm, /*plus_one=*/true);
    K = ggml_rms_norm(ctx, K, h.rms_norm_eps);
    K = norm_scale(ctx, K, L.k_norm, /*plus_one=*/true);

    // mrope. For text-only inference all 4 axes share the same position;
    // we built pos_input as [4 * n_total] in the engine. ggml_rope_multi
    // takes the same pos tensor for both Q and K. n_dims is the number of
    // dims rotated, capped by partial_rotary_factor.
    int sections[4] = {
        h.qwen35_mrope_section[0],
        h.qwen35_mrope_section[1],
        h.qwen35_mrope_section[2],
        0,
    };
    const int rope_n_dims = static_cast<int>(hd * h.qwen35_partial_rotary_factor);
    // GGML rope-type constants are distinct exact mode values (not flags):
    // MROPE=8 (sectioned, axis-by-section), IMROPE=40 (sectioned interleaved
    // by sector mod 3 — what Qwen 3.5 mrope_interleaved=true uses),
    // VISION=24 (asserts n_dims==ne00/2; for vision tower).
    const int rope_mode = h.qwen35_mrope_interleaved
        ? GGML_ROPE_TYPE_IMROPE
        : GGML_ROPE_TYPE_MROPE;

    // Slice the per-request positions out of the global pos_input. For
    // mrope, pos has 4 entries per token: positions for this request
    // start at qo_start_in_full_batch * 4.
    auto* pos_r = ggml_view_1d(ctx, pos_input, n_tokens * 4,
                                static_cast<std::size_t>(qo_start_in_full_batch) * 4 *
                                ggml_element_size(pos_input));

    Q = ggml_rope_multi(ctx, Q, pos_r, /*c=*/nullptr,
                        rope_n_dims, sections, rope_mode,
                        /*n_ctx_orig=*/0, h.rope_theta, /*freq_scale=*/1.0f,
                        /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                        /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
    K = ggml_rope_multi(ctx, K, pos_r, /*c=*/nullptr,
                        rope_n_dims, sections, rope_mode, 0,
                        h.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // KV cache write (uses the global kv_idxs sliced for this request).
    auto* k_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, K), n_embd_gqa, n_tokens);
    auto* v_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, V), n_embd_gqa, n_tokens);
    auto* kv_idxs_r = ggml_view_1d(ctx, kv_idxs_full_batch, n_tokens,
                                    static_cast<std::size_t>(qo_start_in_full_batch) *
                                    ggml_element_size(kv_idxs_full_batch));
    auto* k_cached = ggml_set_rows(ctx, k_cache_layer, k_2d, kv_idxs_r);
    auto* v_cached = ggml_set_rows(ctx, v_cache_layer, v_2d, kv_idxs_r);

    // Per-request flash-attn (head_dim=256 fits ggml's flash_attn).
    // Pass Q in [hd, n_q_h, n_tokens] form via the existing helper —
    // qo_start=0 since we already have a per-request Q.
    auto* attn = build_request_flash_attn(
        ctx, Q, k_cached, v_cached, gather_idx_r, mask_r,
        /*qo_start=*/0, n_tokens, n_kv,
        hd, n_kv_h, n_q_h,
        kq_scale, /*attn_softcap=*/0.0f, /*sinks=*/nullptr);
    // attn shape: [hd, n_q_h, n_tokens, 1].

    // Apply output gate before reshape: attn = attn * sigmoid(gate).
    if (h.qwen35_attn_output_gate) {
        auto* attn_3d = ggml_reshape_3d(ctx, attn, hd, n_q_h, n_tokens);
        auto* gate_sig = ggml_sigmoid(ctx, gate);
        attn_3d = ggml_mul(ctx, attn_3d, gate_sig);
        attn = attn_3d;
    }

    auto* attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn), hd * n_q_h, n_tokens);
    return ggml_mul_mat(ctx, L.o_proj, attn_2d);
}

// SwiGLU FFN (same on both layer types).
ggml_tensor* build_qwen3_5_ffn(ggml_context* ctx,
                               ggml_tensor* cur,
                               const LayerWeights& L) {
    auto* gate = ggml_mul_mat(ctx, L.gate_proj, cur);
    auto* up   = ggml_mul_mat(ctx, L.up_proj,   cur);
    gate = ggml_silu(ctx, gate);
    auto* gated = ggml_mul(ctx, gate, up);
    return ggml_mul_mat(ctx, L.down_proj, gated);
}

}  // namespace

GraphResult build_qwen3_5_graph(ggml_context* ctx,
                                const Model& model,
                                KvCachePaged& kv,
                                StateCache& state,
                                const ForwardEngine::BatchPlan& plan) {
    const auto& h = model.hparams();
    const auto& w = model.weights();
    const std::int32_t n_total  = plan.total_n_tokens;
    const std::int32_t n_req    = static_cast<std::int32_t>(plan.reqs.size());

    auto* gf = ggml_new_graph_custom(ctx, static_cast<int>(GRAPH_MAX_NODES),
                                     /*grads=*/false);

    // Debug tap: env-driven, one tensor from layer 0 of request 0.
    const char*   dbg_tap_name = std::getenv("PIE_QWEN35_DUMP");
    ggml_tensor*  dbg_tap_out  = nullptr;

    // Standard inputs (slow-path: per-request masks + gathers).
    GraphInputs in = declare_graph_inputs(ctx, plan, n_total, n_req);
    // Replace pos_input with a 4×-wide variant for mrope. (ggml_rope_multi
    // expects 4 positions per token.) We'll upload positions repeated 4×.
    in.pos_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                                      static_cast<std::int64_t>(n_total) * 4);
    ggml_set_name(in.pos_input, "inp_pos_mrope");
    ggml_set_input(in.pos_input);

    // Embed.
    auto* embd = ggml_get_rows(ctx, w.tok_embd, in.tok_input);  // [hidden, n_total]
    if (dbg_tap_name && std::strcmp(dbg_tap_name, "embed") == 0 && !dbg_tap_out) {
        dbg_tap_out = ggml_cont(ctx, embd);
        ggml_set_name(dbg_tap_out, "qwen35_dbg_tap_embed");
        ggml_set_output(dbg_tap_out);
        ggml_build_forward_expand(gf, dbg_tap_out);
    }

    // Decide between batched (per-layer outer) and per-request (per-request
    // outer) shapes. Batched mode requires:
    //   (a) every request has the same n_tokens (so x reshapes to a
    //       3D [hidden, n_tokens, n_seqs] tensor cleanly), and
    //   (b) state slots are packed contiguously starting at slot 0
    //       (so the cache slice [..., n_seqs] addresses the right slots).
    // Both hold for the test harness; production runtimes that reuse
    // arbitrary slots fall back to the per-request loop.
    bool batched = (n_req >= 1) && plan.reqs[0].state_slot == 0;
    const std::int32_t n_tokens_r0 = plan.reqs[0].n_tokens;
    for (std::int32_t r = 0; r < n_req && batched; ++r) {
        const auto& R = plan.reqs[r];
        if (R.n_tokens != n_tokens_r0)        batched = false;
        if (R.state_slot != r)                batched = false;
        if (R.qo_start  != r * n_tokens_r0)   batched = false;
    }

    ggml_tensor* finals = nullptr;

    if (batched) {
        // x: [hidden, n_tokens_per_req, n_seqs]. ggml_get_rows produced the
        // tokens in qo_start order so a reshape suffices.
        auto* x = ggml_reshape_3d(ctx, embd, h.hidden_size, n_tokens_r0, n_req);

        for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
            const auto& L = w.layers[il];
            const char kind = h.layer_types[il];
            auto* inpSA = x;

            auto* cur = ggml_rms_norm(ctx, x, h.rms_norm_eps);
            cur = norm_scale(ctx, cur, L.attn_norm, /*plus_one=*/true);

            ggml_tensor* attn_out = nullptr;
            if (kind == 'l') {
                // Batched state slice: contiguous slots [0, n_req).
                auto* ssm  = state.ssm(il);
                auto* conv = state.conv(il);
                auto* ssm_r = ggml_view_4d(
                    ctx, ssm,
                    state.head_k_dim(), state.head_v_dim(), state.n_heads(), n_req,
                    ssm->nb[1], ssm->nb[2], ssm->nb[3], 0);
                auto* conv_r = ggml_view_3d(
                    ctx, conv,
                    state.conv_kernel() - 1, state.conv_dim(), n_req,
                    conv->nb[1], conv->nb[2], 0);
                const bool tap_here = (il == 0);
                attn_out = build_qwen3_5_linear_layer(
                    ctx, gf, cur, L, h, ssm_r, conv_r,
                    tap_here ? dbg_tap_name : nullptr,
                    tap_here ? &dbg_tap_out : nullptr);
            } else {
                // Full-attention layers still need per-request masks/gathers.
                // Slice cur on ne[2]=request, run per-request, concat.
                std::vector<ggml_tensor*> per_req;
                per_req.reserve(n_req);
                for (std::int32_t r = 0; r < n_req; ++r) {
                    const auto& R = plan.reqs[r];
                    auto* cur_r = ggml_view_2d(
                        ctx, cur, h.hidden_size, n_tokens_r0,
                        cur->nb[1],
                        static_cast<std::size_t>(r) * cur->nb[2]);
                    cur_r = ggml_cont(ctx, cur_r);
                    auto* attn_r = build_qwen3_5_full_layer(
                        ctx, cur_r, L, h,
                        kv.k(il), kv.v(il),
                        in.pos_input, in.kv_idxs, R.qo_start,
                        in.gather_idxs[r], in.masks[r],
                        R.n_tokens, R.n_kv);
                    per_req.push_back(attn_r);
                }
                ggml_tensor* concat = per_req[0];
                for (std::int32_t r = 1; r < n_req; ++r) {
                    concat = ggml_concat(ctx, concat, per_req[r], /*dim=*/1);
                }
                attn_out = ggml_reshape_3d(
                    ctx, ggml_cont(ctx, concat),
                    h.hidden_size, n_tokens_r0, n_req);
            }

            auto* ffn_in = ggml_add(ctx, attn_out, inpSA);
            auto* cur2 = ggml_rms_norm(ctx, ffn_in, h.rms_norm_eps);
            cur2 = norm_scale(ctx, cur2, L.ffn_norm, /*plus_one=*/true);
            auto* ffn_out = build_qwen3_5_ffn(ctx, cur2, L);
            x = ggml_add(ctx, ffn_out, ffn_in);
        }
        // Flatten [hidden, n_tokens, n_seqs] back to [hidden, n_total].
        finals = ggml_reshape_2d(ctx, ggml_cont(ctx, x),
                                  h.hidden_size, n_tokens_r0 * n_req);
    } else {
        // Fallback: per-request outer loop (handles non-uniform n_tokens
        // or scattered state slots).
        std::vector<ggml_tensor*> per_req_finals;
        per_req_finals.reserve(n_req);
        const std::size_t embd_stride =
            static_cast<std::size_t>(h.hidden_size) * ggml_type_size(embd->type);

        for (std::int32_t r = 0; r < n_req; ++r) {
            const auto& R = plan.reqs[r];
            auto* x_r = ggml_view_2d(ctx, embd, h.hidden_size, R.n_tokens,
                                      embd->nb[1],
                                      static_cast<std::size_t>(R.qo_start) * embd_stride);
            x_r = ggml_cont(ctx, x_r);
            // Lift to 3D so the linear layer's batched code sees n_seqs=1.
            auto* x_r_3d = ggml_reshape_3d(ctx, x_r, h.hidden_size, R.n_tokens, 1);

            for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
                const auto& L = w.layers[il];
                const char kind = h.layer_types[il];
                auto* inpSA = x_r_3d;

                auto* cur = ggml_rms_norm(ctx, x_r_3d, h.rms_norm_eps);
                cur = norm_scale(ctx, cur, L.attn_norm, /*plus_one=*/true);

                ggml_tensor* attn_out = nullptr;
                if (kind == 'l') {
                    auto* ssm = state.ssm(il);
                    auto* conv = state.conv(il);
                    const std::size_t ssm_slot_bytes =
                        static_cast<std::size_t>(state.head_k_dim()) *
                        state.head_v_dim() * state.n_heads() * sizeof(float);
                    const std::size_t conv_slot_bytes =
                        static_cast<std::size_t>(state.conv_kernel() - 1) *
                        state.conv_dim() * sizeof(float);
                    auto* ssm_r = ggml_view_4d(
                        ctx, ssm,
                        state.head_k_dim(), state.head_v_dim(), state.n_heads(), 1,
                        ssm->nb[1], ssm->nb[2], ssm->nb[3],
                        static_cast<std::size_t>(R.state_slot) * ssm_slot_bytes);
                    auto* conv_r = ggml_view_3d(
                        ctx, conv,
                        state.conv_kernel() - 1, state.conv_dim(), 1,
                        conv->nb[1], conv->nb[2],
                        static_cast<std::size_t>(R.state_slot) * conv_slot_bytes);
                    const bool tap_here = (r == 0 && il == 0);
                    attn_out = build_qwen3_5_linear_layer(
                        ctx, gf, cur, L, h, ssm_r, conv_r,
                        tap_here ? dbg_tap_name : nullptr,
                        tap_here ? &dbg_tap_out : nullptr);
                } else {
                    // Full-attn function expects 2D [hidden, n_tokens].
                    auto* cur_2d = ggml_reshape_2d(ctx, cur, h.hidden_size,
                                                    R.n_tokens);
                    auto* attn_2d = build_qwen3_5_full_layer(
                        ctx, cur_2d, L, h,
                        kv.k(il), kv.v(il),
                        in.pos_input, in.kv_idxs, R.qo_start,
                        in.gather_idxs[r], in.masks[r],
                        R.n_tokens, R.n_kv);
                    attn_out = ggml_reshape_3d(ctx, attn_2d,
                                                h.hidden_size, R.n_tokens, 1);
                }

                auto* ffn_in = ggml_add(ctx, attn_out, inpSA);
                auto* cur2 = ggml_rms_norm(ctx, ffn_in, h.rms_norm_eps);
                cur2 = norm_scale(ctx, cur2, L.ffn_norm, /*plus_one=*/true);
                auto* ffn_out = build_qwen3_5_ffn(ctx, cur2, L);
                x_r_3d = ggml_add(ctx, ffn_out, ffn_in);
            }
            per_req_finals.push_back(
                ggml_reshape_2d(ctx, ggml_cont(ctx, x_r_3d),
                                h.hidden_size, plan.reqs[r].n_tokens));
        }

        // Concatenate per-request finals along the token dim.
        finals = per_req_finals[0];
        for (std::int32_t r = 1; r < n_req; ++r) {
            finals = ggml_concat(ctx, finals, per_req_finals[r], /*dim=*/1);
        }
        finals = ggml_cont(ctx, finals);
    }

    // Final RMSNorm + LM head.
    auto* cur = ggml_rms_norm(ctx, finals, h.rms_norm_eps);
    cur = norm_scale(ctx, cur, w.output_norm, /*plus_one=*/true);
    auto* sampled = ggml_get_rows(ctx, cur, in.out_idx);
    ggml_tensor* lm_head_w = h.tie_word_embeddings ? w.tok_embd : w.output_head;
    auto* logits = ggml_mul_mat(ctx, lm_head_w, sampled);

    // Optional GPU-side sampler (mirrors graph_qwen3 fast paths). Saves
    // the ~vocab*n_slots*4-byte logits download on greedy / uniform-top-K
    // batches; the slow path materializes full logits.
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
    if (plan.all_greedy)             res.tokens_out = tokens_out;
    else if (plan.uniform_top_sample) { res.top_k_idx = top_k_idx; res.top_k_probs = top_k_probs; }
    else                              res.logits = logits;
    res.in = in;
    res.debug_tensor = dbg_tap_out;
    res.debug_name   = dbg_tap_name;
    return res;
}

}  // namespace pie_portable_driver
