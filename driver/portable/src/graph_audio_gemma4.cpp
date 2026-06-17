#include "graph_audio_gemma4.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace pie_portable_driver {

namespace {

// Clipped linear: clamp(x) -> W x -> clamp(out). x is [in, tokens].
ggml_tensor* clip_linear(ggml_context* ctx, const G4AudClipLinear& l,
                         ggml_tensor* x) {
    if (l.has_in) x = ggml_clamp(ctx, x, l.imin, l.imax);
    ggml_tensor* y = ggml_mul_mat(ctx, l.w, x);
    if (l.has_out) y = ggml_clamp(ctx, y, l.omin, l.omax);
    return y;
}

// RMSNorm (variance-only) over ne0, optional gamma. Gemma multiplies by the
// weight directly (no +1); a null weight normalizes only.
ggml_tensor* rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* g, float eps) {
    ggml_tensor* y = ggml_rms_norm(ctx, x, eps);
    if (g) {
        ggml_tensor* gf = g->type == GGML_TYPE_F32 ? g : ggml_cast(ctx, g, GGML_TYPE_F32);
        y = ggml_mul(ctx, y, gf);
    }
    return y;
}

// LayerNorm over ne0 (no bias) + learnable scale + ReLU. Used by SSCP, where the
// channel axis has been permuted to ne0.
ggml_tensor* layernorm_relu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* g,
                            float eps) {
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    ggml_tensor* gf = g->type == GGML_TYPE_F32 ? g : ggml_cast(ctx, g, GGML_TYPE_F32);
    y = ggml_mul(ctx, y, gf);
    return ggml_relu(ctx, y);
}

// Shift along the sequence axis (ne1) by `s`: position i takes x[.., i-s, ..],
// with zeros for i < s. Works for [C, N] and [C, N, H] tensors.
ggml_tensor* shift_seq(ggml_context* ctx, ggml_tensor* x, int s) {
    if (s <= 0) return x;
    const std::int64_t C = x->ne[0], N = x->ne[1], H = x->ne[2];
    ggml_tensor* crop = ggml_cont(ctx, ggml_view_3d(ctx, x, C, N - s, H,
                                                    x->nb[1], x->nb[2], 0));
    ggml_tensor* z = ggml_cont(ctx, ggml_view_3d(ctx, x, C, s, H,
                                                 x->nb[1], x->nb[2], 0));
    z = ggml_scale(ctx, z, 0.0f);
    return ggml_concat(ctx, z, crop, 1);
}

// Logit soft-cap: cap * tanh(s / cap).
ggml_tensor* soft_cap(ggml_context* ctx, ggml_tensor* s, float cap) {
    return ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, s, 1.0f / cap)), cap);
}

}  // namespace

ggml_tensor* build_gemma4_audio_graph(ggml_context* ctx,
                                      const Gemma4AudioWeights& w,
                                      const Hparams& h,
                                      ggml_tensor* features,
                                      ggml_tensor* pe,
                                      ggml_tensor* win_mask,
                                      std::int32_t n_frames,
                                      std::int32_t n_token) {
    const std::int32_t hidden = h.audio_hidden_size;
    const std::int32_t heads  = h.audio_num_heads;
    const std::int32_t hd     = hidden / heads;
    const std::int32_t n_mel  = h.audio_feature_size;
    const std::int32_t C0     = h.audio_sscp_channels0;
    const std::int32_t C1     = h.audio_sscp_channels1;
    const std::int32_t K      = h.audio_conv_kernel;
    const std::int32_t P      = h.audio_context_left;   // pe rows (= context_left)
    const std::int32_t Dwin   = P - 1;                  // window width (max_past)
    const float eps   = h.audio_ln_eps;
    const float cap   = h.audio_logit_cap;
    const float RW    = h.audio_residual_weight;
    const float q_scale = std::pow(static_cast<float>(hd), -0.5f) / std::log(2.0f);
    const float k_scale = std::log(1.0f + static_cast<float>(std::exp(1.0))) /
                          std::log(2.0f);

    // ── 1) SSCP subsampling conv stack ───────────────────────────────────────
    // features [n_mel, n_frames] -> conv input [W=n_mel, H=n_frames, C=1, N=1].
    ggml_tensor* b0 = ggml_reshape_4d(ctx, features, n_mel, n_frames, 1, 1);
    // Conv kernels cast to F32 so ggml_conv_2d's internal im2col + mul_mat run
    // F32xF32 (ggml_conv_2d uses a->type for both operands).
    ggml_tensor* k0 = ggml_cast(ctx, w.sscp0_conv, GGML_TYPE_F32);
    ggml_tensor* c0 = ggml_conv_2d(ctx, k0, b0, 2, 2, 1, 1, 1, 1);  // [F1, T1, C0, 1]
    // LayerNorm-over-channel + ReLU: permute channel to ne0.
    c0 = ggml_cont(ctx, ggml_permute(ctx, c0, 1, 2, 0, 3));         // [C0, F1, T1]
    c0 = layernorm_relu(ctx, c0, w.sscp0_norm, eps);
    c0 = ggml_cont(ctx, ggml_permute(ctx, c0, 2, 0, 1, 3));         // [F1, T1, C0]
    const std::int64_t F1 = c0->ne[0], T1 = c0->ne[1];
    c0 = ggml_reshape_4d(ctx, c0, F1, T1, C0, 1);

    ggml_tensor* k1 = ggml_cast(ctx, w.sscp1_conv, GGML_TYPE_F32);
    ggml_tensor* c1 = ggml_conv_2d(ctx, k1, c0, 2, 2, 1, 1, 1, 1);  // [F2, T2, C1, 1]
    c1 = ggml_cont(ctx, ggml_permute(ctx, c1, 1, 2, 0, 3));         // [C1, F2, T2]
    c1 = layernorm_relu(ctx, c1, w.sscp1_norm, eps);                // norm over C1
    const std::int64_t F2 = c1->ne[1], T2 = c1->ne[2];
    // Flatten to [C1*F2, T2] (channel inner, freq outer) for input_proj.
    ggml_tensor* flat = ggml_reshape_2d(ctx, ggml_cont(ctx, c1), C1 * F2, T2);
    ggml_tensor* cur = ggml_mul_mat(ctx, w.sscp_input_proj, flat);  // [hidden, N]
    const std::int32_t N = static_cast<std::int32_t>(T2);

    // Relative-position cos/sin lives in `pe` [hidden, P] (host sinusoid).
    ggml_tensor* win = ggml_reshape_3d(ctx, win_mask, Dwin, N, 1);

    // ── 2) Conformer layers ──────────────────────────────────────────────────
    auto ffn = [&](const G4AudFfn& ff, ggml_tensor* x) -> ggml_tensor* {
        ggml_tensor* m = rms(ctx, x, ff.pre_ln, eps);
        m = clip_linear(ctx, ff.fc1, m);
        m = ggml_silu(ctx, m);
        m = clip_linear(ctx, ff.fc2, m);
        m = rms(ctx, m, ff.post_ln, eps);
        return ggml_add(ctx, x, ggml_scale(ctx, m, RW));
    };

    for (const G4AudLayer& L : w.layers) {
        // --- feed_forward1 (macaron half) ---
        cur = ffn(L.ff1, cur);

        // --- chunked-local self-attention ---
        ggml_tensor* x = rms(ctx, cur, L.norm_pre_attn, eps);
        ggml_tensor* q = clip_linear(ctx, L.q, x);
        ggml_tensor* k = clip_linear(ctx, L.k, x);
        ggml_tensor* v = clip_linear(ctx, L.v, x);
        q = ggml_reshape_3d(ctx, q, hd, heads, N);
        k = ggml_reshape_3d(ctx, k, hd, heads, N);
        v = ggml_reshape_3d(ctx, v, hd, heads, N);
        // q *= q_scale * softplus(per_dim_scale); k *= k_scale.
        ggml_tensor* pds = L.per_dim_scale->type == GGML_TYPE_F32
                               ? L.per_dim_scale
                               : ggml_cast(ctx, L.per_dim_scale, GGML_TYPE_F32);
        ggml_tensor* qf = ggml_scale(ctx, ggml_softplus(ctx, pds), q_scale);
        qf = ggml_reshape_3d(ctx, qf, hd, 1, 1);
        q = ggml_mul(ctx, q, qf);
        k = ggml_scale(ctx, k, k_scale);
        // Head-major batch layout [hd, N, H].
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        // relk = relative_k_proj(pe) -> [hidden, P] -> [hd, P, H].
        ggml_tensor* relk = ggml_mul_mat(ctx, L.relative_k, pe);   // [hidden, P]
        relk = ggml_reshape_3d(ctx, relk, hd, heads, P);
        relk = ggml_cont(ctx, ggml_permute(ctx, relk, 0, 2, 1, 3));  // [hd, P, H]

        ggml_tensor* SC = nullptr;            // [Dwin, N, H] scores
        std::vector<ggml_tensor*> v_sh(Dwin); // shifted V per offset
        for (std::int32_t d = 0; d < Dwin; ++d) {
            ggml_tensor* k_sh = shift_seq(ctx, k, d);       // k_{i-d}
            v_sh[d] = shift_seq(ctx, v, d);
            ggml_tensor* qk = ggml_sum_rows(ctx, ggml_mul(ctx, q, k_sh));  // [1,N,H]
            // rel term: q_i . relk[(P-1)-d].
            const std::int64_t p = (P - 1) - d;
            ggml_tensor* rcol = ggml_cont(ctx, ggml_view_3d(
                ctx, relk, hd, 1, heads, relk->nb[1], relk->nb[2],
                p * relk->nb[1]));                          // [hd,1,H]
            ggml_tensor* rel = ggml_mul_mat(ctx, rcol, q);  // [1,N,H]
            ggml_tensor* s = soft_cap(ctx, ggml_add(ctx, qk, rel), cap);  // [1,N,H]
            SC = SC ? ggml_concat(ctx, SC, s, 0) : s;
        }
        SC = ggml_add(ctx, SC, win);              // window mask (broadcast over H)
        ggml_tensor* W = ggml_soft_max(ctx, SC);  // softmax over offset axis (ne0)
        ggml_tensor* o = nullptr;
        for (std::int32_t d = 0; d < Dwin; ++d) {
            ggml_tensor* wd = ggml_cont(ctx, ggml_view_3d(
                ctx, W, 1, N, heads, W->nb[1], W->nb[2], d * W->nb[0]));  // [1,N,H]
            ggml_tensor* contrib = ggml_mul(ctx, v_sh[d], wd);           // [hd,N,H]
            o = o ? ggml_add(ctx, o, contrib) : contrib;
        }
        // [hd, N, H] -> [hidden, N].
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));   // [hd, H, N]
        o = ggml_reshape_2d(ctx, o, hidden, N);
        o = clip_linear(ctx, L.post, o);
        o = rms(ctx, o, L.norm_post_attn, eps);
        cur = ggml_add(ctx, cur, o);

        // --- light depthwise-conv module ---
        ggml_tensor* hn = rms(ctx, cur, L.lconv_pre_ln, eps);
        ggml_tensor* start = clip_linear(ctx, L.lconv_start, hn);  // [2*hidden, N]
        ggml_tensor* a = ggml_cont(ctx, ggml_view_2d(ctx, start, hidden, N,
                                                     start->nb[1], 0));
        ggml_tensor* g = ggml_cont(ctx, ggml_view_2d(ctx, start, hidden, N,
                                                     start->nb[1],
                                                     hidden * start->nb[0]));
        ggml_tensor* glu = ggml_mul(ctx, a, ggml_sigmoid(ctx, g));  // [hidden, N]
        // Depthwise causal conv k: out[t] = sum_j w[:,j] * glu[t-(K-1)+j].
        ggml_tensor* wdw = ggml_reshape_2d(ctx, L.depthwise_conv, K, hidden);
        wdw = ggml_cont(ctx, ggml_transpose(ctx, wdw));            // [hidden, K]
        if (wdw->type != GGML_TYPE_F32) wdw = ggml_cast(ctx, wdw, GGML_TYPE_F32);
        ggml_tensor* conv = nullptr;
        for (std::int32_t j = 0; j < K; ++j) {
            ggml_tensor* in_sh = shift_seq(ctx, glu, (K - 1) - j);
            ggml_tensor* wc = ggml_cont(ctx, ggml_view_2d(ctx, wdw, hidden, 1,
                                                          wdw->nb[1],
                                                          j * wdw->nb[1]));  // [hidden,1]
            ggml_tensor* term = ggml_mul(ctx, in_sh, wc);
            conv = conv ? ggml_add(ctx, conv, term) : term;
        }
        conv = rms(ctx, conv, L.lconv_conv_norm, eps);
        conv = ggml_silu(ctx, conv);
        conv = clip_linear(ctx, L.lconv_end, conv);
        cur = ggml_add(ctx, cur, conv);

        // --- feed_forward2 (macaron half) ---
        cur = ffn(L.ff2, cur);

        // --- norm_out ---
        cur = rms(ctx, cur, L.norm_out, eps);
    }

    // ── 3) output_proj (hidden -> out_proj_dims, +bias) ──────────────────────
    ggml_tensor* enc = ggml_mul_mat(ctx, w.output_proj_w, cur);  // [OPD, N]
    ggml_tensor* bias = w.output_proj_b->type == GGML_TYPE_F32
                            ? w.output_proj_b
                            : ggml_cast(ctx, w.output_proj_b, GGML_TYPE_F32);
    enc = ggml_add(ctx, enc, bias);

    // ── 4) embedder: parameterless RMSNorm -> projection -> text hidden ──────
    ggml_tensor* en = rms(ctx, enc, nullptr, eps);
    ggml_tensor* out = ggml_mul_mat(ctx, w.embed_proj, en);      // [text_hidden, N]
    (void)n_token;
    return out;
}

// ── Host helpers ─────────────────────────────────────────────────────────────

std::vector<float> gemma4_audio_rel_pos_enc(std::int32_t P, std::int32_t hidden) {
    std::vector<float> pe(static_cast<std::size_t>(P) * hidden);
    const std::int32_t num_ts = hidden / 2;
    const float log_inc = std::log(10000.0f) /
                          std::max(static_cast<float>(num_ts - 1), 1.0f);
    for (std::int32_t r = 0; r < P; ++r) {
        const float pos = static_cast<float>((P - 1) - r);
        float* row = pe.data() + static_cast<std::size_t>(r) * hidden;
        for (std::int32_t d = 0; d < hidden; ++d) {
            const std::int32_t m = d < num_ts ? d : d - num_ts;
            const float inv = std::exp(static_cast<float>(m) * -log_inc);
            const float t = pos * inv;
            row[d] = d < num_ts ? std::sin(t) : std::cos(t);
        }
    }
    return pe;
}

std::vector<float> gemma4_audio_window_mask(std::int32_t dwin,
                                            std::int32_t n_token) {
    std::vector<float> m(static_cast<std::size_t>(dwin) * n_token, 0.0f);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (std::int32_t d = 0; d < dwin; ++d)
        for (std::int32_t i = 0; i < n_token; ++i)
            if (i < d) m[static_cast<std::size_t>(d) * n_token + i] = ninf;
    return m;
}

}  // namespace pie_portable_driver
