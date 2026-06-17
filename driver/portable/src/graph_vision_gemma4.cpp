#include "graph_vision_gemma4.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pie_portable_driver {

namespace {

// Clipped linear: clamp(x) -> W x -> clamp(out). x is [in, tokens].
ggml_tensor* clip_linear(ggml_context* ctx, const G4VisClipLinear& l,
                         ggml_tensor* x) {
    if (l.has_in) x = ggml_clamp(ctx, x, l.imin, l.imax);
    ggml_tensor* y = ggml_mul_mat(ctx, l.w, x);
    if (l.has_out) y = ggml_clamp(ctx, y, l.omin, l.omax);
    return y;
}

// RMSNorm (variance-only) over ne0, optional gamma. Gemma multiplies by the
// weight directly (no +1); v-norm passes a null weight (normalize only).
ggml_tensor* rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* g, float eps) {
    ggml_tensor* y = ggml_rms_norm(ctx, x, eps);
    if (g) {
        ggml_tensor* gf = g->type == GGML_TYPE_F32 ? g : ggml_cast(ctx, g, GGML_TYPE_F32);
        y = ggml_mul(ctx, y, gf);
    }
    return y;
}

// Blocked rotate for Gemma-4 vision 2D-RoPE: split head_dim (ne0) into four
// 16-chunks [A,B,C,D]; return [-B, A, -D, C]. x is [head_dim, heads, n_patch].
ggml_tensor* rotate_blocked(ggml_context* ctx, ggml_tensor* x) {
    const std::int64_t d = x->ne[0];
    const std::int64_t q = d / 4;  // 16
    auto chunk = [&](std::int64_t off) {
        return ggml_cont(ctx, ggml_view_3d(ctx, x, q, x->ne[1], x->ne[2],
                                            x->nb[1], x->nb[2], off * x->nb[0]));
    };
    ggml_tensor* A = chunk(0), *B = chunk(q), *C = chunk(2 * q), *D = chunk(3 * q);
    ggml_tensor* r = ggml_concat(ctx, ggml_neg(ctx, B), A, 0);   // [-B, A]
    r = ggml_concat(ctx, r, ggml_neg(ctx, D), 0);                // [-B, A, -D]
    r = ggml_concat(ctx, r, C, 0);                               // [-B, A, -D, C]
    return r;
}

ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* cos_b, ggml_tensor* sin_b) {
    return ggml_add(ctx, ggml_mul(ctx, x, cos_b),
                    ggml_mul(ctx, rotate_blocked(ctx, x), sin_b));
}

}  // namespace

ggml_tensor* build_gemma4_vision_graph(ggml_context* ctx,
                                       const Gemma4VisionWeights& w,
                                       const Hparams& h,
                                       ggml_tensor* pixels,
                                       ggml_tensor* pos_embed_in,
                                       ggml_tensor* rope_cos,
                                       ggml_tensor* rope_sin,
                                       ggml_tensor* pool_matrix,
                                       ggml_tensor* attn_mask,
                                       std::int32_t n_patch,
                                       std::int32_t n_token) {
    const std::int32_t hidden  = h.vision_hidden_size;
    const std::int32_t heads   = h.vision_num_heads;
    const std::int32_t head_dim = h.vision_head_dim > 0 ? h.vision_head_dim
                                                        : hidden / heads;
    const float eps   = h.vision_ln_eps;
    // Attention scale is 1.0 (NOT 1/sqrt(head_dim)). HF Gemma3n vision relies on
    // the per-head q/k RMSNorm for magnitude control and applies no extra QK
    // scaling — the parity-verified CUDA reference (k_qk scale=1.0f) confirms
    // this. Using 1/sqrt(head_dim) here flattens the softmax 8x and produces a
    // blurry, garbled encode.
    const float scale = 1.0f;

    ggml_tensor* cos_b = ggml_reshape_3d(ctx, rope_cos, head_dim, 1, n_patch);
    ggml_tensor* sin_b = ggml_reshape_3d(ctx, rope_sin, head_dim, 1, n_patch);

    // Patch proj -> + factored pos-embed (host lookup). The pixel scale
    // 2*(x-0.5) is applied host-side when staging vis_pixels (no-alloc graph
    // ctx can't hold a constant bias tensor).
    ggml_tensor* cur = ggml_mul_mat(ctx, w.patch_w, pixels);  // [hidden, n_patch]
    cur = ggml_add(ctx, cur, pos_embed_in);

    for (const G4VisLayer& L : w.layers) {
        // ---- Attention ----
        ggml_tensor* x = rms(ctx, cur, L.in_ln, eps);
        ggml_tensor* q = clip_linear(ctx, L.q, x);
        ggml_tensor* k = clip_linear(ctx, L.k, x);
        ggml_tensor* v = clip_linear(ctx, L.v, x);
        q = ggml_reshape_3d(ctx, q, head_dim, heads, n_patch);
        k = ggml_reshape_3d(ctx, k, head_dim, heads, n_patch);
        v = ggml_reshape_3d(ctx, v, head_dim, heads, n_patch);
        q = rms(ctx, q, L.q_norm, eps);
        k = rms(ctx, k, L.k_norm, eps);
        v = rms(ctx, v, nullptr, eps);          // v: normalize only (no weight)
        q = apply_rope(ctx, q, cos_b, sin_b);
        k = apply_rope(ctx, k, cos_b, sin_b);

        ggml_tensor* qh = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor* kh = ggml_permute(ctx, k, 0, 2, 1, 3);
        ggml_tensor* kq = ggml_mul_mat(ctx, kh, qh);
        // attn_mask (block-diagonal, -inf across images) gates multi-image
        // batching so images don't cross-attend; null for a single image.
        kq = ggml_soft_max_ext(ctx, kq, attn_mask, scale, 0.0f);
        ggml_tensor* vh = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor* o = ggml_mul_mat(ctx, vh, kq);          // [head_dim, n_patch, heads]
        o = ggml_cont_2d(ctx, ggml_permute(ctx, o, 0, 2, 1, 3), hidden, n_patch);
        o = clip_linear(ctx, L.o, o);
        o = rms(ctx, o, L.post_attn_ln, eps);
        cur = ggml_add(ctx, cur, o);

        // ---- Gated MLP: gelu_tanh(gate) * up -> down ----
        ggml_tensor* m = rms(ctx, cur, L.pre_ff_ln, eps);
        ggml_tensor* gate = clip_linear(ctx, L.gate, m);
        ggml_tensor* up   = clip_linear(ctx, L.up, m);
        m = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
        m = clip_linear(ctx, L.down, m);
        m = rms(ctx, m, L.post_ff_ln, eps);
        cur = ggml_add(ctx, cur, m);
    }

    // 2D avg-pool (group mean * sqrt(hidden)/pool^2, folded into pool_matrix):
    //   pool[hidden, n_token] = h[hidden, n_patch] @ pool_matrix[n_patch, n_token].
    ggml_tensor* hT = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [n_patch, hidden]
    ggml_tensor* pooled = ggml_mul_mat(ctx, hT, pool_matrix);    // [hidden, n_token]
    pooled = rms(ctx, pooled, nullptr, eps);                     // final RMSNorm (no weight)
    ggml_tensor* out = ggml_mul_mat(ctx, w.embed_proj, pooled);  // [text_hidden, n_token]
    (void)n_token;
    return out;
}

// ── Host helpers ─────────────────────────────────────────────────────────────

std::vector<float> gemma4_vision_pos_embed(const float* table, std::int32_t P,
                                           std::int32_t hidden,
                                           const std::int32_t* pos,
                                           std::int32_t n_patch) {
    // table is [2, P, hidden] row-major: tb[t*P*hidden + p*hidden + d].
    std::vector<float> out(static_cast<std::size_t>(n_patch) * hidden);
    for (std::int32_t n = 0; n < n_patch; ++n) {
        std::int32_t x = pos[2 * n], y = pos[2 * n + 1];
        if (x < 0) x = 0; if (x >= P) x = P - 1;
        if (y < 0) y = 0; if (y >= P) y = P - 1;
        const float* tx = table + (static_cast<std::size_t>(x) * hidden);
        const float* ty = table + (static_cast<std::size_t>(P) + y) * hidden;
        float* o = out.data() + static_cast<std::size_t>(n) * hidden;
        for (std::int32_t d = 0; d < hidden; ++d) o[d] = tx[d] + ty[d];
    }
    return out;
}

void gemma4_vision_rope_cos_sin(const std::int32_t* pos, std::int32_t n_patch,
                                std::int32_t head_dim, float theta,
                                std::vector<float>& cos_out,
                                std::vector<float>& sin_out) {
    // Blocked: dims [0,16)&[16,32) use px*invf[c%16]; [32,48)&[48,64) use py.
    const std::int32_t q = head_dim / 4;  // 16
    cos_out.assign(static_cast<std::size_t>(n_patch) * head_dim, 0.0f);
    sin_out.assign(static_cast<std::size_t>(n_patch) * head_dim, 0.0f);
    std::vector<float> invf(static_cast<std::size_t>(q));
    for (std::int32_t c = 0; c < q; ++c)
        invf[c] = std::pow(theta, -static_cast<float>(c) / static_cast<float>(q));
    for (std::int32_t n = 0; n < n_patch; ++n) {
        const float px = static_cast<float>(pos[2 * n]);
        const float py = static_cast<float>(pos[2 * n + 1]);
        float* cp = cos_out.data() + static_cast<std::size_t>(n) * head_dim;
        float* sp = sin_out.data() + static_cast<std::size_t>(n) * head_dim;
        for (std::int32_t j = 0; j < head_dim; ++j) {
            const std::int32_t c = j % q;
            const float coord = (j < 2 * q) ? px : py;
            const float ang = coord * invf[c];
            cp[j] = std::cos(ang);
            sp[j] = std::sin(ang);
        }
    }
}

std::vector<float> gemma4_vision_pool_matrix(const std::int32_t* pos,
                                             std::int32_t n_patch,
                                             std::int32_t grid_w,
                                             std::int32_t grid_h, std::int32_t pool_k,
                                             std::int32_t hidden,
                                             std::int32_t& n_token_out) {
    // grp(p) = (x/pool_k) + gx*(y/pool_k); gx = grid_w/pool_k (integer div,
    // matching the CUDA reference — the Gemma resize makes the grid a multiple
    // of pool_k so there is no remainder).
    const std::int32_t gx = grid_w / pool_k;
    const std::int32_t gy = grid_h / pool_k;
    const std::int32_t n_token = gx * gy;
    n_token_out = n_token;
    // CUDA: sum(h)/pool^2 then * sqrt(hidden). Fold into the matrix weight.
    const float wgt = std::sqrt(static_cast<float>(hidden)) /
                      static_cast<float>(pool_k * pool_k);
    // matrix[token*n_patch + patch].
    std::vector<float> m(static_cast<std::size_t>(n_token) * n_patch, 0.0f);
    for (std::int32_t p = 0; p < n_patch; ++p) {
        const std::int32_t x = pos[2 * p], y = pos[2 * p + 1];
        const std::int32_t tok = (x / pool_k) + gx * (y / pool_k);
        if (tok >= 0 && tok < n_token)
            m[static_cast<std::size_t>(tok) * n_patch + p] = wgt;
    }
    return m;
}

}  // namespace pie_portable_driver
