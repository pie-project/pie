#include "graph_vision_qwen3vl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace pie_portable_driver {

namespace {

// y = W x (+ b). W stored ggml-style [in, out]; x is [in, tokens]; out [out, tokens].
ggml_tensor* linear(ggml_context* ctx, const VisLinear& l, ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(ctx, l.w, x);
    if (l.b) y = ggml_add(ctx, y, l.b);
    return y;
}

// LayerNorm over ne0 (the feature dim): (x - mean)/sqrt(var+eps) * g + b.
// ggml_norm outputs f32; weights may load as bf16 (blocks) or f32 (merger),
// so upcast g/b to f32 to keep the elementwise mul/add a clean f32 op.
ggml_tensor* layer_norm(ggml_context* ctx, const VisLayerNorm& n, ggml_tensor* x,
                        float eps) {
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    ggml_tensor* g = n.g->type == GGML_TYPE_F32 ? n.g
                   : ggml_cast(ctx, n.g, GGML_TYPE_F32);
    y = ggml_mul(ctx, y, g);
    if (n.b) {
        ggml_tensor* b = n.b->type == GGML_TYPE_F32 ? n.b
                       : ggml_cast(ctx, n.b, GGML_TYPE_F32);
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// rotate_half over ne0: cat(-x[d/2:], x[:d/2]). x is [head_dim, heads, n_patch].
ggml_tensor* rotate_half(ggml_context* ctx, ggml_tensor* x) {
    const std::int64_t d = x->ne[0];
    const std::int64_t h = d / 2;
    ggml_tensor* x1 = ggml_view_3d(ctx, x, h, x->ne[1], x->ne[2],
                                   x->nb[1], x->nb[2], 0);
    ggml_tensor* x2 = ggml_view_3d(ctx, x, h, x->ne[1], x->ne[2],
                                   x->nb[1], x->nb[2], h * x->nb[0]);
    // cat(-x2, x1) along ne0.
    return ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, x2)),
                       ggml_cont(ctx, x1), 0);
}

// q_rot = q*cos + rotate_half(q)*sin. cos/sin are [head_dim, 1, n_patch] so they
// broadcast over the heads axis (ne1).
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* cos_b, ggml_tensor* sin_b) {
    ggml_tensor* a = ggml_mul(ctx, x, cos_b);
    ggml_tensor* b = ggml_mul(ctx, rotate_half(ctx, x), sin_b);
    return ggml_add(ctx, a, b);
}

}  // namespace

VisionEncodeResult build_qwen3vl_vision_graph(ggml_context* ctx,
                                              const Qwen3VLVisionWeights& w,
                                              const Hparams& h,
                                              ggml_tensor* pixels,
                                              ggml_tensor* pos_embed_in,
                                              ggml_tensor* rope_cos,
                                              ggml_tensor* rope_sin,
                                              ggml_tensor* attn_mask,
                                              std::int32_t n_patch) {
    const std::int32_t hidden  = h.vision_hidden_size;
    const std::int32_t heads   = h.vision_num_heads;
    const std::int32_t head_dim = h.vision_head_dim > 0
        ? h.vision_head_dim : hidden / heads;
    const std::int32_t merge   = h.vision_spatial_merge_size;
    const std::int32_t merge_u = merge * merge;        // patches per token (4)
    const std::int32_t out_hid = h.vision_out_hidden;
    const float        eps     = h.vision_ln_eps;
    const float        scale   = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (n_patch % merge_u != 0) {
        throw std::runtime_error(
            "qwen3vl vision: n_patch not a multiple of spatial_merge^2");
    }
    const std::int32_t n_token = n_patch / merge_u;

    // cos/sin broadcastable over heads: [head_dim, 1, n_patch].
    ggml_tensor* cos_b = ggml_reshape_3d(ctx, rope_cos, head_dim, 1, n_patch);
    ggml_tensor* sin_b = ggml_reshape_3d(ctx, rope_sin, head_dim, 1, n_patch);

    // Patch embed (Conv3d-as-matmul) + learned abs pos-embed (host-interpolated).
    ggml_tensor* cur = linear(ctx, w.patch, pixels);      // [hidden, n_patch]
    cur = ggml_add(ctx, cur, pos_embed_in);

    std::vector<ggml_tensor*> deepstack_src(w.deepstack.size(), nullptr);

    for (std::size_t li = 0; li < w.blocks.size(); ++li) {
        const VisBlock& blk = w.blocks[li];

        // ---- Attention (pre-norm, full bidirectional) ----
        ggml_tensor* x = layer_norm(ctx, blk.norm1, cur, eps);
        ggml_tensor* qkv = linear(ctx, blk.qkv, x);       // [3*hidden, n_patch]
        ggml_tensor* q = ggml_view_2d(ctx, qkv, hidden, n_patch, qkv->nb[1],
                                      0);
        ggml_tensor* k = ggml_view_2d(ctx, qkv, hidden, n_patch, qkv->nb[1],
                                      (std::size_t)hidden * qkv->nb[0]);
        ggml_tensor* v = ggml_view_2d(ctx, qkv, hidden, n_patch, qkv->nb[1],
                                      (std::size_t)2 * hidden * qkv->nb[0]);
        q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, heads, n_patch);
        k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, heads, n_patch);
        v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, heads, n_patch);

        q = apply_rope(ctx, q, cos_b, sin_b);
        k = apply_rope(ctx, k, cos_b, sin_b);

        // [head_dim, heads, n_patch] -> [head_dim, n_patch, heads]
        ggml_tensor* qh = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor* kh = ggml_permute(ctx, k, 0, 2, 1, 3);
        ggml_tensor* kq = ggml_mul_mat(ctx, kh, qh);      // [n_patch(k), n_patch(q), heads]
        // attn_mask (block-diagonal -inf across images) for multi-image batches;
        // null for a single image (full bidirectional attention).
        kq = ggml_soft_max_ext(ctx, kq, attn_mask, scale, 0.0f);
        // v -> [n_patch, head_dim, heads] so mul_mat(v, kq) -> [head_dim, n_patch(q), heads]
        ggml_tensor* vh = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor* o = ggml_mul_mat(ctx, vh, kq);       // [head_dim, n_patch, heads]
        o = ggml_permute(ctx, o, 0, 2, 1, 3);             // [head_dim, heads, n_patch]
        o = ggml_cont_2d(ctx, o, hidden, n_patch);
        o = linear(ctx, blk.o, o);
        cur = ggml_add(ctx, cur, o);

        // ---- MLP (pre-norm, plain gelu-tanh) ----
        ggml_tensor* m = layer_norm(ctx, blk.norm2, cur, eps);
        m = linear(ctx, blk.fc1, m);
        m = ggml_gelu(ctx, m);                            // gelu_pytorch_tanh
        m = linear(ctx, blk.fc2, m);
        cur = ggml_add(ctx, cur, m);

        // DeepStack source capture (post-block hidden at the configured layers).
        for (std::size_t d = 0; d < w.deepstack.size(); ++d) {
            if (h.vision_deepstack_layers[d] == static_cast<std::int32_t>(li)) {
                deepstack_src[d] = cur;
            }
        }
    }

    // Merger: 2x2 spatial group (4 consecutive patch rows -> one token).
    auto run_merger = [&](const VisMerger& mg, ggml_tensor* feat) -> ggml_tensor* {
        ggml_tensor* x = feat;                            // [hidden, n_patch]
        if (!mg.is_postshuffle) {
            x = layer_norm(ctx, mg.norm, x, eps);         // norm over hidden BEFORE shuffle
        }
        // [hidden, n_patch] -> [hidden*merge_u, n_token] (concat 4 patches' hidden).
        x = ggml_reshape_2d(ctx, ggml_cont(ctx, x), hidden * merge_u, n_token);
        if (mg.is_postshuffle) {
            x = layer_norm(ctx, mg.norm, x, eps);         // norm over 4*hidden AFTER shuffle
        }
        x = linear(ctx, mg.fc1, x);
        x = ggml_gelu_erf(ctx, x);                        // merger uses erf-gelu
        x = linear(ctx, mg.fc2, x);                       // [out_hidden, n_token]
        return x;
    };

    VisionEncodeResult res;
    res.embeddings = run_merger(w.merger, cur);
    res.deepstack.resize(w.deepstack.size());
    for (std::size_t d = 0; d < w.deepstack.size(); ++d) {
        ggml_tensor* src = deepstack_src[d] ? deepstack_src[d] : cur;
        res.deepstack[d] = run_merger(w.deepstack[d], src);
    }
    (void)out_hid;
    return res;
}

// ── Host-side side-input helpers ─────────────────────────────────────────────

std::vector<std::int32_t> qwen3vl_vision_positions(std::int32_t t, std::int32_t h,
                                                   std::int32_t w,
                                                   std::int32_t merge) {
    // Port of transformers.vision_utils.get_vision_position_ids: 2x2 merge-block
    // reorder so each merge block's `merge^2` patches are consecutive.
    std::vector<std::int32_t> out;
    out.reserve(static_cast<std::size_t>(t) * h * w * 2);
    const std::int32_t bh = h / merge, bw = w / merge;
    for (std::int32_t tt = 0; tt < t; ++tt) {
        for (std::int32_t i = 0; i < bh; ++i) {
            for (std::int32_t j = 0; j < bw; ++j) {
                for (std::int32_t mh = 0; mh < merge; ++mh) {
                    for (std::int32_t mw = 0; mw < merge; ++mw) {
                        out.push_back(i * merge + mh);  // row (hpos)
                        out.push_back(j * merge + mw);  // col (wpos)
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> qwen3vl_pos_embed_interp(const float* table, std::int32_t side,
                                            std::int32_t hidden,
                                            const std::int32_t* pos,
                                            std::int32_t n_patch, std::int32_t grid_h,
                                            std::int32_t grid_w) {
    // Port of get_vision_bilinear_indices_and_weights, evaluated per patch from
    // its (row,col): linspace(0, side-1, grid) maps the patch grid onto the
    // side x side learned table, then bilinear-sample the 4 corners.
    std::vector<float> out(static_cast<std::size_t>(n_patch) * hidden);
    const float hden = grid_h > 1 ? static_cast<float>(side - 1) / (grid_h - 1) : 0.0f;
    const float wden = grid_w > 1 ? static_cast<float>(side - 1) / (grid_w - 1) : 0.0f;
    for (std::int32_t p = 0; p < n_patch; ++p) {
        const std::int32_t r = pos[2 * p];
        const std::int32_t c = pos[2 * p + 1];
        const float hg = static_cast<float>(r) * hden;
        const float wg = static_cast<float>(c) * wden;
        const std::int32_t hf = static_cast<std::int32_t>(hg);
        const std::int32_t wf = static_cast<std::int32_t>(wg);
        const std::int32_t hc = std::min(hf + 1, side - 1);
        const std::int32_t wc = std::min(wf + 1, side - 1);
        const float hfr = hg - static_cast<float>(hf);
        const float wfr = wg - static_cast<float>(wf);
        const float w00 = (1.0f - hfr) * (1.0f - wfr);
        const float w01 = (1.0f - hfr) * wfr;
        const float w10 = hfr * (1.0f - wfr);
        const float w11 = hfr * wfr;
        const float* t00 = table + static_cast<std::size_t>(hf * side + wf) * hidden;
        const float* t01 = table + static_cast<std::size_t>(hf * side + wc) * hidden;
        const float* t10 = table + static_cast<std::size_t>(hc * side + wf) * hidden;
        const float* t11 = table + static_cast<std::size_t>(hc * side + wc) * hidden;
        float* o = out.data() + static_cast<std::size_t>(p) * hidden;
        for (std::int32_t d = 0; d < hidden; ++d) {
            o[d] = w00 * t00[d] + w01 * t01[d] + w10 * t10[d] + w11 * t11[d];
        }
    }
    return out;
}

void qwen3vl_rope_cos_sin(const std::int32_t* pos, std::int32_t n_patch,
                          std::int32_t head_dim, float theta,
                          std::vector<float>& cos_out, std::vector<float>& sin_out) {
    // VisionRotaryEmbedding(dim=head_dim/2): inv_freq[k]=theta^(-2k/(head_dim/2)).
    // rotary[p] = [row*invf(0..q-1), col*invf(0..q-1)] (len head_dim/2), then
    // emb = cat([rotary, rotary]) (len head_dim); cos/sin = cos/sin(emb).
    const std::int32_t half = head_dim / 2;     // 32
    const std::int32_t quarter = head_dim / 4;  // 16  (= #inv_freq entries)
    cos_out.assign(static_cast<std::size_t>(n_patch) * head_dim, 0.0f);
    sin_out.assign(static_cast<std::size_t>(n_patch) * head_dim, 0.0f);
    std::vector<float> invf(static_cast<std::size_t>(quarter));
    for (std::int32_t k = 0; k < quarter; ++k) {
        invf[static_cast<std::size_t>(k)] =
            std::pow(theta, -2.0f * static_cast<float>(k) / static_cast<float>(half));
    }
    for (std::int32_t p = 0; p < n_patch; ++p) {
        const float row = static_cast<float>(pos[2 * p]);
        const float col = static_cast<float>(pos[2 * p + 1]);
        float* cp = cos_out.data() + static_cast<std::size_t>(p) * head_dim;
        float* sp = sin_out.data() + static_cast<std::size_t>(p) * head_dim;
        for (std::int32_t j = 0; j < head_dim; ++j) {
            float angle;
            if (j < quarter)             angle = row * invf[static_cast<std::size_t>(j)];
            else if (j < half)           angle = col * invf[static_cast<std::size_t>(j - quarter)];
            else if (j < half + quarter) angle = row * invf[static_cast<std::size_t>(j - half)];
            else                         angle = col * invf[static_cast<std::size_t>(j - half - quarter)];
            cp[j] = std::cos(angle);
            sp[j] = std::sin(angle);
        }
    }
}

}  // namespace pie_portable_driver
