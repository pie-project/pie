#include "graph_csm_mimi.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>

#include "model.hpp"

namespace pie_portable_driver {

namespace {

constexpr int MIMI_GRAPH_NODES = 16384;

// Feature-major [C, T] convention throughout (ne0 = channels, ne1 = time), which
// makes the per-channel layer-scale and the per-tap matmuls natural. All decode
// convs are stride 1 / dilation 1 (causal left-pad k-1); upsampling is via the
// transposed convs (zero-stuff + the same causal conv).

ggml_tensor* shift_t(ggml_context* ctx, ggml_tensor* x, int s) {
    // shift along time (ne1) by s: position t takes x[.., t-s], zeros for t<s.
    if (s <= 0) return x;
    const std::int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor* crop = ggml_cont(ctx, ggml_view_2d(ctx, x, C, T - s, x->nb[1], 0));
    ggml_tensor* z = ggml_scale(ctx, ggml_cont(ctx, ggml_view_2d(ctx, x, C, s, x->nb[1], 0)), 0.0f);
    return ggml_concat(ctx, z, crop, 1);
}

ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

// Permute a conv weight to [Cin, Cout, k] for per-tap matmuls.
//   Conv1d weight       HF [Cout, Cin, k]  -> ggml [k, Cin, Cout] -> perm(2,0,1)
//   ConvTranspose1d wt  HF [Cin, Cout, k]  -> ggml [k, Cout, Cin] -> perm(2,1,0)
ggml_tensor* perm_conv_w(ggml_context* ctx, ggml_tensor* w, bool transposed) {
    ggml_tensor* p = transposed ? ggml_permute(ctx, w, 2, 1, 0, 3)
                                : ggml_permute(ctx, w, 2, 0, 1, 3);
    return ggml_cont(ctx, to_f32(ctx, p));
}

// Per-tap conv core: out = sum_j W_j @ shift_t(x, shift(j)) + bias, wperm
// [Cin, Cout, k]. Forward causal Conv1d (`transpose=false`) reads x[t-(k-1)+j]
// (shift (k-1)-j). Transposed conv (`transpose=true`, on a zero-stuffed input)
// reads x_zs[t-j] (shift j) — the kernel is NOT flipped relative to the forward
// case, so the two differ exactly by the tap order.
ggml_tensor* tap_conv(ggml_context* ctx, ggml_tensor* x, ggml_tensor* wperm,
                      ggml_tensor* bias, int k, bool transpose) {
    const std::int64_t Cin = wperm->ne[0], Cout = wperm->ne[1];
    ggml_tensor* out = nullptr;
    for (int j = 0; j < k; ++j) {
        ggml_tensor* wj = ggml_view_2d(ctx, wperm, Cin, Cout, wperm->nb[1],
                                       static_cast<std::size_t>(j) * wperm->nb[2]);
        ggml_tensor* xs = shift_t(ctx, x, transpose ? j : (k - 1) - j);
        ggml_tensor* term = ggml_mul_mat(ctx, ggml_cont(ctx, wj), xs);  // [Cout, T]
        out = out ? ggml_add(ctx, out, term) : term;
    }
    if (bias) {
        ggml_tensor* b = ggml_reshape_2d(ctx, to_f32(ctx, bias), Cout, 1);
        out = ggml_add(ctx, out, b);
    }
    return out;
}

// Causal Conv1d: x [Cin, T], wperm [Cin, Cout, k] -> [Cout, T]. Left-pad k-1.
ggml_tensor* causal_conv(ggml_context* ctx, ggml_tensor* x, ggml_tensor* wperm,
                         ggml_tensor* bias, int k) {
    return tap_conv(ctx, x, wperm, bias, k, /*transpose=*/false);
}

// Zero-stuff x [C, T] -> [C, T*stride] (x at positions t*stride, else 0).
ggml_tensor* zero_stuff(ggml_context* ctx, ggml_tensor* x, int stride) {
    const std::int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor* y = ggml_reshape_3d(ctx, x, C, 1, T);
    y = ggml_pad(ctx, y, 0, stride - 1, 0, 0);            // [C, stride, T]
    return ggml_reshape_2d(ctx, ggml_cont(ctx, y), C, T * stride);
}

// ConvTranspose1d (groups=1): x [Cin,T], wperm [Cin,Cout,k] -> [Cout, T*stride].
// = zero-stuff(x, stride) then a causal conv with the convtr weight; the left
// T*stride positions are exactly Mimi's right-trimmed output.
ggml_tensor* convtr_g1(ggml_context* ctx, ggml_tensor* x, ggml_tensor* wperm,
                       ggml_tensor* bias, int stride, int k) {
    return tap_conv(ctx, zero_stuff(ctx, x, stride), wperm, bias, k, /*transpose=*/true);
}

// Depthwise ConvTranspose1d: x [C,T], w ggml [k,1,C] -> [C, T*stride].
ggml_tensor* convtr_depthwise(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                              int stride, int k) {
    const std::int64_t C = x->ne[0];
    ggml_tensor* wt = ggml_cont(ctx, ggml_transpose(
        ctx, to_f32(ctx, ggml_reshape_2d(ctx, w, k, C))));   // [C, k]
    ggml_tensor* xz = zero_stuff(ctx, x, stride);            // [C, T*stride]
    ggml_tensor* out = nullptr;
    for (int j = 0; j < k; ++j) {
        ggml_tensor* wc = ggml_cont(ctx, ggml_view_2d(ctx, wt, C, 1, wt->nb[1],
                                                      static_cast<std::size_t>(j) * wt->nb[1]));
        ggml_tensor* term = ggml_mul(ctx, shift_t(ctx, xz, j), wc);  // transpose: shift j
        out = out ? ggml_add(ctx, out, term) : term;
    }
    return out;
}

ggml_tensor* elu(ggml_context* ctx, ggml_tensor* x) { return ggml_elu(ctx, x); }

ggml_tensor* layernorm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                       ggml_tensor* b, float eps) {
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, to_f32(ctx, w));
    return ggml_add(ctx, y, to_f32(ctx, b));
}

// rotate-half (NEOX) for [hd, heads, T].
ggml_tensor* rot_half(ggml_context* ctx, ggml_tensor* x) {
    const std::int64_t hd = x->ne[0], half = hd / 2;
    ggml_tensor* a = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    ggml_tensor* b = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], half * x->nb[0]);
    return ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, b)), ggml_cont(ctx, a), 0);
}

ggml_tensor* rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cos_t, ggml_tensor* sin_t) {
    ggml_tensor* c = ggml_reshape_3d(ctx, cos_t, x->ne[0], 1, x->ne[2]);
    ggml_tensor* s = ggml_reshape_3d(ctx, sin_t, x->ne[0], 1, x->ne[2]);
    return ggml_add(ctx, ggml_mul(ctx, x, c), ggml_mul(ctx, rot_half(ctx, x), s));
}

}  // namespace

void csm_mimi_decode(Model& model, const std::int32_t* codes, int n_frames,
                     std::vector<float>& out_pcm) {
    const auto& h = model.hparams();
    const auto& C = model.weights().csm;
    const auto& mc = h.csm->codec;
    if (n_frames <= 0) { out_pcm.clear(); return; }

    const int DIM = mc.codebook_dim, HID = mc.hidden_size, T = n_frames;
    const int NCB = mc.num_quantizers, NS = mc.num_semantic_quantizers;
    const float eps = mc.norm_eps;
    const int Hh = mc.xf_num_attention_heads, KVH = mc.xf_num_key_value_heads;
    const int hd = mc.xf_head_dim;
    const int window = mc.xf_sliding_window;
    const float xf_theta = mc.xf_rope_theta;

    ggml_backend_t backend = model.backend();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    const std::size_t mem = ggml_tensor_overhead() * MIMI_GRAPH_NODES +
                            ggml_graph_overhead_custom(MIMI_GRAPH_NODES, false);
    ggml_init_params ip{mem, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MIMI_GRAPH_NODES, false);

    // ── code inputs: one I32 [T] per codebook ──
    std::vector<ggml_tensor*> code_in(NCB);
    for (int c = 0; c < NCB; ++c) {
        code_in[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(code_in[c]);
    }
    // RoPE tables for the decoder transformer (full sequence, length Tu = 2T).
    const int Tu = T * 2;
    ggml_tensor* xf_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, Tu);
    ggml_tensor* xf_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, Tu);
    ggml_set_input(xf_cos); ggml_set_input(xf_sin);
    ggml_tensor* xf_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Tu, Tu);
    ggml_set_input(xf_mask);

    // ── 1) RVQ dequantize ──
    auto resolve_cb = [&](int i) {
        ggml_tensor* es = to_f32(ctx, C.codebook_embed_sum[i]);        // [DIM, size]
        ggml_tensor* cu = to_f32(ctx, C.codebook_cluster_usage[i]);    // [size]
        cu = ggml_clamp(ctx, cu, 1e-5f, std::numeric_limits<float>::infinity());
        ggml_tensor* cur = ggml_reshape_2d(ctx, cu, 1, cu->ne[0]);     // [1, size]
        return ggml_div(ctx, es, cur);                                 // [DIM, size]
    };
    auto group_sum = [&](int start, int count) {
        ggml_tensor* acc = nullptr;
        for (int i = start; i < start + count; ++i) {
            ggml_tensor* cb = resolve_cb(i);
            ggml_tensor* e = ggml_get_rows(ctx, cb, code_in[i]);  // [DIM, T]
            acc = acc ? ggml_add(ctx, acc, e) : e;
        }
        return acc;  // [DIM, T]
    };
    ggml_tensor* sem = group_sum(0, NS);
    ggml_tensor* aco = group_sum(NS, NCB - NS);
    // output_proj (1x1 conv = matmul). weight HF [HID, DIM, 1] -> ggml [1,DIM,HID].
    ggml_tensor* sem_w = ggml_cont(ctx, to_f32(ctx, ggml_reshape_2d(ctx, C.semantic_output_proj, DIM, HID)));
    ggml_tensor* aco_w = ggml_cont(ctx, to_f32(ctx, ggml_reshape_2d(ctx, C.acoustic_output_proj, DIM, HID)));
    ggml_tensor* emb = ggml_add(ctx, ggml_mul_mat(ctx, sem_w, sem),
                                ggml_mul_mat(ctx, aco_w, aco));         // [HID, T]

    // ── 2) upsample ConvTranspose1d k4 s2 groups=512 (depthwise) -> [HID, Tu] ──
    ggml_tensor* up = convtr_depthwise(ctx, emb, C.upsample.weight, 2, 4);   // [HID, Tu]

    // ── 3) decoder transformer (8 layers), feature-major [HID, Tu] ──
    const float ascale = 1.0f / std::sqrt(static_cast<float>(hd));
    ggml_tensor* hs = up;
    for (const MimiXfLayer& L : C.xf_layers) {
        ggml_tensor* ln = layernorm(ctx, hs, L.in_ln_w, L.in_ln_b, eps);
        ggml_tensor* q = ggml_mul_mat(ctx, to_f32(ctx, L.q), ln);   // [Hh*hd, Tu]
        ggml_tensor* k = ggml_mul_mat(ctx, to_f32(ctx, L.k), ln);
        ggml_tensor* v = ggml_mul_mat(ctx, to_f32(ctx, L.v), ln);
        q = rope(ctx, ggml_reshape_3d(ctx, q, hd, Hh, Tu), xf_cos, xf_sin);
        k = rope(ctx, ggml_reshape_3d(ctx, k, hd, KVH, Tu), xf_cos, xf_sin);
        v = ggml_reshape_3d(ctx, v, hd, KVH, Tu);
        ggml_tensor* qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [hd,Tu,Hh]
        ggml_tensor* kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd,Tu,KVH]
        ggml_tensor* kq = ggml_mul_mat(ctx, kh, qh);                          // [Tu,Tu,Hh]
        kq = ggml_soft_max_ext(ctx, kq, xf_mask, ascale, 0.0f);
        ggml_tensor* vh = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [Tu,hd,KVH]
        ggml_tensor* o = ggml_mul_mat(ctx, vh, kq);                          // [hd,Tu,Hh]
        o = ggml_cont_2d(ctx, ggml_permute(ctx, o, 0, 2, 1, 3), Hh * hd, Tu);
        o = ggml_mul_mat(ctx, to_f32(ctx, L.o), o);                          // [HID, Tu]
        // per-channel layer-scale (scale indexes ne0=HID) + residual.
        ggml_tensor* asc = ggml_reshape_2d(ctx, to_f32(ctx, L.attn_scale), HID, 1);
        hs = ggml_add(ctx, hs, ggml_mul(ctx, o, asc));
        // MLP (pre-norm) + layer-scale.
        ggml_tensor* mln = layernorm(ctx, hs, L.post_ln_w, L.post_ln_b, eps);
        ggml_tensor* mid = ggml_gelu(ctx, ggml_mul_mat(ctx, to_f32(ctx, L.fc1), mln));
        ggml_tensor* mlp = ggml_mul_mat(ctx, to_f32(ctx, L.fc2), mid);
        ggml_tensor* msc = ggml_reshape_2d(ctx, to_f32(ctx, L.mlp_scale), HID, 1);
        hs = ggml_add(ctx, hs, ggml_mul(ctx, mlp, msc));
    }

    // ── 4) SEANet decoder ──
    auto conv = [&](ggml_tensor* x, const CsmConv& c, int k) {
        return causal_conv(ctx, x, perm_conv_w(ctx, c.weight, false), c.bias, k);
    };
    ggml_tensor* cur = conv(hs, C.seanet_in, mc.kernel_size);   // Conv1d k7 -> [1024, Tu]
    for (std::size_t si = 0; si < C.seanet_stages.size(); ++si) {
        const MimiSeanetStage& st = C.seanet_stages[si];
        const int ratio = mc.upsampling_ratios[si];
        cur = elu(ctx, cur);
        cur = convtr_g1(ctx, cur, perm_conv_w(ctx, st.convtr.weight, true),
                        st.convtr.bias, ratio, 2 * ratio);       // upsample x ratio
        // resnet: out = x + conv2(elu(conv1(elu(x))))
        ggml_tensor* r = cur;
        ggml_tensor* y = elu(ctx, cur);
        y = conv(y, st.resnet_conv1, mc.residual_kernel_size);   // k3
        y = elu(ctx, y);
        y = conv(y, st.resnet_conv2, 1);                         // k1
        cur = ggml_add(ctx, r, y);
    }
    cur = elu(ctx, cur);
    ggml_tensor* wave = conv(cur, C.seanet_out, mc.last_kernel_size);  // Conv1d k3 -> [1, n_samples]
    ggml_set_output(wave);
    ggml_build_forward_expand(gf, wave);

    // ── allocate, upload inputs, compute, read back ──
    ggml_gallocr_alloc_graph(galloc, gf);
    for (int c = 0; c < NCB; ++c) {
        ggml_backend_tensor_set(code_in[c], codes + static_cast<std::size_t>(c) * T,
                                0, T * sizeof(std::int32_t));
    }
    {   // RoPE cos/sin for the transformer (theta xf_theta, rotate-half).
        std::vector<float> cosv(static_cast<std::size_t>(hd) * Tu), sinv(static_cast<std::size_t>(hd) * Tu);
        const int half = hd / 2;
        for (int t = 0; t < Tu; ++t)
            for (int d = 0; d < hd; ++d) {
                const float inv = std::pow(xf_theta, -2.0f * (d % half) / hd);
                const float ang = t * inv;
                cosv[static_cast<std::size_t>(t) * hd + d] = std::cos(ang);
                sinv[static_cast<std::size_t>(t) * hd + d] = std::sin(ang);
            }
        ggml_backend_tensor_set(xf_cos, cosv.data(), 0, cosv.size() * sizeof(float));
        ggml_backend_tensor_set(xf_sin, sinv.data(), 0, sinv.size() * sizeof(float));
    }
    {   // causal + sliding-window mask [Tu(keys), Tu(queries)].
        std::vector<float> m(static_cast<std::size_t>(Tu) * Tu, 0.0f);
        const float ninf = -std::numeric_limits<float>::infinity();
        for (int qi = 0; qi < Tu; ++qi)
            for (int j = 0; j < Tu; ++j)
                if (j > qi || j < qi - window + 1)
                    m[static_cast<std::size_t>(qi) * Tu + j] = ninf;
        ggml_backend_tensor_set(xf_mask, m.data(), 0, m.size() * sizeof(float));
    }
    ggml_backend_graph_compute(backend, gf);

    const int n_samples = static_cast<int>(wave->ne[1]);
    out_pcm.resize(n_samples);
    ggml_backend_tensor_get(wave, out_pcm.data(), 0,
                            static_cast<std::size_t>(n_samples) * sizeof(float));

    ggml_free(ctx);
    ggml_gallocr_free(galloc);
}

}  // namespace pie_portable_driver
