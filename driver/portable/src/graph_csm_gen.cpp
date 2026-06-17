#include "graph_csm_gen.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>

#include "model.hpp"
#include "graph_csm_mimi.hpp"

namespace pie_portable_driver {

namespace {

// The CSM step graphs (prefill / backbone-decode ~420 nodes, depth ~100,
// lm_head ~10) are far smaller than the old 8192 cap, which mallocs a ~2.75MB
// metadata pool per step. 2048 keeps ~4x headroom over the largest while cutting
// the per-step allocation 4x (zero math change). Sync latency still dominates,
// but this trims real CPU overhead on long clips.
constexpr int GEN_GRAPH_NODES = 2048;

// llama3 YaRN scaled inverse-frequency (mirrors CUDA yarn_freq exactly).
float yarn_freq(float base_freq, float factor, float low_freq_factor,
                float high_freq_factor, float orig_max_pos) {
    const float TWO_PI = 6.283185307179586f;
    const float wavelen = TWO_PI / base_freq;
    const float low_wave = orig_max_pos / low_freq_factor;
    const float high_wave = orig_max_pos / high_freq_factor;
    if (wavelen < high_wave) return base_freq;
    if (wavelen > low_wave) return base_freq / factor;
    const float smooth = (orig_max_pos / wavelen - low_freq_factor) /
                         (high_freq_factor - low_freq_factor);
    return (1.0f - smooth) * (base_freq / factor) + smooth * base_freq;
}

// cos/sin tables [hd, R] (row-major hd-fastest) for positions [pos0, pos0+R),
// NEOX rotate-half convention: cos[d] == cos[d+half] == cos(pos * freq_{d}).
struct RopeTab { std::vector<float> cos, sin; };
RopeTab rope_tab(int pos0, int R, int hd, float theta, float factor,
                 float lo, float hi, float orig) {
    const int half = hd / 2;
    RopeTab t;
    t.cos.resize(static_cast<std::size_t>(R) * hd);
    t.sin.resize(static_cast<std::size_t>(R) * hd);
    std::vector<float> freq(half);
    for (int d = 0; d < half; ++d) {
        const float invf = std::pow(theta, -2.0f * d / hd);
        freq[d] = yarn_freq(invf, factor, lo, hi, orig);
    }
    for (int r = 0; r < R; ++r) {
        const float pos = static_cast<float>(pos0 + r);
        for (int d = 0; d < hd; ++d) {
            const float ang = pos * freq[d % half];
            t.cos[static_cast<std::size_t>(r) * hd + d] = std::cos(ang);
            t.sin[static_cast<std::size_t>(r) * hd + d] = std::sin(ang);
        }
    }
    return t;
}

// NEOX rotate-half: split ne0 into [A|B] halves -> [-B, A].
ggml_tensor* rotate_half(ggml_context* ctx, ggml_tensor* x) {
    const std::int64_t hd = x->ne[0], half = hd / 2;
    ggml_tensor* a = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2],
                                  x->nb[1], x->nb[2], 0);
    ggml_tensor* b = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2],
                                  x->nb[1], x->nb[2], half * x->nb[0]);
    return ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, b)),
                       ggml_cont(ctx, a), 0);
}

ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cos_t,
                        ggml_tensor* sin_t) {
    // x [hd, nheads, R]; cos/sin [hd, R] -> broadcast over heads as [hd,1,R].
    ggml_tensor* c = ggml_reshape_3d(ctx, cos_t, x->ne[0], 1, x->ne[2]);
    ggml_tensor* s = ggml_reshape_3d(ctx, sin_t, x->ne[0], 1, x->ne[2]);
    return ggml_add(ctx, ggml_mul(ctx, x, c),
                    ggml_mul(ctx, rotate_half(ctx, x), s));
}

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                      float eps) {
    ggml_tensor* y = ggml_rms_norm(ctx, x, eps);
    ggml_tensor* wf = w->type == GGML_TYPE_F32 ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_mul(ctx, y, wf);
}

// One Llama decoder block over R rows. KV caches kc/vc are persistent 2D
// [KD, maxL] (KD = KV*hd); the R new rows are written at row positions `pos_idx`
// (I64 [R]) via ggml_set_rows, which returns a post-write handle so the
// attention read is ordered after the write. Returns the post-block hidden.
ggml_tensor* llama_block(ggml_context* ctx, const CsmDecLayer& L,
                         ggml_tensor* inpL, ggml_tensor* kc, ggml_tensor* vc,
                         ggml_tensor* cos_t, ggml_tensor* sin_t,
                         ggml_tensor* mask, ggml_tensor* pos_idx,
                         int H, int NH, int KV, int hd,
                         int R, int Lkv, float eps) {
    const int KD = KV * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    ggml_tensor* cur = rms_norm(ctx, inpL, L.in_ln, eps);
    ggml_tensor* q = ggml_mul_mat(ctx, L.q, cur);   // [NH*hd, R]
    ggml_tensor* k = ggml_mul_mat(ctx, L.k, cur);   // [KV*hd, R]
    ggml_tensor* v = ggml_mul_mat(ctx, L.v, cur);
    q = ggml_reshape_3d(ctx, q, hd, NH, R);
    k = ggml_reshape_3d(ctx, k, hd, KV, R);
    q = apply_rope(ctx, q, cos_t, sin_t);
    k = apply_rope(ctx, k, cos_t, sin_t);
    // Write k,v into the cache at the R new positions; read the post-write
    // handle to enforce ordering.
    ggml_tensor* k_kd = ggml_reshape_2d(ctx, ggml_cont(ctx, k), KD, R);
    ggml_tensor* kc2 = ggml_set_rows(ctx, kc, k_kd, pos_idx);   // [KD, maxL]
    ggml_tensor* vc2 = ggml_set_rows(ctx, vc, v, pos_idx);      // [KD, maxL]
    ggml_tensor* kall = ggml_reshape_3d(
        ctx, ggml_view_2d(ctx, kc2, KD, Lkv, kc2->nb[1], 0), hd, KV, Lkv);
    ggml_tensor* vall = ggml_reshape_3d(
        ctx, ggml_view_2d(ctx, vc2, KD, Lkv, vc2->nb[1], 0), hd, KV, Lkv);
    ggml_tensor* qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // [hd,R,NH]
    ggml_tensor* kh = ggml_cont(ctx, ggml_permute(ctx, kall, 0, 2, 1, 3));// [hd,Lkv,KV]
    ggml_tensor* kq = ggml_mul_mat(ctx, kh, qh);                          // [Lkv,R,NH]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    ggml_tensor* vh = ggml_cont(ctx, ggml_permute(ctx, vall, 1, 2, 0, 3));// [Lkv,hd,KV]
    ggml_tensor* o = ggml_mul_mat(ctx, vh, kq);                          // [hd,R,NH]
    o = ggml_cont_2d(ctx, ggml_permute(ctx, o, 0, 2, 1, 3), H, R);
    o = ggml_mul_mat(ctx, L.o, o);
    ggml_tensor* h1 = ggml_add(ctx, inpL, o);
    ggml_tensor* m = rms_norm(ctx, h1, L.post_ln, eps);
    ggml_tensor* gate = ggml_mul_mat(ctx, L.gate, m);
    ggml_tensor* up   = ggml_mul_mat(ctx, L.up, m);
    m = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    m = ggml_mul_mat(ctx, L.down, m);
    return ggml_add(ctx, h1, m);
}

// Threaded-KV Llama block for the FUSED depth decoder: reads/writes the cache
// handles passed in and returns the POST-write handles so the next fused step's
// read is ordered after this step's write (ggml topology only sees the handle
// chain, not aliasing of the underlying buffer). Always R=1 (one token/step).
//
// Reads EXACTLY the `Lkv` valid cache rows (positions 0..Lkv-1) via a
// constant-length view — NOT the full fixed cap with a mask. `Lkv` is a
// compile-time constant per unrolled step (= p+1), so the attention reduction
// width matches the per-step `llama_block` bit-for-bit; a fixed-cap read would
// reduce over zero-padded lanes and round differently, flipping near-tie
// argmaxes downstream. No causal mask needed: the single query at position
// Lkv-1 attends to all Lkv keys.
struct DepthBlockOut { ggml_tensor* hidden; ggml_tensor* kc; ggml_tensor* vc; };
DepthBlockOut depth_block(ggml_context* ctx, const CsmDecLayer& L,
                          ggml_tensor* inpL, ggml_tensor* kc, ggml_tensor* vc,
                          ggml_tensor* cos_t, ggml_tensor* sin_t,
                          ggml_tensor* pos_idx,
                          int H, int NH, int KV, int hd, int Lkv, float eps) {
    const int KD = KV * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    ggml_tensor* cur = rms_norm(ctx, inpL, L.in_ln, eps);
    ggml_tensor* q = ggml_mul_mat(ctx, L.q, cur);
    ggml_tensor* k = ggml_mul_mat(ctx, L.k, cur);
    ggml_tensor* v = ggml_mul_mat(ctx, L.v, cur);
    q = ggml_reshape_3d(ctx, q, hd, NH, 1);
    k = ggml_reshape_3d(ctx, k, hd, KV, 1);
    q = apply_rope(ctx, q, cos_t, sin_t);
    k = apply_rope(ctx, k, cos_t, sin_t);
    ggml_tensor* k_kd = ggml_reshape_2d(ctx, ggml_cont(ctx, k), KD, 1);
    ggml_tensor* kc2 = ggml_set_rows(ctx, kc, k_kd, pos_idx);   // post-write handle
    ggml_tensor* vc2 = ggml_set_rows(ctx, vc, v, pos_idx);
    ggml_tensor* kall = ggml_reshape_3d(                          // exact valid rows
        ctx, ggml_view_2d(ctx, kc2, KD, Lkv, kc2->nb[1], 0), hd, KV, Lkv);
    ggml_tensor* vall = ggml_reshape_3d(
        ctx, ggml_view_2d(ctx, vc2, KD, Lkv, vc2->nb[1], 0), hd, KV, Lkv);
    ggml_tensor* qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // [hd,1,NH]
    ggml_tensor* kh = ggml_cont(ctx, ggml_permute(ctx, kall, 0, 2, 1, 3));// [hd,Lkv,KV]
    ggml_tensor* kq = ggml_mul_mat(ctx, kh, qh);                          // [Lkv,1,NH]
    kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
    ggml_tensor* vh = ggml_cont(ctx, ggml_permute(ctx, vall, 1, 2, 0, 3));// [Lkv,hd,KV]
    ggml_tensor* o = ggml_mul_mat(ctx, vh, kq);                          // [hd,1,NH]
    o = ggml_cont_2d(ctx, ggml_permute(ctx, o, 0, 2, 1, 3), H, 1);
    o = ggml_mul_mat(ctx, L.o, o);
    ggml_tensor* h1 = ggml_add(ctx, inpL, o);
    ggml_tensor* m = rms_norm(ctx, h1, L.post_ln, eps);
    ggml_tensor* gate = ggml_mul_mat(ctx, L.gate, m);
    ggml_tensor* up   = ggml_mul_mat(ctx, L.up, m);
    m = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    m = ggml_mul_mat(ctx, L.down, m);
    return {ggml_add(ctx, h1, m), kc2, vc2};
}

// A compute session: a persistent KV-cache context (allocated on the backend)
// plus a reusable gallocr for per-step compute graphs.
struct Session {
    ggml_backend_t backend;
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<ggml_tensor*> bb_k, bb_v;   // backbone KV cache per layer
    std::vector<ggml_tensor*> dd_k, dd_v;   // depth KV cache per layer

    ~Session() {
        if (galloc) ggml_gallocr_free(galloc);
        if (kv_buf) ggml_backend_buffer_free(kv_buf);
        if (kv_ctx) ggml_free(kv_ctx);
    }
};

// Build a no-alloc compute context sized for `nodes` graph nodes.
ggml_context* new_compute_ctx(int nodes = GEN_GRAPH_NODES) {
    const std::size_t mem = ggml_tensor_overhead() * nodes +
                            ggml_graph_overhead_custom(nodes, false);
    ggml_init_params ip{mem, nullptr, true};
    return ggml_init(ip);
}
// The fused depth decoder (all 31 codebook steps in one graph) is much larger.
constexpr int DEPTH_FUSED_NODES = 16384;

}  // namespace

int csm_generate_audio(Model& model,
                       const std::int32_t* prompt, int n_prompt,
                       int max_frames,
                       std::vector<float>& out_pcm,
                       std::vector<std::int32_t>* out_codes) {
    const auto& h = model.hparams();
    const auto& C = model.weights().csm;
    if (!C.present || !h.csm.has_value()) {
        throw std::runtime_error("csm_generate_audio: model is not CSM");
    }
    if (n_prompt <= 0) throw std::runtime_error("csm_generate_audio: empty prompt");
    if (max_frames <= 0) max_frames = 256;
    const auto& cc = *h.csm;

    // Backbone dims (top-level Llama-3.2-1B).
    const int H = h.hidden_size, NH = h.num_attention_heads;
    const int KV = h.num_key_value_heads;
    const int hd = h.head_dim > 0 ? h.head_dim : H / NH;
    const int NCB = cc.num_codebooks, AV = cc.audio_vocab_size;
    const float eps = h.rms_norm_eps;
    const float bb_theta = h.rope_theta;
    const float bb_factor = h.rope_scaling_factor;
    const float bb_lo = h.rope_scaling_low_freq_factor;
    const float bb_hi = h.rope_scaling_high_freq_factor;
    const float bb_orig = static_cast<float>(h.rope_scaling_original_max_position);
    const int maxL = n_prompt + max_frames + 1;

    // Depth dims.
    const auto& dcfg = cc.depth;
    const int DH = dcfg.hidden_size, DBH = dcfg.backbone_hidden_size;
    const int DNH = dcfg.num_attention_heads, DKV = dcfg.num_key_value_heads;
    const int Dhd = dcfg.head_dim, DV = dcfg.vocab_size, DL = dcfg.num_hidden_layers;
    const float deps = dcfg.rms_norm_eps;
    const float dd_theta = dcfg.rope_theta, dd_factor = dcfg.rope_factor;
    const float dd_lo = dcfg.rope_low_freq_factor, dd_hi = dcfg.rope_high_freq_factor;
    const float dd_orig = static_cast<float>(dcfg.rope_original_max_position);

    Session s;
    s.backend = model.backend();
    s.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));

    // Persistent KV caches: backbone [hd,KV,maxL] x num_layers; depth
    // [Dhd,DKV,NCB] x DL (re-used each frame).
    const int BBL = h.num_hidden_layers;
    {
        const std::size_t n_tensors = 2 * (BBL + DL) + 8;
        ggml_init_params ip{ggml_tensor_overhead() * n_tensors, nullptr, true};
        s.kv_ctx = ggml_init(ip);
    }
    s.bb_k.resize(BBL); s.bb_v.resize(BBL);
    for (int l = 0; l < BBL; ++l) {
        s.bb_k[l] = ggml_new_tensor_2d(s.kv_ctx, GGML_TYPE_F32, KV * hd, maxL);
        s.bb_v[l] = ggml_new_tensor_2d(s.kv_ctx, GGML_TYPE_F32, KV * hd, maxL);
    }
    s.dd_k.resize(DL); s.dd_v.resize(DL);
    for (int l = 0; l < DL; ++l) {
        s.dd_k[l] = ggml_new_tensor_2d(s.kv_ctx, GGML_TYPE_F32, DKV * Dhd, NCB);
        s.dd_v[l] = ggml_new_tensor_2d(s.kv_ctx, GGML_TYPE_F32, DKV * Dhd, NCB);
    }
    s.kv_buf = ggml_backend_alloc_ctx_tensors(s.kv_ctx, s.backend);

    // last_hidden persists on host between backbone step and depth seed.
    std::vector<float> last_hidden(H);

    auto argmax = [](const std::vector<float>& v) {
        int bi = 0; float bv = v[0];
        for (int i = 1; i < static_cast<int>(v.size()); ++i)
            if (v[i] > bv) { bv = v[i]; bi = i; }
        return bi;
    };

    // ── Backbone prefill over the prompt (rows 0..n_prompt-1) ──
    auto backbone_forward = [&](const std::int32_t* tok_ids, int R, int q0,
                                int Lkv, bool want_logits,
                                std::vector<float>& logits_out) {
        ggml_context* ctx = new_compute_ctx();
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, GEN_GRAPH_NODES, false);
        ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R);
        ggml_set_input(ids);
        ggml_tensor* inpL = ggml_get_rows(ctx, C.embed_text, ids);  // [H, R]
        ggml_tensor* cos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, R);
        ggml_tensor* sin_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, R);
        ggml_set_input(cos_t); ggml_set_input(sin_t);
        ggml_tensor* mask = nullptr;
        if (R > 1) {
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lkv, R);
            ggml_set_input(mask);
        }
        ggml_tensor* pos_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, R);
        ggml_set_input(pos_idx);
        for (int l = 0; l < BBL; ++l) {
            inpL = llama_block(ctx, C.backbone_layers[l], inpL, s.bb_k[l],
                               s.bb_v[l], cos_t, sin_t, mask, pos_idx, H, NH, KV,
                               hd, R, Lkv, eps);
        }
        // final norm on the LAST row only.
        ggml_tensor* lastrow = ggml_view_2d(ctx, inpL, H, 1, inpL->nb[1],
                                            static_cast<std::size_t>(R - 1) * inpL->nb[1]);
        ggml_tensor* hn = rms_norm(ctx, ggml_cont(ctx, lastrow), C.bb_norm, eps);  // [H,1]
        ggml_set_output(hn);
        ggml_tensor* logits = nullptr;
        if (want_logits) {
            logits = ggml_mul_mat(ctx, C.lm_head, hn);  // [AV,1]
            ggml_set_output(logits);
            ggml_build_forward_expand(gf, logits);
        }
        ggml_build_forward_expand(gf, hn);
        ggml_gallocr_alloc_graph(s.galloc, gf);
        ggml_backend_tensor_set(ids, tok_ids, 0, R * sizeof(std::int32_t));
        {
            std::vector<std::int64_t> ph(R);
            for (int r = 0; r < R; ++r) ph[r] = q0 + r;
            ggml_backend_tensor_set(pos_idx, ph.data(), 0, R * sizeof(std::int64_t));
        }
        RopeTab rt = rope_tab(q0, R, hd, bb_theta, bb_factor, bb_lo, bb_hi, bb_orig);
        ggml_backend_tensor_set(cos_t, rt.cos.data(), 0, rt.cos.size() * sizeof(float));
        ggml_backend_tensor_set(sin_t, rt.sin.data(), 0, rt.sin.size() * sizeof(float));
        if (mask) {
            std::vector<float> mh(static_cast<std::size_t>(Lkv) * R, 0.0f);
            const float ninf = -std::numeric_limits<float>::infinity();
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < Lkv; ++j)
                    if (j > q0 + r) mh[static_cast<std::size_t>(r) * Lkv + j] = ninf;
            ggml_backend_tensor_set(mask, mh.data(), 0, mh.size() * sizeof(float));
        }
        ggml_backend_graph_compute(s.backend, gf);
        ggml_backend_tensor_get(hn, last_hidden.data(), 0, H * sizeof(float));
        if (want_logits) {
            logits_out.resize(AV);
            ggml_backend_tensor_get(logits, logits_out.data(), 0, AV * sizeof(float));
        }
        ggml_free(ctx);
    };

    if (std::getenv("PIE_CSM_DUMP")) {
        std::fprintf(stderr, "[csm] prompt (%d ids):", n_prompt);
        for (int i = 0; i < n_prompt; ++i) std::fprintf(stderr, " %d", prompt[i]);
        std::fprintf(stderr, "\n");
    }
    std::vector<float> dummy;
    backbone_forward(prompt, n_prompt, 0, n_prompt, false, dummy);
    int Lkv = n_prompt;

    // ── Depth decoder (FUSED): all 31 codebook steps in ONE graph ──
    // In-graph argmax chains each step's code into the next step's embed (via a
    // per-step get_rows over the codebook's embed slab), and the per-layer KV
    // handles are threaded across steps so topology orders each step's read after
    // the prior write. Each unrolled step reads EXACTLY its valid cache prefix
    // (Lkv = p+1) — not a masked fixed cap — so the attention reduction width is
    // bit-identical to the per-step path. This collapses ~31 host-argmax
    // syncs/frame to ONE compute, with output identical to the per-step decoder.
    auto depth_frame = [&](int cb0, std::int32_t* cb_rest) {
        ggml_context* ctx = new_compute_ctx(DEPTH_FUSED_NODES);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, DEPTH_FUSED_NODES, false);
        ggml_tensor* bbh     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DBH, 1);
        ggml_tensor* cb0_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* cos_all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Dhd, NCB);
        ggml_tensor* sin_all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Dhd, NCB);
        for (ggml_tensor* t : {bbh, cb0_in, cos_all, sin_all})
            ggml_set_input(t);
        auto col = [&](ggml_tensor* a, int c) {  // column c of [n,NCB] -> cont [n,1]
            return ggml_cont(ctx, ggml_view_2d(ctx, a, a->ne[0], 1, a->nb[1],
                                  static_cast<std::size_t>(c) * a->nb[1]));
        };
        // Per-step set_rows position index. Metal can't `cont` an I64 view, so
        // mirror the backbone path: one fresh I64 input tensor per step, value
        // set after alloc (see pos_inputs loop below).
        std::vector<std::pair<ggml_tensor*, std::int64_t>> pos_inputs;
        auto posc = [&](int c) {
            ggml_tensor* p = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
            ggml_set_input(p);
            pos_inputs.emplace_back(p, static_cast<std::int64_t>(c));
            return p;
        };
        std::vector<ggml_tensor*> klive(s.dd_k.begin(), s.dd_k.end());
        std::vector<ggml_tensor*> vlive(s.dd_v.begin(), s.dd_v.end());
        auto step = [&](ggml_tensor* embed_raw, int p) -> ggml_tensor* {
            ggml_tensor* inpL = ggml_mul_mat(ctx, C.depth_inputs_proj, embed_raw);
            // Hoist the per-step column views (same for all DL layers).
            ggml_tensor* cosc = col(cos_all, p), *sinc = col(sin_all, p);
            ggml_tensor* pidx = posc(p);
            for (int l = 0; l < DL; ++l) {
                DepthBlockOut o = depth_block(ctx, C.depth_layers[l], inpL,
                    klive[l], vlive[l], cosc, sinc, pidx,
                    DH, DNH, DKV, Dhd, p + 1, deps);  // Lkv = p+1 valid positions
                inpL = o.hidden; klive[l] = o.kc; vlive[l] = o.vc;
            }
            return rms_norm(ctx, inpL, C.depth_norm, deps);  // [DH,1]
        };
        step(bbh, 0);  // seed (pos 0): no head, just writes KV pos 0.
        ggml_tensor* prev_code = cb0_in;
        std::vector<ggml_tensor*> codes;
        for (int p = 1; p < NCB; ++p) {
            // embed slab for codebook p-1: depth_embed_tokens cols [(p-1)*V, p*V).
            ggml_tensor* eview = ggml_view_2d(
                ctx, C.depth_embed_tokens, DBH, DV, C.depth_embed_tokens->nb[1],
                static_cast<std::size_t>(p - 1) * DV * C.depth_embed_tokens->nb[1]);
            ggml_tensor* embed = ggml_get_rows(ctx, eview, prev_code);  // [DBH,1]
            ggml_tensor* hn = step(embed, p);
            ggml_tensor* slab = ggml_view_2d(
                ctx, C.depth_codebooks_head, DV, DH, C.depth_codebooks_head->nb[1],
                static_cast<std::size_t>(p - 1) * C.depth_codebooks_head->nb[2]);
            ggml_tensor* slabT = ggml_cont(ctx, ggml_transpose(ctx, slab));  // [DH,V]
            ggml_tensor* logits = ggml_mul_mat(ctx, slabT, hn);  // [V,1]
            ggml_tensor* code = ggml_argmax(ctx, logits);  // [1] I32
            codes.push_back(code);
            prev_code = code;
        }
        ggml_tensor* codes_t = codes[0];
        for (std::size_t i = 1; i < codes.size(); ++i)
            codes_t = ggml_concat(ctx, codes_t, codes[i], 0);  // [NCB-1] I32
        ggml_set_output(codes_t);
        ggml_build_forward_expand(gf, codes_t);
        ggml_gallocr_alloc_graph(s.galloc, gf);
        ggml_backend_tensor_set(bbh, last_hidden.data(), 0, DBH * sizeof(float));
        std::int32_t cb0v = cb0;
        ggml_backend_tensor_set(cb0_in, &cb0v, 0, sizeof(std::int32_t));
        {   std::vector<float> ca(static_cast<std::size_t>(Dhd) * NCB), sa(ca.size());
            for (int p = 0; p < NCB; ++p) {
                RopeTab rt = rope_tab(p, 1, Dhd, dd_theta, dd_factor, dd_lo, dd_hi, dd_orig);
                for (int d = 0; d < Dhd; ++d) {
                    ca[static_cast<std::size_t>(p) * Dhd + d] = rt.cos[d];
                    sa[static_cast<std::size_t>(p) * Dhd + d] = rt.sin[d];
                }
            }
            ggml_backend_tensor_set(cos_all, ca.data(), 0, ca.size() * sizeof(float));
            ggml_backend_tensor_set(sin_all, sa.data(), 0, sa.size() * sizeof(float));
        }
        for (auto& [pt, pv] : pos_inputs)
            ggml_backend_tensor_set(pt, &pv, 0, sizeof(std::int64_t));
        ggml_backend_graph_compute(s.backend, gf);
        std::vector<std::int32_t> codes_h(static_cast<std::size_t>(NCB - 1));
        ggml_backend_tensor_get(codes_t, codes_h.data(), 0,
                                static_cast<std::size_t>(NCB - 1) * sizeof(std::int32_t));
        for (int p = 1; p < NCB; ++p) cb_rest[p - 1] = codes_h[static_cast<std::size_t>(p - 1)];
        ggml_free(ctx);
    };

    // ── Frame loop ──
    std::vector<std::int32_t> all_codes;       // [n_frames * NCB] frame-major
    std::vector<std::int32_t> frame(NCB);
    int n_frames = 0;
    std::vector<float> lm_logits;
    for (int f = 0; f < max_frames; ++f) {
        // cb0 from the already-computed last_hidden (prefill seeded the first).
        // Compute lm_head logits via a tiny graph over last_hidden.
        {
            ggml_context* ctx = new_compute_ctx();
            ggml_cgraph* gf = ggml_new_graph_custom(ctx, 64, false);
            ggml_tensor* hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(hin);
            ggml_tensor* logits = ggml_mul_mat(ctx, C.lm_head, hin);
            ggml_set_output(logits);
            ggml_build_forward_expand(gf, logits);
            ggml_gallocr_alloc_graph(s.galloc, gf);
            ggml_backend_tensor_set(hin, last_hidden.data(), 0, H * sizeof(float));
            ggml_backend_graph_compute(s.backend, gf);
            lm_logits.resize(AV);
            ggml_backend_tensor_get(logits, lm_logits.data(), 0, AV * sizeof(float));
            ggml_free(ctx);
        }
        int cb0 = argmax(lm_logits);

        std::int32_t cb_rest[64];
        depth_frame(cb0, cb_rest);
        frame[0] = cb0;
        for (int c = 1; c < NCB; ++c) frame[c] = cb_rest[c - 1];

        bool all_eos = true;
        for (int c = 0; c < NCB - 1; ++c)
            if (frame[c] != cc.codebook_eos_token_id) { all_eos = false; break; }
        if (all_eos) break;

        for (int c = 0; c < NCB; ++c) all_codes.push_back(frame[c]);
        if (out_codes) for (int c = 0; c < NCB; ++c) out_codes->push_back(frame[c]);
        if (std::getenv("PIE_CSM_DUMP")) {
            std::fprintf(stderr, "[csm] frame %d:", n_frames);
            for (int c = 0; c < NCB; ++c) std::fprintf(stderr, " %d", frame[c]);
            std::fprintf(stderr, "\n");
        }
        ++n_frames;
        if (f + 1 >= max_frames) break;

        // Re-embed the 32-code frame as the next backbone row and decode.
        {
            ggml_context* ctx = new_compute_ctx();
            ggml_cgraph* gf = ggml_new_graph_custom(ctx, GEN_GRAPH_NODES, false);
            ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NCB);
            ggml_set_input(idx);
            ggml_tensor* rows = ggml_get_rows(ctx, C.embed_audio, idx);  // [H, NCB]
            // sum over the NCB rows -> [H,1]: transpose -> [NCB,H], sum_rows -> [1,H].
            ggml_tensor* summed = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, rows)));
            ggml_tensor* inpL = ggml_cont(ctx, ggml_transpose(ctx, summed));  // [H,1]
            const int q0 = Lkv, newL = Lkv + 1;
            ggml_tensor* cos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, 1);
            ggml_tensor* sin_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, 1);
            ggml_set_input(cos_t); ggml_set_input(sin_t);
            ggml_tensor* pos_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
            ggml_set_input(pos_idx);
            for (int l = 0; l < BBL; ++l) {
                inpL = llama_block(ctx, C.backbone_layers[l], inpL, s.bb_k[l],
                                   s.bb_v[l], cos_t, sin_t, nullptr, pos_idx, H,
                                   NH, KV, hd, 1, newL, eps);
            }
            ggml_tensor* hn = rms_norm(ctx, inpL, C.bb_norm, eps);
            ggml_set_output(hn);
            ggml_build_forward_expand(gf, hn);
            ggml_gallocr_alloc_graph(s.galloc, gf);
            std::vector<std::int32_t> idxh(NCB);
            for (int c = 0; c < NCB; ++c) idxh[c] = frame[c] + c * AV;
            ggml_backend_tensor_set(idx, idxh.data(), 0, NCB * sizeof(std::int32_t));
            { std::int64_t pp = q0; ggml_backend_tensor_set(pos_idx, &pp, 0, sizeof(std::int64_t)); }
            RopeTab rt = rope_tab(q0, 1, hd, bb_theta, bb_factor, bb_lo, bb_hi, bb_orig);
            ggml_backend_tensor_set(cos_t, rt.cos.data(), 0, rt.cos.size() * sizeof(float));
            ggml_backend_tensor_set(sin_t, rt.sin.data(), 0, rt.sin.size() * sizeof(float));
            ggml_backend_graph_compute(s.backend, gf);
            ggml_backend_tensor_get(hn, last_hidden.data(), 0, H * sizeof(float));
            ggml_free(ctx);
            Lkv = newL;
        }
    }

    // ── Mimi decode: codes [NCB, n_frames] (codebook-major) -> PCM ──
    if (n_frames > 0) {
        std::vector<std::int32_t> codes_cb(static_cast<std::size_t>(NCB) * n_frames);
        for (int f = 0; f < n_frames; ++f)
            for (int c = 0; c < NCB; ++c)
                codes_cb[static_cast<std::size_t>(c) * n_frames + f] =
                    all_codes[static_cast<std::size_t>(f) * NCB + c];
        csm_mimi_decode(model, codes_cb.data(), n_frames, out_pcm);
    } else {
        out_pcm.clear();
    }
    return n_frames;
}

}  // namespace pie_portable_driver
