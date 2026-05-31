// Standalone self-test for the full Gemma-3/4-family decoder forward
// (gemma_forward_bf16).
//
// SELF-CONSISTENCY test (NOT external ground truth): a CPU reference implements
// the EXACT SAME composed formulation as gemma_forward_bf16 —
//   embed * √H  ->  N x gemma sandwich layer (4 (1+w) norms, GQA attn with
//   per-layer sliding window + attention softcap, GeGLU-tanh MLP)  ->  final
//   rmsnorm_gemma (1+w)  ->  lm_head  ->  final logit softcap  ->  argmax
// re-using the per-layer math chained by decoder_layer_gemma_inplace, with bf16
// rounding at the same materialization boundaries (every gemm output, every
// rmsnorm output, every rope output, the attention output, the residual add).
// We check the GPU forward's logits AND argmax ids against it. This validates
// the WIRING — shapes, per-layer KV strides, sliding/full alternation, softcap
// placement, the √H embed scale — not agreement with a real Gemma checkpoint.
// See gemma_forward.cuh "ARCHITECTURAL ASSUMPTIONS" for what is DEFERRED.
//
// Build (H100, sm_90):
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I .../device/include -I .../device/src \
//     -o /tmp/gemma_fwd_selftest \
//     src/forward/gemma_forward.cu src/forward/gemma_forward_selftest.cu \
//     src/forward/gemma_layer.cu src/forward/llama_layer.cu \
//     src/kernels/gemma.cu src/kernels/rmsnorm.cu src/kernels/rope.cu \
//     src/kernels/rope_partial.cu src/kernels/residual_add.cu \
//     src/kernels/swiglu.cu src/kernels/embed.cu src/kernels/argmax.cu \
//     src/ops/attention_naive_paged.cu src/kernels/kv_append.cu \
//     src/ops/gemm.cpp src/kernels/dtype_cast.cu -lcublas
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/gemma_fwd_selftest

#include "gemma_forward.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::forward::GemmaForwardDims;
using pie_cuda_device::forward::GemmaForwardWeights;
using pie_cuda_device::forward::GemmaLayerWeights;
using pie_cuda_device::forward::gemma_forward_bf16;

// --- bf16 round-trip (round to nearest even), matching __float2bfloat16. -----
static float bf16_rt(float f) { return __bfloat162float(__float2bfloat16(f)); }
static __nv_bfloat16 to_bf16(float f) { return __float2bfloat16(f); }

#define CK(call)                                                               \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_),    \
                        __FILE__, __LINE__);                                   \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static __nv_bfloat16* up_bf16(const std::vector<float>& h) {
    std::vector<__nv_bfloat16> tmp(h.size());
    for (size_t i = 0; i < h.size(); ++i) tmp[i] = to_bf16(h[i]);
    __nv_bfloat16* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(__nv_bfloat16));
    cudaMemcpy(d, tmp.data(), h.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    return d;
}

// Per-layer host weights (fp32 masters; rounded to bf16 on use / upload).
struct HLayer {
    std::vector<float> input_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln;
    std::vector<float> wq, wk, wv, wo, w_gate, w_up, w_down;
};

int main() {
    // ---- Tiny dims per the spec. ----
    const int T = 4;
    const int R = 1;
    const int H = 64;
    const int n_q = 4;
    const int n_kv = 2;
    const int hd = 16;
    const int I = 128;
    const int n_layers = 2;
    const int vocab = 32;
    const int page_size = 16;
    const float rms_eps = 1e-6f;
    const float rope_theta = 10000.f;
    const float attn_softcap = 30.f;
    const float final_softcap = 50.f;
    const float embed_scale = std::sqrt((float)H);

    const int Hq = n_q * hd;
    const int Hkv = n_kv * hd;
    const float sm_scale = 1.f / std::sqrt((float)hd);  // banked uses sm_scale=-1 ⇒ 1/√hd

    // Sliding-window alternation (A5): layer 0 sliding (window=2), layer 1 full.
    std::vector<int> window_left = {2, -1};

    const int num_pages = (T + page_size - 1) / page_size;  // = 1 per layer

    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> dist(-0.10f, 0.10f);
    auto rnd = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    // ---- Model-level weights / inputs. Norm gains live around ~0 (the (1+w)
    //      convention adds the +1 inside the kernel), matching real Gemma where
    //      the stored gain is the delta from 1. ----
    std::vector<float> embed = rnd(vocab * H);
    std::vector<float> final_norm = rnd(H);
    std::vector<float> lm_head = rnd(vocab * H);

    std::vector<HLayer> layers(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        HLayer& ly = layers[L];
        ly.input_ln = rnd(H);
        ly.post_attn_ln = rnd(H);
        ly.pre_ffn_ln = rnd(H);
        ly.post_ffn_ln = rnd(H);
        ly.wq = rnd(Hq * H);
        ly.wk = rnd(Hkv * H);
        ly.wv = rnd(Hkv * H);
        ly.wo = rnd(H * Hq);
        ly.w_gate = rnd(I * H);
        ly.w_up = rnd(I * H);
        ly.w_down = rnd(H * I);
    }

    std::vector<int32_t> token_ids(T);
    std::uniform_int_distribution<int> tdist(0, vocab - 1);
    for (int t = 0; t < T; ++t) token_ids[t] = tdist(rng);
    std::vector<int32_t> positions(T);
    for (int t = 0; t < T; ++t) positions[t] = t;

    // CSR page bookkeeping (single fresh request, slot 0) — shared per layer.
    std::vector<uint32_t> qo_indptr = {0, (uint32_t)T};
    std::vector<uint32_t> kv_page_indptr = {0, (uint32_t)num_pages};
    std::vector<uint32_t> kv_page_indices(num_pages);
    for (int p = 0; p < num_pages; ++p) kv_page_indices[p] = p;
    std::vector<uint32_t> kv_last_page_lens = {
        (uint32_t)(T - (num_pages - 1) * page_size)};

    // ============================ CPU REFERENCE ===========================
    // rmsnorm_gemma: (1 + w) gain. y = x * rsqrt(mean(x^2)+eps) * (1+w).
    auto rmsnorm_gemma = [&](const std::vector<float>& x,
                             const std::vector<float>& wt, int rows, int dim) {
        std::vector<float> y(rows * dim);
        for (int r = 0; r < rows; ++r) {
            float ss = 0.f;
            for (int i = 0; i < dim; ++i) {
                float v = bf16_rt(x[r * dim + i]);
                ss += v * v;
            }
            float inv = 1.f / std::sqrt(ss / dim + rms_eps);
            for (int i = 0; i < dim; ++i) {
                float xv = bf16_rt(x[r * dim + i]);
                float wv = bf16_rt(wt[i]) + 1.f;  // (1 + w)
                y[r * dim + i] = bf16_rt(xv * inv * wv);
            }
        }
        return y;
    };
    // y = act @ wt^T, act [M,K], wt [N,K]; bf16 inputs, fp32 accum, bf16 out.
    auto gemm = [&](const std::vector<float>& act, const std::vector<float>& wt,
                    int M, int N, int K) {
        std::vector<float> out(M * N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float acc = 0.f;
                for (int k = 0; k < K; ++k)
                    acc += bf16_rt(act[m * K + k]) * bf16_rt(wt[n * K + k]);
                out[m * N + n] = bf16_rt(acc);
            }
        return out;
    };
    // RoPE (NeoX pairing, full slice): angle = pos * theta^(-2i/d).
    auto apply_rope = [&](std::vector<float>& x, int heads, int d) {
        int half = d / 2;
        for (int t = 0; t < T; ++t) {
            int pos = positions[t];
            for (int h = 0; h < heads; ++h) {
                float* p = &x[(t * heads + h) * d];
                for (int i = 0; i < half; ++i) {
                    float freq = std::pow(rope_theta, -2.f * i / (float)d);
                    float ang = pos * freq;
                    float cs = std::cos(ang), sn = std::sin(ang);
                    float a = bf16_rt(p[i]);
                    float bb = bf16_rt(p[i + half]);
                    p[i] = bf16_rt(a * cs - bb * sn);
                    p[i + half] = bf16_rt(bb * cs + a * sn);
                }
            }
        }
    };
    // GeLU(tanh) gate * up, elementwise (matches geglu_tanh_bf16_kernel).
    auto geglu = [&](const std::vector<float>& gate, const std::vector<float>& up) {
        const float c = 0.7978845608028654f;
        std::vector<float> y(gate.size());
        for (size_t idx = 0; idx < gate.size(); ++idx) {
            float g = bf16_rt(gate[idx]);
            float u = bf16_rt(up[idx]);
            float gelu = 0.5f * g * (1.f + std::tanh(c * (g + 0.044715f * g * g * g)));
            y[idx] = bf16_rt(gelu * u);
        }
        return y;
    };
    // transform_logit: cap*tanh(dot*scale/cap) if cap>0 else dot*scale.
    auto transform_logit = [&](float dot, float scale, float cap) {
        dot *= scale;
        if (cap > 0.f) dot = cap * std::tanh(dot / cap);
        return dot;
    };

    // One Gemma sandwich layer, in place on `hidden` [T,H]. Mirrors
    // decoder_layer_gemma_inplace exactly.
    auto gemma_layer_cpu = [&](std::vector<float>& hidden, const HLayer& ly,
                               int win_left) {
        // --- attention block: input-norm -> attn -> post-attn-norm -> residual ---
        std::vector<float> hn = rmsnorm_gemma(hidden, ly.input_ln, T, H);
        std::vector<float> q = gemm(hn, ly.wq, T, Hq, H);
        std::vector<float> k = gemm(hn, ly.wk, T, Hkv, H);
        std::vector<float> v = gemm(hn, ly.wv, T, Hkv, H);
        apply_rope(q, n_q, hd);
        apply_rope(k, n_kv, hd);
        // KV is freshly written for this request (pre_kv = 0), so KV row j == token j.
        // Causal naive paged attention with optional sliding window + softcap.
        std::vector<float> attn(T * Hq);  // [T, n_q, hd]
        const int gqa = n_q / n_kv;
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < n_q; ++h) {
                int kv_head = h / gqa;
                // kv_lim mirrors the kernel: kv_total - qo_len + qo_off + 1.
                // pre_kv=0 ⇒ kv_total = T, qo_len = T, qo_off = t ⇒ kv_lim = t+1.
                int kv_lim = t + 1;
                float mx = -1e30f;
                std::vector<float> logit(kv_lim, -1e30f);
                std::vector<char> active(kv_lim, 0);
                for (int j = 0; j < kv_lim; ++j) {
                    // Sliding window: skip kv < kv_lim - 1 - win_left.
                    if (win_left >= 0 && j < kv_lim - 1 - win_left) continue;
                    float dot = 0.f;
                    for (int dd = 0; dd < hd; ++dd)
                        dot += bf16_rt(q[(t * n_q + h) * hd + dd]) *
                               bf16_rt(k[(j * n_kv + kv_head) * hd + dd]);
                    float lg = transform_logit(dot, sm_scale, attn_softcap);
                    logit[j] = lg;
                    active[j] = 1;
                    if (lg > mx) mx = lg;
                }
                float den = 0.f;
                std::vector<float> p(kv_lim, 0.f);
                for (int j = 0; j < kv_lim; ++j)
                    if (active[j]) { p[j] = std::exp(logit[j] - mx); den += p[j]; }
                float inv = den > 0.f ? 1.f / den : 0.f;
                for (int dd = 0; dd < hd; ++dd) {
                    float acc = 0.f;
                    for (int j = 0; j < kv_lim; ++j)
                        if (active[j])
                            acc += p[j] * bf16_rt(v[(j * n_kv + kv_head) * hd + dd]);
                    attn[(t * n_q + h) * hd + dd] = bf16_rt(acc * inv);
                }
            }
        std::vector<float> o = gemm(attn, ly.wo, T, H, Hq);
        o = rmsnorm_gemma(o, ly.post_attn_ln, T, H);  // post-norm in place
        for (int i = 0; i < T * H; ++i)
            hidden[i] = bf16_rt(bf16_rt(hidden[i]) + bf16_rt(o[i]));

        // --- MLP block: pre-ffn-norm -> GeGLU -> post-ffn-norm -> residual ---
        std::vector<float> mn = rmsnorm_gemma(hidden, ly.pre_ffn_ln, T, H);
        std::vector<float> gate = gemm(mn, ly.w_gate, T, I, H);
        std::vector<float> up = gemm(mn, ly.w_up, T, I, H);
        std::vector<float> mlp = geglu(gate, up);
        std::vector<float> mlp_out = gemm(mlp, ly.w_down, T, H, I);
        mlp_out = rmsnorm_gemma(mlp_out, ly.post_ffn_ln, T, H);  // post-norm
        for (int i = 0; i < T * H; ++i)
            hidden[i] = bf16_rt(bf16_rt(hidden[i]) + bf16_rt(mlp_out[i]));
    };

    // embed * √H -> N layers -> final norm -> lm_head -> final softcap -> argmax
    std::vector<float> hidden(T * H);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < H; ++i)
            hidden[t * H + i] =
                bf16_rt(bf16_rt(embed[(size_t)token_ids[t] * H + i]) * embed_scale);
    for (int L = 0; L < n_layers; ++L)
        gemma_layer_cpu(hidden, layers[L], window_left[L]);
    std::vector<float> normed = rmsnorm_gemma(hidden, final_norm, T, H);
    std::vector<float> ref_logits = gemm(normed, lm_head, T, vocab, H);
    for (auto& lg : ref_logits)  // final logit softcap (A6)
        lg = bf16_rt(final_softcap * std::tanh(bf16_rt(lg) / final_softcap));
    std::vector<int32_t> ref_ids(T);
    for (int t = 0; t < T; ++t) {
        int best = 0;
        float bestv = ref_logits[t * vocab + 0];
        for (int vv = 1; vv < vocab; ++vv)
            if (ref_logits[t * vocab + vv] > bestv) {
                bestv = ref_logits[t * vocab + vv];
                best = vv;
            }
        ref_ids[t] = best;
    }

    // ============================== GPU RUN ===============================
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    cublasSetStream(cublas, stream);

    std::vector<GemmaLayerWeights> dw(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        const HLayer& ly = layers[L];
        dw[L].input_ln = up_bf16(ly.input_ln);
        dw[L].post_attn_ln = up_bf16(ly.post_attn_ln);
        dw[L].pre_ffn_ln = up_bf16(ly.pre_ffn_ln);
        dw[L].post_ffn_ln = up_bf16(ly.post_ffn_ln);
        dw[L].wq = up_bf16(ly.wq);
        dw[L].wk = up_bf16(ly.wk);
        dw[L].wv = up_bf16(ly.wv);
        dw[L].wo = up_bf16(ly.wo);
        dw[L].w_gate = up_bf16(ly.w_gate);
        dw[L].w_up = up_bf16(ly.w_up);
        dw[L].w_down = up_bf16(ly.w_down);
    }

    GemmaForwardWeights w;
    w.embed = up_bf16(embed);
    w.layers = dw.data();
    w.final_norm = up_bf16(final_norm);
    w.lm_head = up_bf16(lm_head);

    GemmaForwardDims dims;
    dims.n_layers = n_layers;
    dims.hidden = H;
    dims.n_q_heads = n_q;
    dims.n_kv_heads = n_kv;
    dims.head_dim = hd;
    dims.intermediate = I;
    dims.vocab = vocab;
    dims.page_size = page_size;
    dims.num_pages = num_pages;
    dims.window_left = window_left.data();  // per-layer array (A5)
    dims.window_left_all = -1;
    dims.attn_logit_softcap = attn_softcap;
    dims.final_logit_softcap = final_softcap;
    dims.embed_scale = embed_scale;
    dims.rms_eps = rms_eps;
    dims.rope_theta = rope_theta;
    dims.qk_norm = 0;            // A3: deferred/off
    dims.altup_num_inputs = 1;   // A8: deferred/off

    const size_t kv_total = (size_t)n_layers * num_pages * page_size * Hkv;
    __nv_bfloat16* d_k = nullptr;
    __nv_bfloat16* d_v = nullptr;
    CK(cudaMalloc(&d_k, kv_total * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_v, kv_total * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_k, 0, kv_total * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_v, 0, kv_total * sizeof(__nv_bfloat16)));

    int32_t* d_tokens = nullptr;
    CK(cudaMalloc(&d_tokens, T * sizeof(int32_t)));
    CK(cudaMemcpy(d_tokens, token_ids.data(), T * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    int32_t* d_pos = nullptr;
    CK(cudaMalloc(&d_pos, T * sizeof(int32_t)));
    CK(cudaMemcpy(d_pos, positions.data(), T * sizeof(int32_t),
                  cudaMemcpyHostToDevice));

    auto up_u32 = [](const std::vector<uint32_t>& vv) {
        uint32_t* d = nullptr;
        cudaMalloc(&d, vv.size() * sizeof(uint32_t));
        cudaMemcpy(d, vv.data(), vv.size() * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
        return d;
    };
    uint32_t* d_qo = up_u32(qo_indptr);
    uint32_t* d_kpi = up_u32(kv_page_indices);
    uint32_t* d_kpp = up_u32(kv_page_indptr);
    uint32_t* d_klp = up_u32(kv_last_page_lens);

    __nv_bfloat16* d_logits = nullptr;
    int32_t* d_ids = nullptr;
    CK(cudaMalloc(&d_logits, (size_t)T * vocab * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_ids, T * sizeof(int32_t)));

    cudaError_t rc = gemma_forward_bf16(
        cublas, stream, d_tokens, w, d_pos, d_k, d_v,
        d_qo, d_kpi, d_kpp, d_klp, d_logits, d_ids, T, R, dims);
    CK(rc);

    std::vector<__nv_bfloat16> got_logits(T * vocab);
    CK(cudaMemcpy(got_logits.data(), d_logits,
                  T * vocab * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<int32_t> got_ids(T);
    CK(cudaMemcpy(got_ids.data(), d_ids, T * sizeof(int32_t),
                  cudaMemcpyDeviceToHost));

    // ============================== COMPARE ===============================
    float max_abs = 0.f, max_rel = 0.f;
    bool pass = true;
    for (int i = 0; i < T * vocab; ++i) {
        float got = __bfloat162float(got_logits[i]);
        float want = ref_logits[i];
        float ae = std::fabs(got - want);
        float tol = 0.08f * std::fabs(want) + 0.3f;
        float re = ae / (std::fabs(want) + 1e-6f);
        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, re);
        if (ae > tol) pass = false;
    }
    int id_mismatch = 0;
    for (int t = 0; t < T; ++t)
        if (got_ids[t] != ref_ids[t]) ++id_mismatch;
    if (id_mismatch) pass = false;

    std::printf("Gemma forward self-test: logits max_abs=%.6f max_rel=%.6f, "
                "argmax mismatches=%d/%d -> %s\n",
                max_abs, max_rel, id_mismatch, T, pass ? "PASS" : "FAIL");
    if (id_mismatch) {
        for (int t = 0; t < T; ++t)
            std::printf("  token %d: got id %d, want id %d\n", t, got_ids[t],
                        ref_ids[t]);
    }

    cublasDestroy(cublas);
    return pass ? 0 : 1;
}
