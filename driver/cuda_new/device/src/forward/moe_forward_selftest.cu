// Standalone self-test for the dense-MoE transformer forward (prefill, bf16).
//
// Self-consistency test: a CPU reference implements the EXACT SAME formulation
// as moe_forward_bf16 (same weight layouts, NeoX rope pairing, GQA causal paged
// attention, dense top-K MoE FFN, bf16 rounding at the same materialization
// boundaries). We check the GPU logits AND the greedy argmax token ids against
// it. This validates wiring / shapes / kernel-call correctness, not external
// ground truth.
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I .../device/include -I .../device/src \
//     -o /tmp/moe_selftest \
//     src/forward/moe_forward.cu src/forward/moe_forward_selftest.cu \
//     src/forward/moe_mlp.cu src/forward/llama_layer.cu \
//     src/ops/attention_naive_paged.cu src/ops/gemm.cpp \
//     src/kernels/rmsnorm.cu src/kernels/rope.cu src/kernels/residual_add.cu \
//     src/kernels/embed.cu src/kernels/argmax.cu src/kernels/kv_append.cu \
//     src/kernels/moe.cu src/kernels/swiglu.cu src/kernels/dtype_cast.cu -lcublas
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/moe_selftest

#include "moe_forward.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::forward::MoeForwardDims;
using pie_cuda_device::forward::MoeForwardWeights;
using pie_cuda_device::forward::MoeLayerWeights;
using pie_cuda_device::forward::moe_forward_bf16;

// bf16 round-trip (round to nearest even), matching __float2bfloat16.
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
    cudaMemcpy(d, tmp.data(), h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    return d;
}

int main() {
    // ---- Dims (tiny). ----
    const int T = 4;        // tokens (1 request)
    const int R = 1;        // requests
    const int H = 64;       // hidden
    const int n_q = 4;      // q heads
    const int n_kv = 2;     // kv heads (GQA group = 2)
    const int hd = 16;      // head dim
    const int E = 4;        // experts
    const int K = 2;        // top-k
    const int I = 128;      // expert intermediate
    const int n_layers = 2;
    const int vocab = 32;
    const int page_size = 16;
    const float rms_eps = 1e-6f;
    const float rope_theta = 10000.f;

    const int Hq = n_q * hd;
    const int Hkv = n_kv * hd;
    const int grp = n_q / n_kv;  // q-heads per kv-head
    const float sm_scale = 1.f / std::sqrt((float)hd);

    const int num_pages = (T + page_size - 1) / page_size;  // 1

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    auto rnd = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    // ---- Host weights / inputs (fp32 master, rounded to bf16 on upload). ----
    std::vector<float> embed = rnd(vocab * H);
    std::vector<float> final_norm = rnd(H);
    for (auto& x : final_norm) x += 1.f;
    std::vector<float> lm_head = rnd(vocab * H);

    struct HLayer {
        std::vector<float> attn_norm, wq, wk, wv, wo, ffn_norm, router, wgu, wdown;
    };
    std::vector<HLayer> hl(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        hl[L].attn_norm = rnd(H);
        for (auto& x : hl[L].attn_norm) x += 1.f;
        hl[L].wq = rnd(Hq * H);
        hl[L].wk = rnd(Hkv * H);
        hl[L].wv = rnd(Hkv * H);
        hl[L].wo = rnd(H * Hq);
        hl[L].ffn_norm = rnd(H);
        for (auto& x : hl[L].ffn_norm) x += 1.f;
        hl[L].router = rnd(E * H);
        hl[L].wgu = rnd(E * 2 * I * H);
        hl[L].wdown = rnd(E * H * I);
    }

    std::vector<int32_t> token_ids(T);
    {
        std::uniform_int_distribution<int> td(0, vocab - 1);
        for (int t = 0; t < T; ++t) token_ids[t] = td(rng);
    }
    std::vector<int32_t> positions(T);
    for (int t = 0; t < T; ++t) positions[t] = t;

    std::vector<uint32_t> qo_indptr = {0, (uint32_t)T};
    std::vector<uint32_t> kv_page_indptr = {0, (uint32_t)num_pages};
    std::vector<uint32_t> kv_page_indices(num_pages);
    for (int p = 0; p < num_pages; ++p) kv_page_indices[p] = p;
    std::vector<uint32_t> kv_last_page_lens = {(uint32_t)(T - (num_pages - 1) * page_size)};

    // ============================ CPU REFERENCE ===========================
    auto rmsnorm = [&](const std::vector<float>& x, const std::vector<float>& wt,
                       int rows, int dim) {
        std::vector<float> y(rows * dim);
        for (int r = 0; r < rows; ++r) {
            float ss = 0.f;
            for (int i = 0; i < dim; ++i) {
                float v = bf16_rt(x[r * dim + i]);
                ss += v * v;
            }
            float inv = 1.f / std::sqrt(ss / dim + rms_eps);
            for (int i = 0; i < dim; ++i)
                y[r * dim + i] = bf16_rt(bf16_rt(x[r * dim + i]) * inv * bf16_rt(wt[i]));
        }
        return y;
    };
    // out[M,N] = act[M,K] @ w[N,K]^T
    auto gemm = [&](const std::vector<float>& act, const std::vector<float>& wt,
                    int M, int N, int Kk) {
        std::vector<float> out(M * N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float acc = 0.f;
                for (int k = 0; k < Kk; ++k)
                    acc += bf16_rt(act[m * Kk + k]) * bf16_rt(wt[n * Kk + k]);
                out[m * N + n] = bf16_rt(acc);
            }
        return out;
    };
    auto silu = [](float x) { return x / (1.f + std::exp(-x)); };

    // NeoX rope in place on [T, heads, hd].
    auto apply_rope = [&](std::vector<float>& x, int heads) {
        int half = hd / 2;
        for (int t = 0; t < T; ++t) {
            int pos = positions[t];
            for (int h = 0; h < heads; ++h) {
                float* p = &x[(size_t)(t * heads + h) * hd];
                for (int i = 0; i < half; ++i) {
                    float freq = std::pow(rope_theta, -2.f * i / (float)hd);
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

    // hidden = embed(tokens)
    std::vector<float> hidden(T * H);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < H; ++i)
            hidden[t * H + i] = bf16_rt(embed[(size_t)token_ids[t] * H + i]);

    for (int L = 0; L < n_layers; ++L) {
        // --- attention ---
        std::vector<float> hn = rmsnorm(hidden, hl[L].attn_norm, T, H);
        std::vector<float> q = gemm(hn, hl[L].wq, T, Hq, H);    // [T, n_q, hd]
        std::vector<float> k = gemm(hn, hl[L].wk, T, Hkv, H);   // [T, n_kv, hd]
        std::vector<float> v = gemm(hn, hl[L].wv, T, Hkv, H);   // [T, n_kv, hd]
        apply_rope(q, n_q);
        apply_rope(k, n_kv);

        // GQA causal paged attention. pre_kv=0, so query t attends j in [0,t].
        std::vector<float> attn(T * Hq);  // [T, n_q, hd]
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < n_q; ++h) {
                int kvh = h / grp;
                int j_end = t + 1;
                std::vector<float> sc(j_end);
                float mx = -1e30f;
                for (int j = 0; j < j_end; ++j) {
                    float dp = 0.f;
                    for (int e = 0; e < hd; ++e)
                        dp += bf16_rt(q[(size_t)(t * n_q + h) * hd + e]) *
                              bf16_rt(k[(size_t)(j * n_kv + kvh) * hd + e]);
                    sc[j] = dp * sm_scale;
                    mx = std::max(mx, sc[j]);
                }
                float den = 0.f;
                std::vector<float> p(j_end);
                for (int j = 0; j < j_end; ++j) {
                    p[j] = std::exp(sc[j] - mx);
                    den += p[j];
                }
                for (int e = 0; e < hd; ++e) {
                    float acc = 0.f;
                    for (int j = 0; j < j_end; ++j)
                        acc += p[j] * bf16_rt(v[(size_t)(j * n_kv + kvh) * hd + e]);
                    attn[(size_t)(t * n_q + h) * hd + e] = bf16_rt(acc / den);
                }
            }
        std::vector<float> o = gemm(attn, hl[L].wo, T, H, Hq);
        for (int i = 0; i < T * H; ++i)
            hidden[i] = bf16_rt(bf16_rt(hidden[i]) + bf16_rt(o[i]));

        // --- MoE FFN ---
        std::vector<float> hn2 = rmsnorm(hidden, hl[L].ffn_norm, T, H);
        std::vector<float> logits = gemm(hn2, hl[L].router, T, E, H);  // [T, E]
        // topk_softmax: softmax over all E, take top-K (by prob, ties low idx),
        // renormalize the K weights. Match topk_softmax_bf16 semantics.
        std::vector<int> topk_idx(T * K);
        std::vector<float> topk_w(T * K);
        for (int t = 0; t < T; ++t) {
            float mx = -1e30f;
            for (int e = 0; e < E; ++e) mx = std::max(mx, logits[t * E + e]);
            std::vector<float> prob(E);
            float den = 0.f;
            for (int e = 0; e < E; ++e) { prob[e] = std::exp(logits[t * E + e] - mx); den += prob[e]; }
            for (int e = 0; e < E; ++e) prob[e] /= den;
            std::vector<bool> used(E, false);
            float wsum = 0.f;
            for (int kk = 0; kk < K; ++kk) {
                int best = -1;
                float bestp = -1.f;
                for (int e = 0; e < E; ++e) {
                    if (used[e]) continue;
                    if (prob[e] > bestp) { bestp = prob[e]; best = e; }
                }
                used[best] = true;
                topk_idx[t * K + kk] = best;
                topk_w[t * K + kk] = prob[best];
                wsum += prob[best];
            }
            for (int kk = 0; kk < K; ++kk) topk_w[t * K + kk] /= wsum;
        }
        // dense experts: ffn_all[e,t] = down(silu(gate)*up)
        // wgu[e] is [2I, H] (gate rows [0,I), up rows [I,2I)); wdown[e] is [H,I].
        std::vector<float> moe_out(T * H, 0.f);
        for (int e = 0; e < E; ++e) {
            std::vector<float> wgu_e(hl[L].wgu.begin() + (size_t)e * 2 * I * H,
                                     hl[L].wgu.begin() + (size_t)(e + 1) * 2 * I * H);
            std::vector<float> wdown_e(hl[L].wdown.begin() + (size_t)e * H * I,
                                       hl[L].wdown.begin() + (size_t)(e + 1) * H * I);
            std::vector<float> gu = gemm(hn2, wgu_e, T, 2 * I, H);  // [T, 2I]
            std::vector<float> mlp(T * I);
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < I; ++i)
                    mlp[t * I + i] = bf16_rt(silu(bf16_rt(gu[t * 2 * I + i])) *
                                             bf16_rt(gu[t * 2 * I + I + i]));
            std::vector<float> ffn_e = gemm(mlp, wdown_e, T, H, I);  // [T, H]
            // combine: out[t] += w[t,k] * ffn_e[t] for any k with idx==e (fp32 weight).
            for (int t = 0; t < T; ++t) {
                float wt = 0.f;
                for (int kk = 0; kk < K; ++kk)
                    if (topk_idx[t * K + kk] == e) wt += topk_w[t * K + kk];
                if (wt != 0.f)
                    for (int i = 0; i < H; ++i)
                        moe_out[t * H + i] += wt * bf16_rt(ffn_e[t * H + i]);
            }
        }
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < H; ++i)
                hidden[t * H + i] = bf16_rt(bf16_rt(hidden[t * H + i]) +
                                            bf16_rt(moe_out[t * H + i]));
    }

    // final norm -> lm_head -> argmax
    std::vector<float> fn = rmsnorm(hidden, final_norm, T, H);
    std::vector<float> ref_logits = gemm(fn, lm_head, T, vocab, H);
    std::vector<int32_t> ref_tok(T);
    for (int t = 0; t < T; ++t) {
        int best = 0;
        float bv = ref_logits[t * vocab + 0];
        for (int e = 1; e < vocab; ++e)
            if (ref_logits[t * vocab + e] > bv) { bv = ref_logits[t * vocab + e]; best = e; }
        ref_tok[t] = best;
    }

    // ============================== GPU RUN ===============================
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    cublasSetStream(cublas, stream);

    MoeForwardWeights w{};
    w.embed = up_bf16(embed);
    w.final_norm = up_bf16(final_norm);
    w.lm_head = up_bf16(lm_head);
    w.n_layers = n_layers;
    std::vector<MoeLayerWeights> dl(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        dl[L].attn_norm = up_bf16(hl[L].attn_norm);
        dl[L].wq = up_bf16(hl[L].wq);
        dl[L].wk = up_bf16(hl[L].wk);
        dl[L].wv = up_bf16(hl[L].wv);
        dl[L].wo = up_bf16(hl[L].wo);
        dl[L].ffn_norm = up_bf16(hl[L].ffn_norm);
        dl[L].router_w = up_bf16(hl[L].router);
        dl[L].wgu = up_bf16(hl[L].wgu);
        dl[L].wdown = up_bf16(hl[L].wdown);
    }
    w.layers = dl.data();

    MoeForwardDims d{};
    d.hidden_size = H;
    d.n_q_heads = n_q;
    d.n_kv_heads = n_kv;
    d.head_dim = hd;
    d.intermediate = I;
    d.num_experts = E;
    d.top_k = K;
    d.n_layers = n_layers;
    d.vocab = vocab;
    d.page_size = page_size;
    d.rms_eps = rms_eps;
    d.rope_theta = rope_theta;

    int32_t* d_tok = nullptr;
    CK(cudaMalloc(&d_tok, T * sizeof(int32_t)));
    CK(cudaMemcpy(d_tok, token_ids.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice));
    int32_t* d_pos = nullptr;
    CK(cudaMalloc(&d_pos, T * sizeof(int32_t)));
    CK(cudaMemcpy(d_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Per-layer KV pools (NHD): [n_layers][num_pages, page_size, n_kv, hd].
    const size_t kv_elems = (size_t)n_layers * num_pages * page_size * Hkv;
    __nv_bfloat16* d_kv_k = nullptr;
    __nv_bfloat16* d_kv_v = nullptr;
    CK(cudaMalloc(&d_kv_k, kv_elems * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_kv_v, kv_elems * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_kv_k, 0, kv_elems * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_kv_v, 0, kv_elems * sizeof(__nv_bfloat16)));

    auto up_u32 = [](const std::vector<uint32_t>& vv) {
        uint32_t* dd = nullptr;
        cudaMalloc(&dd, vv.size() * sizeof(uint32_t));
        cudaMemcpy(dd, vv.data(), vv.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        return dd;
    };
    uint32_t* d_qo = up_u32(qo_indptr);
    uint32_t* d_kpi = up_u32(kv_page_indices);
    uint32_t* d_kpp = up_u32(kv_page_indptr);
    uint32_t* d_klp = up_u32(kv_last_page_lens);

    __nv_bfloat16* d_logits = nullptr;
    int32_t* d_out_tok = nullptr;
    CK(cudaMalloc(&d_logits, (size_t)T * vocab * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_out_tok, T * sizeof(int32_t)));

    cudaError_t rc = moe_forward_bf16(
        cublas, stream, d_tok, w, d_pos, d_kv_k, d_kv_v,
        d_qo, d_kpi, d_kpp, d_klp, d_logits, d_out_tok,
        T, R, num_pages, d);
    CK(rc);

    std::vector<__nv_bfloat16> got_logits(T * vocab);
    std::vector<int32_t> got_tok(T);
    CK(cudaMemcpy(got_logits.data(), d_logits, (size_t)T * vocab * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got_tok.data(), d_out_tok, T * sizeof(int32_t), cudaMemcpyDeviceToHost));

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
    bool tok_match = true;
    for (int t = 0; t < T; ++t)
        if (got_tok[t] != ref_tok[t]) tok_match = false;

    std::printf("argmax tokens  ref:");
    for (int t = 0; t < T; ++t) std::printf(" %d", ref_tok[t]);
    std::printf("  got:");
    for (int t = 0; t < T; ++t) std::printf(" %d", got_tok[t]);
    std::printf("\n");
    std::printf("MoE forward self-test: max_abs=%.6f max_rel=%.6f argmax_match=%s -> %s\n",
                max_abs, max_rel, tok_match ? "yes" : "NO",
                (pass && tok_match) ? "PASS" : "FAIL");

    cublasDestroy(cublas);
    return (pass && tok_match) ? 0 : 1;
}
