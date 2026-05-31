// Standalone self-test for the full DeepSeek-V3/V4 decoder forward
// (deepseek_forward_bf16).
//
// Self-consistency test: a CPU reference implements the EXACT SAME composed
// formulation as deepseek_forward_bf16 — embed -> N x (absorbed MLA block + FFN
// sublayer) -> final rmsnorm -> lm_head -> argmax — re-using the per-layer
// absorbed MLA math validated by mla_block_selftest.cu and the dense-MoE combine
// validated by moe_forward_selftest.cu (same weight layouts, same RoPE pairing,
// same causal latent attention, same topk_softmax routing + weighted combine,
// same dense SwiGLU, bf16 rounding at the same materialization boundaries). The
// tiny dims exercise BOTH FFN paths: with n_layers=2, first_k_dense=1, layer 0
// is the dense SwiGLU MLP and layer 1 is the sparse top-K MoE MLP. We check the
// GPU forward's logits AND argmax ids against the reference. This validates
// wiring / shapes / per-layer cache strides / kernel-call correctness, not
// external ground truth.
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I .../device/include -I .../device/src \
//     -o /tmp/deepseek_fwd_selftest \
//     src/forward/deepseek_forward.cu src/forward/deepseek_forward_selftest.cu \
//     src/forward/mla_block.cu src/ops/mla_paged.cu src/kernels/mla_write.cu \
//     src/forward/moe_mlp.cu src/kernels/moe.cu \
//     src/ops/gemm.cpp src/kernels/rmsnorm.cu src/kernels/rope_partial.cu \
//     src/kernels/residual_add.cu src/kernels/swiglu.cu src/kernels/dtype_cast.cu \
//     src/kernels/embed.cu src/kernels/argmax.cu -lcublas
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/deepseek_fwd_selftest

#include "deepseek_forward.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::forward::DeepseekDims;
using pie_cuda_device::forward::DeepseekLayerWeights;
using pie_cuda_device::forward::DeepseekWeights;
using pie_cuda_device::forward::MlaLayerWeights;
using pie_cuda_device::forward::deepseek_forward_bf16;

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

// Upload an fp32 host vector as bf16 to a fresh device buffer.
static __nv_bfloat16* up_bf16(const std::vector<float>& h) {
    std::vector<__nv_bfloat16> tmp(h.size());
    for (size_t i = 0; i < h.size(); ++i) tmp[i] = to_bf16(h[i]);
    __nv_bfloat16* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(__nv_bfloat16));
    cudaMemcpy(d, tmp.data(), h.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    return d;
}

// ---- Per-layer host weights (fp32 masters; rounded to bf16 on use/upload). --
struct HLayer {
    // MLA attention.
    std::vector<float> attn_norm, W_q_a, q_a_ln, W_q_b, W_kv_a, kv_a_ln,
        W_uk, W_uv, W_o;
    // FFN.
    std::vector<float> ffn_norm;
    std::vector<float> w_gate, w_up, w_down;        // dense path
    std::vector<float> router, wgu, wdown;          // MoE path
};

int main() {
    // ---- Dims (honor kv_lora_rank % 128 == 0, /128 <= 8). ----
    const int T = 4;
    const int R = 1;
    const int H = 256;
    const int nh = 2;
    const int q_lora = 96;
    const int ckv = 128;       // kv_lora_rank
    const int nope = 128;      // qk_nope_head_dim
    const int rope = 64;       // qk_rope_head_dim
    const int vhd = 128;       // v_head_dim
    const int n_layers = 2;
    const int first_k_dense = 1;  // layer 0 dense, layer 1 MoE
    const int dense_inter = 128;
    const int moe_inter = 128;
    const int E = 4;
    const int K = 2;
    const int vocab = 32;
    const int page_size = 16;
    const int qk = nope + rope;
    const int q_b_out = nh * qk;
    const int kv_a_out = ckv + rope;
    const int ov_width = nh * vhd;
    const float rms_eps = 1e-6f;
    const float rope_theta = 10000.f;
    const float sm_scale = 1.f / std::sqrt((float)(ckv + rope));

    const int num_pages = (T + page_size - 1) / page_size;  // = 1 per layer

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    auto rnd = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    // ---- Model-level weights / inputs. ----
    std::vector<float> embed = rnd(vocab * H);
    std::vector<float> final_norm = rnd(H);
    std::vector<float> lm_head = rnd(vocab * H);
    for (auto& x : final_norm) x += 1.f;  // norm gain ~1

    std::vector<HLayer> layers(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        HLayer& ly = layers[L];
        ly.attn_norm = rnd(H);
        ly.W_q_a = rnd(q_lora * H);
        ly.q_a_ln = rnd(q_lora);
        ly.W_q_b = rnd(q_b_out * q_lora);
        ly.W_kv_a = rnd(kv_a_out * H);
        ly.kv_a_ln = rnd(ckv);
        ly.W_uk = rnd(nh * ckv * nope);   // [nh, ckv, nope] transposed
        ly.W_uv = rnd(nh * vhd * ckv);    // [nh, vhd, ckv] transposed
        ly.W_o = rnd(H * ov_width);
        ly.ffn_norm = rnd(H);
        for (auto& x : ly.attn_norm) x += 1.f;
        for (auto& x : ly.q_a_ln) x += 1.f;
        for (auto& x : ly.kv_a_ln) x += 1.f;
        for (auto& x : ly.ffn_norm) x += 1.f;
        if (L < first_k_dense) {
            ly.w_gate = rnd(dense_inter * H);
            ly.w_up = rnd(dense_inter * H);
            ly.w_down = rnd(H * dense_inter);
        } else {
            ly.router = rnd(E * H);
            ly.wgu = rnd(E * 2 * moe_inter * H);   // [E, 2*moe_inter, H]
            ly.wdown = rnd(E * H * moe_inter);     // [E, H, moe_inter]
        }
    }

    // Token ids in [0, vocab) and contiguous RoPE positions.
    std::vector<int32_t> token_ids(T);
    std::uniform_int_distribution<int> tdist(0, vocab - 1);
    for (int t = 0; t < T; ++t) token_ids[t] = tdist(rng);
    std::vector<int32_t> positions(T);
    for (int t = 0; t < T; ++t) positions[t] = t;

    // CSR page bookkeeping (single request, fresh, slot 0) — shared per layer.
    std::vector<uint32_t> qo_indptr = {0, (uint32_t)T};
    std::vector<uint32_t> kv_page_indptr = {0, (uint32_t)num_pages};
    std::vector<uint32_t> kv_page_indices(num_pages);
    for (int p = 0; p < num_pages; ++p) kv_page_indices[p] = p;
    std::vector<uint32_t> kv_last_page_lens = {
        (uint32_t)(T - (num_pages - 1) * page_size)};

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
            for (int i = 0; i < dim; ++i) {
                float xv = bf16_rt(x[r * dim + i]);
                float wv = bf16_rt(wt[i]);
                y[r * dim + i] = bf16_rt(xv * inv * wv);
            }
        }
        return y;
    };
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
    // RoPE (NeoX pairing, full slice): angle = pos * theta^(-2*i/d).
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

    // One absorbed MLA block, in place on `hidden` [T,H]. Mirrors mla_block_bf16
    // (and the validated mla_block_selftest reference) exactly.
    auto mla_block_cpu = [&](std::vector<float>& hidden, const HLayer& ly) {
        std::vector<float> hn = rmsnorm(hidden, ly.attn_norm, T, H);
        std::vector<float> q_a = gemm(hn, ly.W_q_a, T, q_lora, H);
        q_a = rmsnorm(q_a, ly.q_a_ln, T, q_lora);
        std::vector<float> q_full = gemm(q_a, ly.W_q_b, T, q_b_out, q_lora);
        std::vector<float> q_nope(T * nh * nope), q_pe(T * nh * rope);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < nh; ++h) {
                const float* src = &q_full[(t * nh + h) * qk];
                for (int i = 0; i < nope; ++i)
                    q_nope[(t * nh + h) * nope + i] = bf16_rt(src[i]);
                for (int i = 0; i < rope; ++i)
                    q_pe[(t * nh + h) * rope + i] = bf16_rt(src[nope + i]);
            }
        std::vector<float> kv_a = gemm(hn, ly.W_kv_a, T, kv_a_out, H);
        std::vector<float> ckv_v(T * ckv), k_pe(T * rope);
        for (int t = 0; t < T; ++t) {
            const float* src = &kv_a[t * kv_a_out];
            for (int i = 0; i < ckv; ++i) ckv_v[t * ckv + i] = bf16_rt(src[i]);
            for (int i = 0; i < rope; ++i) k_pe[t * rope + i] = bf16_rt(src[ckv + i]);
        }
        ckv_v = rmsnorm(ckv_v, ly.kv_a_ln, T, ckv);
        apply_rope(q_pe, nh, rope);
        apply_rope(k_pe, 1, rope);
        std::vector<float> q_lat(T * nh * ckv);
        for (int h = 0; h < nh; ++h) {
            const float* Wh = &ly.W_uk[(size_t)h * ckv * nope];
            for (int t = 0; t < T; ++t)
                for (int l = 0; l < ckv; ++l) {
                    float acc = 0.f;
                    for (int d = 0; d < nope; ++d)
                        acc += bf16_rt(q_nope[(t * nh + h) * nope + d]) *
                               bf16_rt(Wh[l * nope + d]);
                    q_lat[(t * nh + h) * ckv + l] = bf16_rt(acc);
                }
        }
        std::vector<float> o_lat(T * nh * ckv);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < nh; ++h) {
                int j_end = t + 1;
                std::vector<float> sc(j_end);
                float mx = -1e30f;
                for (int j = 0; j < j_end; ++j) {
                    float pd = 0.f;
                    for (int l = 0; l < ckv; ++l)
                        pd += bf16_rt(q_lat[(t * nh + h) * ckv + l]) *
                              bf16_rt(ckv_v[j * ckv + l]);
                    for (int l = 0; l < rope; ++l)
                        pd += bf16_rt(q_pe[(t * nh + h) * rope + l]) *
                              bf16_rt(k_pe[j * rope + l]);
                    sc[j] = pd * sm_scale;
                    mx = std::max(mx, sc[j]);
                }
                float den = 0.f;
                std::vector<float> p(j_end);
                for (int j = 0; j < j_end; ++j) {
                    p[j] = std::exp(sc[j] - mx);
                    den += p[j];
                }
                for (int l = 0; l < ckv; ++l) {
                    float acc = 0.f;
                    for (int j = 0; j < j_end; ++j)
                        acc += p[j] * bf16_rt(ckv_v[j * ckv + l]);
                    o_lat[(t * nh + h) * ckv + l] = bf16_rt(acc / den);
                }
            }
        std::vector<float> o_v(T * nh * vhd);
        for (int h = 0; h < nh; ++h) {
            const float* Wh = &ly.W_uv[(size_t)h * vhd * ckv];
            for (int t = 0; t < T; ++t)
                for (int vd = 0; vd < vhd; ++vd) {
                    float acc = 0.f;
                    for (int l = 0; l < ckv; ++l)
                        acc += bf16_rt(o_lat[(t * nh + h) * ckv + l]) *
                               bf16_rt(Wh[vd * ckv + l]);
                    o_v[(t * nh + h) * vhd + vd] = bf16_rt(acc);
                }
        }
        std::vector<float> o_proj = gemm(o_v, ly.W_o, T, H, ov_width);
        for (int i = 0; i < T * H; ++i)
            hidden[i] = bf16_rt(bf16_rt(hidden[i]) + bf16_rt(o_proj[i]));
    };

    // FFN sublayer in place on hidden. Dense SwiGLU for L<first_k_dense (mirrors
    // llama_layer.cu / swiglu_bf16), else top-K MoE (mirrors moe_mlp.cu +
    // topk_softmax + the weighted top-K combine).
    auto ffn_cpu = [&](std::vector<float>& hidden, const HLayer& ly, int L) {
        std::vector<float> hn = rmsnorm(hidden, ly.ffn_norm, T, H);
        std::vector<float> ffn_out(T * H);
        if (L < first_k_dense) {
            const int I = dense_inter;
            std::vector<float> gate = gemm(hn, ly.w_gate, T, I, H);
            std::vector<float> up = gemm(hn, ly.w_up, T, I, H);
            std::vector<float> mlp(T * I);
            for (int i = 0; i < T * I; ++i)
                mlp[i] = bf16_rt(silu(bf16_rt(gate[i])) * bf16_rt(up[i]));
            ffn_out = gemm(mlp, ly.w_down, T, H, I);
        } else {
            const int I = moe_inter;
            std::vector<float> logits = gemm(hn, ly.router, T, E, H);  // [T,E]
            // topk_softmax_bf16 semantics: softmax over all E, top-K by prob
            // (ties -> lowest index), renormalize the K weights.
            std::vector<int> topk_idx(T * K);
            std::vector<float> topk_w(T * K);
            for (int t = 0; t < T; ++t) {
                float mx = -1e30f;
                for (int e = 0; e < E; ++e) mx = std::max(mx, logits[t * E + e]);
                std::vector<float> prob(E);
                float den = 0.f;
                for (int e = 0; e < E; ++e) {
                    prob[e] = std::exp(logits[t * E + e] - mx);
                    den += prob[e];
                }
                for (int e = 0; e < E; ++e) prob[e] /= den;
                float wsum = 0.f;
                for (int kk = 0; kk < K; ++kk) {
                    int best = -1;
                    float bestp = -1.f;
                    for (int e = 0; e < E; ++e)
                        if (prob[e] > bestp) { bestp = prob[e]; best = e; }
                    topk_idx[t * K + kk] = best;
                    topk_w[t * K + kk] = prob[best];
                    wsum += prob[best];
                    prob[best] = -1.f;  // exclude on next pass
                }
                for (int kk = 0; kk < K; ++kk) topk_w[t * K + kk] /= wsum;
            }
            // Dense per-expert FFN: ffn_all[e] = down(silu(gate)*up).
            // wgu[e] is [2I, H]; gate = rows [0,I), up = rows [I,2I).
            std::vector<float> ffn_all((size_t)E * T * H);
            for (int e = 0; e < E; ++e) {
                const float* wgu_e = &ly.wgu[(size_t)e * 2 * I * H];
                const float* wdown_e = &ly.wdown[(size_t)e * H * I];
                std::vector<float> gate_up =
                    gemm(hn, std::vector<float>(wgu_e, wgu_e + (size_t)2 * I * H),
                         T, 2 * I, H);
                std::vector<float> mlp(T * I);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < I; ++i)
                        mlp[t * I + i] = bf16_rt(silu(bf16_rt(gate_up[t * 2 * I + i])) *
                                                 bf16_rt(gate_up[t * 2 * I + I + i]));
                std::vector<float> ffn_e =
                    gemm(mlp, std::vector<float>(wdown_e, wdown_e + (size_t)H * I),
                         T, H, I);
                for (int i = 0; i < T * H; ++i)
                    ffn_all[(size_t)e * T * H + i] = ffn_e[i];
            }
            // Weighted top-K combine in fp32, round once to bf16 (matches
            // moe_combine_topk_kernel).
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h) {
                    float acc = 0.f;
                    for (int kk = 0; kk < K; ++kk) {
                        int e = topk_idx[t * K + kk];
                        acc += topk_w[t * K + kk] *
                               ffn_all[((size_t)e * T + t) * H + h];
                    }
                    ffn_out[t * H + h] = bf16_rt(acc);
                }
        }
        for (int i = 0; i < T * H; ++i)
            hidden[i] = bf16_rt(bf16_rt(hidden[i]) + bf16_rt(ffn_out[i]));
    };

    // embed -> N layers (MLA attn + FFN) -> final norm -> lm_head -> argmax
    std::vector<float> hidden(T * H);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < H; ++i)
            hidden[t * H + i] = bf16_rt(embed[(size_t)token_ids[t] * H + i]);
    for (int L = 0; L < n_layers; ++L) {
        mla_block_cpu(hidden, layers[L]);
        ffn_cpu(hidden, layers[L], L);
    }
    std::vector<float> normed = rmsnorm(hidden, final_norm, T, H);
    std::vector<float> ref_logits = gemm(normed, lm_head, T, vocab, H);
    std::vector<int32_t> ref_ids(T);
    for (int t = 0; t < T; ++t) {
        int best = 0;
        float bestv = ref_logits[t * vocab + 0];
        for (int v = 1; v < vocab; ++v)
            if (ref_logits[t * vocab + v] > bestv) {
                bestv = ref_logits[t * vocab + v];
                best = v;
            }
        ref_ids[t] = best;
    }

    // ============================== GPU RUN ===============================
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    cublasSetStream(cublas, stream);

    std::vector<DeepseekLayerWeights> dw(n_layers);
    for (int L = 0; L < n_layers; ++L) {
        const HLayer& ly = layers[L];
        dw[L].attn.attn_norm = up_bf16(ly.attn_norm);
        dw[L].attn.W_q_a = up_bf16(ly.W_q_a);
        dw[L].attn.q_a_ln = up_bf16(ly.q_a_ln);
        dw[L].attn.W_q_b = up_bf16(ly.W_q_b);
        dw[L].attn.W_kv_a = up_bf16(ly.W_kv_a);
        dw[L].attn.kv_a_ln = up_bf16(ly.kv_a_ln);
        dw[L].attn.W_uk = up_bf16(ly.W_uk);
        dw[L].attn.W_uv = up_bf16(ly.W_uv);
        dw[L].attn.W_o = up_bf16(ly.W_o);
        dw[L].ffn_norm = up_bf16(ly.ffn_norm);
        if (L < first_k_dense) {
            dw[L].w_gate = up_bf16(ly.w_gate);
            dw[L].w_up = up_bf16(ly.w_up);
            dw[L].w_down = up_bf16(ly.w_down);
            dw[L].router_w = nullptr;
            dw[L].wgu = nullptr;
            dw[L].wdown = nullptr;
        } else {
            dw[L].w_gate = nullptr;
            dw[L].w_up = nullptr;
            dw[L].w_down = nullptr;
            dw[L].router_w = up_bf16(ly.router);
            dw[L].wgu = up_bf16(ly.wgu);
            dw[L].wdown = up_bf16(ly.wdown);
        }
    }

    DeepseekWeights w;
    w.embed = up_bf16(embed);
    w.layers = dw.data();   // host array of per-layer device-pointer structs
    w.final_norm = up_bf16(final_norm);
    w.lm_head = up_bf16(lm_head);

    DeepseekDims dims;
    dims.n_layers = n_layers;
    dims.first_k_dense = first_k_dense;
    dims.hidden = H;
    dims.num_heads = nh;
    dims.q_lora_rank = q_lora;
    dims.kv_lora_rank = ckv;
    dims.qk_nope_head_dim = nope;
    dims.qk_rope_head_dim = rope;
    dims.v_head_dim = vhd;
    dims.dense_inter = dense_inter;
    dims.moe_inter = moe_inter;
    dims.num_experts = E;
    dims.top_k = K;
    dims.vocab = vocab;
    dims.page_size = page_size;
    dims.num_pages = num_pages;
    dims.rms_eps = rms_eps;
    dims.sm_scale = sm_scale;
    dims.rope_theta = rope_theta;

    // Per-layer paged MLA cache (zeroed; written by each block for all T tokens).
    const size_t ckv_total = (size_t)n_layers * num_pages * page_size * ckv;
    const size_t kpe_total = (size_t)n_layers * num_pages * page_size * rope;
    __nv_bfloat16* d_ckv_pages = nullptr;
    __nv_bfloat16* d_kpe_pages = nullptr;
    CK(cudaMalloc(&d_ckv_pages, ckv_total * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_kpe_pages, kpe_total * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_ckv_pages, 0, ckv_total * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_kpe_pages, 0, kpe_total * sizeof(__nv_bfloat16)));

    int32_t* d_tokens = nullptr;
    CK(cudaMalloc(&d_tokens, T * sizeof(int32_t)));
    CK(cudaMemcpy(d_tokens, token_ids.data(), T * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    int32_t* d_pos = nullptr;
    CK(cudaMalloc(&d_pos, T * sizeof(int32_t)));
    CK(cudaMemcpy(d_pos, positions.data(), T * sizeof(int32_t),
                  cudaMemcpyHostToDevice));

    auto up_u32 = [](const std::vector<uint32_t>& v) {
        uint32_t* d = nullptr;
        cudaMalloc(&d, v.size() * sizeof(uint32_t));
        cudaMemcpy(d, v.data(), v.size() * sizeof(uint32_t),
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

    cudaError_t rc = deepseek_forward_bf16(
        cublas, stream, d_tokens, w, d_pos, d_ckv_pages, d_kpe_pages,
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

    std::printf("DeepSeek forward self-test: logits max_abs=%.6f max_rel=%.6f, "
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
