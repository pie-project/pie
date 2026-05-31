// Standalone self-test for the absorbed-form MLA block forward.
//
// Self-consistency test: a CPU reference implements the EXACT SAME absorbed
// formulation as mla_block_bf16 (same weight layouts, same RoPE pairing, same
// causal latent attention, bf16 rounding at the same materialization
// boundaries) and we check the GPU block output against it. This validates
// wiring / shapes / kernel-call correctness, not external ground truth.
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I .../device/include -I .../device/src \
//     -o /tmp/mla_selftest src/forward/mla_block.cu src/forward/mla_block_selftest.cu \
//     src/ops/mla_paged.cu src/kernels/mla_write.cu src/ops/gemm.cpp \
//     src/kernels/rmsnorm.cu src/kernels/rope_partial.cu \
//     src/kernels/residual_add.cu src/kernels/dtype_cast.cu -lcublas
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/mla_selftest

#include "mla_block.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::forward::MlaLayerWeights;
using pie_cuda_device::forward::mla_block_bf16;

// --- bf16 round-trip (round to nearest even), matching __float2bfloat16. -----
static float bf16_rt(float f) {
    return __bfloat162float(__float2bfloat16(f));
}
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
    const int page_size = 16;
    const int qk = nope + rope;
    const int q_b_out = nh * qk;
    const int kv_a_out = ckv + rope;
    const int ov_width = nh * vhd;
    const float rms_eps = 1e-6f;
    const float rope_theta = 10000.f;
    const float sm_scale = 1.f / std::sqrt((float)(ckv + rope));

    // Pages: enough for T tokens of one request from slot 0.
    const int num_pages = (T + page_size - 1) / page_size;  // = 1

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    auto rnd = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    // ---- Host weights / inputs (fp32 master, rounded to bf16 on upload). ----
    std::vector<float> hidden = rnd(T * H);
    std::vector<float> attn_norm = rnd(H);
    std::vector<float> W_q_a = rnd(q_lora * H);
    std::vector<float> q_a_ln = rnd(q_lora);
    std::vector<float> W_q_b = rnd(q_b_out * q_lora);
    std::vector<float> W_kv_a = rnd(kv_a_out * H);
    std::vector<float> kv_a_ln = rnd(ckv);
    std::vector<float> W_uk = rnd(nh * ckv * nope);   // [nh, ckv, nope] transposed
    std::vector<float> W_uv = rnd(nh * vhd * ckv);    // [nh, vhd, ckv] transposed
    std::vector<float> W_o = rnd(H * ov_width);
    // make norm gains ~1 so rmsnorm scales aren't tiny
    for (auto& x : attn_norm) x += 1.f;
    for (auto& x : q_a_ln) x += 1.f;
    for (auto& x : kv_a_ln) x += 1.f;

    std::vector<int32_t> positions(T);
    for (int t = 0; t < T; ++t) positions[t] = t;

    // CSR page bookkeeping (single request, fresh, slot 0).
    std::vector<uint32_t> qo_indptr = {0, (uint32_t)T};
    std::vector<uint32_t> kv_page_indptr = {0, (uint32_t)num_pages};
    std::vector<uint32_t> kv_page_indices(num_pages);
    for (int p = 0; p < num_pages; ++p) kv_page_indices[p] = p;
    std::vector<uint32_t> kv_last_page_lens = {
        (uint32_t)(T - (num_pages - 1) * page_size)};

    // ============================ CPU REFERENCE ===========================
    // All round to bf16 at materialization boundaries (matching GPU storage).
    // rmsnorm: y[r,:] = x[r,:]*rsqrt(mean(x^2)+eps)*w  (bf16 inputs, bf16 out)
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
    // gemm: out[M,N] = act[M,K] @ w[N,K]^T  (bf16 inputs, bf16 out)
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

    // 1. hn = rmsnorm(hidden, attn_norm)
    std::vector<float> hn = rmsnorm(hidden, attn_norm, T, H);
    // 2. q_a = hn @ W_q_a^T ; rmsnorm ; q = q_a @ W_q_b^T
    std::vector<float> q_a = gemm(hn, W_q_a, T, q_lora, H);
    q_a = rmsnorm(q_a, q_a_ln, T, q_lora);
    std::vector<float> q_full = gemm(q_a, W_q_b, T, q_b_out, q_lora);  // [T, nh*qk]
    // split into q_nope [T,nh,nope], q_pe [T,nh,rope]
    std::vector<float> q_nope(T * nh * nope), q_pe(T * nh * rope);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < nh; ++h) {
            const float* src = &q_full[(t * nh + h) * qk];
            for (int i = 0; i < nope; ++i)
                q_nope[(t * nh + h) * nope + i] = bf16_rt(src[i]);
            for (int i = 0; i < rope; ++i)
                q_pe[(t * nh + h) * rope + i] = bf16_rt(src[nope + i]);
        }
    // 3. kv_a = hn @ W_kv_a^T ; split ; rmsnorm(ckv)
    std::vector<float> kv_a = gemm(hn, W_kv_a, T, kv_a_out, H);
    std::vector<float> ckv_v(T * ckv), k_pe(T * rope);
    for (int t = 0; t < T; ++t) {
        const float* src = &kv_a[t * kv_a_out];
        for (int i = 0; i < ckv; ++i) ckv_v[t * ckv + i] = bf16_rt(src[i]);
        for (int i = 0; i < rope; ++i) k_pe[t * rope + i] = bf16_rt(src[ckv + i]);
    }
    ckv_v = rmsnorm(ckv_v, kv_a_ln, T, ckv);
    // 4. RoPE on q_pe (per head) and k_pe (shared), NeoX pairing, full slice.
    //    rope_partial: half = rope/2, angle = pos * theta^(-2*dimpair/rope)
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
    apply_rope(q_pe, nh, rope);
    apply_rope(k_pe, 1, rope);
    // 5. Absorb q_nope -> q_lat [T,nh,ckv]: per head q_nope[:,h,:] @ W_uk[h]
    //    W_uk[h] is [ckv, nope] transposed; gemm out[t,l]=sum_d qn[t,d]*W_uk[h][l,d]
    std::vector<float> q_lat(T * nh * ckv);
    for (int h = 0; h < nh; ++h) {
        const float* Wh = &W_uk[(size_t)h * ckv * nope];
        for (int t = 0; t < T; ++t)
            for (int l = 0; l < ckv; ++l) {
                float acc = 0.f;
                for (int d = 0; d < nope; ++d)
                    acc += bf16_rt(q_nope[(t * nh + h) * nope + d]) *
                           bf16_rt(Wh[l * nope + d]);
                q_lat[(t * nh + h) * ckv + l] = bf16_rt(acc);
            }
    }
    // 6/7. Latent paged MLA attention. Cache holds bf16(ckv_v) / bf16(k_pe).
    //   score_j = (q_lat[t,h] . ckv[j] + q_pe[t,h] . k_pe[j]) * sm_scale
    //   causal: j in [0, abs_q]; here pre_kv=0 so abs_q = t.
    //   o_lat[t,h,:] = softmax_j(score) weighted sum of ckv[j].
    std::vector<float> o_lat(T * nh * ckv);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < nh; ++h) {
            int j_end = t + 1;  // causal, pre_kv=0
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
    // 8a. Absorb V: o_v[:,h,:] = o_lat[:,h,:] @ W_uv[h]  (W_uv[h] [vhd,ckv])
    std::vector<float> o_v(T * nh * vhd);
    for (int h = 0; h < nh; ++h) {
        const float* Wh = &W_uv[(size_t)h * vhd * ckv];
        for (int t = 0; t < T; ++t)
            for (int vd = 0; vd < vhd; ++vd) {
                float acc = 0.f;
                for (int l = 0; l < ckv; ++l)
                    acc += bf16_rt(o_lat[(t * nh + h) * ckv + l]) *
                           bf16_rt(Wh[vd * ckv + l]);
                o_v[(t * nh + h) * vhd + vd] = bf16_rt(acc);
            }
    }
    // 8b. out = o_v[T, nh*vhd] @ W_o^T -> [T,H]
    std::vector<float> o_proj = gemm(o_v, W_o, T, H, ov_width);
    // 9. residual add
    std::vector<float> ref(T * H);
    for (int i = 0; i < T * H; ++i)
        ref[i] = bf16_rt(bf16_rt(hidden[i]) + bf16_rt(o_proj[i]));

    // ============================== GPU RUN ===============================
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    cublasSetStream(cublas, stream);

    __nv_bfloat16* d_hidden = up_bf16(hidden);
    MlaLayerWeights w;
    w.attn_norm = up_bf16(attn_norm);
    w.W_q_a = up_bf16(W_q_a);
    w.q_a_ln = up_bf16(q_a_ln);
    w.W_q_b = up_bf16(W_q_b);
    w.W_kv_a = up_bf16(W_kv_a);
    w.kv_a_ln = up_bf16(kv_a_ln);
    w.W_uk = up_bf16(W_uk);
    w.W_uv = up_bf16(W_uv);
    w.W_o = up_bf16(W_o);

    __nv_bfloat16* d_ckv_pages = nullptr;
    __nv_bfloat16* d_kpe_pages = nullptr;
    CK(cudaMalloc(&d_ckv_pages, (size_t)num_pages * page_size * ckv * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_kpe_pages, (size_t)num_pages * page_size * rope * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_ckv_pages, 0, (size_t)num_pages * page_size * ckv * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_kpe_pages, 0, (size_t)num_pages * page_size * rope * sizeof(__nv_bfloat16)));

    int32_t* d_pos = nullptr;
    CK(cudaMalloc(&d_pos, T * sizeof(int32_t)));
    CK(cudaMemcpy(d_pos, positions.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice));

    auto up_u32 = [](const std::vector<uint32_t>& v) {
        uint32_t* d = nullptr;
        cudaMalloc(&d, v.size() * sizeof(uint32_t));
        cudaMemcpy(d, v.data(), v.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        return d;
    };
    uint32_t* d_qo = up_u32(qo_indptr);
    uint32_t* d_kpi = up_u32(kv_page_indices);
    uint32_t* d_kpp = up_u32(kv_page_indptr);
    uint32_t* d_klp = up_u32(kv_last_page_lens);

    cudaError_t rc = mla_block_bf16(
        cublas, stream, d_hidden, w, d_pos, d_ckv_pages, d_kpe_pages,
        d_qo, d_kpi, d_kpp, d_klp, T, R, H, nh, q_lora, ckv, nope, rope, vhd,
        page_size, rms_eps, sm_scale, rope_theta);
    CK(rc);

    std::vector<__nv_bfloat16> got_bf(T * H);
    CK(cudaMemcpy(got_bf.data(), d_hidden, T * H * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));

    // ============================== COMPARE ===============================
    float max_abs = 0.f, max_rel = 0.f;
    bool pass = true;
    for (int i = 0; i < T * H; ++i) {
        float got = __bfloat162float(got_bf[i]);
        float want = ref[i];
        float ae = std::fabs(got - want);
        float tol = 0.08f * std::fabs(want) + 0.2f;
        float re = ae / (std::fabs(want) + 1e-6f);
        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, re);
        if (ae > tol) pass = false;
    }
    std::printf("MLA block self-test: max_abs=%.6f max_rel=%.6f -> %s\n",
                max_abs, max_rel, pass ? "PASS" : "FAIL");

    cublasDestroy(cublas);
    return pass ? 0 : 1;
}
