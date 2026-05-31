// Standalone self-test for the Nemotron-H whole-model hybrid forward
// (nemotron_forward_bf16). Self-consistency: a CPU reference implements the
// EXACT SAME formulation in fp32 with bf16 rounding at the same boundaries the
// kernels round at, over a SHORT hybrid schedule [M, A, F, M] that exercises
// each of the three layer kinds. Validates wiring / layout / composition
// correctness (and the per-layer-kind dispatch), not external ground truth.
//
// Build (from driver/cuda_new/device):
//   export LD_LIBRARY_PATH=/usr/local/cuda/lib64
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 -I include -I src \
//     -o /tmp/nemotron_forward_selftest \
//     src/forward/nemotron_forward.cu src/forward/nemotron_forward_selftest.cu \
//     src/forward/nemotron_block.cu \
//     src/kernels/mamba_proj.cu src/kernels/ssm_scan.cu \
//     src/kernels/causal_conv1d.cu src/ops/attention_naive_paged.cu \
//     src/ops/gemm.cpp src/kernels/rmsnorm.cu src/kernels/rope.cu \
//     src/kernels/residual_add.cu src/kernels/swiglu.cu src/kernels/embed.cu \
//     src/kernels/argmax.cu src/kernels/kv_append.cu -lcublas
//   /tmp/nemotron_forward_selftest
//
// (llama_layer.cu / moe_mlp.cu / moe.cu / dtype_cast.cu are NOT needed: the
// attention and SwiGLU-FFN sublayers are composed directly from the lifted
// kernels in nemotron_forward.cu.)

#include "nemotron_forward.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using namespace pie_cuda_device::forward;

static float bf16_rt(float f) { return __bfloat162float(__float2bfloat16(f)); }
static __nv_bfloat16 to_bf16(float f) { return __float2bfloat16(f); }
static float from_bf16(__nv_bfloat16 v) { return __bfloat162float(v); }
static float softplus_ref(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
static float silu_ref(float x) { return x / (1.f + std::exp(-x)); }

#define CK(call)                                                               \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(e), __FILE__, __LINE__);           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// ---- dims ----------------------------------------------------------------
static const int T = 4;
static const int H = 64;
static const int V = 32;
static const float EPS = 1e-5f;
static const float TSMIN = 0.f;
static const float ROPE_THETA = 10000.f;

// Mamba
static const int M_HEADS = 4, M_HD = 16, M_STATE = 16, M_GROUPS = 2, M_K = 4;
static const int M_INTER = M_HEADS * M_HD;                       // 64
static const int M_CONV_DIM = M_INTER + 2 * M_GROUPS * M_STATE;  // 64+64=128
static const int M_DPROJ = M_INTER + M_CONV_DIM + M_HEADS;       // 64+128+4=196

// Attention
static const int A_NQ = 4, A_NKV = 2, A_HD = 16, PAGE = 16;
static const int A_HQ = A_NQ * A_HD;    // 64
static const int A_HKV = A_NKV * A_HD;  // 32

// FFN
static const int F_I = 96;

// ---- helpers -------------------------------------------------------------
static std::mt19937 rng(20260531);
static std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
static std::vector<float> rvec(int n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = bf16_rt(scale * ud(rng));
    return v;
}

static __nv_bfloat16* upload(const std::vector<float>& host) {
    std::vector<__nv_bfloat16> tmp(host.size());
    for (size_t i = 0; i < host.size(); ++i) tmp[i] = to_bf16(host[i]);
    __nv_bfloat16* d = nullptr;
    cudaMalloc(&d, tmp.size() * sizeof(__nv_bfloat16));
    cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    return d;
}

// y[t,n] = bf16( sum_k act[t,k]*W[n,k] ) + (beta? base[t,n] : 0)  (HF: act@W^T)
static std::vector<float> gemm_wt(const std::vector<float>& act,
                                  const std::vector<float>& W, int Tn, int N, int Kd) {
    std::vector<float> y(Tn * N);
    for (int t = 0; t < Tn; ++t)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < Kd; ++k) acc += act[t * Kd + k] * W[n * Kd + k];
            y[t * N + n] = bf16_rt(acc);
        }
    return y;
}

static std::vector<float> rmsnorm(const std::vector<float>& x,
                                  const std::vector<float>& w, int Tn, int Hn) {
    std::vector<float> y(Tn * Hn);
    for (int t = 0; t < Tn; ++t) {
        float ss = 0.f;
        for (int h = 0; h < Hn; ++h) {
            float v = bf16_rt(x[t * Hn + h]);
            ss += v * v;
        }
        float inv = 1.f / std::sqrt(ss / Hn + EPS);
        for (int h = 0; h < Hn; ++h)
            y[t * Hn + h] = bf16_rt(bf16_rt(x[t * Hn + h]) * inv * w[h]);
    }
    return y;
}

// ---- per-kind CPU references (operate in place on residual `hid` [T,H]) ----

struct MambaW {
    std::vector<float> pre_norm, in_proj, conv_w, conv_b, A_log, D, dt_bias, norm_w, out_proj;
};
static void cpu_mamba(std::vector<float>& hid, const MambaW& w) {
    // pre-norm
    std::vector<float> nx = rmsnorm(hid, w.pre_norm, T, H);
    // in_proj
    std::vector<float> proj = gemm_wt(nx, w.in_proj, T, M_DPROJ, H);
    // split: conv_in = cols [M_INTER, M_INTER+M_CONV_DIM); dt = last M_HEADS;
    //        z (gate) = first M_INTER cols (read in place from proj).
    std::vector<float> conv_in(T * M_CONV_DIM), dt_raw(T * M_HEADS);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < M_CONV_DIM; ++c)
            conv_in[t * M_CONV_DIM + c] = proj[t * M_DPROJ + M_INTER + c];
        for (int h = 0; h < M_HEADS; ++h)
            dt_raw[t * M_HEADS + h] = proj[t * M_DPROJ + M_INTER + M_CONV_DIM + h];
    }
    // depthwise causal conv + silu
    std::vector<float> conv_out(T * M_CONV_DIM);
    for (int c = 0; c < M_CONV_DIM; ++c)
        for (int t = 0; t < T; ++t) {
            float acc = w.conv_b[c];
            for (int k = 0; k < M_K; ++k) {
                int st = t - (M_K - 1) + k;
                float xv = (st < 0) ? 0.f : conv_in[st * M_CONV_DIM + c];
                acc += w.conv_w[c * M_K + k] * xv;
            }
            conv_out[t * M_CONV_DIM + c] = bf16_rt(silu_ref(acc));
        }
    // params
    std::vector<float> A(M_HEADS);
    for (int h = 0; h < M_HEADS; ++h) A[h] = -std::exp(w.A_log[h]);
    // sequential SSM
    std::vector<float> scan_y(T * M_INTER, 0.f);
    std::vector<float> state(M_HEADS * M_HD * M_STATE, 0.f);
    const int hpg = M_HEADS / M_GROUPS;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < M_HEADS; ++h) {
            int g = h / hpg;
            float A_h = A[h], D_h = w.D[h], dt_b = w.dt_bias[h];
            int bc_base = M_INTER + g * M_STATE;
            int c_base = M_INTER + M_GROUPS * M_STATE + g * M_STATE;
            float dt = std::fmax(softplus_ref(bf16_rt(dt_raw[t * M_HEADS + h]) + dt_b), TSMIN);
            float dA = std::exp(dt * A_h);
            float* stp = state.data() + (long long)h * M_HD * M_STATE;
            for (int dim = 0; dim < M_HD; ++dim) {
                float x = bf16_rt(conv_out[t * M_CONV_DIM + h * M_HD + dim]);
                float sum = 0.f;
                for (int s = 0; s < M_STATE; ++s) {
                    float bb = bf16_rt(conv_out[t * M_CONV_DIM + bc_base + s]);
                    float cc = bf16_rt(conv_out[t * M_CONV_DIM + c_base + s]);
                    int idx = dim * M_STATE + s;
                    float old = bf16_rt(stp[idx]);
                    float next = old * dA + (dt * bb) * x;
                    stp[idx] = bf16_rt(next);
                    sum += next * cc;
                }
                scan_y[t * M_INTER + h * M_HD + dim] = bf16_rt(sum + D_h * x);
            }
        }
    // gated rmsnorm (z read in place from proj's first M_INTER cols)
    const int gsize = M_INTER / M_GROUPS;
    std::vector<float> core(T * M_INTER);
    for (int t = 0; t < T; ++t)
        for (int g = 0; g < M_INTER / gsize; ++g) {
            float ss = 0.f;
            for (int i = 0; i < gsize; ++i) {
                int ch = g * gsize + i;
                float xv = bf16_rt(scan_y[t * M_INTER + ch]);
                float gv = bf16_rt(proj[t * M_DPROJ + ch]);
                float v = xv * silu_ref(gv);
                ss += v * v;
            }
            float inv = 1.f / std::sqrt(ss / gsize + EPS);
            for (int i = 0; i < gsize; ++i) {
                int ch = g * gsize + i;
                float xv = bf16_rt(scan_y[t * M_INTER + ch]);
                float gv = bf16_rt(proj[t * M_DPROJ + ch]);
                float v = xv * silu_ref(gv) * inv;
                core[t * M_INTER + ch] = bf16_rt(v * w.norm_w[ch]);
            }
        }
    // out_proj + residual: hid += out_proj(core)
    std::vector<float> op = gemm_wt(core, w.out_proj, T, H, M_INTER);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            hid[t * H + h] = bf16_rt(bf16_rt(hid[t * H + h]) + bf16_rt(op[t * H + h]));
}

struct AttnW { std::vector<float> norm, wq, wk, wv, wo; };
static void rope_pair(std::vector<float>& v, int Tn, int nheads, int hd) {
    // NeoX half/half: pair i with i+hd/2, freq theta^(-2i/hd), pos = token idx.
    int half = hd / 2;
    for (int t = 0; t < Tn; ++t)
        for (int h = 0; h < nheads; ++h)
            for (int i = 0; i < half; ++i) {
                float freq = std::pow(ROPE_THETA, -2.f * i / hd);
                float ang = t * freq;
                float c = std::cos(ang), s = std::sin(ang);
                int base = (t * nheads + h) * hd;
                float a = v[base + i], b = v[base + i + half];
                v[base + i] = bf16_rt(a * c - b * s);
                v[base + i + half] = bf16_rt(b * c + a * s);
            }
}
static void cpu_attn(std::vector<float>& hid, const AttnW& w) {
    std::vector<float> nx = rmsnorm(hid, w.norm, T, H);
    std::vector<float> q = gemm_wt(nx, w.wq, T, A_HQ, H);
    std::vector<float> k = gemm_wt(nx, w.wk, T, A_HKV, H);
    std::vector<float> vv = gemm_wt(nx, w.wv, T, A_HKV, H);
    rope_pair(q, T, A_NQ, A_HD);
    rope_pair(k, T, A_NKV, A_HD);
    // GQA causal attention. sm_scale = 1/sqrt(hd).
    const float scale = 1.f / std::sqrt((float)A_HD);
    const int gpkv = A_NQ / A_NKV;
    std::vector<float> attn(T * A_HQ, 0.f);
    for (int qh = 0; qh < A_NQ; ++qh) {
        int kvh = qh / gpkv;
        for (int t = 0; t < T; ++t) {
            // scores over keys [0, t]
            std::vector<float> sc(t + 1);
            float mx = -1e30f;
            for (int j = 0; j <= t; ++j) {
                float dot = 0.f;
                for (int d = 0; d < A_HD; ++d)
                    dot += bf16_rt(q[(t * A_NQ + qh) * A_HD + d]) *
                           bf16_rt(k[(j * A_NKV + kvh) * A_HD + d]);
                sc[j] = dot * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            float den = 0.f;
            for (int j = 0; j <= t; ++j) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
            for (int d = 0; d < A_HD; ++d) {
                float acc = 0.f;
                for (int j = 0; j <= t; ++j)
                    acc += sc[j] * bf16_rt(vv[(j * A_NKV + kvh) * A_HD + d]);
                attn[(t * A_NQ + qh) * A_HD + d] = bf16_rt(acc / den);
            }
        }
    }
    std::vector<float> o = gemm_wt(attn, w.wo, T, H, A_HQ);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            hid[t * H + h] = bf16_rt(bf16_rt(hid[t * H + h]) + bf16_rt(o[t * H + h]));
}

struct FfnW { std::vector<float> norm, gate, up, down; };
static void cpu_ffn(std::vector<float>& hid, const FfnW& w) {
    std::vector<float> nx = rmsnorm(hid, w.norm, T, H);
    std::vector<float> g = gemm_wt(nx, w.gate, T, F_I, H);
    std::vector<float> u = gemm_wt(nx, w.up, T, F_I, H);
    std::vector<float> mlp(T * F_I);
    for (int i = 0; i < T * F_I; ++i)
        mlp[i] = bf16_rt(silu_ref(bf16_rt(g[i])) * bf16_rt(u[i]));
    std::vector<float> o = gemm_wt(mlp, w.down, T, H, F_I);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            hid[t * H + h] = bf16_rt(bf16_rt(hid[t * H + h]) + bf16_rt(o[t * H + h]));
}

int main() {
    const char kinds[4] = {'M', 'A', 'F', 'M'};
    const int n_layers = 4;

    // ---- weights (bf16-faithful fp32 mirrors) ----
    std::vector<float> embed = rvec(V * H, 1.0f);
    std::vector<float> final_norm = rvec(H, 0.f);
    for (auto& x : final_norm) x = bf16_rt(1.f + 0.2f * ud(rng));
    std::vector<float> lm_head = rvec(V * H, 0.1f);

    // token ids / positions
    std::vector<std::int32_t> tokens(T);
    for (int t = 0; t < T; ++t) tokens[t] = (int)(rng() % V);
    std::vector<std::int32_t> positions(T);
    for (int t = 0; t < T; ++t) positions[t] = t;

    auto mk_norm = [&] { auto v = rvec(H, 0.f); for (auto& x : v) x = bf16_rt(1.f + 0.2f * ud(rng)); return v; };

    // Per-layer weight builders
    MambaW m0, m3;
    auto build_mamba = [&](MambaW& m) {
        m.pre_norm = mk_norm();
        m.in_proj = rvec(M_DPROJ * H, 0.1f);
        m.conv_w = rvec(M_CONV_DIM * M_K, 0.3f);
        m.conv_b = rvec(M_CONV_DIM, 0.1f);
        m.A_log = rvec(M_HEADS, 0.5f);
        m.D = rvec(M_HEADS, 1.0f);
        m.dt_bias = rvec(M_HEADS, 0.3f);
        m.norm_w = mk_norm(); m.norm_w.resize(M_INTER);
        for (auto& x : m.norm_w) x = bf16_rt(1.f + 0.2f * ud(rng));
        m.out_proj = rvec(H * M_INTER, 0.1f);
    };
    build_mamba(m0);
    build_mamba(m3);

    AttnW a1;
    a1.norm = mk_norm();
    a1.wq = rvec(A_HQ * H, 0.1f);
    a1.wk = rvec(A_HKV * H, 0.1f);
    a1.wv = rvec(A_HKV * H, 0.1f);
    a1.wo = rvec(H * A_HQ, 0.1f);

    FfnW f2;
    f2.norm = mk_norm();
    f2.gate = rvec(F_I * H, 0.1f);
    f2.up = rvec(F_I * H, 0.1f);
    f2.down = rvec(H * F_I, 0.1f);

    // ====================== CPU reference ======================
    std::vector<float> hid(T * H);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h)
            hid[t * H + h] = bf16_rt(embed[tokens[t] * H + h]);
    cpu_mamba(hid, m0);   // L0 M
    cpu_attn(hid, a1);    // L1 A
    cpu_ffn(hid, f2);     // L2 F
    cpu_mamba(hid, m3);   // L3 M
    std::vector<float> fn = rmsnorm(hid, final_norm, T, H);
    std::vector<float> logits_ref = gemm_wt(fn, lm_head, T, V, H);
    std::vector<std::int32_t> argmax_ref(T);
    for (int t = 0; t < T; ++t) {
        int best = 0; float bv = -1e30f;
        for (int v = 0; v < V; ++v) {
            float lv = bf16_rt(logits_ref[t * V + v]);
            if (lv > bv) { bv = lv; best = v; }
        }
        argmax_ref[t] = best;
    }

    // ====================== GPU run ======================
    NemotronForwardWeights gw{};
    gw.embed = upload(embed);
    gw.final_norm = upload(final_norm);
    gw.lm_head = upload(lm_head);

    std::vector<NemotronLayerWeights> layers(n_layers);
    auto up_mamba = [&](NemotronLayerWeights& Lw, const MambaW& m) {
        Lw.kind = 'M';
        Lw.mamba_pre_norm = upload(m.pre_norm);
        Lw.mamba.in_proj_w = upload(m.in_proj);
        Lw.mamba.conv_w = upload(m.conv_w);
        Lw.mamba.conv_bias = upload(m.conv_b);
        Lw.mamba.A_log = upload(m.A_log);
        Lw.mamba.D = upload(m.D);
        Lw.mamba.dt_bias = upload(m.dt_bias);
        Lw.mamba.norm_weight = upload(m.norm_w);
        Lw.mamba.out_proj_w = upload(m.out_proj);
    };
    up_mamba(layers[0], m0);
    layers[1].kind = 'A';
    layers[1].attn.attn_norm = upload(a1.norm);
    layers[1].attn.wq = upload(a1.wq);
    layers[1].attn.wk = upload(a1.wk);
    layers[1].attn.wv = upload(a1.wv);
    layers[1].attn.wo = upload(a1.wo);
    layers[2].kind = 'F';
    layers[2].ffn.ffn_norm = upload(f2.norm);
    layers[2].ffn.w_gate = upload(f2.gate);
    layers[2].ffn.w_up = upload(f2.up);
    layers[2].ffn.w_down = upload(f2.down);
    up_mamba(layers[3], m3);
    gw.layers = layers.data();

    NemotronForwardDims dims{};
    dims.n_layers = n_layers;
    dims.kinds = kinds;
    dims.hidden = H;
    dims.vocab = V;
    dims.mamba_num_heads = M_HEADS;
    dims.mamba_head_dim = M_HD;
    dims.mamba_state_size = M_STATE;
    dims.mamba_n_groups = M_GROUPS;
    dims.mamba_conv_kernel = M_K;
    dims.time_step_min = TSMIN;
    dims.attn_n_q_heads = A_NQ;
    dims.attn_n_kv_heads = A_NKV;
    dims.attn_head_dim = A_HD;
    dims.page_size = PAGE;
    dims.rope_theta = ROPE_THETA;
    dims.ffn_intermediate = F_I;
    dims.rms_eps = EPS;

    std::int32_t* d_tokens = nullptr;
    std::int32_t* d_pos = nullptr;
    CK(cudaMalloc(&d_tokens, T * sizeof(std::int32_t)));
    CK(cudaMalloc(&d_pos, T * sizeof(std::int32_t)));
    CK(cudaMemcpy(d_tokens, tokens.data(), T * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, positions.data(), T * sizeof(std::int32_t), cudaMemcpyHostToDevice));

    void* d_logits = nullptr;
    std::int32_t* d_argmax = nullptr;
    CK(cudaMalloc(&d_logits, (size_t)T * V * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_argmax, T * sizeof(std::int32_t)));

    cublasHandle_t cublas;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    CK(cudaStreamCreate(&stream));
    cublasSetStream(cublas, stream);

    CK(nemotron_forward_bf16(cublas, stream, d_tokens, gw, d_pos,
                             d_logits, d_argmax, T, dims));

    std::vector<__nv_bfloat16> logits_gpu(T * V);
    std::vector<std::int32_t> argmax_gpu(T);
    CK(cudaMemcpy(logits_gpu.data(), d_logits, logits_gpu.size() * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(argmax_gpu.data(), d_argmax, T * sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost));

    // ====================== compare ======================
    float max_abs = 0.f, max_rel = 0.f;
    int n_bad = 0;
    for (int i = 0; i < T * V; ++i) {
        float want = logits_ref[i];
        float got = from_bf16(logits_gpu[i]);
        float abs_err = std::fabs(got - want);
        float tol = 0.08f * std::fabs(want) + 0.3f;
        float rel = abs_err / (std::fabs(want) + 1e-6f);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel > max_rel) max_rel = rel;
        if (abs_err > tol) {
            if (n_bad < 8)
                std::printf("  MISMATCH logits[%d] (t=%d,v=%d): got=%.5f want=%.5f abs=%.5f tol=%.5f\n",
                            i, i / V, i % V, got, want, abs_err, tol);
            ++n_bad;
        }
    }
    int argmax_bad = 0;
    for (int t = 0; t < T; ++t) {
        if (argmax_gpu[t] != argmax_ref[t]) {
            ++argmax_bad;
            std::printf("  ARGMAX MISMATCH t=%d: got=%d want=%d\n", t, argmax_gpu[t], argmax_ref[t]);
        }
    }

    std::printf("nemotron_forward_bf16 selftest: schedule=[M,A,F,M] T=%d H=%d V=%d\n", T, H, V);
    std::printf("  mamba(heads=%d hd=%d state=%d groups=%d K=%d)  attn(nq=%d nkv=%d hd=%d page=%d)  ffn(I=%d)\n",
                M_HEADS, M_HD, M_STATE, M_GROUPS, M_K, A_NQ, A_NKV, A_HD, PAGE, F_I);
    std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  logits_bad=%d/%d  argmax_bad=%d/%d\n",
                max_abs, max_rel, n_bad, T * V, argmax_bad, T);
    bool pass = (n_bad == 0) && (argmax_bad == 0);
    std::printf("%s\n", pass ? "PASS" : "FAIL");

    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    return pass ? 0 : 1;
}
