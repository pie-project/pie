// Standalone self-test for the composed Nemotron-H / Mamba-2 mixer block
// (nemotron_mamba_block_bf16). Self-consistency: a CPU reference implements the
// EXACT SAME formulation (in_proj -> split -> depthwise causal conv -> sequential
// SSM recurrence -> gated RMSNorm -> out_proj -> residual add) in fp32 with bf16
// rounding at the same boundaries the kernels round at. Validates wiring /
// layout / composition correctness, not external ground truth.
//
// Build (from driver/cuda_new/device):
//   export LD_LIBRARY_PATH=/usr/local/cuda/lib64
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 -I include -I src \
//     -o /tmp/nemotron_block_selftest \
//     src/forward/nemotron_block.cu src/forward/nemotron_block_selftest.cu \
//     src/kernels/mamba_proj.cu src/kernels/ssm_scan.cu \
//     src/kernels/causal_conv1d.cu src/ops/gemm.cpp \
//     src/kernels/rmsnorm.cu src/kernels/residual_add.cu \
//     src/kernels/dtype_cast.cu -lcublas
//   /tmp/nemotron_block_selftest

#include "nemotron_block.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::forward::NemotronMambaWeights;
using pie_cuda_device::forward::nemotron_mamba_block_bf16;

static float bf16_rt(float f) { return __bfloat162float(__float2bfloat16(f)); }
static __nv_bfloat16 to_bf16(float f) { return __float2bfloat16(f); }
static float from_bf16(__nv_bfloat16 v) { return __bfloat162float(v); }

#define CK(call)                                                               \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(e), __FILE__, __LINE__);           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static float softplus_ref(float x) {
    return x > 20.f ? x : std::log1p(std::exp(x));
}
static float silu_ref(float x) { return x / (1.f + std::exp(-x)); }

int main() {
    // Tiny dims (one request).
    const int T = 6;
    const int H = 64;
    const int num_heads = 4;
    const int head_dim = 16;
    const int state_size = 16;
    const int n_groups = 2;
    const int conv_kernel = 4;
    const float rms_eps = 1e-5f;
    const float time_step_min = 0.f;

    const int intermediate = num_heads * head_dim;              // 64
    const int conv_dim = intermediate + 2 * n_groups * state_size;  // 64 + 64 = 128
    const int d_in_proj = intermediate + conv_dim + num_heads;  // 64+128+4 = 196
    const int K = conv_kernel;

    std::mt19937 rng(20260530);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    auto rnd = [&] { return bf16_rt(ud(rng)); };

    // --- Host weights / inputs (bf16-faithful fp32 mirrors) ---------------
    std::vector<float> hidden(T * H);
    for (auto& v : hidden) v = rnd();
    std::vector<float> in_proj_w(d_in_proj * H);
    for (auto& v : in_proj_w) v = bf16_rt(0.1f * ud(rng));
    std::vector<float> conv_w(conv_dim * K);
    for (auto& v : conv_w) v = bf16_rt(0.3f * ud(rng));
    std::vector<float> conv_bias(conv_dim);
    for (auto& v : conv_bias) v = bf16_rt(0.1f * ud(rng));
    std::vector<float> A_log(num_heads);
    for (auto& v : A_log) v = bf16_rt(0.5f * ud(rng));
    std::vector<float> D(num_heads);
    for (auto& v : D) v = rnd();
    std::vector<float> dt_bias(num_heads);
    for (auto& v : dt_bias) v = bf16_rt(0.3f * ud(rng));
    std::vector<float> norm_weight(intermediate);
    for (auto& v : norm_weight) v = bf16_rt(1.f + 0.2f * ud(rng));
    std::vector<float> out_proj_w(H * intermediate);
    for (auto& v : out_proj_w) v = bf16_rt(0.1f * ud(rng));

    // ===================== CPU reference (fp32, bf16-faithful) ============
    // 1. in_proj: projected[t, j] = bf16( sum_k hidden[t,k] * in_proj_w[j,k] )
    std::vector<float> projected(T * d_in_proj);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < d_in_proj; ++j) {
            float acc = 0.f;
            for (int k = 0; k < H; ++k)
                acc += hidden[t * H + k] * in_proj_w[j * H + k];
            projected[t * d_in_proj + j] = bf16_rt(acc);
        }

    // 2. split: z = first `intermediate` cols (read in place); conv_in = next
    //    conv_dim cols; dt = last num_heads cols.
    std::vector<float> conv_in(T * conv_dim);
    std::vector<float> dt_raw(T * num_heads);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < conv_dim; ++c)
            conv_in[t * conv_dim + c] =
                projected[t * d_in_proj + intermediate + c];
        for (int h = 0; h < num_heads; ++h)
            dt_raw[t * num_heads + h] =
                projected[t * d_in_proj + intermediate + conv_dim + h];
    }

    // 3. depthwise causal conv (state = zero pad) with silu:
    //    y[t,c] = silu( sum_{k} W[c,k]*x[t-(K-1)+k,c] + bias[c] )
    std::vector<float> conv_out(T * conv_dim);
    for (int c = 0; c < conv_dim; ++c)
        for (int t = 0; t < T; ++t) {
            float acc = conv_bias[c];
            for (int k = 0; k < K; ++k) {
                const int src_t = t - (K - 1) + k;
                const float xv =
                    (src_t < 0) ? 0.f : conv_in[src_t * conv_dim + c];
                acc += conv_w[c * K + k] * xv;
            }
            conv_out[t * conv_dim + c] = bf16_rt(silu_ref(acc));
        }

    // 4. params: A = -exp(A_log), D, dt_bias (fp32).
    std::vector<float> A(num_heads);
    for (int h = 0; h < num_heads; ++h) A[h] = -std::exp(A_log[h]);
    // D, dt_bias already fp32 mirrors of bf16.

    // 5. sequential SSM recurrence (state starts zero). Matches the warp kernel:
    //    dt = max(softplus(dt_raw + dt_bias), tmin); dA = exp(dt*A);
    //    state' = state*dA + (dt*B[s])*x; y[dim] = sum_s state'*C[s] + D*x.
    std::vector<float> scan_y(T * intermediate, 0.f);
    std::vector<float> state(num_heads * head_dim * state_size, 0.f);
    const int heads_per_group = num_heads / n_groups;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const int group = h / heads_per_group;
            const float A_h = A[h];
            const float D_h = D[h];
            const float dt_b = dt_bias[h];
            const int bc_base = intermediate + group * state_size;
            const int c_base = intermediate + n_groups * state_size +
                               group * state_size;
            const float dt = std::fmax(
                softplus_ref(bf16_rt(dt_raw[t * num_heads + h]) + dt_b),
                time_step_min);
            const float dA = std::exp(dt * A_h);
            float* st = state.data() +
                        (long long)h * head_dim * state_size;
            for (int dim = 0; dim < head_dim; ++dim) {
                const float x = bf16_rt(conv_out[t * conv_dim + h * head_dim + dim]);
                float sum = 0.f;
                for (int s = 0; s < state_size; ++s) {
                    const float bb = bf16_rt(conv_out[t * conv_dim + bc_base + s]);
                    const float cc = bf16_rt(conv_out[t * conv_dim + c_base + s]);
                    const int idx = dim * state_size + s;
                    const float old = bf16_rt(st[idx]);
                    const float next = old * dA + (dt * bb) * x;
                    st[idx] = bf16_rt(next);     // state persisted bf16
                    sum += next * cc;
                }
                scan_y[t * intermediate + h * head_dim + dim] =
                    bf16_rt(sum + D_h * x);
            }
        }
    }

    // 6. gated RMSNorm over groups of `group_size` channels:
    //    v = scan_y * silu(z); y = v * rsqrt(mean(v^2)+eps) * weight.
    const int group_size = intermediate / n_groups;
    const int groups = intermediate / group_size;
    std::vector<float> core(T * intermediate);
    for (int t = 0; t < T; ++t)
        for (int g = 0; g < groups; ++g) {
            float ss = 0.f;
            for (int i = 0; i < group_size; ++i) {
                const int ch = g * group_size + i;
                const float xv = bf16_rt(scan_y[t * intermediate + ch]);
                // z lives in projected's first `intermediate` cols.
                const float gv = bf16_rt(projected[t * d_in_proj + ch]);
                const float v = xv * silu_ref(gv);
                ss += v * v;
            }
            const float inv_rms = 1.f / std::sqrt(ss / group_size + rms_eps);
            for (int i = 0; i < group_size; ++i) {
                const int ch = g * group_size + i;
                const float xv = bf16_rt(scan_y[t * intermediate + ch]);
                const float gv = bf16_rt(projected[t * d_in_proj + ch]);
                const float v = xv * silu_ref(gv) * inv_rms;
                core[t * intermediate + ch] = bf16_rt(v * norm_weight[ch]);
            }
        }

    // 7. out_proj + residual: out[t,h] = bf16( hidden[t,h] +
    //    bf16(sum_i core[t,i]*out_proj_w[h,i]) ).  (beta=1 fused add)
    std::vector<float> out_ref(T * H);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            float acc = 0.f;
            for (int i = 0; i < intermediate; ++i)
                acc += bf16_rt(core[t * intermediate + i]) * out_proj_w[h * intermediate + i];
            out_ref[t * H + h] = bf16_rt(bf16_rt(hidden[t * H + h]) + bf16_rt(acc));
        }

    // ====================== GPU run =======================================
    auto upload_bf16 = [](const std::vector<float>& host) {
        std::vector<__nv_bfloat16> tmp(host.size());
        for (size_t i = 0; i < host.size(); ++i) tmp[i] = to_bf16(host[i]);
        __nv_bfloat16* d = nullptr;
        cudaMalloc(&d, tmp.size() * sizeof(__nv_bfloat16));
        cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice);
        return d;
    };

    __nv_bfloat16* d_hidden = upload_bf16(hidden);
    NemotronMambaWeights w{};
    w.in_proj_w = upload_bf16(in_proj_w);
    w.conv_w = upload_bf16(conv_w);
    w.conv_bias = upload_bf16(conv_bias);
    w.A_log = upload_bf16(A_log);
    w.D = upload_bf16(D);
    w.dt_bias = upload_bf16(dt_bias);
    w.norm_weight = upload_bf16(norm_weight);
    w.out_proj_w = upload_bf16(out_proj_w);

    cublasHandle_t cublas;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    CK(cudaStreamCreate(&stream));
    cublasSetStream(cublas, stream);

    CK(nemotron_mamba_block_bf16(
        cublas, stream, d_hidden, w,
        T, H, num_heads, head_dim, state_size, n_groups, conv_kernel,
        rms_eps, time_step_min));

    std::vector<__nv_bfloat16> out_gpu(T * H);
    CK(cudaMemcpy(out_gpu.data(), d_hidden,
                  out_gpu.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // ====================== Compare =======================================
    float max_abs = 0.f, max_rel = 0.f;
    int n_bad = 0;
    const float atol = 0.1f, rtol = 0.05f;
    for (size_t i = 0; i < out_ref.size(); ++i) {
        const float want = out_ref[i];
        const float got = from_bf16(out_gpu[i]);
        const float abs_err = std::fabs(got - want);
        const float tol = rtol * std::fabs(want) + atol;
        const float rel = abs_err / (std::fabs(want) + 1e-6f);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel > max_rel) max_rel = rel;
        if (abs_err > tol) {
            if (n_bad < 8)
                std::printf("  MISMATCH out[%zu]: got=%.5f want=%.5f abs=%.5f tol=%.5f\n",
                            i, got, want, abs_err, tol);
            ++n_bad;
        }
    }

    std::printf("nemotron_mamba_block_bf16 selftest: T=%d H=%d heads=%d head_dim=%d "
                "state=%d groups=%d K=%d d_in_proj=%d conv_dim=%d\n",
                T, H, num_heads, head_dim, state_size, n_groups, conv_kernel,
                d_in_proj, conv_dim);
    std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  bad=%d/%zu\n",
                max_abs, max_rel, n_bad, out_ref.size());
    std::printf("%s\n", n_bad == 0 ? "PASS" : "FAIL");

    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    return n_bad == 0 ? 0 : 1;
}
