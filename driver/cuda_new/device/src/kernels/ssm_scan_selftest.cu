// Standalone self-test for the Mamba-2 / SSD selective-scan recurrence
// (ssm_selective_scan_bf16, lifted from nemotron_h.cu's warp kernel).
//
// Self-consistency test: a CPU reference implements the EXACT SAME per-channel
// recurrence sequentially in fp32, with bf16 rounding at the same boundaries the
// kernel rounds at (state read/write, x/B/C reads, dt raw read, y write). It
// validates wiring / layout / recurrence correctness, not external ground truth.
//
//     dt        = max( softplus(dt_raw + dt_bias_h), time_step_min )
//     dA        = exp(dt * A_h)
//     state'    = state * dA + (dt * B[s]) * x          (per dim, per state s)
//     y[dim]    = sum_s state' * C[s]  +  D_h * x[dim]
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 -I include -I src \
//     -o /tmp/ssm_selftest src/kernels/ssm_scan.cu src/kernels/ssm_scan_selftest.cu
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/ssm_selftest

#include "ssm_scan.cuh"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::kernels::ssm_selective_scan_bf16;

// bf16 round-trip (round to nearest even), matching __float2bfloat16.
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

int main() {
    // Tiny dims. Match the kernel's grouped-B/C contract.
    const int R = 1;            // one request
    const int seq_len = 8;      // tokens
    const int num_heads = 4;
    const int head_dim = 4;     // channels = num_heads*head_dim = 16
    const int state_size = 8;
    const int n_groups = 2;
    const int intermediate = num_heads * head_dim;            // 16
    const int conv_dim = intermediate + 2 * n_groups * state_size;  // 16 + 32 = 48
    const int N = seq_len;
    const int num_slots = 1;
    const float time_step_min = 1e-3f;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    // --- Host inputs (bf16-quantized fp32 mirrors) -------------------------
    std::vector<float> conv_out(N * conv_dim);
    for (auto& v : conv_out) v = bf16_rt(ud(rng));

    std::vector<float> dt_raw(N * num_heads);
    for (auto& v : dt_raw) v = bf16_rt(ud(rng));

    // fp32 params (host precomputes A = -exp(A_log)).
    std::vector<float> A(num_heads), D(num_heads), dt_bias(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        A[h] = -std::exp(0.5f * ud(rng));   // negative, as -exp(A_log)
        D[h] = ud(rng);
        dt_bias[h] = 0.3f * ud(rng);
    }

    // Initial recurrent state (bf16-quantized).
    std::vector<float> state0(num_slots * num_heads * head_dim * state_size);
    for (auto& v : state0) v = bf16_rt(0.3f * ud(rng));

    // qo_indptr: single request covering all tokens.
    std::vector<std::uint32_t> qo_indptr = {0u, static_cast<std::uint32_t>(N)};
    std::vector<std::int32_t> slot_ids = {0};

    // ====================== CPU reference (fp32, sequential) ==============
    std::vector<float> y_ref(N * intermediate, 0.f);
    std::vector<float> state = state0;  // mutated in place
    const int heads_per_group = num_heads / n_groups;
    for (int local_t = 0; local_t < N; ++local_t) {
        const int row = local_t;  // request 0, t0=0
        for (int h = 0; h < num_heads; ++h) {
            const int group = h / heads_per_group;
            const float A_h = A[h];
            const float D_h = D[h];
            const float dt_b = dt_bias[h];
            const int bc_base = intermediate + group * state_size;
            const int c_base = intermediate + n_groups * state_size +
                               group * state_size;
            // dt / dA computed inline (matches dt_precomputed == nullptr path).
            const float dt = std::fmax(
                softplus_ref(bf16_rt(dt_raw[row * num_heads + h]) + dt_b),
                time_step_min);
            const float dA = std::exp(dt * A_h);
            const int slot = 0;
            float* st = state.data() +
                        (static_cast<long long>(slot) * num_heads * head_dim *
                             state_size +
                         static_cast<long long>(h) * head_dim * state_size);
            for (int dim = 0; dim < head_dim; ++dim) {
                const float x = bf16_rt(conv_out[row * conv_dim + h * head_dim + dim]);
                float sum = 0.f;
                for (int s = 0; s < state_size; ++s) {
                    const float b = bf16_rt(conv_out[row * conv_dim + bc_base + s]);
                    const float c = bf16_rt(conv_out[row * conv_dim + c_base + s]);
                    const int idx = dim * state_size + s;
                    const float old = bf16_rt(st[idx]);
                    const float next = old * dA + (dt * b) * x;
                    st[idx] = bf16_rt(next);   // state persisted as bf16
                    sum += next * c;
                }
                y_ref[row * intermediate + h * head_dim + dim] =
                    bf16_rt(sum + D_h * x);
            }
        }
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
    auto upload_f32 = [](const std::vector<float>& host) {
        float* d = nullptr;
        cudaMalloc(&d, host.size() * sizeof(float));
        cudaMemcpy(d, host.data(), host.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        return d;
    };

    __nv_bfloat16* d_conv = upload_bf16(conv_out);
    __nv_bfloat16* d_dt = upload_bf16(dt_raw);
    float* d_A = upload_f32(A);
    float* d_D = upload_f32(D);
    float* d_dt_bias = upload_f32(dt_bias);
    __nv_bfloat16* d_state = upload_bf16(state0);  // gets updated in place
    __nv_bfloat16* d_y = nullptr;
    CK(cudaMalloc(&d_y, static_cast<size_t>(N) * intermediate * sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_y, 0, static_cast<size_t>(N) * intermediate * sizeof(__nv_bfloat16)));

    std::uint32_t* d_qo = nullptr;
    CK(cudaMalloc(&d_qo, qo_indptr.size() * sizeof(std::uint32_t)));
    CK(cudaMemcpy(d_qo, qo_indptr.data(), qo_indptr.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    std::int32_t* d_slot = nullptr;
    CK(cudaMalloc(&d_slot, slot_ids.size() * sizeof(std::int32_t)));
    CK(cudaMemcpy(d_slot, slot_ids.data(), slot_ids.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));

    ssm_selective_scan_bf16(
        d_conv, d_dt, d_A, d_D, d_dt_bias,
        /*dt_precomputed=*/nullptr, /*dA_precomputed=*/nullptr,
        d_state, d_slot, d_qo, d_y,
        R, num_heads, head_dim, state_size, n_groups,
        conv_dim, intermediate, time_step_min,
        /*stream=*/0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> y_gpu_bf(static_cast<size_t>(N) * intermediate);
    CK(cudaMemcpy(y_gpu_bf.data(), d_y,
                  y_gpu_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // ====================== Compare =======================================
    float max_abs = 0.f, max_rel = 0.f;
    int n_bad = 0;
    const float atol = 0.05f, rtol = 0.03f;
    for (size_t i = 0; i < y_ref.size(); ++i) {
        const float want = y_ref[i];
        const float got = from_bf16(y_gpu_bf[i]);
        const float abs_err = std::fabs(got - want);
        const float tol = rtol * std::fabs(want) + atol;
        const float rel = abs_err / (std::fabs(want) + 1e-6f);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel > max_rel) max_rel = rel;
        if (abs_err > tol) {
            if (n_bad < 8) {
                std::printf("  MISMATCH y[%zu]: got=%.5f want=%.5f abs=%.5f tol=%.5f\n",
                            i, got, want, abs_err, tol);
            }
            ++n_bad;
        }
    }

    std::printf("ssm_selective_scan_bf16 selftest: N=%d heads=%d head_dim=%d "
                "state=%d groups=%d\n",
                N, num_heads, head_dim, state_size, n_groups);
    std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  bad=%d/%zu\n",
                max_abs, max_rel, n_bad, y_ref.size());
    std::printf("%s\n", n_bad == 0 ? "PASS" : "FAIL");

    cudaFree(d_conv); cudaFree(d_dt); cudaFree(d_A); cudaFree(d_D);
    cudaFree(d_dt_bias); cudaFree(d_state); cudaFree(d_y);
    cudaFree(d_qo); cudaFree(d_slot);
    return n_bad == 0 ? 0 : 1;
}
