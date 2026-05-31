// Standalone self-test for the Qwen3 per-head q/k RMSNorm (qk_norm_bf16).
//
// A CPU reference implements the EXACT SAME per-(token,head) RMSNorm over the
// head_dim axis in fp32, with bf16 rounding at the same boundaries the kernel
// rounds at (q/k reads, weight reads, output writes). It validates wiring /
// layout / math, not external ground truth.
//
//   y[d] = x[d] * rsqrt(mean_d(x[d]^2) + eps) * weight[d]
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I /root/pie/driver/cuda_new/device/include \
//     -I /root/pie/driver/cuda_new/device/src \
//     -o /tmp/qknorm_selftest \
//     src/kernels/qk_norm.cu src/kernels/qk_norm_selftest.cu src/kernels/rmsnorm.cu \
//     -lcublas
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/qknorm_selftest

#include "qk_norm.cuh"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::kernels::qk_norm_bf16;

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

// Per (token,head) RMSNorm over head_dim, fp32 with bf16 rounding at read/write.
static void rmsnorm_ref(const std::vector<float>& x, const std::vector<float>& w,
                        std::vector<float>& y, int num_rows, int head_dim,
                        float eps) {
    for (int r = 0; r < num_rows; ++r) {
        const float* xr = x.data() + static_cast<size_t>(r) * head_dim;
        float* yr = y.data() + static_cast<size_t>(r) * head_dim;
        float acc = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            const float v = bf16_rt(xr[d]);
            acc += v * v;
        }
        const float inv_rms = 1.f / std::sqrt(acc / static_cast<float>(head_dim) + eps);
        for (int d = 0; d < head_dim; ++d) {
            const float xv = bf16_rt(xr[d]);
            const float wv = bf16_rt(w[d]);
            yr[d] = bf16_rt(xv * inv_rms * wv);
        }
    }
}

int main() {
    const int T = 4;             // tokens
    const int num_q_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 16;
    const float eps = 1e-6f;

    const int q_rows = T * num_q_heads;   // 16
    const int k_rows = T * num_kv_heads;  // 8
    const size_t q_n = static_cast<size_t>(q_rows) * head_dim;
    const size_t k_n = static_cast<size_t>(k_rows) * head_dim;

    std::mt19937 rng(20260531);
    std::uniform_real_distribution<float> ud(-1.5f, 1.5f);

    // --- Host inputs (bf16-quantized fp32 mirrors) -------------------------
    std::vector<float> q(q_n), k(k_n);
    for (auto& v : q) v = bf16_rt(ud(rng));
    for (auto& v : k) v = bf16_rt(ud(rng));
    std::vector<float> qw(head_dim), kw(head_dim);
    for (auto& v : qw) v = bf16_rt(0.5f + 0.5f * ud(rng));  // gains near 1
    for (auto& v : kw) v = bf16_rt(0.5f + 0.5f * ud(rng));

    // ====================== CPU reference =================================
    std::vector<float> q_ref(q_n), k_ref(k_n);
    rmsnorm_ref(q, qw, q_ref, q_rows, head_dim, eps);
    rmsnorm_ref(k, kw, k_ref, k_rows, head_dim, eps);

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

    __nv_bfloat16* d_q = upload_bf16(q);  // updated in place
    __nv_bfloat16* d_k = upload_bf16(k);  // updated in place
    __nv_bfloat16* d_qw = upload_bf16(qw);
    __nv_bfloat16* d_kw = upload_bf16(kw);

    qk_norm_bf16(d_q, d_k, d_qw, d_kw,
                 T, num_q_heads, num_kv_heads, head_dim, eps, /*stream=*/0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> q_gpu(q_n), k_gpu(k_n);
    CK(cudaMemcpy(q_gpu.data(), d_q, q_n * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(k_gpu.data(), d_k, k_n * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));

    // ====================== Compare =======================================
    float max_abs = 0.f, max_rel = 0.f;
    int n_bad = 0;
    auto cmp = [&](const std::vector<float>& ref,
                   const std::vector<__nv_bfloat16>& gpu, const char* tag) {
        for (size_t i = 0; i < ref.size(); ++i) {
            const float want = ref[i];
            const float got = from_bf16(gpu[i]);
            const float abs_err = std::fabs(got - want);
            const float tol = 0.02f + 0.02f * std::fabs(want);
            const float rel = abs_err / (std::fabs(want) + 1e-6f);
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel > max_rel) max_rel = rel;
            if (abs_err > tol) {
                if (n_bad < 8) {
                    std::printf("  MISMATCH %s[%zu]: got=%.5f want=%.5f abs=%.5f tol=%.5f\n",
                                tag, i, got, want, abs_err, tol);
                }
                ++n_bad;
            }
        }
    };
    cmp(q_ref, q_gpu, "q");
    cmp(k_ref, k_gpu, "k");

    std::printf("qk_norm_bf16 selftest: T=%d q_heads=%d kv_heads=%d head_dim=%d eps=%g\n",
                T, num_q_heads, num_kv_heads, head_dim, eps);
    std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  bad=%d/%zu\n",
                max_abs, max_rel, n_bad, q_n + k_n);
    std::printf("%s\n", n_bad == 0 ? "PASS" : "FAIL");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_qw); cudaFree(d_kw);
    return n_bad == 0 ? 0 : 1;
}
