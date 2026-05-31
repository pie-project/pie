// Standalone self-test for the broadcast bias-add (add_bias_bf16).
//
// A CPU reference implements the EXACT SAME per-element broadcast add over a
// [num_tokens, dim] bf16 row-major tensor, in fp32 with bf16 rounding at the
// same boundaries the kernel rounds at (x reads, bias reads, output writes):
//
//   x[t*dim + i] = round_bf16(decode(x[t*dim + i]) + decode(bias[i]))
//
// It validates wiring / layout / math (the bias broadcasts across rows), not
// external ground truth.
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I /root/pie/driver/cuda_new/device/include \
//     -I /root/pie/driver/cuda_new/device/src \
//     -o /tmp/addbias_selftest \
//     src/kernels/add_bias.cu src/kernels/add_bias_selftest.cu
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/addbias_selftest

#include "add_bias.cuh"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using pie_cuda_device::kernels::add_bias_bf16;

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

int main() {
    const int num_tokens = 5;
    const int dim = 64;
    const size_t n = static_cast<size_t>(num_tokens) * dim;

    std::mt19937 rng(20260531);
    std::uniform_real_distribution<float> ud(-1.5f, 1.5f);

    // --- Host inputs (bf16-quantized fp32 mirrors) -------------------------
    std::vector<float> x(n), bias(dim);
    for (auto& v : x) v = bf16_rt(ud(rng));
    for (auto& v : bias) v = bf16_rt(ud(rng));

    // ====================== CPU reference =================================
    std::vector<float> ref(n);
    for (int t = 0; t < num_tokens; ++t) {
        for (int i = 0; i < dim; ++i) {
            const size_t idx = static_cast<size_t>(t) * dim + i;
            ref[idx] = bf16_rt(bf16_rt(x[idx]) + bf16_rt(bias[i]));
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

    __nv_bfloat16* d_x = upload_bf16(x);  // updated in place
    __nv_bfloat16* d_bias = upload_bf16(bias);

    add_bias_bf16(d_x, d_bias, num_tokens, dim, /*stream=*/0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> x_gpu(n);
    CK(cudaMemcpy(x_gpu.data(), d_x, n * sizeof(__nv_bfloat16),
                  cudaMemcpyDeviceToHost));

    // ====================== Compare =======================================
    float max_abs = 0.f, max_rel = 0.f;
    int n_bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const float want = ref[i];
        const float got = from_bf16(x_gpu[i]);
        const float abs_err = std::fabs(got - want);
        const float tol = 0.02f + 0.02f * std::fabs(want);
        const float rel = abs_err / (std::fabs(want) + 1e-6f);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel > max_rel) max_rel = rel;
        if (abs_err > tol) {
            if (n_bad < 8) {
                std::printf("  MISMATCH x[%zu]: got=%.5f want=%.5f abs=%.5f tol=%.5f\n",
                            i, got, want, abs_err, tol);
            }
            ++n_bad;
        }
    }

    std::printf("add_bias_bf16 selftest: num_tokens=%d dim=%d\n", num_tokens, dim);
    std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  bad=%d/%zu\n",
                max_abs, max_rel, n_bad, n);
    std::printf("%s\n", n_bad == 0 ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_bias);
    return n_bad == 0 ? 0 : 1;
}
