// Standalone self-test for the grouped per-expert bf16 GEMM (ops/grouped_gemm).
//
// Build / run:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I include -I src \
//     -o /tmp/ggemm_selftest \
//     src/ops/grouped_gemm.cu src/ops/grouped_gemm_selftest.cu src/ops/gemm.cpp \
//     -lcublas \
//   && LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/ggemm_selftest
//
// Shape: E=3 experts, K=32, N=16, group sizes [5, 0, 7] -> total_rows=12.
//   The MIDDLE group is EMPTY (M_e=0) to exercise the skip path.
// Random bf16 x [total_rows, K] (grouped by expert) and W [E, N, K]. Run the
// grouped GEMM, compare against a CPU oracle decoding bf16 exactly as the GPU:
//   for row r owned by expert e:  y[r,n] = sum_k decode(x[r,k]) * decode(W[e,n,k])
// PASS iff every element is within |got-want| <= 0.05*|want| + 0.1.

#include "ops/grouped_gemm.cuh"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using pie_cuda_device::ops::grouped_gemm_bf16;

#define CK(expr)                                                              \
  do {                                                                        \
    cudaError_t e_ = (expr);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      std::printf("CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,  \
                  cudaGetErrorString(e_));                                    \
      return 1;                                                               \
    }                                                                         \
  } while (0)

int main() {
  const int E = 3;
  const int K = 32;
  const int N = 16;
  const int group_sizes[E] = {5, 0, 7};  // middle group is empty

  // Host prefix-sum offsets [E+1]; expert e owns rows [off[e], off[e+1]).
  std::vector<int32_t> off(E + 1, 0);
  for (int e = 0; e < E; ++e) off[e + 1] = off[e] + group_sizes[e];
  const int total_rows = off[E];
  std::printf("--- grouped_gemm: E=%d K=%d N=%d groups=[%d,%d,%d] total_rows=%d ---\n",
              E, K, N, group_sizes[0], group_sizes[1], group_sizes[2], total_rows);
  std::printf("    (expert 1 is empty -> exercises the M_e==0 skip)\n");

  std::mt19937 rng(20260530u);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  auto to_bf16 = [](float f) { return __float2bfloat16(f); };

  // x [total_rows, K] bf16, grouped by expert (rows already in expert order).
  std::vector<float> x_f((size_t)total_rows * K);
  for (auto& v : x_f) v = dist(rng);
  std::vector<__nv_bfloat16> x_bf(x_f.size());
  for (size_t i = 0; i < x_f.size(); ++i) x_bf[i] = to_bf16(x_f[i]);

  // W [E, N, K] bf16 contiguous per expert.
  std::vector<float> w_f((size_t)E * N * K);
  for (auto& v : w_f) v = dist(rng);
  std::vector<__nv_bfloat16> w_bf(w_f.size());
  for (size_t i = 0; i < w_f.size(); ++i) w_bf[i] = to_bf16(w_f[i]);

  // Device buffers.
  __nv_bfloat16* d_x = nullptr;
  __nv_bfloat16* d_w = nullptr;
  __nv_bfloat16* d_y = nullptr;
  CK(cudaMalloc(&d_x, x_bf.size() * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_w, w_bf.size() * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_y, (size_t)total_rows * N * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(d_x, x_bf.data(), x_bf.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_w, w_bf.data(), w_bf.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
  // Poison the output so a no-op kernel would be caught.
  CK(cudaMemset(d_y, 0x7f, (size_t)total_rows * N * sizeof(__nv_bfloat16)));

  cublasHandle_t cublas = nullptr;
  if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
    std::printf("cublasCreate failed\n");
    return 1;
  }

  // expert_offsets is a HOST pointer (see grouped_gemm.cuh): pass off.data().
  CK(grouped_gemm_bf16(cublas, /*stream=*/0, d_x, d_w, off.data(), d_y,
                       total_rows, E, N, K));
  CK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> y_bf((size_t)total_rows * N);
  CK(cudaMemcpy(y_bf.data(), d_y, (size_t)total_rows * N * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));

  // CPU oracle, decoding bf16 exactly as the device sees it.
  std::vector<float> x_dec(x_bf.size());
  for (size_t i = 0; i < x_bf.size(); ++i) x_dec[i] = __bfloat162float(x_bf[i]);
  std::vector<float> w_dec(w_bf.size());
  for (size_t i = 0; i < w_bf.size(); ++i) w_dec[i] = __bfloat162float(w_bf[i]);

  double max_abs = 0.0, max_rel = 0.0;
  bool ok = true;
  for (int e = 0; e < E; ++e) {
    for (int r = off[e]; r < off[e + 1]; ++r) {
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += (double)x_dec[(size_t)r * K + k] *
                 (double)w_dec[((size_t)e * N + n) * K + k];
        }
        float want = (float)acc;
        float got = __bfloat162float(y_bf[(size_t)r * N + n]);
        float ad = std::fabs(got - want);
        float rd = (want != 0.0f) ? ad / std::fabs(want) : ad;
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (ad > 0.05f * std::fabs(want) + 0.1f) {
          if (ok) {
            std::printf("  FAIL first mismatch at (e=%d,r=%d,n=%d): got=%.5f "
                        "want=%.5f abs=%.5f rel=%.5f\n",
                        e, r, n, got, want, ad, rd);
          }
          ok = false;
        }
      }
    }
  }

  std::printf("  max_abs_err=%.6f  max_rel_err=%.6f\n", max_abs, max_rel);
  std::printf("==== %s ==== (empty group %d skipped, %d non-empty experts run)\n",
              ok ? "PASS" : "FAIL", /*empty=*/1, /*non-empty=*/2);

  cublasDestroy(cublas);
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_y);
  return ok ? 0 : 1;
}
