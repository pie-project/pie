// Standalone self-test for the sparse (token-dispatched) MoE block
// (forward/moe_sparse). ORACLE = the dense block forward/moe_mlp
// (moe_mlp_block_bf16): both compute the SAME MoE math on identical inputs;
// the sparse path just routes each (token,expert) pair once instead of running
// all E experts on all T tokens. Differences are only bf16 / summation order.
//
// Build / run:
//   /usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_90 -O2 \
//     -I include -I src \
//     -o /tmp/moe_sparse_selftest \
//     src/forward/moe_sparse.cu src/forward/moe_sparse_selftest.cu \
//     src/forward/moe_mlp.cu src/kernels/moe_dispatch.cu \
//     src/ops/grouped_gemm.cu src/ops/gemm.cpp src/kernels/moe.cu \
//     src/kernels/swiglu.cu src/kernels/dtype_cast.cu \
//     -lcublas \
//   && LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/moe_sparse_selftest
//
// Dims: T=8, H=64, I=128, E=4, top_k=2. Random bf16 hidden + weights.
// PASS iff |sparse - dense| <= 0.03*|dense| + 0.05 elementwise.

#include "forward/moe_mlp.cuh"
#include "forward/moe_sparse.cuh"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using pie_cuda_device::forward::moe_mlp_block_bf16;
using pie_cuda_device::forward::moe_sparse_block_bf16;

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
  const int T = 8, H = 64, I = 128, E = 4, K = 2;
  constexpr size_t es = sizeof(__nv_bfloat16);
  std::printf("--- moe_sparse vs dense oracle: T=%d H=%d I=%d E=%d top_k=%d ---\n",
              T, H, I, E, K);

  std::mt19937 rng(20260530u);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  auto to_bf16 = [](float f) { return __float2bfloat16(f); };

  auto make_bf16 = [&](size_t n, float scale) {
    std::vector<__nv_bfloat16> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = to_bf16(dist(rng) * scale);
    return v;
  };

  // Random inputs (shared by both paths). Scale weights down a bit so the
  // GEMM/silu products stay in a sane range for bf16.
  std::vector<__nv_bfloat16> hidden   = make_bf16((size_t)T * H, 1.0f);
  std::vector<__nv_bfloat16> router_w = make_bf16((size_t)E * H, 0.5f);
  std::vector<__nv_bfloat16> wgu      = make_bf16((size_t)E * 2 * I * H, 0.25f);
  std::vector<__nv_bfloat16> wdown    = make_bf16((size_t)E * H * I, 0.25f);

  __nv_bfloat16 *d_hidden=nullptr, *d_router=nullptr, *d_wgu=nullptr, *d_wdown=nullptr;
  __nv_bfloat16 *d_out_dense=nullptr, *d_out_sparse=nullptr;
  CK(cudaMalloc(&d_hidden, hidden.size() * es));
  CK(cudaMalloc(&d_router, router_w.size() * es));
  CK(cudaMalloc(&d_wgu,    wgu.size() * es));
  CK(cudaMalloc(&d_wdown,  wdown.size() * es));
  CK(cudaMalloc(&d_out_dense,  (size_t)T * H * es));
  CK(cudaMalloc(&d_out_sparse, (size_t)T * H * es));
  CK(cudaMemcpy(d_hidden, hidden.data(),   hidden.size() * es,   cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_router, router_w.data(), router_w.size() * es, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_wgu,    wgu.data(),      wgu.size() * es,      cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_wdown,  wdown.data(),    wdown.size() * es,    cudaMemcpyHostToDevice));
  // Poison outputs so a no-op kernel is caught.
  CK(cudaMemset(d_out_dense,  0x7f, (size_t)T * H * es));
  CK(cudaMemset(d_out_sparse, 0x7f, (size_t)T * H * es));

  cublasHandle_t cublas = nullptr;
  if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
    std::printf("cublasCreate failed\n");
    return 1;
  }
  cudaStream_t stream = nullptr;
  CK(cudaStreamCreate(&stream));

  // ORACLE: dense block.
  CK(moe_mlp_block_bf16(cublas, stream, d_hidden, d_router, d_wgu, d_wdown,
                        d_out_dense, T, H, I, E, K));
  // Under test: sparse routed block, identical inputs.
  CK(moe_sparse_block_bf16(cublas, stream, d_hidden, d_router, d_wgu, d_wdown,
                           d_out_sparse, T, H, I, E, K));
  CK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> out_dense((size_t)T * H), out_sparse((size_t)T * H);
  CK(cudaMemcpy(out_dense.data(),  d_out_dense,  (size_t)T * H * es, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out_sparse.data(), d_out_sparse, (size_t)T * H * es, cudaMemcpyDeviceToHost));

  double max_abs = 0.0, max_rel = 0.0;
  bool ok = true;
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      const size_t idx = (size_t)t * H + h;
      const float want = __bfloat162float(out_dense[idx]);   // oracle
      const float got  = __bfloat162float(out_sparse[idx]);
      const float ad = std::fabs(got - want);
      const float rd = (want != 0.0f) ? ad / std::fabs(want) : ad;
      if (ad > max_abs) max_abs = ad;
      if (rd > max_rel) max_rel = rd;
      if (ad > 0.03f * std::fabs(want) + 0.05f) {
        if (ok) {
          std::printf("  FAIL first mismatch at (t=%d,h=%d): sparse=%.5f dense=%.5f "
                      "abs=%.5f rel=%.5f\n", t, h, got, want, ad, rd);
        }
        ok = false;
      }
    }
  }

  std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  (tol = 0.03*|dense| + 0.05)\n",
              max_abs, max_rel);
  std::printf("==== %s ====\n", ok ? "PASS" : "FAIL");

  cublasDestroy(cublas);
  cudaStreamDestroy(stream);
  cudaFree(d_hidden); cudaFree(d_router); cudaFree(d_wgu); cudaFree(d_wdown);
  cudaFree(d_out_dense); cudaFree(d_out_sparse);
  return ok ? 0 : 1;
}
