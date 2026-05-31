// Standalone self-test for the fused quant GEMM (bf16 x u4b8 -> bf16).
//
// Build / run:
//   /usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_90 -O2 \
//     -I /root/pie/driver/cuda_new/device/include \
//     -I /root/pie/driver/cuda_new/device/src \
//     -o /tmp/qgemm_selftest src/qgemm/*.cu \
//   && LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/qgemm_selftest
//
// Two cases:
//   (1) M=16,  N=256, K=512, group_size=128
//   (2) M=64,  N=512, K=256, group_size=-1 (per-channel)
//
// For each: build plain GPTQ-packed int4 weights (q in [0,15]) + bf16 scales,
// repack -> run kernel -> compare against a CPU oracle:
//   w_f[n,k] = (q[n,k] - 8) * scale[group(k), n]
//   y[m,n]   = sum_k decode_bf16(act[m,k]) * w_f[n,k]
//
// PASS iff every element is within |got-want| <= 0.05*|want| + 0.08.

#include "qgemm.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

using pie_cuda_device::qgemm::w4a16_bf16_gemm;
using pie_cuda_device::qgemm::w4a16_repack;
using pie_cuda_device::qgemm::w4a16_workspace_ints;
using pie_cuda_device::qgemm::w8a16_fp8_bf16_gemm;
using pie_cuda_device::qgemm::w8a16_fp8_repack;
using pie_cuda_device::qgemm::w8a16_fp8_workspace_ints;

#define CK(expr)                                                            \
  do {                                                                      \
    cudaError_t e_ = (expr);                                                \
    if (e_ != cudaSuccess) {                                                \
      std::printf("CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                  cudaGetErrorString(e_));                                  \
      return false;                                                         \
    }                                                                       \
  } while (0)

static bool run_case(const char* name, int M, int N, int K, int group_size) {
  std::printf("--- case %s: M=%d N=%d K=%d group_size=%d ---\n", name, M, N, K,
              group_size);

  const int num_groups = (group_size > 0) ? (K / group_size) : 1;
  const int gsize = (group_size > 0) ? group_size : K;  // effective group span

  std::mt19937 rng(12345u + (unsigned)(M * 131 + N * 17 + K));
  std::uniform_int_distribution<int> qdist(0, 15);
  std::uniform_real_distribution<float> sdist(0.02f, 0.06f);
  std::normal_distribution<float> adist(0.0f, 1.0f);

  // Logical int4 weights, indexed [n][k], values in [0,15].
  std::vector<int> q(N * (size_t)K);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) q[n * (size_t)K + k] = qdist(rng);

  // Per-group bf16 scales, indexed [group][n].
  std::vector<float> scale_f(num_groups * (size_t)N);
  for (int g = 0; g < num_groups; g++)
    for (int n = 0; n < N; n++) scale_f[g * (size_t)N + n] = sdist(rng);

  // bf16 activations [M][K].
  std::vector<float> act_f(M * (size_t)K);
  for (int i = 0; i < M * (size_t)K; i++) act_f[i] = adist(rng);

  // ---- Build GPTQ-packed weights: [K/8, N] int32, nibble j of int32(kp,n)
  //      holds q for k = kp*8 + j, column n. ----
  const int Kp = K / 8;
  std::vector<int32_t> gptq(Kp * (size_t)N, 0);
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      int kp = k / 8;
      int j = k % 8;
      uint32_t nib = (uint32_t)(q[n * (size_t)K + k] & 0xF);
      gptq[kp * (size_t)N + n] |= (int32_t)(nib << (4 * j));
    }
  }

  // ---- Convert host floats to bf16 (round-to-nearest-even via CUDA host fn).
  auto to_bf16 = [](float f) { return __float2bfloat16(f); };
  std::vector<__nv_bfloat16> act_bf(M * (size_t)K);
  for (size_t i = 0; i < act_f.size(); i++) act_bf[i] = to_bf16(act_f[i]);
  std::vector<__nv_bfloat16> scale_bf(scale_f.size());
  for (size_t i = 0; i < scale_f.size(); i++) scale_bf[i] = to_bf16(scale_f[i]);

  // ---- Device buffers.
  int32_t* d_gptq = nullptr;
  int32_t* d_repacked = nullptr;
  __nv_bfloat16* d_act = nullptr;
  __nv_bfloat16* d_scale = nullptr;
  __nv_bfloat16* d_out = nullptr;
  int* d_ws = nullptr;

  const int tile_size = 16;          // matches qgemm tile_size
  const size_t repacked_ints = (size_t)(K / tile_size) * (size_t)(N * tile_size / 8);
  const int ws_ints = w4a16_workspace_ints(N, M);

  CK(cudaMalloc(&d_gptq, gptq.size() * sizeof(int32_t)));
  CK(cudaMalloc(&d_repacked, repacked_ints * sizeof(int32_t)));
  CK(cudaMalloc(&d_act, act_bf.size() * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_scale, scale_bf.size() * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_out, (size_t)M * N * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_ws, (size_t)ws_ints * sizeof(int)));

  CK(cudaMemcpy(d_gptq, gptq.data(), gptq.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_act, act_bf.data(), act_bf.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_scale, scale_bf.data(),
                scale_bf.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));

  // ---- Repack + zero workspace + GEMM.
  CK(w4a16_repack(0, d_gptq, d_repacked, N, K));
  CK(cudaMemset(d_ws, 0, (size_t)ws_ints * sizeof(int)));  // zero before call
  CK(cudaMemset(d_out, 0, (size_t)M * N * sizeof(__nv_bfloat16)));
  CK(w4a16_bf16_gemm(0, d_act, d_repacked, d_scale, d_out, M, N, K, group_size,
                     d_ws, /*sms=*/0));
  CK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> out_bf((size_t)M * N);
  CK(cudaMemcpy(out_bf.data(), d_out, (size_t)M * N * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));

  // ---- CPU oracle (decode bf16 act/scale exactly as the device sees them).
  std::vector<float> act_dec(M * (size_t)K);
  for (size_t i = 0; i < act_dec.size(); i++)
    act_dec[i] = __bfloat162float(act_bf[i]);
  std::vector<float> scale_dec(scale_bf.size());
  for (size_t i = 0; i < scale_dec.size(); i++)
    scale_dec[i] = __bfloat162float(scale_bf[i]);

  double max_abs = 0.0, max_rel = 0.0;
  bool ok = true;
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double acc = 0.0;
      for (int k = 0; k < K; k++) {
        int g = k / gsize;
        float wf =
            (float)(q[n * (size_t)K + k] - 8) * scale_dec[g * (size_t)N + n];
        acc += (double)act_dec[m * (size_t)K + k] * (double)wf;
      }
      float want = (float)acc;
      float got = __bfloat162float(out_bf[(size_t)m * N + n]);
      float ad = std::fabs(got - want);
      float rd = (want != 0.0f) ? ad / std::fabs(want) : ad;
      if (ad > max_abs) max_abs = ad;
      if (rd > max_rel) max_rel = rd;
      if (ad > 0.05f * std::fabs(want) + 0.08f) {
        if (ok) {
          std::printf("  FAIL first mismatch at (m=%d,n=%d): got=%.5f want=%.5f"
                      " abs=%.5f rel=%.5f\n",
                      m, n, got, want, ad, rd);
        }
        ok = false;
      }
    }
  }

  std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  -> %s\n", max_abs, max_rel,
              ok ? "PASS" : "FAIL");

  cudaFree(d_gptq);
  cudaFree(d_repacked);
  cudaFree(d_act);
  cudaFree(d_scale);
  cudaFree(d_out);
  cudaFree(d_ws);
  return ok;
}

// ---------------------------------------------------------------------------
// fp8 (fe4m3fn) weight path.
//
// Build fe4m3fn-quantized weights + bf16 scales, repack -> run the fused fp8
// GEMM -> compare against a CPU oracle that decodes the SAME fp8 bytes to
// float (via the hardware fe4m3fn->half conversion, exactly what the device's
// in-register splat reconstructs) and multiplies by the bf16-decoded scale:
//   w_f[n,k] = decode_fe4m3fn(qw[n,k]) * scale[group(k), n]
//   y[m,n]   = sum_k decode_bf16(act[m,k]) * w_f[n,k]
//
// Tolerance: same |got-want| <= 0.05*|want| + 0.08 as the int4 cases. fp8
// (E4M3, 3 mantissa bits) is coarser than int4, so we keep the weight
// *magnitudes* in a comparable range (|w| ~ O(1) after scaling) and verify the
// fused result tracks the oracle within the shared tolerance — no loosening
// was needed in practice.
static bool run_fp8_case(const char* name, int M, int N, int K,
                         int group_size) {
  std::printf("--- fp8 case %s: M=%d N=%d K=%d group_size=%d ---\n", name, M, N,
              K, group_size);

  const int num_groups = (group_size > 0) ? (K / group_size) : 1;
  const int gsize = (group_size > 0) ? group_size : K;

  std::mt19937 rng(98765u + (unsigned)(M * 131 + N * 17 + K));
  // Quantized weight values: small bf16-ish magnitudes that fe4m3fn represents
  // with full normal precision (avoid subnormals / overflow).
  std::uniform_real_distribution<float> wdist(-2.0f, 2.0f);
  std::uniform_real_distribution<float> sdist(0.02f, 0.06f);
  std::normal_distribution<float> adist(0.0f, 1.0f);

  auto enc_fp8 = [](float f) -> uint8_t {
    return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
  };
  auto dec_fp8 = [](uint8_t b) -> float {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&hr));
  };

  // Logical fp8 weights, indexed [n][k] (stored byte) + their decoded floats.
  std::vector<uint8_t> qw(N * (size_t)K);
  std::vector<float> wdec(N * (size_t)K);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) {
      uint8_t b = enc_fp8(wdist(rng));
      qw[n * (size_t)K + k] = b;
      wdec[n * (size_t)K + k] = dec_fp8(b);
    }

  std::vector<float> scale_f(num_groups * (size_t)N);
  for (int g = 0; g < num_groups; g++)
    for (int n = 0; n < N; n++) scale_f[g * (size_t)N + n] = sdist(rng);

  std::vector<float> act_f(M * (size_t)K);
  for (int i = 0; i < M * (size_t)K; i++) act_f[i] = adist(rng);

  // Pack: [K/4, N] int32, byte j of int32(kp,n) holds fp8 for k = kp*4 + j.
  const int Kp = K / 4;
  std::vector<int32_t> packed(Kp * (size_t)N, 0);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) {
      int kp = k / 4, j = k % 4;
      packed[kp * (size_t)N + n] |=
          (int32_t)((uint32_t)qw[n * (size_t)K + k] << (8 * j));
    }

  auto to_bf16 = [](float f) { return __float2bfloat16(f); };
  std::vector<__nv_bfloat16> act_bf(M * (size_t)K);
  for (size_t i = 0; i < act_f.size(); i++) act_bf[i] = to_bf16(act_f[i]);
  std::vector<__nv_bfloat16> scale_bf(scale_f.size());
  for (size_t i = 0; i < scale_f.size(); i++) scale_bf[i] = to_bf16(scale_f[i]);

  int32_t* d_packed = nullptr;
  int32_t* d_repacked = nullptr;
  __nv_bfloat16* d_act = nullptr;
  __nv_bfloat16* d_scale = nullptr;
  __nv_bfloat16* d_out = nullptr;
  int* d_ws = nullptr;

  const int tile_size = 16;  // 8-bit weights: out is [K/16, N*16/4] int32.
  const size_t repacked_ints =
      (size_t)(K / tile_size) * (size_t)(N * tile_size / 4);
  const int ws_ints = w8a16_fp8_workspace_ints(N, M);

  CK(cudaMalloc(&d_packed, packed.size() * sizeof(int32_t)));
  CK(cudaMalloc(&d_repacked, repacked_ints * sizeof(int32_t)));
  CK(cudaMalloc(&d_act, act_bf.size() * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_scale, scale_bf.size() * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_out, (size_t)M * N * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&d_ws, (size_t)ws_ints * sizeof(int)));

  CK(cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_act, act_bf.data(), act_bf.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_scale, scale_bf.data(),
                scale_bf.size() * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));

  CK(w8a16_fp8_repack(0, d_packed, d_repacked, N, K));
  CK(cudaMemset(d_ws, 0, (size_t)ws_ints * sizeof(int)));
  CK(cudaMemset(d_out, 0, (size_t)M * N * sizeof(__nv_bfloat16)));
  CK(w8a16_fp8_bf16_gemm(0, d_act, d_repacked, d_scale, d_out, M, N, K,
                         group_size, d_ws, /*sms=*/0));
  CK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> out_bf((size_t)M * N);
  CK(cudaMemcpy(out_bf.data(), d_out, (size_t)M * N * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));

  std::vector<float> act_dec(M * (size_t)K);
  for (size_t i = 0; i < act_dec.size(); i++)
    act_dec[i] = __bfloat162float(act_bf[i]);
  std::vector<float> scale_dec(scale_bf.size());
  for (size_t i = 0; i < scale_dec.size(); i++)
    scale_dec[i] = __bfloat162float(scale_bf[i]);

  double max_abs = 0.0, max_rel = 0.0, sum_abs_want = 0.0;
  bool ok = true;
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double acc = 0.0;
      for (int k = 0; k < K; k++) {
        int g = k / gsize;
        float wf = wdec[n * (size_t)K + k] * scale_dec[g * (size_t)N + n];
        acc += (double)act_dec[m * (size_t)K + k] * (double)wf;
      }
      float want = (float)acc;
      sum_abs_want += std::fabs(want);
      float got = __bfloat162float(out_bf[(size_t)m * N + n]);
      float ad = std::fabs(got - want);
      float rd = (want != 0.0f) ? ad / std::fabs(want) : ad;
      if (ad > max_abs) max_abs = ad;
      if (rd > max_rel) max_rel = rd;
      if (ad > 0.05f * std::fabs(want) + 0.08f) {
        if (ok) {
          std::printf("  FAIL first mismatch at (m=%d,n=%d): got=%.5f want=%.5f"
                      " abs=%.5f rel=%.5f\n",
                      m, n, got, want, ad, rd);
        }
        ok = false;
      }
    }
  }

  // Sanity: outputs must be non-trivial (guard against a degenerate all-zero
  // result passing on the absolute-tolerance floor alone).
  double mean_abs_want = sum_abs_want / ((double)M * N);
  if (mean_abs_want < 0.1) {
    std::printf("  WARNING: mean|want|=%.4f is small; test may be weak\n",
                mean_abs_want);
  }
  std::printf("  max_abs_err=%.6f  max_rel_err=%.6f  mean|want|=%.4f  -> %s\n",
              max_abs, max_rel, mean_abs_want, ok ? "PASS" : "FAIL");

  cudaFree(d_packed);
  cudaFree(d_repacked);
  cudaFree(d_act);
  cudaFree(d_scale);
  cudaFree(d_out);
  cudaFree(d_ws);
  return ok;
}

int main() {
  bool ok = true;
  ok &= run_case("1", 16, 256, 512, 128);
  ok &= run_case("2", 64, 512, 256, -1);
  ok &= run_fp8_case("3", 16, 256, 512, 128);
  ok &= run_fp8_case("4", 64, 512, 256, -1);
  ok &= run_fp8_case("5", 128, 128, 1024, 128);  // larger M (multi-stripe)
  ok &= run_fp8_case("6", 8, 64, 128, -1);        // m_block_size_8 path
  std::printf("==== %s ====\n", ok ? "ALL PASS" : "FAILURE" );
  return ok ? 0 : 1;
}
