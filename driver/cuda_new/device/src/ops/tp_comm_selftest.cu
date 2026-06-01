// Standalone self-test for the TP NCCL communicator wrapper (ops/tp_comm).
//
// Build / run (NCCL present):
//   /usr/local/cuda/bin/nvcc -arch=sm_90 -std=c++20 -O2 \
//     -I include -I src \
//     -o /tmp/tpcomm_selftest \
//     src/ops/tp_comm.cu src/ops/tp_comm_selftest.cu \
//     -lnccl \
//   && LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/tpcomm_selftest
//
// Build / run (NCCL absent): drop `-lnccl`; the wrapper compiles to the
// single-rank identity stub and the test runs only its world_size==1 checks.
//
// Tests:
//   1. Single process, single GPU, world_size=1: init a 1-rank comm via the
//      unique-id path, all-reduce a bf16 buffer, assert it is UNCHANGED
//      (sum over one rank is the identity). Same for fp32.
//   2. If 2+ GPUs are visible AND NCCL is present: spin 2 ranks across 2
//      threads, init each comm with ncclCommInitAll over the shared id minted
//      by rank 0 + broadcast (here just shared in-process), all-reduce a bf16
//      buffer where rank r holds value (r+1), assert the result == sum over
//      ranks. With 2 ranks holding {1,2} the reduced value is 3 on both.
//      (The task's "2x input" is the special case where every rank holds the
//      same input; we use distinct per-rank values to also catch a no-op.)

#include "ops/tp_comm.cuh"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using pie_cuda_device::ops::TpComm;
using pie_cuda_device::ops::tp_comm_init;
using pie_cuda_device::ops::tp_comm_get_unique_id;
using pie_cuda_device::ops::tp_all_reduce_bf16;
using pie_cuda_device::ops::tp_all_reduce_fp32;
using pie_cuda_device::ops::tp_comm_destroy;

#define CK(expr)                                                              \
  do {                                                                        \
    cudaError_t e_ = (expr);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      std::printf("CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,  \
                  cudaGetErrorString(e_));                                    \
      return false;                                                           \
    }                                                                         \
  } while (0)

// -------- test 1: single-rank identity (always runs) ------------------------
static bool test_single_rank() {
  std::printf("--- test 1: world_size=1 identity all-reduce ---\n");

  const int N = 1024;
  std::vector<__nv_bfloat16> h_in(N);
  for (int i = 0; i < N; ++i) {
    h_in[i] = __float2bfloat16(0.5f + 0.001f * static_cast<float>(i));
  }

  __nv_bfloat16* d_buf = nullptr;
  CK(cudaMalloc(&d_buf, N * sizeof(__nv_bfloat16)));
  CK(cudaMemcpy(d_buf, h_in.data(), N * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));

  // unique-id path (id may be ignored for world_size==1, but exercise it).
  ncclUniqueId id;
  cudaError_t id_rc = tp_comm_get_unique_id(&id);
  const bool have_nccl = (id_rc == cudaSuccess);
  std::printf("    tp_comm_get_unique_id -> %s\n",
              have_nccl ? "ok (NCCL present)" : "not-supported (NCCL absent)");

  TpComm* c = tp_comm_init(/*rank=*/0, /*world_size=*/1,
                           have_nccl ? &id : nullptr);
  if (c == nullptr) {
    std::printf("    FAIL: tp_comm_init(world_size=1) returned nullptr\n");
    cudaFree(d_buf);
    return false;
  }

  CK(tp_all_reduce_bf16(c, d_buf, N, /*stream=*/nullptr));
  CK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_out(N);
  CK(cudaMemcpy(h_out.data(), d_buf, N * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < N; ++i) {
    float want = __bfloat162float(h_in[i]);
    float got = __bfloat162float(h_out[i]);
    if (got != want) {
      if (ok) {
        std::printf("    FAIL bf16 mismatch at %d: got=%.6f want=%.6f\n", i,
                    got, want);
      }
      ok = false;
    }
  }

  // fp32 variant.
  const int Nf = 256;
  std::vector<float> hf_in(Nf);
  for (int i = 0; i < Nf; ++i) hf_in[i] = -3.0f + 0.01f * static_cast<float>(i);
  float* d_f = nullptr;
  CK(cudaMalloc(&d_f, Nf * sizeof(float)));
  CK(cudaMemcpy(d_f, hf_in.data(), Nf * sizeof(float), cudaMemcpyHostToDevice));
  CK(tp_all_reduce_fp32(c, d_f, Nf, /*stream=*/nullptr));
  CK(cudaDeviceSynchronize());
  std::vector<float> hf_out(Nf);
  CK(cudaMemcpy(hf_out.data(), d_f, Nf * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < Nf; ++i) {
    if (hf_out[i] != hf_in[i]) {
      if (ok) {
        std::printf("    FAIL fp32 mismatch at %d: got=%.6f want=%.6f\n", i,
                    hf_out[i], hf_in[i]);
      }
      ok = false;
    }
  }

  tp_comm_destroy(c);
  cudaFree(d_buf);
  cudaFree(d_f);
  std::printf("    %s: buffer unchanged after 1-rank all-reduce\n",
              ok ? "ok" : "BAD");
  return ok;
}

// -------- test 2: 2-rank sum all-reduce (only with 2+ GPUs + NCCL) ----------
#if PIE_CUDA_DEVICE_HAS_NCCL
static bool test_two_rank(int n_gpus) {
  std::printf("--- test 2: world_size=2 sum all-reduce across %d-GPU box ---\n",
              n_gpus);

  const int N = 4096;
  const int world = 2;

  // Rank 0 mints the id; both ranks share it (in a real deployment the Rust
  // control plane broadcasts these 128 bytes out-of-band — here threads share
  // the same stack object, which is the same raw-bytes contract).
  ncclUniqueId id;
  if (tp_comm_get_unique_id(&id) != cudaSuccess) {
    std::printf("    SKIP: could not mint unique id\n");
    return true;  // not a failure of the wrapper
  }

  std::vector<int> results(world, 0);  // 1 = ok, -1 = fail
  auto rank_fn = [&](int rank) {
    if (cudaSetDevice(rank) != cudaSuccess) {
      results[rank] = -1;
      return;
    }
    TpComm* c = tp_comm_init(rank, world, &id);
    if (c == nullptr) {
      results[rank] = -1;
      return;
    }
    // Rank r fills its buffer with value (r+1): {1.0, 2.0}. Sum = 3.0 on all.
    const float my_val = static_cast<float>(rank + 1);
    std::vector<__nv_bfloat16> h(N, __float2bfloat16(my_val));
    __nv_bfloat16* d = nullptr;
    if (cudaMalloc(&d, N * sizeof(__nv_bfloat16)) != cudaSuccess) {
      tp_comm_destroy(c);
      results[rank] = -1;
      return;
    }
    cudaMemcpy(d, h.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    if (tp_all_reduce_bf16(c, d, N, /*stream=*/nullptr) != cudaSuccess ||
        cudaStreamSynchronize(c->stream) != cudaSuccess) {
      cudaFree(d);
      tp_comm_destroy(c);
      results[rank] = -1;
      return;
    }

    std::vector<__nv_bfloat16> out(N);
    cudaMemcpy(out.data(), d, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    const float want = 3.0f;  // 1 + 2
    bool ok = true;
    for (int i = 0; i < N && ok; ++i) {
      if (std::fabs(__bfloat162float(out[i]) - want) > 1e-2f) ok = false;
    }
    std::printf("    rank %d: reduced[0]=%.3f (want %.1f) -> %s\n", rank,
                __bfloat162float(out[0]), want, ok ? "ok" : "BAD");

    cudaFree(d);
    tp_comm_destroy(c);
    results[rank] = ok ? 1 : -1;
  };

  std::vector<std::thread> threads;
  for (int r = 0; r < world; ++r) threads.emplace_back(rank_fn, r);
  for (auto& t : threads) t.join();

  bool ok = true;
  for (int r = 0; r < world; ++r) ok = ok && (results[r] == 1);
  return ok;
}
#endif  // PIE_CUDA_DEVICE_HAS_NCCL

int main() {
  std::printf("==== ops/tp_comm selftest (NCCL %s) ====\n",
              PIE_CUDA_DEVICE_HAS_NCCL ? "ENABLED" : "DISABLED (stub)");

  int n_gpus = 0;
  if (cudaGetDeviceCount(&n_gpus) != cudaSuccess) n_gpus = 0;
  std::printf("    GPUs visible: %d\n", n_gpus);

  if (n_gpus < 1) {
    std::printf("==== SKIP ==== no CUDA device visible\n");
    return 0;  // environment skip, not a wrapper failure
  }

  bool ok = test_single_rank();

#if PIE_CUDA_DEVICE_HAS_NCCL
  if (n_gpus >= 2) {
    ok = test_two_rank(n_gpus) && ok;
  } else {
    std::printf("--- test 2: SKIP (need 2+ GPUs, have %d) ---\n", n_gpus);
  }
#else
  std::printf("--- test 2: SKIP (NCCL disabled) ---\n");
#endif

  std::printf("==== %s ====\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
