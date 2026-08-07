// Standalone microbenchmark for the MoE decode GEMM path.
//
// The engine build is minutes (marlin alone is ~142s of it), which is far too
// slow to iterate a kernel against. This links only the kernels under test.
//
// Build (fast path, no marlin -- ~20s):
//   nvcc -O3 -std=c++20 -arch=sm_80 --expt-relaxed-constexpr \
//        -I driver/cuda/src -o /tmp/moe_bench \
//        driver/cuda/bench/moe_bench.cu \
//        driver/cuda/src/kernels/dequant_fp4.cu \
//        driver/cuda/src/kernels/moe_dispatch.cu
//
// Build with the Marlin MoE candidate (~2-3 min, only when it changes):
//   nvcc -O3 -std=c++20 -arch=sm_80 --expt-relaxed-constexpr -DWITH_MARLIN_MOE \
//        -I driver/cuda/src -I driver/cuda/third_party/marlin_moe \
//        -o /tmp/moe_bench driver/cuda/bench/moe_bench.cu \
//        driver/cuda/src/kernels/dequant_fp4.cu \
//        driver/cuda/src/kernels/moe_dispatch.cu \
//        driver/cuda/third_party/marlin_moe/ops.cu \
//        driver/cuda/third_party/marlin_moe/marlin_moe_wrapper.cpp
//
// Run:
//   /tmp/moe_bench gptoss 32      # N=32 decode rows at gpt-oss-20b's shape
//   /tmp/moe_bench gptoss 1,8,32,64,128
//
// Weights are random bytes. That is exact for TIMING (the kernels are
// data-independent) and is why the parity check here only compares the two
// GEMV variants against each other -- they read the same packed bytes the same
// way, whereas Marlin wants its own repacked layout that this does not build.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

#include <cuda_runtime.h>

#include "kernels/dequant_fp4.hpp"
#include "kernels/moe_dispatch.hpp"

#ifdef WITH_MARLIN_MOE
  #include "marlin_moe_wrapper.hpp"
#endif

#define CHECK(x)                                                             \
  do {                                                                       \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,       \
                   cudaGetErrorString(e));                                   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

struct Shape {
  const char* name;
  int hidden;
  int inter;        // per-expert intermediate (unpadded)
  int inter_pad;    // what the packed weights are padded to
  int experts;
  int top_k;
};

// The three models this has to win on.
const Shape kShapes[] = {
    {"gptoss", 2880, 2880, 2944, 32, 4},    // gpt-oss-20b, MXFP4
    {"gemma4", 2816, 704, 704, 128, 8},     // gemma-4-26B-A4B
    {"qwen36", 2048, 512, 512, 256, 8},     // Qwen3.6-35B-A3B
};

void* dalloc(std::size_t bytes) {
  void* p = nullptr;
  CHECK(cudaMalloc(&p, bytes));
  CHECK(cudaMemset(p, 0, bytes));
  return p;
}

template <class T>
void* dupload(const std::vector<T>& v) {
  void* p = dalloc(v.size() * sizeof(T));
  CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

// Median of repeated timings. This pod refuses `nvidia-smi -lgc`, so the SM
// clock drifts and a single timing is worth little; the engine-level harness
// has the same problem and solves it the same way.
float time_ms(const std::function<void(cudaStream_t)>& fn, cudaStream_t s,
              int iters = 30, int warmup = 10) {
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  for (int i = 0; i < warmup; ++i) fn(s);
  CHECK(cudaStreamSynchronize(s));
  std::vector<float> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CHECK(cudaEventRecord(a, s));
    fn(s);
    CHECK(cudaEventRecord(b, s));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    samples.push_back(ms);
  }
  std::sort(samples.begin(), samples.end());
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  return samples[samples.size() / 2];
}

void run_shape(const Shape& sh, const std::vector<int>& batch_sizes) {
  const int H = sh.hidden;
  const int Ip = sh.inter_pad;
  const int E = sh.experts;
  const int K = sh.top_k;

  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> expert_dist(0, E - 1);

  // Packed MXFP4 expert stack: [E, Ip, H/2] bytes, scales [E, Ip, H/32].
  const std::size_t w_bytes = std::size_t(E) * Ip * (H / 2);
  const std::size_t s_bytes = std::size_t(E) * Ip * (H / 32);
  std::vector<std::uint8_t> hw(w_bytes), hs(s_bytes);
  for (auto& x : hw) x = static_cast<std::uint8_t>(byte_dist(rng));
  // E8M0 scales near 1.0 keep the products in bf16 range.
  for (auto& x : hs) x = 127;
  void* d_w = dupload(hw);
  void* d_s = dupload(hs);

  // One device pointer per expert, as the GEMV kernels index them.
  std::vector<const std::uint8_t*> hw_ptrs(E), hs_ptrs(E);
  for (int e = 0; e < E; ++e) {
    hw_ptrs[e] = static_cast<const std::uint8_t*>(d_w) +
                 std::size_t(e) * Ip * (H / 2);
    hs_ptrs[e] = static_cast<const std::uint8_t*>(d_s) +
                 std::size_t(e) * Ip * (H / 32);
  }
  void* d_w_ptrs = dupload(hw_ptrs);
  void* d_s_ptrs = dupload(hs_ptrs);

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  std::printf("\n== %s  H=%d I=%d(pad %d) E=%d top_k=%d\n", sh.name, H,
              sh.inter, Ip, E, K);
  std::printf("%6s %14s %14s %10s %13s %10s %13s %10s\n", "N",
              "per-route(ms)", "grouped(ms)", "speedup", "marlin2x(ms)",
              "vs grouped", "marlin1x(ms)", "vs 2x");

  for (int N : batch_sizes) {
    const int routes = N * K;
    std::vector<std::int32_t> h_topk(routes);
    for (auto& x : h_topk) x = expert_dist(rng);
    void* d_topk = dupload(h_topk);

    void* d_act = dalloc(std::size_t(N) * H * 2);      // fp16 activations
    void* d_gate = dalloc(std::size_t(routes) * Ip * 2);
    void* d_up = dalloc(std::size_t(routes) * Ip * 2);
    void* d_sorted = dalloc(std::size_t(routes) * 4);
    void* d_r2r = dalloc(std::size_t(routes) * 4);
    void* d_counts = dalloc(std::size_t(E) * 4);

    using namespace pie_cuda_driver;
    auto per_route = [&](cudaStream_t s) {
      kernels::launch_mxfp4_moe_gate_up_decode_bf16(
          d_act, static_cast<const std::int32_t*>(d_topk),
          static_cast<const std::uint8_t* const*>(d_w_ptrs),
          static_cast<const std::uint8_t* const*>(d_s_ptrs),
          nullptr, nullptr, d_gate, d_up, N, K, H, Ip, s);
    };
    auto grouped = [&](cudaStream_t s) {
      kernels::launch_moe_bucket_exact(
          static_cast<const std::int32_t*>(d_topk),
          static_cast<std::int32_t*>(d_sorted),
          static_cast<std::int32_t*>(d_r2r),
          static_cast<std::int32_t*>(d_counts), routes, E, s);
      kernels::launch_mxfp4_moe_gate_up_decode_grouped_bf16(
          d_act, static_cast<const std::int32_t*>(d_sorted),
          static_cast<const std::int32_t*>(d_counts),
          static_cast<const std::uint8_t* const*>(d_w_ptrs),
          static_cast<const std::uint8_t* const*>(d_s_ptrs),
          nullptr, nullptr, d_gate, d_up, E, K, H, Ip, s);
    };

    const float t_pr = time_ms(per_route, stream);
    const float t_gr = time_ms(grouped, stream);
    float t_ml = 0.f;
    float t_mf = 0.f;
#ifdef WITH_MARLIN_MOE
    {
      // Marlin's own block-sorted metadata. Timing only: the weights are
      // random bytes rather than a real repack, which the kernel cannot tell
      // apart because its addressing does not depend on the values.
      const int block = 16;
      const int max_blocks = (routes + E * (block - 1) + block - 1) / block;
      const int padded = max_blocks * block;
      void* d_msorted = dalloc(std::size_t(padded) * 4);
      void* d_mexpert = dalloc(std::size_t(max_blocks) * 4);
      void* d_npast = dalloc(sizeof(std::int32_t));
      void* d_out = dalloc(std::size_t(routes) * Ip * 2);
      void* d_out2 = dalloc(std::size_t(routes) * Ip * 2);
      void* d_ws = dalloc(marlin_moe::marlin_moe_workspace_bytes(Ip, block));
      kernels::launch_moe_align_decode(
          static_cast<const std::int32_t*>(d_topk),
          static_cast<std::int32_t*>(d_msorted),
          static_cast<std::int32_t*>(d_mexpert), nullptr, routes, E, block,
          max_blocks, static_cast<std::int32_t*>(d_npast), stream);
      CHECK(cudaStreamSynchronize(stream));
      // TWO calls, to match what the GEMV kernels above produce in one: they
      // emit gate and up together, while Marlin does one projection per
      // launch (GPT-OSS stores gate and up as separate expert slabs). Timing
      // a single call against them overstates the speedup ~2x, which is
      // exactly the gap that showed up between this bench and the engine.
      auto marlin = [&](cudaStream_t s) {
        for (int half = 0; half < 2; ++half) {
          marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16(
              d_act, d_w, d_s, nullptr, half == 0 ? d_out : d_out2, nullptr,
              d_ws, static_cast<const std::int32_t*>(d_msorted),
              static_cast<const std::int32_t*>(d_mexpert),
              static_cast<const std::int32_t*>(d_npast), nullptr, block, E, K,
              false, N, Ip, H, s);
        }
      };
      t_ml = time_ms(marlin, stream);
      // The question the contract change turns on: GPT-OSS ships gate and up
      // as ONE `[E, 2I, H]` slab and the contract splits them, so fc1 costs
      // two launches. Concatenating the two repacked halves back along n is a
      // valid Marlin packing (I is padded to a multiple of 128, and marlin
      // tiles n by 64), so the same work is one launch at prob_n = 2*Ip.
      // Worth doing only if one wide launch beats two narrow ones.
      {
        // The fused arm reads twice the rows, so it needs its own weight and
        // scale stacks at `[E, 2*Ip, ...]` -- the shared ones above are sized
        // for one projection.
        std::vector<std::uint8_t> hw2(std::size_t(E) * (2 * Ip) * (H / 2));
        std::vector<std::uint8_t> hs2(std::size_t(E) * (2 * Ip) * (H / 32),
                                      std::uint8_t{127});
        for (auto& x : hw2) x = static_cast<std::uint8_t>(byte_dist(rng));
        void* d_w2 = dupload(hw2);
        void* d_s2 = dupload(hs2);
        void* d_wide = dalloc(std::size_t(routes) * (2 * Ip) * 2);
        void* d_ws2 =
            dalloc(marlin_moe::marlin_moe_workspace_bytes(2 * Ip, block));
        auto marlin_fused = [&](cudaStream_t s) {
          marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16(
              d_act, d_w2, d_s2, nullptr, d_wide, nullptr, d_ws2,
              static_cast<const std::int32_t*>(d_msorted),
              static_cast<const std::int32_t*>(d_mexpert),
              static_cast<const std::int32_t*>(d_npast), nullptr, block, E, K,
              false, N, 2 * Ip, H, s);
        };
        t_mf = time_ms(marlin_fused, stream);
        CHECK(cudaFree(d_wide));
        CHECK(cudaFree(d_ws2));
        CHECK(cudaFree(d_w2));
        CHECK(cudaFree(d_s2));
      }
      CHECK(cudaFree(d_msorted)); CHECK(cudaFree(d_mexpert));
      CHECK(cudaFree(d_npast)); CHECK(cudaFree(d_out));
      CHECK(cudaFree(d_out2)); CHECK(cudaFree(d_ws));
    }
#endif
    std::printf("%6d %14.4f %14.4f %9.2fx %13.4f %9.2fx %13.4f %9.2fx\n",
                N, t_pr, t_gr, t_gr > 0 ? t_pr / t_gr : 0.f, t_ml,
                t_ml > 0 ? t_gr / t_ml : 0.f, t_mf,
                t_mf > 0 ? t_ml / t_mf : 0.f);

    CHECK(cudaFree(d_topk));
    CHECK(cudaFree(d_act));
    CHECK(cudaFree(d_gate));
    CHECK(cudaFree(d_up));
    CHECK(cudaFree(d_sorted));
    CHECK(cudaFree(d_r2r));
    CHECK(cudaFree(d_counts));
  }

  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(d_w));
  CHECK(cudaFree(d_s));
  CHECK(cudaFree(d_w_ptrs));
  CHECK(cudaFree(d_s_ptrs));
}

std::vector<int> parse_list(const char* s) {
  std::vector<int> out;
  const char* p = s;
  while (*p) {
    out.push_back(std::atoi(p));
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string which = argc > 1 ? argv[1] : "gptoss";
  const std::vector<int> batches =
      argc > 2 ? parse_list(argv[2]) : std::vector<int>{1, 8, 32, 64, 128};

  bool any = false;
  for (const auto& sh : kShapes) {
    if (which == "all" || which == sh.name) {
      run_shape(sh, batches);
      any = true;
    }
  }
  if (!any) {
    std::fprintf(stderr, "unknown shape %s (gptoss|gemma4|qwen36|all)\n",
                 which.c_str());
    return 2;
  }
  return 0;
}
