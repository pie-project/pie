// WS6 FlashInfer baseline (lane L7 / hotel): the **hardwired top-k / top-p**
// latency baseline for delta's IR-vs-FlashInfer head-to-head.
//
// delta's `bench_sampling_ir.cu` measures the IR-JIT samplers and has a hardwired
// baseline only for argmax/temp/min-p (the `sample_temp` kernel). top-k / top-p
// have no hardwired baseline there, because their production kernel is FlashInfer
// (`launch_sample_topk_topp_bf16`), which needs the FlashInfer headers + a probs
// scratch buffer. This harness supplies exactly that number, in delta's CSV
// schema, so the bench wiki can show IR top-k/p vs FlashInfer top-k/p.
//
// Output CSV columns mirror delta's harness (the hardwired columns; the IR
// columns are filled by delta's run, joined on gpu,vocab,sampler,batch):
//   gpu,vocab,sampler,batch,hardwired_us_per_fire
//
// Measures `launch_sample_topk_topp_bf16` over [B, vocab] in one fused launch
// (the real production path), mean over timed iters after warmup, CUDA-event
// timed. Run: `bench_flashinfer_baseline [out.csv]`.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/sample_flashinfer.hpp"

using pie_cuda_driver::kernels::launch_sample_topk_topp_bf16;

namespace {

void rt_check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL rt: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(expr) rt_check((expr), #expr, __LINE__)

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}

double ms_event(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f;
    RT(cudaEventElapsedTime(&ms, a, b));
    return static_cast<double>(ms);
}

constexpr int kWarmupIters = 5;
constexpr int kTimedIters = 30;

struct Cfg {
    const char* name;
    int top_k;     // 0 = disabled
    float top_p;   // 1.0 = disabled
    float temp;
};

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    RT(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    RT(cudaGetDeviceProperties(&prop, dev));

    const int vocab = 151936;
    const int kBatches[] = {1, 8, 32, 128, 256};
    const int max_b = 256;

    // top-k=50 and top-p=0.9 — the canonical truncation settings, matching the
    // IR bench programs delta runs.
    const Cfg cfgs[] = {
        {"top-k", 50, 1.0f, 0.9f},
        {"top-p", 0, 0.9f, 0.9f},
    };

    // ── Device buffers (sized for the largest batch). ──
    std::vector<std::uint16_t> h_logits(static_cast<size_t>(max_b) * vocab);
    std::mt19937 rng(0xB0BACAFE);
    std::normal_distribution<float> nd(0.f, 3.f);
    for (auto& v : h_logits) v = f32_to_bf16(nd(rng));

    void* d_logits = nullptr;
    float* d_probs = nullptr;        // fp32 scratch [B, vocab]
    float* d_temps = nullptr;
    std::int32_t* d_row_idx = nullptr;
    std::int32_t* d_top_k = nullptr;
    float* d_top_p = nullptr;
    std::uint64_t* d_seed = nullptr;
    bool* d_valid = nullptr;
    std::int32_t* d_out = nullptr;

    RT(cudaMalloc(&d_logits, sizeof(std::uint16_t) * max_b * vocab));
    RT(cudaMalloc(&d_probs, sizeof(float) * max_b * vocab));
    RT(cudaMalloc(&d_temps, sizeof(float) * max_b));
    RT(cudaMalloc(&d_row_idx, sizeof(std::int32_t) * max_b));
    RT(cudaMalloc(&d_top_k, sizeof(std::int32_t) * max_b));
    RT(cudaMalloc(&d_top_p, sizeof(float) * max_b));
    RT(cudaMalloc(&d_seed, sizeof(std::uint64_t) * max_b));
    RT(cudaMalloc(&d_valid, sizeof(bool) * max_b));
    RT(cudaMalloc(&d_out, sizeof(std::int32_t) * max_b));
    RT(cudaMemcpy(d_logits, h_logits.data(),
                  sizeof(std::uint16_t) * max_b * vocab, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    cudaEvent_t ev0, ev1;
    RT(cudaEventCreate(&ev0));
    RT(cudaEventCreate(&ev1));

    std::string csv =
        "gpu,vocab,sampler,batch,hardwired_us_per_fire\n";
    std::fprintf(stderr, "GPU=%s vocab=%d (FlashInfer top-k/p baseline)\n",
                 prop.name, vocab);

    for (const Cfg& c : cfgs) {
        for (int B : kBatches) {
            // Per-row params + identity sample-row indices.
            std::vector<float> temps(B, c.temp);
            std::vector<std::int32_t> row_idx(B), top_k(B, c.top_k);
            std::vector<float> top_p(B, c.top_p);
            std::vector<std::uint64_t> seeds(B);
            for (int i = 0; i < B; ++i) {
                row_idx[i] = i;
                seeds[i] = 0x9E3779B97F4A7C15ull * (i + 1);
            }
            RT(cudaMemcpy(d_temps, temps.data(), sizeof(float) * B, cudaMemcpyHostToDevice));
            RT(cudaMemcpy(d_row_idx, row_idx.data(), sizeof(std::int32_t) * B, cudaMemcpyHostToDevice));
            RT(cudaMemcpy(d_top_k, top_k.data(), sizeof(std::int32_t) * B, cudaMemcpyHostToDevice));
            RT(cudaMemcpy(d_top_p, top_p.data(), sizeof(float) * B, cudaMemcpyHostToDevice));
            RT(cudaMemcpy(d_seed, seeds.data(), sizeof(std::uint64_t) * B, cudaMemcpyHostToDevice));

            auto fire = [&](std::uint64_t off) {
                launch_sample_topk_topp_bf16(
                    d_logits, d_probs, d_temps, d_row_idx, d_top_k, d_top_p,
                    d_seed, d_valid, d_out,
                    /*num_rows=*/B, /*num_samples=*/B, vocab, off, stream);
            };

            for (int i = 0; i < kWarmupIters; ++i) fire(static_cast<std::uint64_t>(i));
            RT(cudaStreamSynchronize(stream));

            RT(cudaEventRecord(ev0, stream));
            for (int i = 0; i < kTimedIters; ++i) fire(static_cast<std::uint64_t>(100 + i));
            RT(cudaEventRecord(ev1, stream));
            RT(cudaEventSynchronize(ev1));

            const double hw_us = ms_event(ev0, ev1) * 1000.0 / kTimedIters;

            char line[256];
            std::snprintf(line, sizeof(line), "%s,%d,%s,%d,%.2f\n",
                          prop.name, vocab, c.name, B, hw_us);
            csv += line;
            std::fprintf(stderr, "  %-6s B=%-4d FlashInfer=%.1f us/fire\n",
                         c.name, B, hw_us);
        }
    }

    std::fputs(csv.c_str(), stdout);
    if (argc > 1) {
        FILE* f = std::fopen(argv[1], "w");
        if (f) {
            std::fputs(csv.c_str(), f);
            std::fclose(f);
            std::fprintf(stderr, "wrote %s\n", argv[1]);
        }
    }

    cudaFree(d_logits); cudaFree(d_probs); cudaFree(d_temps);
    cudaFree(d_row_idx); cudaFree(d_top_k); cudaFree(d_top_p);
    cudaFree(d_seed); cudaFree(d_valid); cudaFree(d_out);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
