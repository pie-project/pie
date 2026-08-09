// What does pie's decode small-op tail actually cost, and why?
//
// A CUPTI trace of gpt-oss decode on B200 puts 627 us/step -- 26% of decode
// kernel time -- in kernels that move almost no memory: rmsnorm 149 us over
// 49 calls, topk_softmax 128 over 24, write_kv 89, rope 63, attn_sink 52,
// bf16_to_fp16 42. That is the shape of pie's 88%-GPU-busy / 18%-memory-busy
// signature against SGLang's 100% / 39%.
//
// Two very different causes fit those numbers and they want opposite fixes:
//
//   * the kernels are SLOW -- 3.05 us to normalise 2880 bf16 values is 5.8 KB
//     of traffic, which even at 1 TB/s is nanoseconds. If a standalone launch
//     also takes 3 us, the launch configuration is wrong and can be fixed in
//     place, with no change to how the model is composed.
//   * the kernels are MERELY NUMEROUS -- if an EMPTY kernel also costs ~3 us
//     here, then nothing about these kernels is wrong and the only fix is to
//     have fewer of them. topk_softmax.hpp already records that fusing the
//     router projection into the top-K cost 2x (gpt-oss 291 -> 134 tok/s),
//     because top-K is one block per token and dragged the projection down to
//     one block. So "fuse it" is not free advice; it has already backfired
//     once here, and the empty-kernel floor is what says whether there is
//     anything left to win.
//
// The empty-kernel row is therefore the whole point of this bench: it is the
// floor every other row is measured against.
//
// Decode is ONE token, so every shape here is N=1. That is not a simplifying
// assumption, it is the regime: a kernel that looks fine at N=256 can be pure
// latency at N=1, and single-stream latency is what the table measures.
//
// pie replays a captured CUDA graph, so both an eager and a graph-replay
// number are reported. Eager includes per-launch host cost that pie does not
// pay; graph replay is the honest comparison.
//
// Build (~20s):
//   nvcc -O3 -std=c++20 -arch=sm_100 --expt-relaxed-constexpr \
//        -I driver/cuda/src -o /tmp/smallop_bench \
//        driver/cuda/bench/smallop_bench.cu \
//        driver/cuda/src/norm/rmsnorm.cu \
//        driver/cuda/src/moe/topk_softmax.cu
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attn/kv_paged.hpp"
#include "norm/rmsnorm.hpp"
#include "rope/rope.hpp"
#include "moe/topk_softmax.hpp"

namespace kernels = pie_cuda_driver::kernels;

namespace {

#define CHECK(x)                                                            \
    do {                                                                    \
        cudaError_t e_ = (x);                                               \
        if (e_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(e_));                           \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

constexpr int kWarmup = 50;
constexpr int kIters = 500;

__global__ void empty_kernel() {}

// Time `fn` two ways: launched one at a time, and replayed from a captured
// graph of kIters launches. pie decodes through graph replay, so the second
// number is the one its trace should be compared against.
template <class F>
void bench(const char* what, F&& fn) {
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    for (int i = 0; i < kWarmup; ++i) fn(s);
    CHECK(cudaStreamSynchronize(s));

    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < kIters; ++i) fn(s);
    CHECK(cudaEventRecord(b, s));
    CHECK(cudaEventSynchronize(b));
    float eager_ms = 0.f;
    CHECK(cudaEventElapsedTime(&eager_ms, a, b));

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < kIters; ++i) fn(s);
    CHECK(cudaStreamEndCapture(s, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK(cudaGraphLaunch(exec, s));
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaEventRecord(a, s));
    CHECK(cudaGraphLaunch(exec, s));
    CHECK(cudaEventRecord(b, s));
    CHECK(cudaEventSynchronize(b));
    float graph_ms = 0.f;
    CHECK(cudaEventElapsedTime(&graph_ms, a, b));

    std::printf("  %-34s eager %6.2f us   graph %6.2f us\n", what,
                eager_ms * 1e3 / kIters, graph_ms * 1e3 / kIters);

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    CHECK(cudaStreamDestroy(s));
}

void* dalloc(size_t bytes, int fill = 0x3c) {
    void* p = nullptr;
    CHECK(cudaMalloc(&p, bytes));
    CHECK(cudaMemset(p, fill, bytes));
    return p;
}

// Run both forms over random logits and require identical expert choices.
// Weights may differ in the last bits -- the two reduction orders are not the
// same -- so those are compared with a tolerance, but indices must match
// exactly.
bool verify() {
    std::printf("\nverifying warp form against block form:\n");
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.f, 2.f);
    bool ok = true;
    // (num_experts, K) across the benchmark models: gpt-oss 32/4,
    // gemma-4-26B-A4B and Qwen3.6-35B-A3B use larger pools, plus the
    // boundaries of each PER_LANE specialisation and a non-multiple of 32.
    const int cases[][2] = {{32, 4}, {32, 1}, {31, 4},  {33, 2},
                            {64, 8}, {65, 4}, {128, 8}, {127, 6},
                            // Qwen3.6-35B-A3B routes through >128 experts and
                            // was falling back to the block form; Kimi K2.6
                            // uses 384. 512 is MAX_EXPERTS.
                            {129, 8}, {256, 8}, {384, 6}, {512, 8}, {511, 4}};
    for (const auto& c : cases) {
        const int E = c[0], K = c[1], rows = 7;
        std::vector<__nv_bfloat16> h(static_cast<size_t>(rows) * E);
        for (auto& v : h) v = __float2bfloat16(dist(rng));
        void* d = nullptr;
        CHECK(cudaMalloc(&d, h.size() * sizeof(__nv_bfloat16)));
        CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
        std::int32_t* di[2];
        float* dw[2];
        std::vector<std::int32_t> hi[2];
        std::vector<float> hw[2];
        for (int form = 0; form < 2; ++form) {
            CHECK(cudaMalloc(&di[form], sizeof(std::int32_t) * rows * K));
            CHECK(cudaMalloc(&dw[form], sizeof(float) * rows * K));
            // Explicit selector, NOT the env-gated entry point: that one
            // caches its choice in a function-local static, so flipping
            // PIE_TOPK_WARP here would run the block form twice and the
            // comparison would pass without comparing anything.
            pie_cuda_driver::kernels::moe::topk_softmax_bf16_form(
                d, di[form], dw[form], rows, E, K, /*use_warp=*/form == 1,
                nullptr);
            CHECK(cudaDeviceSynchronize());
            hi[form].resize(static_cast<size_t>(rows) * K);
            hw[form].resize(static_cast<size_t>(rows) * K);
            CHECK(cudaMemcpy(hi[form].data(), di[form],
                             sizeof(std::int32_t) * rows * K,
                             cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hw[form].data(), dw[form], sizeof(float) * rows * K,
                             cudaMemcpyDeviceToHost));
        }
        int bad_i = 0;
        float max_dw = 0.f;
        for (size_t j = 0; j < hi[0].size(); ++j) {
            if (hi[0][j] != hi[1][j]) ++bad_i;
            max_dw = fmaxf(max_dw, std::fabs(hw[0][j] - hw[1][j]));
        }
        const bool pass = bad_i == 0 && max_dw < 1e-3f;
        ok = ok && pass;
        std::printf("  E=%-4d K=%-2d  %s  idx_mismatch=%d  max|dw|=%.2e\n", E, K,
                    pass ? "OK  " : "FAIL", bad_i, max_dw);
        for (int form = 0; form < 2; ++form) {
            CHECK(cudaFree(di[form]));
            CHECK(cudaFree(dw[form]));
        }
        CHECK(cudaFree(d));
    }
    return ok;
}

// RoPE + write_kv, fused against the real two-kernel pair.
//
// This one writes into the paged KV cache, so "faster" is worthless without
// "identical": a wrong page index does not crash, it stores a plausible key
// in the wrong slot and the model quietly degrades. Both paths run over the
// same paging inputs and every output -- q, k, and both page arrays -- is
// compared bit-for-bit.
struct KvFixture {
    int N, R, hq, hkv, d, page_size, pages_per_req;
    bool hnd;
    const char* what;
};

bool bench_rope_write_kv() {
    std::printf("\nrope + write_kv, fused vs the two-kernel pair:\n");
    const KvFixture fx[] = {
        {1, 1, 64, 8, 64, 16, 2, false, "gpt-oss decode NHD"},
        {1, 1, 64, 8, 64, 16, 2, true,  "gpt-oss decode HND"},
        {3, 3, 64, 8, 64, 16, 2, false, "3 requests NHD"},
        {1, 1, 32, 8, 128, 32, 3, false, "d=128 NHD"},
    };
    bool all_ok = true;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (const KvFixture& f : fx) {
        const size_t qn = (size_t)f.N * f.hq * f.d;
        const size_t kn = (size_t)f.N * f.hkv * f.d;
        const int total_pages = f.R * f.pages_per_req;
        const size_t pn = (size_t)total_pages * f.page_size * f.hkv * f.d;

        std::vector<__nv_bfloat16> hq_(qn), hk_(kn), hv_(kn);
        for (auto& x : hq_) x = __float2bfloat16(dist(rng));
        for (auto& x : hk_) x = __float2bfloat16(dist(rng));
        for (auto& x : hv_) x = __float2bfloat16(dist(rng));
        std::vector<std::int32_t> hpos(f.N);
        for (int i = 0; i < f.N; ++i) hpos[i] = 20 + i;
        // One new token per request, each with pages_per_req pages already
        // allocated and 5 rows used in the last one.
        std::vector<std::uint32_t> hqo(f.R + 1), hpi(total_pages),
            hpp(f.R + 1), hll(f.R);
        for (int r = 0; r <= f.R; ++r) hqo[r] = r;
        for (int r = 0; r <= f.R; ++r) hpp[r] = r * f.pages_per_req;
        for (int i = 0; i < total_pages; ++i) hpi[i] = total_pages - 1 - i;
        for (int r = 0; r < f.R; ++r) hll[r] = 5;

        auto up = [](const void* h, size_t bytes) {
            void* d = nullptr;
            CHECK(cudaMalloc(&d, bytes));
            CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
            return d;
        };
        void* dpos = up(hpos.data(), hpos.size() * 4);
        void* dqo = up(hqo.data(), hqo.size() * 4);
        void* dpi = up(hpi.data(), hpi.size() * 4);
        void* dpp = up(hpp.data(), hpp.size() * 4);
        void* dll = up(hll.data(), hll.size() * 4);

        void* q[2];
        void* k[2];
        void* v = up(hv_.data(), kn * 2);
        void* kp[2];
        void* vp[2];
        for (int i = 0; i < 2; ++i) {
            q[i] = up(hq_.data(), qn * 2);
            k[i] = up(hk_.data(), kn * 2);
            CHECK(cudaMalloc(&kp[i], pn * 2));
            CHECK(cudaMalloc(&vp[i], pn * 2));
            CHECK(cudaMemset(kp[i], 0, pn * 2));
            CHECK(cudaMemset(vp[i], 0, pn * 2));
        }
        namespace K = pie_cuda_driver::kernels;
        // Reference: the two kernels the engine runs today.
        kernels::rope::rope_bf16(q[0], k[0], (const std::int32_t*)dpos, f.N, f.hq,
                            f.hkv, f.d, 10000.f, nullptr);
        kernels::attn::write_kv_to_pages_bf16(
            kp[0], vp[0], k[0], v, (const std::uint32_t*)dqo,
            (const std::uint32_t*)dpi, (const std::uint32_t*)dpp,
            (const std::uint32_t*)dll, f.N, f.R, f.page_size, f.hkv, f.d,
            f.hnd, nullptr, nullptr);
        kernels::rope::rope_write_kv_bf16(
            q[1], k[1], v, (const std::int32_t*)dpos, kp[1], vp[1],
            (const std::uint32_t*)dqo, (const std::uint32_t*)dpi,
            (const std::uint32_t*)dpp, (const std::uint32_t*)dll, nullptr,
            f.N, f.R, f.page_size, f.hq, f.hkv, f.d, 10000.f, f.hnd, nullptr);
        CHECK(cudaDeviceSynchronize());

        auto down = [](void* d, size_t n_) {
            std::vector<std::uint16_t> h(n_);
            CHECK(cudaMemcpy(h.data(), d, n_ * 2, cudaMemcpyDeviceToHost));
            return h;
        };
        int bad = 0;
        auto cmp = [&](void* a, void* b, size_t n_, const char* /*tag*/) {
            auto x = down(a, n_), y = down(b, n_);
            for (size_t i = 0; i < n_; ++i) if (x[i] != y[i]) ++bad;
        };
        cmp(q[0], q[1], qn, "q");
        cmp(k[0], k[1], kn, "k");
        cmp(kp[0], kp[1], pn, "k_pages");
        cmp(vp[0], vp[1], pn, "v_pages");
        all_ok = all_ok && bad == 0;
        std::printf("  %-22s %s  mismatched_bf16=%d\n", f.what,
                    bad == 0 ? "OK  " : "FAIL", bad);

        if (bad == 0 && f.N == 1 && !f.hnd && f.d == 64) {
            bench("  rope + write_kv (2 kernels)", [&](cudaStream_t st) {
                kernels::rope::rope_bf16(q[0], k[0], (const std::int32_t*)dpos, f.N,
                                    f.hq, f.hkv, f.d, 10000.f, st);
                kernels::attn::write_kv_to_pages_bf16(
                    kp[0], vp[0], k[0], v, (const std::uint32_t*)dqo,
                    (const std::uint32_t*)dpi, (const std::uint32_t*)dpp,
                    (const std::uint32_t*)dll, f.N, f.R, f.page_size, f.hkv,
                    f.d, f.hnd, st, nullptr);
            });
            bench("  rope_write_kv (fused)", [&](cudaStream_t st) {
                kernels::rope::rope_write_kv_bf16(
                    q[1], k[1], v, (const std::int32_t*)dpos, kp[1], vp[1],
                    (const std::uint32_t*)dqo, (const std::uint32_t*)dpi,
                    (const std::uint32_t*)dpp, (const std::uint32_t*)dll,
                    nullptr, f.N, f.R, f.page_size, f.hq, f.hkv, f.d, 10000.f,
                    f.hnd, st);
            });
        }
        for (int i = 0; i < 2; ++i) {
            CHECK(cudaFree(q[i])); CHECK(cudaFree(k[i]));
            CHECK(cudaFree(kp[i])); CHECK(cudaFree(vp[i]));
        }
        CHECK(cudaFree(v)); CHECK(cudaFree(dpos)); CHECK(cudaFree(dqo));
        CHECK(cudaFree(dpi)); CHECK(cudaFree(dpp)); CHECK(cudaFree(dll));
    }
    return all_ok;
}

}  // namespace

int main() {
    // gpt-oss-20b decode: one token, hidden 2880, 32 experts, top-4.
    const int H = 2880, E = 32, K = 4, N = 1;
    std::printf("gpt-oss decode small ops, N=1 token (hidden=%d, experts=%d, "
                "top_k=%d)\n", H, E, K);
    std::printf("  traced in-engine: rmsnorm 3.05 us/call, topk_softmax 5.35, "
                "rope 2.62, write_kv 3.69\n\n");

    void* x = dalloc(static_cast<size_t>(N) * H * sizeof(__nv_bfloat16));
    void* w = dalloc(static_cast<size_t>(H) * sizeof(__nv_bfloat16));
    void* y = dalloc(static_cast<size_t>(N) * H * sizeof(__nv_bfloat16));
    void* yf = dalloc(static_cast<size_t>(N) * H * sizeof(__half));
    void* logits = dalloc(static_cast<size_t>(N) * E * sizeof(__nv_bfloat16));
    void* big_logits = dalloc(static_cast<size_t>(N) * 512 * sizeof(__nv_bfloat16));
    auto* idx = static_cast<std::int32_t*>(
        dalloc(static_cast<size_t>(N) * 8 * sizeof(std::int32_t), 0));
    auto* wt = static_cast<float*>(
        dalloc(static_cast<size_t>(N) * 8 * sizeof(float), 0));

    // THE FLOOR. Everything below is only interesting relative to this.
    bench("empty kernel (1 block)",
          [](cudaStream_t s) { empty_kernel<<<1, 128, 0, s>>>(); });
    bench("empty kernel (49 blocks)",
          [](cudaStream_t s) { empty_kernel<<<49, 128, 0, s>>>(); });

    bench("rmsnorm_bf16", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::norm::rmsnorm_bf16(x, w, y, N, H, 1e-6f, s);
    });
    // pie runs a separate bf16_to_fp16_kernel once per layer (42 us/step).
    // This variant already folds that copy into the norm; if it costs the
    // same as the plain one, that whole kernel is deletable.
    bench("rmsnorm_bf16_with_fp16 (fused cast)", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::norm::rmsnorm_bf16_with_fp16(
            x, w, y, yf, N, H, 1e-6f, s);
    });
    // The warp form must pick the SAME experts as the block form. A routing
    // difference is a different model, not a faster kernel, so this gate runs
    // before any timing is reported and over the expert counts and K values
    // the five benchmark models actually use.
    if (!verify()) {
        std::printf("\nCORRECTNESS FAILED -- timings withheld\n");
        return 1;
    }
    bench("topk_softmax (block, old)", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::moe::topk_softmax_bf16_form(
            logits, idx, wt, N, E, K, /*use_warp=*/false, s);
    });
    bench("topk_softmax (warp, new)", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::moe::topk_softmax_bf16_form(
            logits, idx, wt, N, E, K, /*use_warp=*/true, s);
    });
    // Qwen3.6-35B-A3B and gemma-4-26B-A4B route through bigger expert pools,
    // where the block form had more parallelism to lose.
    bench("topk_softmax E=256 K=8 (block)", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::moe::topk_softmax_bf16_form(
            big_logits, idx, wt, N, 256, 8, /*use_warp=*/false, s);
    });
    bench("topk_softmax E=256 K=8 (warp)", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::moe::topk_softmax_bf16_form(
            big_logits, idx, wt, N, 256, 8, /*use_warp=*/true, s);
    });
    bench("topk_softmax E=128 K=8 (block)", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::moe::topk_softmax_bf16_form(
            big_logits, idx, wt, N, 128, 8, /*use_warp=*/false, s);
    });
    bench("topk_softmax E=128 K=8 (warp)", [&](cudaStream_t s) {
        pie_cuda_driver::kernels::moe::topk_softmax_bf16_form(
            big_logits, idx, wt, N, 128, 8, /*use_warp=*/true, s);
    });
    if (!bench_rope_write_kv()) {
        std::printf("\nrope+write_kv FUSION IS NOT IDENTICAL -- do not ship\n");
        return 1;
    }
    return 0;
}
