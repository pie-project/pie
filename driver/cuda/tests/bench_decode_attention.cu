// Decode-attention microbenchmark: FlashInfer's paged-decode kernel against
// the SM90 (FA3) path, at one layer's shape, with no model and no engine.
//
// Why it exists: the difference between this driver and vLLM on gemma-4 is one
// stage, and measuring it through `pie run` costs a 40-minute build plus a
// 4-minute generation per data point. This runs both kernels over the same KV
// pages in a few seconds, so a kernel change can be judged before it is wired
// into a model.
//
//   bench_decode_attention [kv_heads] [gqa] [head_dim] [window_left]
//
// Defaults are gemma-4-26B-A4B's sliding layer: 8 KV heads, GQA 2,
// head_dim 256, window 1024. Sweeps context and prints per-call microseconds
// and the effective KV bandwidth each kernel reaches.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_flashinfer_hopper.hpp"

using pie_cuda_driver::AttentionWorkspace;
namespace ops = pie_cuda_driver::ops;

namespace {

constexpr int kPageSize = 32;
constexpr int kIters = 50;
constexpr int kWarmup = 10;

// bf16 noise, so a split's output can be compared against the unsplit one.
void* device_randn(std::size_t bytes) {
    std::vector<std::uint16_t> h(bytes / 2);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& x : h) {
        const float f = d(rng);
        std::uint32_t bits;
        std::memcpy(&bits, &f, 4);
        x = static_cast<std::uint16_t>(bits >> 16);
    }
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice));
    return p;
}

void* device_zeros(std::size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemset(p, 0, bytes));
    return p;
}

float time_us(const std::function<void(cudaStream_t)>& fn, cudaStream_t s) {
    for (int i = 0; i < kWarmup; ++i) fn(s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < kIters; ++i) fn(s);
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms * 1000.f / kIters;
}

}  // namespace

int main(int argc, char** argv) {
    const int kv_heads = argc > 1 ? std::atoi(argv[1]) : 8;
    const int gqa = argc > 2 ? std::atoi(argv[2]) : 2;
    const int head_dim = argc > 3 ? std::atoi(argv[3]) : 256;
    const int window_left = argc > 4 ? std::atoi(argv[4]) : 1024;
    const int q_heads = kv_heads * gqa;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device=%s sm=%d kv_heads=%d gqa=%d head_dim=%d window=%d\n",
                prop.name, prop.multiProcessorCount, kv_heads, gqa, head_dim,
                window_left);
    const int splits = std::getenv("SPLITS") ? std::atoi(std::getenv("SPLITS")) : 8;
    std::printf("%8s %12s %12s %12s %10s\n",
                "ctx", "flashinfer", "fa3_sm90", "fa3_split", "max|diff|");
    std::printf("(split = %d-way KV split expressed as %d one-token requests, "
                "each over its own page range)\n", splits, splits);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto workspace = AttentionWorkspace::allocate();

    for (int ctx : {256, 512, 1024, 2048, 4096}) {
        const int pages = (ctx + kPageSize - 1) / kPageSize;
        const std::size_t page_elems =
            static_cast<std::size_t>(kPageSize) * kv_heads * head_dim;
        const std::size_t kv_bytes = page_elems * pages * sizeof(std::uint16_t);

        void* k_pages = device_randn(kv_bytes);
        void* v_pages = device_randn(kv_bytes);
        void* q = device_randn(static_cast<std::size_t>(q_heads) * head_dim * 2);
        void* out = device_zeros(static_cast<std::size_t>(q_heads) * head_dim * 2);

        std::vector<std::uint32_t> idx_h(pages);
        for (int i = 0; i < pages; ++i) idx_h[i] = static_cast<std::uint32_t>(i);
        std::vector<std::uint32_t> indptr_h{0, static_cast<std::uint32_t>(pages)};
        std::vector<std::uint32_t> lastlen_h{
            static_cast<std::uint32_t>(ctx - (pages - 1) * kPageSize)};

        auto upload = [&](const std::vector<std::uint32_t>& v) {
            void* d = nullptr;
            CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice));
            return static_cast<std::uint32_t*>(d);
        };
        auto* idx_d = upload(idx_h);
        auto* indptr_d = upload(indptr_h);
        auto* lastlen_d = upload(lastlen_h);

        auto decode_plan = ops::make_decode_plan();
        ops::plan_attention_flashinfer_decode(
            *decode_plan, indptr_h.data(), 1, q_heads, kv_heads, head_dim,
            kPageSize, workspace, stream, /*enable_cuda_graph=*/true,
            /*full_attention_variant=*/window_left < 0, /*hnd_layout=*/false);
        const float fi_us = time_us([&](cudaStream_t s) {
            ops::dispatch_attention_flashinfer_decode_bf16(
                *decode_plan, q, k_pages, v_pages, out, idx_d, indptr_d,
                lastlen_d, workspace, s, window_left, 0.f, 1.0f);
        }, stream);

        float fa3_us = -1.f;
        if (ops::hopper_prefill_supported(head_dim, window_left, 1, 1)) {
            std::vector<std::uint32_t> qo_h{0, 1};
            ops::HopperPrefillPlan hplan;
            ops::plan_attention_flashinfer_prefill_sm90_bf16(
                hplan, qo_h.data(), indptr_h.data(), lastlen_h.data(), 1, 1,
                q_heads, kv_heads, head_dim, kPageSize, workspace, stream,
                /*enable_cuda_graph=*/true, /*causal=*/true, window_left,
                workspace.int_bytes() / 2);
            if (hplan.valid) {
                fa3_us = time_us([&](cudaStream_t s) {
                    ops::dispatch_attention_flashinfer_prefill_sm90_bf16(
                        hplan, q, k_pages, v_pages, out, idx_d, workspace, s,
                        0.f, 1.0f);
                }, stream);
            }
        }

        // KV split as a batch split: `splits` pseudo-requests, each one token
        // of Q over a disjoint slice of the pages. Decode puts the query at
        // the end of whatever range it is given, so a causal q_len=1 fire over
        // a slice attends to all of that slice -- which is exactly the partial
        // this needs. Merging the partials is `MergeStates` over `splits`
        // index sets, tiny next to the attention itself.
        float split_us = -1.f;
        float max_abs_diff = -1.f;
        if (ops::hopper_prefill_supported(head_dim, -1, splits, splits) &&
            pages >= splits) {
            // Only the in-window tail is worth splitting. The oldest chunk
            // gets one extra page and carries the window: with qo_len = 1 the
            // kernel's first visible index is `kv_len - 1 - window_left`, so
            // setting `window_left = kv_len_0 - 1 - skip` starts it exactly at
            // the window, and the extra page keeps that same `window_left`
            // from masking anything in the later (shorter) chunks.
            const int win_start = (window_left >= 0)
                ? std::max(0, ctx - window_left) : 0;
            const int first_page = win_start / kPageSize;
            const int skip = win_start - first_page * kPageSize;
            const int live = pages - first_page;
            const int chunk = std::max(1, (live + splits - 1) / splits);
            std::vector<std::uint32_t> sq(splits + 1), sindptr(splits + 1),
                slast(splits);
            for (int i = 0; i <= splits; ++i) sq[i] = static_cast<std::uint32_t>(i);
            sindptr[0] = static_cast<std::uint32_t>(first_page);
            int cursor = first_page;
            int split_window = -1;
            for (int i = 0; i < splits; ++i) {
                const int extra = (i == 0 && skip > 0) ? 1 : 0;
                const int lo = cursor;
                const int hi = std::min(lo + chunk + extra, pages);
                cursor = hi;
                sindptr[i + 1] = static_cast<std::uint32_t>(hi);
                const bool last = (hi == pages);
                const int len = (hi > lo)
                    ? ((hi - lo - 1) * kPageSize +
                       (last ? (ctx - (pages - 1) * kPageSize) : kPageSize))
                    : 0;
                slast[i] = (hi > lo)
                    ? static_cast<std::uint32_t>(
                          last ? (ctx - (pages - 1) * kPageSize) : kPageSize)
                    : 1u;
                if (i == 0) split_window = len - 1 - skip;
            }
            auto* sindptr_d = upload(sindptr);
            auto* slast_d = upload(slast);
            void* sq_dev = device_zeros(
                static_cast<std::size_t>(splits) * q_heads * head_dim * 2);
            void* spart = device_zeros(
                static_cast<std::size_t>(splits) * q_heads * head_dim * 2);
            void* slse = device_zeros(
                static_cast<std::size_t>(splits) * q_heads * sizeof(float));
            ops::HopperPrefillPlan sp;
            ops::plan_attention_flashinfer_prefill_sm90_bf16(
                sp, sq.data(), sindptr.data(), slast.data(), splits, splits,
                q_heads, kv_heads, head_dim, kPageSize, workspace, stream,
                /*enable_cuda_graph=*/true, /*causal=*/true,
                /*window_left=*/(skip > 0 ? split_window : -1),
                workspace.int_bytes() / 2);
            void* smerged = device_zeros(
                static_cast<std::size_t>(q_heads) * head_dim * 2);
            void* slse_m = device_zeros(
                static_cast<std::size_t>(q_heads) * sizeof(float));
            if (sp.valid) {
                split_us = time_us([&](cudaStream_t st) {
                    ops::dispatch_attention_flashinfer_prefill_sm90_bf16(
                        sp, q, k_pages, v_pages, spart, idx_d, workspace,
                        st, 0.f, 1.0f, static_cast<float*>(slse),
                        /*broadcast_q=*/true);
                    ops::merge_attention_states_bf16(
                        spart, static_cast<float*>(slse), smerged,
                        static_cast<float*>(slse_m), splits, 1, q_heads,
                        head_dim, st);
                }, stream);
                // Correctness: the merged split must match the unsplit answer.
                std::vector<std::uint16_t> a(
                    static_cast<std::size_t>(q_heads) * head_dim), b(a.size());
                CUDA_CHECK(cudaMemcpy(a.data(), out, a.size() * 2,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(b.data(), smerged, b.size() * 2,
                                      cudaMemcpyDeviceToHost));
                auto tof = [](std::uint16_t h) {
                    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
                    float f; std::memcpy(&f, &bits, 4); return f;
                };
                float worst = 0.f;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    worst = std::max(worst, std::fabs(tof(a[i]) - tof(b[i])));
                }
                max_abs_diff = worst;
            }
            CUDA_CHECK(cudaFree(smerged));
            CUDA_CHECK(cudaFree(slse_m));
            for (void* p : {sq_dev, spart, slse}) CUDA_CHECK(cudaFree(p));
            for (void* p : {static_cast<void*>(sindptr_d),
                            static_cast<void*>(slast_d)}) CUDA_CHECK(cudaFree(p));
        }

        // Only the in-window tail is useful work; that is what a kernel could
        // read if it bounded its scan, and what the bandwidth column charges.
        const int scanned = (window_left >= 0) ? std::min(ctx, window_left) : ctx;
        const double useful_bytes =
            2.0 * scanned * kv_heads * head_dim * sizeof(std::uint16_t);
        auto gbs = [&](float us) {
            return us > 0.f ? useful_bytes / (us * 1e3) : 0.0;
        };
        std::printf("%8d %10.1fus %10.1fus %10.1fus %10.4f\n",
                    ctx, fi_us, fa3_us, split_us, max_abs_diff);

        for (void* p : {k_pages, v_pages, q, out}) CUDA_CHECK(cudaFree(p));
        for (void* p : {static_cast<void*>(idx_d), static_cast<void*>(indptr_d),
                        static_cast<void*>(lastlen_d)}) {
            CUDA_CHECK(cudaFree(p));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
