// Standalone device verify for the #6 P2 next-input link machinery (next_input.cpp).
// Needs a GPU. Checks inject (sampled[src_row] → next_input[dest_pos], skip lanes)
// and the event-ordered inject (cross-stream: producer write → event → inject waits).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#include "sampling_ir/next_input.hpp"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)
#define RT(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
}  // namespace

int main() {
    // Producer pi.sampled [5] i32 tokens.
    const std::vector<std::int32_t> sampled = {100, 200, 300, 400, 500};
    std::int32_t* d_sampled = nullptr;
    RT(cudaMalloc(&d_sampled, sampled.size() * sizeof(std::int32_t)));

    // Consumer next_input [4] i32, pre-filled with a -7 marker to prove the skip
    // lane and untouched positions are left intact.
    const int dst_n = 4;
    std::int32_t* d_next = nullptr;
    RT(cudaMalloc(&d_next, dst_n * sizeof(std::int32_t)));

    // Links: row2→pos0, row4→pos3, row0→IGNORE (skip), row1→pos1. pos2 untouched.
    const std::vector<NextInputLink> links = {
        {2, 0}, {4, 3}, {0, kIgnorePosition}, {1, 1}};

    // ── (a) plain inject (single stream) ──────────────────────────────────────
    RT(cudaMemcpy(d_sampled, sampled.data(), sampled.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));
    {
        std::vector<std::int32_t> init(dst_n, -7);
        RT(cudaMemcpy(d_next, init.data(), dst_n * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice));
    }
    inject_next_input(d_sampled, std::span<const NextInputLink>(links.data(), links.size()),
                      d_next, /*stream=*/nullptr);
    RT(cudaDeviceSynchronize());
    {
        std::vector<std::int32_t> out(dst_n);
        RT(cudaMemcpy(out.data(), d_next, dst_n * sizeof(std::int32_t),
                      cudaMemcpyDeviceToHost));
        CHECK(out[0] == 300);  // ← sampled[2]
        CHECK(out[1] == 200);  // ← sampled[1]
        CHECK(out[2] == -7);   // untouched (no link)
        CHECK(out[3] == 500);  // ← sampled[4]
        // row0→IGNORE wrote nothing.
    }

    // ── (b) event-ordered inject (cross-stream: producer write on A → event → the
    //        inject on B waits it, so B sees A's tokens) ────────────────────────
    cudaStream_t prod = nullptr, cons = nullptr;
    cudaEvent_t done = nullptr;
    RT(cudaStreamCreate(&prod));
    RT(cudaStreamCreate(&cons));
    RT(cudaEventCreate(&done));
    {
        std::vector<std::int32_t> init(dst_n, -7);
        RT(cudaMemcpyAsync(d_next, init.data(), dst_n * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice, cons));
        // Producer "sample": (re)write sampled on stream `prod`, then record the
        // completion event on `prod`.
        RT(cudaMemcpyAsync(d_sampled, sampled.data(), sampled.size() * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice, prod));
        RT(cudaEventRecord(done, prod));
        // Consumer inject on `cons` gated behind `done` — no host await.
        inject_next_input_after(
            d_sampled, std::span<const NextInputLink>(links.data(), links.size()), d_next,
            done, cons);
        RT(cudaStreamSynchronize(cons));

        std::vector<std::int32_t> out(dst_n);
        RT(cudaMemcpy(out.data(), d_next, dst_n * sizeof(std::int32_t),
                      cudaMemcpyDeviceToHost));
        CHECK(out[0] == 300);
        CHECK(out[1] == 200);
        CHECK(out[2] == -7);
        CHECK(out[3] == 500);
    }
    RT(cudaEventDestroy(done));
    RT(cudaStreamDestroy(prod));
    RT(cudaStreamDestroy(cons));

    RT(cudaFree(d_sampled));
    RT(cudaFree(d_next));

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_next_input_device: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_next_input_device: %d failure(s)\n", g_failures);
    return 1;
}
