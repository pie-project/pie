// Chunked (vocab-streaming) argmax equivalence test.
//
// The whole point of the chunked accumulator (§20.36) is that streaming the
// vocab in slabs must produce BIT-IDENTICAL token ids to argmaxing a
// materialised logits buffer -- including the tie-break (lowest index wins),
// which is what makes it substitutable for the sampler. So the reference here
// is `launch_argmax_bf16` itself, not a host reimplementation, and the gate is
// exact equality rather than a tolerance.
//
// Covers: chunk widths that divide the vocab and ones that leave a ragged
// remainder, a width below one vector load, a single-chunk degenerate case,
// and a slab whose row stride defeats the vectorised path.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/argmax.hpp"

using pie_cuda_driver::kernels::kArgmaxAccumSlots;
using pie_cuda_driver::kernels::launch_argmax_accumulate_bf16;
using pie_cuda_driver::kernels::launch_argmax_bf16;
using pie_cuda_driver::kernels::launch_argmax_finalize_bf16;

namespace {

int g_fail = 0;

void rt(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, line, what,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(e) rt((e), #e, __LINE__)

std::uint32_t rng_state = 0x9e3779b9u;
std::uint32_t next_rand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// Run the chunked path over `logits` in slabs of `chunk` columns and compare
// against `reference`. `row_stride` lets the caller hand in a slab whose rows
// are padded, which is what a real chunk buffer looks like.
void check_chunked(const char* name, const __nv_bfloat16* d_logits, int rows,
                   int vocab, int row_stride, int chunk,
                   const std::vector<std::int32_t>& reference) {
    float* d_acc_val = nullptr;
    std::int32_t* d_acc_idx = nullptr;
    std::int32_t* d_tokens = nullptr;
    const std::size_t slots =
        static_cast<std::size_t>(rows) * kArgmaxAccumSlots;
    RT(cudaMalloc(&d_acc_val, slots * sizeof(float)));
    RT(cudaMalloc(&d_acc_idx, slots * sizeof(std::int32_t)));
    RT(cudaMalloc(&d_tokens, rows * sizeof(std::int32_t)));
    // Poison the accumulator so a missing `init` on the first slab shows up.
    RT(cudaMemset(d_acc_val, 0x7f, slots * sizeof(float)));
    RT(cudaMemset(d_acc_idx, 0xff, slots * sizeof(std::int32_t)));

    int chunks = 0;
    for (int base = 0; base < vocab; base += chunk) {
        const int width = (base + chunk <= vocab) ? chunk : (vocab - base);
        launch_argmax_accumulate_bf16(d_logits + base, rows, width, row_stride,
                                      base, d_acc_val, d_acc_idx, base == 0,
                                      nullptr);
        ++chunks;
    }
    launch_argmax_finalize_bf16(d_acc_val, d_acc_idx, d_tokens, rows, nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::int32_t> got(rows);
    RT(cudaMemcpy(got.data(), d_tokens, rows * sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost));

    int mismatches = 0;
    int first = -1;
    for (int r = 0; r < rows; ++r) {
        if (got[r] != reference[r]) {
            if (first < 0) first = r;
            ++mismatches;
        }
    }
    std::printf("[%s] %s (chunk=%d, %d slabs)\n", mismatches ? "FAIL" : " ok ",
                name, chunk, chunks);
    if (mismatches) {
        std::printf("        %d/%d rows differ; first row %d: got %d want %d\n",
                    mismatches, rows, first, got[first], reference[first]);
        ++g_fail;
    }

    cudaFree(d_acc_val);
    cudaFree(d_acc_idx);
    cudaFree(d_tokens);
}

}  // namespace

int main() {
    // Production decode shape: R=512 rows over a Qwen3 vocab.
    constexpr int kRows = 512;
    constexpr int kVocab = 151936;

    std::vector<__nv_bfloat16> host(static_cast<std::size_t>(kRows) * kVocab);
    for (std::size_t i = 0; i < host.size(); ++i) {
        // Coarse quantisation on purpose: bf16 has 8 mantissa bits, so ties
        // are common in real logits and the tie-break must be exercised.
        const float v =
            static_cast<float>(static_cast<int>(next_rand() % 4096) - 2048) /
            256.0f;
        host[i] = __float2bfloat16(v);
    }

    __nv_bfloat16* d_logits = nullptr;
    RT(cudaMalloc(&d_logits, host.size() * sizeof(__nv_bfloat16)));
    RT(cudaMemcpy(d_logits, host.data(), host.size() * sizeof(__nv_bfloat16),
                  cudaMemcpyHostToDevice));

    std::int32_t* d_ref = nullptr;
    RT(cudaMalloc(&d_ref, kRows * sizeof(std::int32_t)));
    launch_argmax_bf16(d_logits, d_ref, kRows, kVocab, nullptr);
    RT(cudaDeviceSynchronize());
    std::vector<std::int32_t> reference(kRows);
    RT(cudaMemcpy(reference.data(), d_ref, kRows * sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost));

    // vocab % chunk != 0 for every one of these except the last, so the
    // ragged final slab is the common case, not the exception.
    check_chunked("ragged slabs", d_logits, kRows, kVocab, kVocab, 16384,
                  reference);
    check_chunked("ragged slabs", d_logits, kRows, kVocab, kVocab, 8192,
                  reference);
    // 151936 = 8192 * 18 + 4480, and 4480 % 8 == 0, so also exercise a chunk
    // whose remainder is NOT a multiple of the 8-wide vector load.
    check_chunked("non-vector remainder", d_logits, kRows, kVocab, kVocab, 8191,
                  reference);
    check_chunked("sub-vector slabs", d_logits, kRows, kVocab, kVocab, 5,
                  reference);
    check_chunked("single slab", d_logits, kRows, kVocab, kVocab, kVocab,
                  reference);

    // A padded slab: rows no longer start on a 16-byte boundary, so the
    // launcher must fall back to the scalar path and still agree.
    {
        constexpr int kNarrowVocab = 4093;
        constexpr int kStride = kNarrowVocab + 3;
        std::vector<__nv_bfloat16> padded(
            static_cast<std::size_t>(kRows) * kStride, __float2bfloat16(0.0f));
        for (int r = 0; r < kRows; ++r)
            for (int c = 0; c < kNarrowVocab; ++c)
                padded[static_cast<std::size_t>(r) * kStride + c] =
                    host[static_cast<std::size_t>(r) * kVocab + c];

        __nv_bfloat16* d_padded = nullptr;
        RT(cudaMalloc(&d_padded, padded.size() * sizeof(__nv_bfloat16)));
        RT(cudaMemcpy(d_padded, padded.data(),
                      padded.size() * sizeof(__nv_bfloat16),
                      cudaMemcpyHostToDevice));

        // Reference over the same logical region, densely packed.
        std::vector<__nv_bfloat16> dense(
            static_cast<std::size_t>(kRows) * kNarrowVocab);
        for (int r = 0; r < kRows; ++r)
            for (int c = 0; c < kNarrowVocab; ++c)
                dense[static_cast<std::size_t>(r) * kNarrowVocab + c] =
                    host[static_cast<std::size_t>(r) * kVocab + c];
        __nv_bfloat16* d_dense = nullptr;
        RT(cudaMalloc(&d_dense, dense.size() * sizeof(__nv_bfloat16)));
        RT(cudaMemcpy(d_dense, dense.data(),
                      dense.size() * sizeof(__nv_bfloat16),
                      cudaMemcpyHostToDevice));
        launch_argmax_bf16(d_dense, d_ref, kRows, kNarrowVocab, nullptr);
        RT(cudaDeviceSynchronize());
        std::vector<std::int32_t> narrow_ref(kRows);
        RT(cudaMemcpy(narrow_ref.data(), d_ref, kRows * sizeof(std::int32_t),
                      cudaMemcpyDeviceToHost));

        check_chunked("padded row stride", d_padded, kRows, kNarrowVocab,
                      kStride, 1024, narrow_ref);

        cudaFree(d_padded);
        cudaFree(d_dense);
    }

    cudaFree(d_logits);
    cudaFree(d_ref);

    std::printf(g_fail ? "\nCHUNKED_ARGMAX FAILED (%d)\n"
                       : "\nALL PASS (0 failures)\n",
                g_fail);
    return g_fail ? 1 : 0;
}
