//===-- test_attn_score_fold.cu ---------------------------------*- CUDA -*-===//
//
// Head-fold parity for the `AttnScore` capture (design doc §3).
//
// `test_attn_score_capture` proves the kernel deposits the right
// `p[head, kv_idx]`. This proves the second half of the path: folding that
// `[num_q_heads, kv_len]` block down to the `[kv_len]` row a PTIR program
// actually reads.
//
// Three claims, all of which have a silent-failure mode if wrong:
//
//  1. The fold averages the right elements. A `[H, kv_len]` block is row-major
//     per request, so folding position `i` means striding by `kv_len` -- and
//     `kv_len` is DERIVED here from the page CSR, exactly as the driver derives
//     it everywhere else. If the kernel took a length argument instead, a
//     disagreement between the two would mis-attribute every score past the
//     first request without ever faulting.
//
//  2. Each folded row sums to 1. Every head's row is a softmax distribution, so
//     the mean over heads is one too. This is the property every Track B policy
//     leans on when it thresholds the row in absolute terms.
//
//  3. The ragged offsets line up across requests. The folded offset is the raw
//     offset divided by `num_q_heads`; if that identity broke, request `r`'s
//     scores would land in request `r-1`'s row. Requests are given DIFFERENT
//     `kv_len`s here precisely so a stride bug cannot hide.
//
//===----------------------------------------------------------------------===//
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "ops/attention_flashinfer.hpp"

namespace ops = pie_cuda_driver::ops;

namespace {

int g_failures = 0;

void rt_check(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL rt %s:%d: %s -> %s\n", __FILE__, line, expr,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}

#define RT(expr) rt_check((expr), #expr, __LINE__)

void expect(bool condition, const char* what) {
    if (condition) {
        std::printf("  ok   %s\n", what);
        return;
    }
    std::printf("  FAIL %s\n", what);
    ++g_failures;
}

// Deliberately uneven: a partial last page, an exactly-full last page, and a
// single-page request. Anything that assumes a uniform `kv_len` breaks here.
struct Case {
    int pages;
    int last_page_len;
};

void run(int page_size, int num_q_heads, const std::vector<Case>& cases) {
    const int requests = static_cast<int>(cases.size());
    std::printf("[fold] page_size=%d heads=%d requests=%d\n", page_size,
                num_q_heads, requests);

    std::vector<std::uint32_t> page_indptr(requests + 1, 0);
    std::vector<std::uint32_t> last_page_lens(requests, 0);
    std::vector<int> kv_lens(requests, 0);
    std::vector<std::int32_t> raw_indptr(requests + 1, 0);
    std::vector<std::uint32_t> folded_indptr(requests + 1, 0);
    std::uint32_t pages_total = 0;
    std::int64_t raw_total = 0;
    std::int64_t folded_total = 0;
    for (int r = 0; r < requests; ++r) {
        page_indptr[r] = pages_total;
        pages_total += static_cast<std::uint32_t>(cases[r].pages);
        last_page_lens[r] = static_cast<std::uint32_t>(cases[r].last_page_len);
        kv_lens[r] = (cases[r].pages - 1) * page_size + cases[r].last_page_len;
        raw_indptr[r] = static_cast<std::int32_t>(raw_total);
        folded_indptr[r] = static_cast<std::uint32_t>(folded_total);
        raw_total += static_cast<std::int64_t>(kv_lens[r]) * num_q_heads;
        folded_total += kv_lens[r];
    }
    page_indptr[requests] = pages_total;
    raw_indptr[requests] = static_cast<std::int32_t>(raw_total);
    folded_indptr[requests] = static_cast<std::uint32_t>(folded_total);

    // Fill each (request, head) row with a real softmax distribution so the
    // sum-to-one claim is meaningful rather than tautological.
    std::mt19937 rng(0x5eed1234u);
    std::uniform_real_distribution<float> logit(-4.f, 4.f);
    std::vector<float> raw_h(static_cast<std::size_t>(raw_total), 0.f);
    for (int r = 0; r < requests; ++r) {
        for (int h = 0; h < num_q_heads; ++h) {
            float* row = raw_h.data() + raw_indptr[r] +
                         static_cast<std::size_t>(h) * kv_lens[r];
            float max_logit = -1e30f;
            for (int i = 0; i < kv_lens[r]; ++i) {
                row[i] = logit(rng);
                max_logit = std::fmax(max_logit, row[i]);
            }
            float sum = 0.f;
            for (int i = 0; i < kv_lens[r]; ++i) {
                row[i] = std::exp(row[i] - max_logit);
                sum += row[i];
            }
            for (int i = 0; i < kv_lens[r]; ++i) row[i] /= sum;
        }
    }

    float* raw_d = nullptr;
    float* folded_d = nullptr;
    std::int32_t* raw_indptr_d = nullptr;
    std::uint32_t* page_indptr_d = nullptr;
    std::uint32_t* last_page_lens_d = nullptr;
    RT(cudaMalloc(&raw_d, raw_h.size() * sizeof(float)));
    RT(cudaMalloc(&folded_d,
                  static_cast<std::size_t>(folded_total) * sizeof(float)));
    RT(cudaMalloc(&raw_indptr_d, raw_indptr.size() * sizeof(std::int32_t)));
    RT(cudaMalloc(&page_indptr_d, page_indptr.size() * sizeof(std::uint32_t)));
    RT(cudaMalloc(&last_page_lens_d,
                  last_page_lens.size() * sizeof(std::uint32_t)));
    RT(cudaMemcpy(raw_d, raw_h.data(), raw_h.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(raw_indptr_d, raw_indptr.data(),
                  raw_indptr.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(page_indptr_d, page_indptr.data(),
                  page_indptr.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(last_page_lens_d, last_page_lens.data(),
                  last_page_lens.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    // A poison fill: the kernel must write every folded slot, so a leftover
    // value would show up as a large error rather than as a plausible zero.
    RT(cudaMemset(folded_d, 0x7f,
                  static_cast<std::size_t>(folded_total) * sizeof(float)));

    ops::launch_attn_score_fold_heads(raw_d, raw_indptr_d, page_indptr_d,
                                      last_page_lens_d, page_size, requests,
                                      num_q_heads, folded_d, nullptr);
    RT(cudaGetLastError());
    RT(cudaDeviceSynchronize());

    std::vector<float> folded_h(static_cast<std::size_t>(folded_total), 0.f);
    RT(cudaMemcpy(folded_h.data(), folded_d, folded_h.size() * sizeof(float),
                  cudaMemcpyDeviceToHost));

    double worst_value = 0.0;
    double worst_sum = 0.0;
    for (int r = 0; r < requests; ++r) {
        // Claim 3: the folded offset is the raw offset over the head count.
        const std::uint32_t derived =
            static_cast<std::uint32_t>(raw_indptr[r]) /
            static_cast<std::uint32_t>(num_q_heads);
        if (derived != folded_indptr[r]) {
            std::printf("  FAIL request %d offset %u != %u\n", r, derived,
                        folded_indptr[r]);
            ++g_failures;
        }
        double row_sum = 0.0;
        for (int i = 0; i < kv_lens[r]; ++i) {
            // Claim 1: reference mean over heads, computed independently.
            double reference = 0.0;
            for (int h = 0; h < num_q_heads; ++h) {
                reference += raw_h[static_cast<std::size_t>(raw_indptr[r]) +
                                   static_cast<std::size_t>(h) * kv_lens[r] +
                                   static_cast<std::size_t>(i)];
            }
            reference /= num_q_heads;
            const double got = folded_h[folded_indptr[r] + i];
            worst_value = std::fmax(worst_value, std::fabs(got - reference));
            row_sum += got;
        }
        worst_sum = std::fmax(worst_sum, std::fabs(row_sum - 1.0));
    }

    std::printf("  max|fold-ref| = %.3e   max|rowsum-1| = %.3e\n", worst_value,
                worst_sum);
    expect(worst_value < 1e-6, "folded row matches the per-head mean");
    expect(worst_sum < 1e-5, "each folded row is still a distribution");

    RT(cudaFree(raw_d));
    RT(cudaFree(folded_d));
    RT(cudaFree(raw_indptr_d));
    RT(cudaFree(page_indptr_d));
    RT(cudaFree(last_page_lens_d));
}

}  // namespace

int main() {
    // Mixed geometry: partial last page, exactly-full last page, single page.
    run(/*page_size=*/16, /*num_q_heads=*/8,
        {{5, 7}, {1, 16}, {3, 1}, {12, 16}});
    // A head count that is not a warp multiple, and a wide row that forces the
    // grid-stride loop in the kernel to take more than one step.
    run(/*page_size=*/32, /*num_q_heads=*/5, {{100, 3}, {2, 32}});
    // Single request, single head: the degenerate case where a stride bug
    // cannot be seen, kept as a floor.
    run(/*page_size=*/8, /*num_q_heads=*/1, {{4, 5}});

    if (g_failures != 0) {
        std::printf("FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
