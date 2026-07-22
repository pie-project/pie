// Direct-kernel regression for the AWQ -> Marlin weight-lowering that Pie's
// cuda_native driver applies to asymmetric AWQ-INT4 checkpoints (quant_method
// "awq", zero_point=true, e.g. Qwen2.5-72B-Instruct-AWQ).
//
// The zero-point kernel (launch_awq_qzero_to_marlin_w4) must reproduce vLLM's
// `awq_to_marlin_zero_points`: AWQ undo-interleave, the 64-wide Marlin
// scale permutation, the 8-wide Marlin interleave [0,2,4,6,1,3,5,7], then a
// linear int4 repack. Omitting the 8-wide interleave silently corrupts every
// group's zero-points -> (q - z)*s is wrong on every output channel and decode
// collapses to a single token. This test allocates device AWQ qzeros, runs the
// kernel, copies the result back, and compares against an INDEPENDENT host
// reference of the same pipeline; it also asserts that the naive linear-slot
// mapping (no 8-wide interleave) would differ, so a regression is caught.
//
// The AWQ qweight path (launch_awq_qweight_to_gptq_w4) is an existing working
// path and is covered here too against an independent plain-GPTQ host unpack.
// The scale path (launch_marlin_permute_scales_bf16) is the shared Marlin
// scale permutation exercised by every GPTQ/AWQ checkpoint and is verified by
// test_transcode_fused; it is intentionally not duplicated here.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/dtype_cast.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                            \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA FAIL %s:%d: %s (%s)\n", __FILE__,    \
                         __LINE__, #expr, cudaGetErrorString(_err));        \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

template <typename T>
T* device_from_host(const std::vector<T>& host) {
    T* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(
        device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template <typename T>
std::vector<T> host_from_device(const T* device, std::size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(
        host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

// Deterministic 4-bit fill packed 8 nibbles/int32 along N: shape [rows, cols/8].
std::vector<std::uint32_t> make_packed(int rows, int cols, std::uint32_t seed) {
    const int packed = cols / 8;
    std::vector<std::uint32_t> out(static_cast<std::size_t>(rows) * packed, 0);
    std::uint32_t s = seed;
    auto next_nibble = [&]() -> std::uint32_t {
        s = s * 1664525u + 1013904223u;
        return (s >> 12) & 0xFu;
    };
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < packed; ++b) {
            std::uint32_t v = 0;
            for (int i = 0; i < 8; ++i) v |= next_nibble() << (4 * i);
            out[static_cast<std::size_t>(r) * packed + b] = v;
        }
    }
    return out;
}

// Independent host reference for vLLM `awq_to_marlin_zero_points`.
// AWQ qzeros [groups, N/8] int32 -> marlin qzeros [groups, N/8] int32.
// `marlin_interleave=false` reproduces the pre-fix linear-slot mapping.
std::vector<std::uint32_t> ref_awq_qzero_to_marlin(
    const std::vector<std::uint32_t>& in, int groups, int N,
    bool marlin_interleave) {
    const int n8 = N / 8;
    const int total = groups * N;
    static const int undo[8]  = {0, 4, 1, 5, 2, 6, 3, 7};  // undo AWQ interleave
    static const int inter[8] = {0, 2, 4, 6, 1, 3, 5, 7};  // marlin_zero_points
    // 1. unpack_cols (linear nibble order).
    std::vector<std::uint8_t> q(total);
    for (int r = 0; r < groups; ++r)
        for (int col = 0; col < N; ++col)
            q[r * N + col] =
                (in[r * n8 + col / 8] >> (4 * (col % 8))) & 0xFu;
    // 2. undo AWQ interleave over each 8-wide block.
    std::vector<std::uint8_t> a(total);
    for (int blk = 0; blk < total / 8; ++blk)
        for (int m = 0; m < 8; ++m) a[blk * 8 + m] = q[blk * 8 + undo[m]];
    // 3a. 64-wide scale_perm: perm[i*8+j] = i + 8*j.
    int sp[64];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) sp[i * 8 + j] = i + 8 * j;
    std::vector<std::uint8_t> b(total);
    for (int blk = 0; blk < total / 64; ++blk)
        for (int p = 0; p < 64; ++p) b[blk * 64 + p] = a[blk * 64 + sp[p]];
    // 3b. 8-wide marlin interleave (or linear for the pre-fix reference).
    std::vector<std::uint8_t> c(total);
    for (int blk = 0; blk < total / 8; ++blk)
        for (int m = 0; m < 8; ++m)
            c[blk * 8 + m] = marlin_interleave ? b[blk * 8 + inter[m]]
                                               : b[blk * 8 + m];
    // 3c. pack_cols (linear).
    std::vector<std::uint32_t> out(static_cast<std::size_t>(groups) * n8, 0);
    for (int r = 0; r < groups; ++r)
        for (int base = 0; base < n8; ++base) {
            std::uint32_t v = 0;
            for (int i = 0; i < 8; ++i)
                v |= static_cast<std::uint32_t>(c[r * N + base * 8 + i] & 0xF)
                     << (4 * i);
            out[static_cast<std::size_t>(r) * n8 + base] = v;
        }
    return out;
}

// Independent host reference for AWQ qweight -> plain-GPTQ (the input Pie feeds
// to the shared gptq_marlin_repack). AWQ qweight [K, N/8] -> GPTQ [K/8, N].
std::vector<std::uint32_t> ref_awq_qweight_to_gptq(
    const std::vector<std::uint32_t>& in, int K, int N) {
    const int n8 = N / 8;
    static const int reverse[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<std::uint32_t> out(static_cast<std::size_t>(K / 8) * N, 0);
    for (int k8 = 0; k8 < K / 8; ++k8)
        for (int n = 0; n < N; ++n) {
            std::uint32_t v = 0;
            for (int i = 0; i < 8; ++i) {
                const int k = k8 * 8 + i;
                const std::uint32_t w =
                    (in[static_cast<std::size_t>(k) * n8 + n / 8] >>
                     (4 * reverse[n % 8])) & 0xFu;
                v |= w << (4 * i);
            }
            out[static_cast<std::size_t>(k8) * N + n] = v;
        }
    return out;
}

// groups corresponds to K / group_size (e.g. groups=8 <-> K=1024 at gs=128).
void run_zero_point(int groups, int N, std::uint32_t seed) {
    const int n8 = N / 8;
    const auto awq = make_packed(groups, N, seed);
    std::uint32_t* d_in = device_from_host(awq);
    std::uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, awq.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, awq.size() * sizeof(std::uint32_t)));
    pie_cuda_driver::kernels::launch_awq_qzero_to_marlin_w4(
        d_in, d_out, groups, N, /*stream=*/0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    const auto got = host_from_device(d_out, awq.size());
    const auto want = ref_awq_qzero_to_marlin(awq, groups, N, /*interleave=*/true);
    const auto pre_fix = ref_awq_qzero_to_marlin(awq, groups, N, /*interleave=*/false);

    bool eq = true;
    std::size_t diff_words = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) eq = false;
        if (want[i] != pre_fix[i]) ++diff_words;
    }
    // Kernel output must match vLLM's awq_to_marlin_zero_points bit-for-bit.
    CHECK(eq);
    // The 8-wide interleave [0,2,4,6,1,3,5,7] relocates six of every eight
    // lanes, so the correct layout must differ from the naive linear-slot
    // mapping across most words. Requiring a large mismatch count both proves
    // the interleave is load-bearing and confirms the deterministic inputs
    // disambiguate the lanes (identical inputs would make this fail).
    CHECK(diff_words >= want.size() / 2);
    if (!eq)
        std::fprintf(stderr, "  zero_point groups=%d N=%d mismatch\n", groups, N);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    (void)n8;
}

void run_qweight(int K, int N, std::uint32_t seed) {
    const auto awq = make_packed(K, N, seed);
    std::uint32_t* d_in = device_from_host(awq);
    std::uint32_t* d_out = nullptr;
    const std::size_t out_count = static_cast<std::size_t>(K / 8) * N;
    CUDA_CHECK(cudaMalloc(&d_out, out_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, out_count * sizeof(std::uint32_t)));
    pie_cuda_driver::kernels::launch_awq_qweight_to_gptq_w4(
        d_in, d_out, K, N, /*stream=*/0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    const auto got = host_from_device(d_out, out_count);
    const auto want = ref_awq_qweight_to_gptq(awq, K, N);
    bool eq = true;
    for (std::size_t i = 0; i < want.size(); ++i)
        if (got[i] != want[i]) eq = false;
    CHECK(eq);
    if (!eq) std::fprintf(stderr, "  qweight K=%d N=%d mismatch\n", K, N);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

}  // namespace

int main() {
    // Deterministic shapes; N must be a multiple of 64 (Marlin). Includes the
    // checkpoint-relevant group-size-128 geometry (groups = K / 128).
    run_zero_point(/*groups=*/2, /*N=*/128, 1u);
    run_zero_point(/*groups=*/4, /*N=*/256, 2u);
    run_zero_point(/*groups=*/8, /*N=*/512, 3u);   // K=1024 @ gs=128
    run_zero_point(/*groups=*/5, /*N=*/320, 4u);

    run_qweight(/*K=*/64, /*N=*/128, 5u);
    run_qweight(/*K=*/256, /*N=*/256, 6u);

    if (g_failures) {
        std::fprintf(stderr, "test_awq_marlin_zero_points: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::printf("test_awq_marlin_zero_points: OK\n");
    return 0;
}
