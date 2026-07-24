// Ordering regression for the AWQ repack-source materialization in
// TranscodeEngine::repack_tile_map / materialize_repack_source.
//
// The repack kernels (e.g. launch_awq_qweight_to_gptq_w4) run on stream 0 and
// read a device "scratch" buffer that materialize_repack_source uploads from the
// checkpoint. That upload MUST be ordered before the kernel. The original code
// used the deferred, cross-stream copy_engine_.queue(), which only executes at
// the next copy_engine_.flush() -- after the kernel has launched and after the
// scratch DeviceTensor is freed. The kernel therefore read uninitialized device
// memory (compute-sanitizer initcheck: "Uninitialized __global__ memory read" in
// awq_qweight_to_gptq_w4_kernel), producing corrupted weights, and the deferred
// reader-lane copy later wrote to freed memory (a pre-readiness SIGSEGV). The fix
// issues the upload with copy_engine_.queue_on_stream(..., /*stream=*/0) so the
// stream-0 kernel sees the fully uploaded source via implicit stream ordering,
// mirroring the encode/fp8 materialize paths.
//
// This test reproduces that ordering invariant deterministically with the REAL
// kernel: when the H2D upload is ordered before the kernel on stream 0 the output
// matches an independent host reference; when the kernel is (mis)ordered ahead of
// the upload it reads the poison the scratch was left with and the output is
// wrong. The exact call-site fails-before/passes-after is additionally covered by
// the compute-sanitizer initcheck re-gate on the real 72B load.

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
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < packed; ++b) {
            std::uint32_t v = 0;
            for (int i = 0; i < 8; ++i) v |= next_nibble() << (4 * i);
            out[static_cast<std::size_t>(r) * packed + b] = v;
        }
    return out;
}

// Independent host reference for AWQ qweight [K, N/8] -> plain-GPTQ [K/8, N],
// matching awq_qweight_to_gptq_w4_kernel.
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

// The 0xFF poison the scratch is left with stands in for uninitialized device
// memory; repacking it gives a deterministic "wrong" output distinct from the
// real source's repack.
constexpr std::uint32_t kPoison = 0xFFFFFFFFu;

void run_ordering(int K, int N, std::uint32_t seed) {
    const auto awq = make_packed(K, N, seed);            // AWQ qweight [K, N/8]
    const std::size_t span_words = awq.size();
    const std::size_t span_bytes = span_words * sizeof(std::uint32_t);
    const std::size_t out_count = static_cast<std::size_t>(K / 8) * N;

    const auto want = ref_awq_qweight_to_gptq(awq, K, N);
    const std::vector<std::uint32_t> poison(span_words, kPoison);
    const auto want_poison = ref_awq_qweight_to_gptq(poison, K, N);

    std::uint32_t* scratch = nullptr;
    std::uint32_t* out = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, span_bytes));
    CUDA_CHECK(cudaMalloc(&out, out_count * sizeof(std::uint32_t)));

    // ORDERED (the fix): poison, upload the source on stream 0, THEN launch the
    // kernel on stream 0. Stream ordering guarantees the kernel reads the source.
    CUDA_CHECK(cudaMemset(scratch, 0xFF, span_bytes));
    CUDA_CHECK(cudaMemcpyAsync(scratch, awq.data(), span_bytes,
                               cudaMemcpyHostToDevice, /*stream=*/0));
    pie_cuda_driver::kernels::launch_awq_qweight_to_gptq_w4(
        scratch, out, K, N, /*stream=*/0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    const auto got_ordered = host_from_device(out, out_count);

    // MISORDERED (the pre-fix hazard): the kernel is enqueued on stream 0 BEFORE
    // the upload, so it reads the poison the scratch still holds.
    CUDA_CHECK(cudaMemset(scratch, 0xFF, span_bytes));
    pie_cuda_driver::kernels::launch_awq_qweight_to_gptq_w4(
        scratch, out, K, N, /*stream=*/0);
    CUDA_CHECK(cudaMemcpyAsync(scratch, awq.data(), span_bytes,
                               cudaMemcpyHostToDevice, /*stream=*/0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    const auto got_misordered = host_from_device(out, out_count);

    // The fix: the ordered upload yields the correct repack bit-for-bit.
    bool ordered_ok = (got_ordered == want);
    CHECK(ordered_ok);
    // The hazard is real and this test setup actually distinguishes the two
    // orderings: the misordered run repacks the poison, not the source, so it is
    // wrong and differs from the ordered run. (Guards against a test that would
    // pass even if ordering were ignored.)
    CHECK(got_misordered == want_poison);
    CHECK(got_misordered != want);
    CHECK(got_ordered != got_misordered);
    if (!ordered_ok)
        std::fprintf(stderr, "  ordering K=%d N=%d ordered-upload mismatch\n", K, N);

    CUDA_CHECK(cudaFree(scratch));
    CUDA_CHECK(cudaFree(out));
}

}  // namespace

int main() {
    // N multiple of 64 (Marlin), K multiple of 8 (kernel packs 8 along K).
    run_ordering(/*K=*/64, /*N=*/128, 5u);
    run_ordering(/*K=*/256, /*N=*/256, 6u);
    run_ordering(/*K=*/1024, /*N=*/512, 7u);  // gs-128-relevant geometry

    if (g_failures) {
        std::fprintf(stderr, "test_awq_repack_source_ordering: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::printf("test_awq_repack_source_ordering: OK\n");
    return 0;
}
