// The Tier A baseline: what `<<<>>>` costs to issue, for the same kernel.
//
// The number this prints is the one `tests/tier_a_pilot.rs`'s
// `the_issue_cost_of_a_stated_launch` is compared against, and the two are
// written to measure the same thing: the host cost of getting one launch onto
// a stream, including whatever the caller must compute to do it.
//
// So this loop is not a bare `<<<>>>`. It calls `tanh_bf16` -- the real
// launcher, out of `norm/altup_aux.cu`, with its `T <= 0` guard, its `BLOCK`
// constant, its round-up and its `static_cast` -- because that is the work
// `bind::launch::eval` plus `bind::device::Args::bind` replaces. Timing a
// launcher against a launch primitive would be measuring the wrong pair.
//
//   nvcc -O2 -std=c++20 -arch=sm_89 -I crates/kernels-cuda/csrc/src \
//        scripts/tier_a_baseline.cu crates/kernels-cuda/csrc/src/norm/altup_aux.cu \
//        -o /tmp/tier_a_baseline && /tmp/tier_a_baseline

#include <chrono>
#include <cstdio>

#include <cuda_runtime.h>

#include "norm/altup_aux.hpp"

int main() {
    constexpr int T = 16;
    constexpr int H = 2048;
    constexpr int N = 2000;

    void* x = nullptr;
    if (cudaMalloc(&x, static_cast<size_t>(T) * H * 2) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }
    cudaMemset(x, 0, static_cast<size_t>(T) * H * 2);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // The same warm-up the Rust side takes, and for the same reason: the
    // first launch of a kernel pays for its module's lazy load.
    for (int i = 0; i < 50; ++i) {
        pie_cuda_driver::kernels::norm::tanh_bf16(x, T * H, stream);
    }
    cudaStreamSynchronize(stream);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        pie_cuda_driver::kernels::norm::tanh_bf16(x, T * H, stream);
    }
    const auto issued = std::chrono::steady_clock::now();
    cudaStreamSynchronize(stream);
    const auto retired = std::chrono::steady_clock::now();

    const double us_issue =
        std::chrono::duration<double, std::micro>(issued - start).count() / N;
    const double us_retire =
        std::chrono::duration<double, std::micro>(retired - start).count() / N;
    std::printf("baseline issue cost: %.3f us/launch (issue), %.3f us/launch (to retire), n=%d\n",
                us_issue, us_retire, N);

    cudaStreamDestroy(stream);
    cudaFree(x);
    return 0;
}
