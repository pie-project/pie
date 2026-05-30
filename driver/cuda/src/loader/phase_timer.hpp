#pragma once

// Tiny scoped timer that adds its lifetime (ms) to *sink on destruction. Used to
// attribute a weight load to its phases (alloc / transfer / transform) for the
// PIE_WEIGHT_LOADER_PROFILE output. Shared by the storage executor and the copy
// engine so both report into the same LoadExecutionStats phase counters.

#include <chrono>

namespace pie_cuda_driver {

struct PhaseTimer {
    std::chrono::steady_clock::time_point t0;
    double* sink;
    explicit PhaseTimer(double* s)
        : t0(std::chrono::steady_clock::now()), sink(s) {}
    ~PhaseTimer()
    {
        *sink += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }
};

}  // namespace pie_cuda_driver
