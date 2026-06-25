// harness.cpp — latency / micro-bench driver (pure C++ over RawMetalContext).

#include "harness.hpp"

#include <algorithm>
#include <cstdio>

namespace pie_metal_driver::raw_metal {

namespace {
double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}
}  // namespace

BenchResult LatencyHarness::time_step(const std::string& label,
                                      const std::function<void(StepEncoder&)>& encode_fn,
                                      int iters, int warmup) {
    std::vector<double> enc, gpu;
    enc.reserve(iters);
    gpu.reserve(iters);
    for (int it = 0; it < iters + warmup; ++it) {
        StepTiming t = ctx_.run_step(encode_fn, /*ab=*/it & 1);
        if (it >= warmup) { enc.push_back(t.encode_ms); gpu.push_back(t.gpu_exec_ms); }
    }
    BenchResult r;
    r.label = label;
    r.iters = iters;
    r.warmup = warmup;
    r.median.encode_ms   = median(enc);
    r.median.gpu_exec_ms = median(gpu);
    r.p10.encode_ms      = percentile(enc, 0.10);
    r.p10.gpu_exec_ms    = percentile(gpu, 0.10);
    return r;
}

BenchResult LatencyHarness::bench_kernel(const std::string& label, Pso pso,
                                         Kernel argtable_kernel, int layer, Grid grid,
                                         Threadgroup tg, int iters, int warmup) {
    auto encode_fn = [&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable(argtable_kernel, layer);
        se.dispatch(grid, tg);
        se.barrier();
    };
    return time_step(label, encode_fn, iters, warmup);
}

void print_result(const BenchResult& r) {
    printf("  %-22s encode %.4f ms | gpu-exec %.4f ms | total %.4f ms  "
           "(p10 enc %.4f / gpu %.4f, n=%d)\n",
           r.label.c_str(), r.median.encode_ms, r.median.gpu_exec_ms,
           r.median.total_ms(), r.p10.encode_ms, r.p10.gpu_exec_ms, r.iters);
}

}  // namespace pie_metal_driver::raw_metal
