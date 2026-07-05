#pragma once
// harness.hpp — single-step latency rig + single-kernel micro-bench for the raw-Metal
// decode path. Reports encode-ms vs GPU-exec-ms SEPARATELY (manager's ask: both counters).
//
//   * LatencyHarness::time_step — drives RawMetalContext::run_step over warmup + N iters,
//     returns median encode_ms / gpu_exec_ms. This is the per-token decode-step timer
//     beta fills with the real DAG (it gets the byte-identical CB via I1/I2).
//   * LatencyHarness::bench_kernel — times a SINGLE dispatch (one PSO + one arg table)
//     so delta can A/B each ported kernel's exec-ms against MLX's kernel.

#include <cstdint>
#include <string>
#include <vector>

#include "mtl4_context.hpp"

namespace pie_metal_driver::raw_metal {

struct BenchResult {
    StepTiming median;
    StepTiming p10;   // best-case (10th pct) encode/exec
    int        iters   = 0;
    int        warmup  = 0;
    std::string label;
};

class LatencyHarness {
  public:
    explicit LatencyHarness(RawMetalContext& ctx) : ctx_(ctx) {}

    // Time a full decode step: `encode_fn` issues the whole per-step DAG.
    // Alternates the double-buffered allocator across iters (encode/GPU overlap-ready).
    BenchResult time_step(const std::string& label,
                          const std::function<void(StepEncoder&)>& encode_fn,
                          int iters = 200, int warmup = 40);

    // Micro-bench a single kernel: binds `argtable_kernel`@`layer`, dispatches once.
    // The arg table must already be bound (ctx.arg_bind) + ctx.make_resident() called.
    BenchResult bench_kernel(const std::string& label, Pso pso, Kernel argtable_kernel,
                             int layer, Grid grid, Threadgroup tg,
                             int iters = 500, int warmup = 50);

  private:
    RawMetalContext& ctx_;
};

// Pretty one-line print (encode-ms vs gpu-exec-ms split).
void print_result(const BenchResult& r);

}  // namespace pie_metal_driver::raw_metal
