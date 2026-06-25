#pragma once
// decode_timing.hpp — beta's GPU-exec attribution for the raw-Metal decode step.
//
// Purpose (manager's optimization-phase prep): once the step is correct (argmax 264),
// we need to know WHICH dispatches dominate gpu-exec-ms so we know what to fuse/cut to
// BEAT mlx_lm (not just match). This module attributes the single-CB step's GPU time
// down to per-dispatch / per-kernel-kind / per-layer granularity via MTL4 timestamp
// counter-sampling.
//
// Mechanism (validated end-to-end in files/icb-probes/mtl4_tsattrib.mm):
//   * A timestamp MTL4CounterHeap with (dag.size()+1) entries.
//   * The DAG walker (encode_decode_step) emits a timestamp mark at boundary i BEFORE
//     dispatch i, plus a final mark after the last dispatch. Boundary i == dispatch
//     ordinal i. Diffing consecutive resolved timestamps yields dispatch i's GPU time.
//   * On this box the GPU timestamp domain is NANOSECONDS (calibrated 1.0 ns/tick via
//     MTLDevice sampleTimestamps:gpuTimestamp: — see probe), so the resolved uint64s are
//     ns directly; attribute_step still takes an explicit ns/tick for portability.
//
// Ownership seam: this module + the mark-scheduling in encode_decode_step are beta's
// (device-agnostic analysis). The ~15-line Obj-C timestamp plumbing
// (newCounterHeapWithDescriptor / writeTimestampWithGranularity:intoHeap:atIndex: /
// resolveCounterRange:) is alpha's StepEncoder/RawMetalContext seam — see
// StepTimingHook below + the request posted to #mac. Zero perturbation when unused
// (the hook is null on the production path; default build never marks).

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "decode_step.hpp"  // Dispatch, kKernelKindCount, Kernel

namespace pie_metal_driver::raw_metal {

// Human-readable kernel name (for the attribution report / charlie's dump tags).
const char* kernel_name(Kernel k);

// One dispatch's attributed GPU time.
struct DispatchAttribution {
    int    ordinal = 0;
    Kernel kind    = Kernel::EmbedGather;
    int    layer   = -1;
    double gpu_ms  = 0.0;
};

// Full per-step attribution: per-dispatch + aggregates by kernel-kind and by layer.
struct StepAttribution {
    std::vector<DispatchAttribution>           per_dispatch;
    std::array<double, kKernelKindCount>       by_kind{};    // summed gpu_ms per kind
    std::array<int,    kKernelKindCount>       count_kind{}; // #dispatches per kind
    double                                     total_gpu_ms = 0.0;
    bool                                       valid        = false;
};

// Attribute a step from resolved timestamp boundaries.
//   dag            — the encoded DAG (kind/ordinal/layer per dispatch).
//   boundary_ticks — n_boundaries raw timestamps, where n_boundaries == dag.size()+1.
//                    boundary_ticks[i] = GPU time the walker reached dispatch i;
//                    boundary_ticks[dag.size()] = GPU time after the last dispatch.
//   ns_per_tick    — GPU-tick→ns scale (1.0 on this box; calibrate via sampleTimestamps).
// Returns an attribution with valid=false if the boundary count doesn't match the DAG.
StepAttribution attribute_step(const std::vector<Dispatch>& dag,
                               const uint64_t* boundary_ticks,
                               size_t n_boundaries,
                               double ns_per_tick = 1.0);

// Print a fusion/cut-oriented report: per-kind totals sorted DESC (the optimization
// targets), per-layer rollup, and the top-N hottest individual dispatches.
void print_attribution(const StepAttribution& a, const char* title,
                       int top_n = 16, FILE* out = stdout);

}  // namespace pie_metal_driver::raw_metal
