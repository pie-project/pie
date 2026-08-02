#pragma once

// Region support: the host-side analysis and ABI the Metal launch path still
// performs on a decoded plan.
//
// This is what survived the in-driver emitter (`m1_codegen.{cpp,hpp}`, deleted
// in the compiler/ consolidation). That file was a pure MSL string builder
// and its output now comes from the host (`compiler/codegen`, shipped in
// `PieProgramDesc::emitted_kernels`); the constants, ABI structs and the
// metadata walker below are NOT emission and are still called at bind and at
// launch:
//
//   * `M1ChannelEffect` — per-channel readiness/commit effect derived from
//     the bound trace; the M1 host readiness check and the effect-kernel
//     binding both read it.
//   * `M1OpMeta` — one entry per singleton region, populated once at bind
//     and read at execute to fill the per-op `DeviceOpParams`. The walk is
//     a deterministic function of the plan; the host runs the same walk in
//     `compiler/codegen/src/metal/validate.rs` before it emits.
//   * `kMetalM1EmitterVersion` — pins this driver's compile cache identity
//     to the host emitter it was verified against, so a stale mtl4archive
//     never survives a host-side emitter bump.
//   * `kMetalM1MaxChannels` — the hard channel count the M1 lane table can
//     address per lane; a stage above this is refused at register.
//   * `kMetalM2MaxFusedChannels` — the direct-binding limit past which a
//     fused kernel would need more argument buffers than the driver wires
//     up. The host honours the same limit and pushes a KERNEL_FUSED entry
//     with an error when the stage's binding count exceeds it, so the
//     driver's fallback to per-op M1 launches remains identical.
//
// Under the launch-package north star (`ptir-refactor.md` §3′) even this
// header goes away; until then it is the only place these live.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pie/driver/launch/plan.hpp"

namespace pie::metal::pipeline {

inline constexpr std::uint16_t kMetalM1EmitterVersion = 23;
inline constexpr std::size_t kMetalM1MaxChannels = 29;
inline constexpr std::size_t kMetalM2MaxFusedChannels = 12;

struct M1ChannelEffect {
    bool requires_full = false;
    bool requires_empty = false;
    bool take = false;
    bool put = false;
    std::uint32_t capacity = 1;
};

struct M1OpMeta {
    std::uint32_t node = 0;
    std::uint32_t result_base = 0;
    pie::driver::launch::plan::PlanOp op;
};

// Walk the plan's singleton partition and populate `M1OpMeta` per op. This
// is not validation: the host runs the equivalent check in
// `compiler/codegen/src/metal/validate.rs` and signals a stage-level
// rejection through the KERNEL_SINGLETON slot at region 0. If that signal
// arrives, the caller must reject before this walker ever runs; the walker
// itself assumes the plan is well-formed and result bases are the running
// prefix sum of `op.results` in container order.
inline std::vector<M1OpMeta> collect_singleton_metadata(
    const pie::driver::launch::plan::StagePlan& plan) {
    std::vector<M1OpMeta> operations;
    operations.reserve(plan.ops.size());
    std::uint32_t result_base = 0;
    for (std::size_t node = 0; node < plan.ops.size(); ++node) {
        const auto& op = plan.ops[node].op;
        operations.push_back(
            {static_cast<std::uint32_t>(node), result_base, op});
        result_base += op.results;
    }
    return operations;
}

}  // namespace pie::metal::pipeline
