// Per-device tuning constants.
//
// Every crossover in this driver was measured on one machine -- an M1 Max --
// and then written down as a `constexpr`. That was honest while there was one
// machine. On an M4 Pro the same constants are wrong in the direction that
// costs the most: the GEMM crossover sits three rows too high, so the batches
// where the GEMM already wins are still served by the GEMV.
//
// The rule this file exists to keep: a DEFAULT-CONSTRUCTED `DeviceTuning`
// reproduces the M1 numbers EXACTLY. Adding a device may never change what an
// unrecognised one does, and a machine this file has never heard of gets the
// constants that were measured rather than an extrapolation. Overrides are
// per-generation and each carries the measurement that justifies it.

#pragma once

#include <cstdint>

namespace pie::metal {

/// What the driver knows about the GPU it is running on. Queried once and
/// cached; every field has a value on every path, including the one where the
/// query fails.
struct DeviceInfo {
    /// `MTLGPUFamilyApple<N>`, resolved NEWEST-FIRST: the families are
    /// cumulative, so an M4 answers `supportsFamily:` for Apple7 as well as
    /// Apple9 and an oldest-first probe would call every Apple Silicon device
    /// an Apple7. 0 when no Metal device answered.
    int apple_family = 0;
    /// From IOKit's `gpu-core-count`, which is the only place the count is
    /// published -- `MTLDevice` does not expose it. 0 when absent.
    int gpu_core_count = 0;
};

/// The tuned constants, defaulted to the M1 Max measurements.
struct DeviceTuning {
    /// The batch at which the ported steel GEMM overtakes the batched GEMV.
    ///
    /// M1 Max: 12. Measured -- pie's per-step cost beats mlx-lm's at every
    /// batch up to 8 with the GEMV and only loses above it.
    ///
    /// M4 Pro (Apple9, 20 cores): 8. Measured on gemma-4-E4B at the batch
    /// that discriminates -- concurrency 8, where 12 takes the GEMV and 8
    /// takes the GEMM -- same binary via the env override, arms alternated,
    /// quiet host: 12 -> 140.88, 136.92 tok/s; 8 -> 144.44, 143.63. Means
    /// 138.90 vs 144.04, +3.7%. (At concurrency 16 both settings take the
    /// same path and measure the same, which is the check that the difference
    /// above is the crossover and not the weather.) The M4's wider per-core
    /// matrix throughput moves the crossover down; the GEMV's advantage at
    /// small N is a memory-bound one and does not.
    int qmm_min_batch = 12;

    /// The threadgroup count at which the unsplit GEMM's BN=32 tile overtakes
    /// BN=16.
    ///
    /// M1 Max: 160. Measured with `roofline_probe` over fifteen (M, N) pairs
    /// and then checked against a real llama-3.2-1B prefill, which agrees --
    /// 2565.8 / 2663.7 / 2578.3 tok/s at BN 16/32/64, 448 rows. The sweep
    /// brackets the crossover between 144 threadgroups and 192, so this sits in
    /// the gap.
    ///
    /// A machine with more cores saturates later and wants this HIGHER, since
    /// the narrow tile's only advantage is that it makes more threadgroups.
    int qmm_bn_crossover_tg = 160;

    /// Rows an expert's run must hold before the mixture's GEMM takes a wider
    /// row tile: 32 above `moe_tile_mid_per`, 64 above `moe_tile_wide_per`.
    ///
    /// M1 Max: 12 and 88. Measured end to end over eight (rows, expert count)
    /// pairs on gpt-oss-20b and gemma-4-26B; the table is in
    /// `shared_kernels.hpp`. Measured END TO END on purpose --
    /// `roofline_probe` predicts the dense tile correctly and this one wrong
    /// three times over, because its single hot expert cannot show what
    /// thirty-two cold ones cost.
    ///
    /// These move with per-core matrix throughput the way `qmm_min_batch`
    /// does: a machine that runs a wide tile relatively faster wants both
    /// thresholds LOWER.
    int moe_tile_mid_per = 12;
    int moe_tile_wide_per = 88;
};

/// The platform query. `device_tuning_apple.mm` on Apple, a stub in
/// `device_tuning.cpp` elsewhere. Call `device_info()` instead; this is the
/// seam, not the accessor.
DeviceInfo query_device_info();

/// The device this process is running on. Queried once on first call.
const DeviceInfo& device_info();

/// The tuning for this device. Queried once on first call.
///
/// `PIE_METAL_QMM_MIN_BATCH` overrides `qmm_min_batch` for a run, which is how
/// the crossover above was measured: a rebuild between arms is a different
/// binary, and a different binary is a different measurement.
const DeviceTuning& device_tuning();

/// The GEMM crossover for this device. A function and not a constant because
/// it is a property of the machine; see `DeviceTuning::qmm_min_batch`.
int qmm_min_batch();

/// The unsplit GEMM's BN crossover, in threadgroups.
int qmm_bn_crossover_tg();

/// The mixture's two row-tile crossovers, in rows per expert.
int moe_tile_mid_per();
int moe_tile_wide_per();

}  // namespace pie::metal
