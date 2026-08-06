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

    /// Threadgroups (counted at BN=32) past which BN=32 beats BN=16.
    ///
    /// M1 Max: 160. The sweep in `decode_dispatch_mb.hpp` brackets it between
    /// 144 (16 still wins) and 192 (32 wins).
    ///
    /// M4 Pro (Apple9, 20 cores): 96. Re-measured with `roofline_probe` on
    /// this machine, BM=64, GFLOP/s, the same projections the M1 sweep used.
    /// Threadgroups here are `(N_out/32) * ceil(M/64)`:
    ///
    ///     tg@32    BN=16    BN=32    delta
    ///        64     5004     4615    -7.8%   (M=128, N=1024)
    ///        96     5540     5863    +5.8%   (M=192, N=1024)
    ///       128     5829     5846    +0.3%   (M=256, N=1024)
    ///       160     5987     6411    +7.1%   (M=320, N=1024)
    ///       224     5980     6196    +3.6%   (M=448, N=1024)
    ///
    /// The crossover moved DOWN, and it moved down for the reason the machine
    /// has fewer cores: 20 against the M1 Max's 32, so it takes fewer
    /// threadgroups to fill and the wide tile's smaller grid stops being a
    /// starvation risk sooner. 160 is not merely conservative here, it is on
    /// the wrong side of a 5.8% gap at tg=96. The bracket is (64, 96]; 96 is
    /// the measured point and not the midpoint, which is the honest choice
    /// when only one side of the gap was sampled.
    ///
    /// End-to-end this is NOT a win, and the entry says so rather than
    /// implying one. Same binary, arms alternated on `PIE_METAL_QMM_BN_CROSSOVER_TG`,
    /// 160 against 96: gpt-oss-20b-Q4 179.68 -> 180.03 tok/s (3 reps),
    /// gemma-4-E2B 621.60 -> 622.67 (2), Llama-3.2-3B 558.40 -> 560.85 (3),
    /// and Llama-3.2-3B on a 3000-word prefill 13.09 -> 13.14 (2). Every
    /// workload moves the same direction and none of them moves far enough to
    /// clear the noise; what carries the change is the kernel measurement plus
    /// four out of four agreeing signs, not any one of these numbers. The
    /// reason the GEMM's 5.8% does not survive to the step is in the probe's
    /// other column: these projections run at ~5% of the streaming roof, so
    /// the step is bound by weight traffic and not by how the tiles are cut.
    int qmm_bn_crossover_tg = 160;
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

/// The BN=16 -> BN=32 tile crossover, in threadgroups counted at BN=32.
/// See `DeviceTuning::qmm_bn_crossover_tg`.
int qmm_bn_crossover_tg();

}  // namespace pie::metal
