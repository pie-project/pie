//! Per-device tuning constants — every crossover this plane decides, and the
//! measurement that decided it.
//!
//! # The rule this module exists to keep
//!
//! A DEFAULT-CONSTRUCTED [`DeviceTuning`] reproduces the M1 Max numbers
//! EXACTLY. Adding a device may never change what an unrecognised one does,
//! and a machine this file has never heard of gets the constants that were
//! MEASURED rather than an extrapolation from the ones that were. Every
//! per-family override below carries the run that justifies it, and a family
//! with no entry inherits the default because nothing has measured it apart —
//! not because the default is thought to generalize.
//!
//! # No environment variables
//!
//! The reference driver swept these with a `PIE_METAL_*` override apiece, and
//! that mechanism does not come across: a shell here reads no environment
//! (constitution art. 9). What replaces it is [`Overrides`] — typed data, one
//! `Option` per field, filled from the boot document's own `[metal.tuning]`
//! table by `engine_metal::boot`. The sweep property that mattered is
//! preserved (the same binary, two arms, no rebuild between them); what is
//! gone is the ambient channel.
//!
//! The reference also recorded, at length, that a knob which silently reports
//! its default when you set it is worse than no knob — it spent two false
//! conclusions on an integer parser that folded `0` in with "not a number",
//! so `moe_batch_min_per_expert = 0` never once took the path the experiment
//! claimed to be testing. An `Option<u32>` cannot make that mistake: the
//! absence of a value and the value zero are different inhabitants.
//!
//! # How it reaches a call site
//!
//! One process-wide cell ([`install`], [`current`]), the way the reference
//! held one function-local `static`. The alternative — threading a
//! `&DeviceTuning` through every entry function — would put a parameter on
//! forty signatures to carry a value that is constant for the life of the
//! process, and `kernels-metal` names no device to hang it off instead.

use std::sync::OnceLock;

/// What this plane knows about the GPU it is running on.
///
/// Queried once by the shell that binds the device and installed here; every
/// field has a value on every path, including the one where nothing was
/// asked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `MTLGPUFamilyApple<N>`. **PROBE IT NEWEST-FIRST**: the families are
    /// cumulative, so an M4 answers `supportsFamily:` for Apple7 as well as
    /// Apple9, and an oldest-first probe would report every Apple Silicon GPU
    /// ever made as an Apple7 and hand all of them the M1 constants — a bug
    /// that looks exactly like this module not existing. 0 when nothing
    /// answered, which selects the defaults.
    pub apple_family: u32,

    /// GPU cores. Recorded, and read by nothing here: the crossovers below
    /// are set by per-core matrix throughput, which the FAMILY names and the
    /// count does not — see [`DeviceTuning::qmm_min_batch`]'s Apple9 entry.
    /// It is carried because the next constant to be measured may want it.
    /// 0 when absent.
    pub gpu_core_count: u32,
}

impl DeviceInfo {
    /// The family a `MTLDevice`'s own name implies, for a shell that has a
    /// name and no `supportsFamily:` probe.
    ///
    /// **A NAME IS A WEAKER ANSWER THAN THE PROBE AND IS NOT A SUBSTITUTE FOR
    /// IT.** It cannot report the core count at all — Metal does not publish
    /// one and IOKit's `gpu-core-count` is the only place it lives — and it
    /// answers 0 for any silicon minted after this table was written, which
    /// is the same answer a failed probe gives and lands on the same
    /// measured defaults. What it does get right is the axis the table
    /// actually branches on.
    #[must_use]
    pub fn of_name(name: &str) -> Self {
        let family = if name.contains("M1") {
            7
        } else if name.contains("M2") {
            8
        } else if name.contains("M3") || name.contains("M4") {
            9
        } else {
            0
        };
        Self {
            apple_family: family,
            gpu_core_count: 0,
        }
    }
}

/// The tuned constants, defaulted to the M1 Max measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceTuning {
    /// The batch at which the tiled GEMM overtakes the batched GEMV, for a
    /// checkpoint whose GEMM reaches the FP16 matrix path. See
    /// [`qmm_min_batch_emulated`](Self::qmm_min_batch_emulated) for the one
    /// that does not.
    ///
    /// M1 Max: 8. It read 12 while the batched GEMM was still emulating a
    /// bfloat matrix unit; it is not any more, and the same sweep re-run
    /// against the same binary — arms alternated twice, eight lanes,
    /// 128-token prompt, 64 decode steps, aggregate tok/s — gives
    /// Qwen3.6-27B 23.0 → 32.1 (+40%), gemma-4-31b 17.9 → 30.0 (+68%),
    /// gemma-4-26b-a4b 107.3 → 130.1 (+21%), gpt-oss-20b 108.2 → 116.2 (+7%).
    /// What moved is not the GEMM's speed but which side of the crossover
    /// eight sits on: the old number was measuring a GEMM the device could
    /// not really run.
    ///
    /// M2 Max (Apple8, 38 cores) agrees at 8 — the first batch where the GEMM
    /// wins on all four dense checkpoints (Llama-1B +17.6%, Llama-3B +19.2%,
    /// Qwen3-1.7B +14.4%, gemma-4-E2B +4.6%) and the first with no measured
    /// regression on any. M4 Pro (Apple9, 20 cores) agrees at 8 too, measured
    /// on gemma-4-E4B at concurrency 8, the batch where 12 and 8 provably
    /// take different paths: 138.90 against 144.04 tok/s, +3.7%.
    ///
    /// DENSE only: see [`qmm_min_batch_moe`](Self::qmm_min_batch_moe).
    pub qmm_min_batch: u32,

    /// The same crossover for a checkpoint whose FFN is ROUTED.
    ///
    /// A separate number because the measurement says so, not for symmetry.
    /// In a mixture the dense projections this governs are the attention four
    /// and the head; the FFN — the largest weights in the layer — is routed
    /// and takes [`moe_tile_mid_per`](Self::moe_tile_mid_per)'s decision
    /// instead. So the GEMM here pays its padding without the matrices that
    /// repay it.
    ///
    /// M1 Max: 8, for the same reason the dense number moved — with the
    /// routed expert GEMM on FP16 the sweep reverses (gemma-4-26b-a4b
    /// 107.3 → 130.1, gpt-oss-20b 108.2 → 116.2, eight lanes). The M2 Max
    /// keeps 12: at the batches where the dense value would have switched the
    /// GEMV won or tied on every mixture measured — Qwen3-30B by 8%,
    /// gemma-4-26B by 12%, gpt-oss-20B by nothing either way — and nothing
    /// has re-run that machine since the FP16 wiring.
    pub qmm_min_batch_moe: u32,

    /// The same crossover for a checkpoint whose quantization does NOT reach
    /// the FP16 matrix path — anything but 4-bit at group 64.
    ///
    /// Twelve, and it is the same measurement rather than a leftover: on the
    /// M1 Max at eight lanes Llama-1B at group 128 turns in 440.5 and 439.4
    /// tok/s on the GEMV against 403.9 and 402.5 on the GEMM, arms
    /// alternated, −8.4%. The GEMM it runs is the one the old sweep measured,
    /// so it keeps the old sweep's answer.
    ///
    /// One number for dense and routed both, because nothing has measured
    /// them apart and inventing a split would be a guess with a field to live
    /// in.
    pub qmm_min_batch_emulated: u32,

    /// The threadgroup count at which the unsplit GEMM's BN=32 tile overtakes
    /// BN=16.
    ///
    /// M1 Max: 160. Measured over fifteen (M, N) pairs and checked against a
    /// real llama-3.2-1B prefill, which agrees — 2565.8 / 2663.7 / 2578.3
    /// tok/s at BN 16/32/64, 448 rows. The sweep brackets the crossover
    /// between 144 threadgroups and 192, so this sits in the gap.
    ///
    /// A machine with more cores saturates later and wants this HIGHER, since
    /// the narrow tile's only advantage is that it makes more threadgroups.
    /// The M4 Pro reads that sentence backwards: twenty cores against thirty-
    /// two fill sooner, so the narrow tile runs out of reasons sooner, and
    /// the re-sweep puts the bracket at (64, 96].
    pub qmm_bn_crossover_tg: u32,

    /// Rows an expert's run must hold before the mixture's GEMM takes the
    /// 32-row tile, and then the 64-row one.
    ///
    /// M1 Max: 32, and 64 never. These were 12 and 88, and they moved for the
    /// reason [`qmm_min_batch`](Self::qmm_min_batch) did — a tile's
    /// ARITHMETIC got ~40% cheaper on the FP16 matrix unit and the rows it
    /// pads did not, so the padding a wide tile wastes buys less than it used
    /// to. Re-swept, prefill tok/s, best starred:
    ///
    /// ```text
    ///  per   model     rows    BM=16    BM=32    BM=64
    ///   12   26b        192   *642.4*   616.0       --
    ///   16   gptoss     128   *594.8*   551.8       --
    ///   24   gptoss     192   *642.3*   622.9       --
    ///   32   gptoss     256     654.2  *664.2*      --
    ///   64   gptoss     512        --  *721.4*      --
    ///  128   gptoss    1024        --  *725.1*   716.6
    ///  256   gptoss    2048        --  *656.1*   654.0
    /// ```
    ///
    /// So the mid threshold moves 12 → 32 and the wide one has nowhere left
    /// to fire: 64 loses at 128 rows an expert and ties at 256, already past
    /// any batch this plane sees. It is set OUT OF REACH rather than deleted,
    /// because the field is what a device with different matrix throughput
    /// would answer differently and no other device has been re-measured
    /// since the routed GEMM changed underneath it.
    ///
    /// Measured end to end on purpose: a roofline probe predicts the dense
    /// tile correctly and this one wrong three times over, because its single
    /// hot expert cannot show what thirty-two cold ones cost.
    pub moe_tile_mid_per: u32,

    /// The 64-row rung's threshold — see
    /// [`moe_tile_mid_per`](Self::moe_tile_mid_per). Out of reach on the M1
    /// Max, which is the finding and not an omission.
    pub moe_tile_wide_per: u32,

    /// Whether a g64/b4 projection stages its input to FP16 and feeds native
    /// FP16 simdgroup MMA instead of BF16.
    ///
    /// M1 Max: on, and it is the largest single win recorded — roughly 40% on
    /// the GEMM at every shape measured, which on gemma-4 was 938 → 1298
    /// tok/s of prefill.
    ///
    /// It exists BECAUSE of the machine. M1 and M2 have no native bfloat16
    /// matrix path and emulate it; Metal 3.1 and Apple9 (M3, M4) do have one,
    /// and on those the staging pass has nothing left to buy and is a
    /// dispatch, a barrier and a buffer per projection. So this is the one
    /// tuned field whose default may be actively WRONG on newer silicon
    /// rather than merely unmeasured, and the first thing to check on an M3
    /// or an M4.
    pub fp16_qmm: bool,

    /// Rows a request must contribute before its attention takes the tiled
    /// kernel instead of the per-row one.
    ///
    /// M1 Max: 32, which is also the tile's height — a fire earns the tiled
    /// shape by filling a tile. The two are separate numbers even so: the
    /// height is the simdgroup count and cannot move, while this is a
    /// crossover. Measured on llama-1B, where a 32-request fleet of one-row
    /// members runs 370 tok/s tiled against 728 per-row (and at 64, 480
    /// against 915), so the threshold has to keep the fleet off it. A machine
    /// whose shuffles are cheaper relative to its FMAs wants it HIGHER, since
    /// the tiled kernel's whole advantage is that it removes reductions.
    pub sdpa_tile_min_rows_per_request: u32,

    /// Whether a tiled prefill attention runs on the simdgroup MATRIX unit
    /// rather than the scalar path.
    ///
    /// M1 Max: on. The scalar tiled kernel computes Q·Kᵀ and P·V as
    /// hand-walked dot products — measured at 35.8% of a 2048-token gpt-oss
    /// prefill, running near 0.5 TFLOP/s while the quantized GEMM one
    /// dispatch away reaches ~5.6 on the same silicon. The arithmetic is a
    /// matmul; issuing it as one is what `sdpa_paged_mma.metal` is.
    ///
    /// A PREFILL switch. The predicate that earns the tiled shape at all
    /// ([`sdpa_tile_min_rows_per_request`](Self::sdpa_tile_min_rows_per_request))
    /// is unchanged and still keeps a fleet of decodes on the per-row kernel.
    ///
    /// It is a switch rather than a replacement because the matrix path
    /// depends on the register layout of `simdgroup_matrix<T,8,8>`, and a
    /// machine whose layout differs would produce WRONG numbers rather than
    /// slow ones. Turning it off is the way back.
    pub sdpa_mma: bool,

    /// Lanes that share one value row of the gated-delta scan.
    ///
    /// M1 Max: 32. The scan is latency-bound on two cross-lane reductions per
    /// token, not on arithmetic (7% of ALU peak) and not on bandwidth
    /// (staging q/k in threadgroup memory made it slower) — which reads like
    /// an argument for NARROWING the row, and is wrong. Sweeping down from 16
    /// lanes: 16 → 92.4 tok/s, 8 → 85.6, 4 → 92.6. Removing a reduction round
    /// does not help because the latency it removes was already hidden by
    /// occupancy, and the registers it costs take that occupancy away.
    ///
    /// Going UP is where the win is: a full simdgroup per dv row is the
    /// shortest q and k row a lane can read, and the extra xor round it costs
    /// is cheaper than the reads it removes — 3.6% once swept together with
    /// [`gdn_scan_rows`](Self::gdn_scan_rows).
    pub gdn_scan_lanes: u32,

    /// Value rows one lane group of that scan walks, sharing the q and k it
    /// read for all of them.
    ///
    /// M1 Max: 4. Qwen3.6-27B, 128-token prefill, both axes swept:
    ///
    /// ```text
    ///  lanes | rows | float/lane | tok/s
    ///  ------|------|------------|-------
    ///     8  |   1  |     16     |  89.3
    ///     8  |   2  |     32     |  94.6
    ///    16  |   1  |      8     |  99.4
    ///    16  |   2  |     16     | 100.9
    ///    16  |   4  |     32     |  97.0
    ///    32  |   2  |      8     | 103.8
    ///    32  |   4  |     16     | 104.5
    ///    32  |   8  |     32     | 102.7
    /// ```
    ///
    /// Read down the `float/lane` column and the two knobs separate cleanly:
    /// at a fixed register cost more lanes is always faster, because
    /// `Dk/LANES` is how much q and k a lane reads; along the other axis 16
    /// floats a lane beats both 8 and 32 at every lane count, because too few
    /// rows do not amortize the reads and too many spend the occupancy that
    /// hides them.
    pub gdn_scan_rows: u32,

    /// Rows an expert's run must hold before the mixture sorts and batches at
    /// all, rather than running the routed projections as matvecs.
    ///
    /// M1 Max: 1. It was 8 — half a narrow tile — on the model that the
    /// padding is wasted work and the break-even is where a run half fills a
    /// tile. Measured, that model is wrong in the direction that matters: a
    /// 4-bit mixture is bandwidth-bound, and what batching buys is reading
    /// each expert's slice ONCE instead of once per pair, which is worth far
    /// more than the arithmetic a half-empty tile throws away.
    ///
    /// The cost of a high value is a step function on a SERVING fleet, not a
    /// rounding error. gpt-oss-20b is 32 experts at top-4, so `4·n ≥ 32·4`
    /// first holds at exactly 32 lanes — and until it does, the largest
    /// weights in the model run as a per-row matvec. Aggregate tok/s,
    /// 128-token prompt, 64 decode steps, arms alternated:
    ///
    /// ```text
    ///  model              lanes       4        1     delta
    ///  gpt-oss-20b            4   103.5    103.3       -0%
    ///  gpt-oss-20b            8   116.9    174.7      +49%
    ///  gpt-oss-20b           16   134.1    310.7     +132%
    ///  gemma-4-26b-a4b        8   128.9    130.3       +1%
    ///  gemma-4-26b-a4b       16   174.3    269.3      +55%
    /// ```
    ///
    /// The two neutral rows are neutral off the arithmetic and not the
    /// weather: the gate compares `n_pairs` to `n_experts · this`, and at
    /// those widths neither fire clears it either way.
    ///
    /// ZERO IS A VALUE, not a typo: "batch at any width" is the only way to
    /// reach the routed GEMM below its crossover, which is how a wrong-answer
    /// bug in that kernel gets bisected.
    pub moe_batch_min_per_expert: u32,
}

impl Default for DeviceTuning {
    fn default() -> Self {
        Self {
            qmm_min_batch: 8,
            qmm_min_batch_moe: 8,
            qmm_min_batch_emulated: 12,
            qmm_bn_crossover_tg: 160,
            moe_tile_mid_per: 32,
            moe_tile_wide_per: 1 << 24,
            fp16_qmm: true,
            sdpa_tile_min_rows_per_request: 32,
            sdpa_mma: true,
            gdn_scan_lanes: 32,
            gdn_scan_rows: 4,
            moe_batch_min_per_expert: 1,
        }
    }
}

impl DeviceTuning {
    /// The table, read at one device.
    ///
    /// A family with no arm inherits the defaults, which are the M1 Max's
    /// measurements — see the module header for why that is the rule rather
    /// than an accident.
    #[must_use]
    pub fn of(info: DeviceInfo) -> Self {
        let mut t = Self::default();
        match info.apple_family {
            // M3/M4. Both crossovers are the struct's defaults now — the M1
            // measured its way to the same pair — so this entry no longer
            // restates them. What is left is the tile crossover: with twenty
            // cores rather than thirty-two the wide tile's smaller grid fills
            // the machine sooner, so it moves DOWN. Re-swept at BM=64,
            // GFLOP/s, threadgroups counted as `(N/32)·ceil(M/64)`:
            //
            //   tg@32    BN=16    BN=32    delta
            //      64     5004     4615    -7.8%
            //      96     5540     5863    +5.8%
            //     128     5829     5846    +0.3%
            //     160     5987     6411    +7.1%
            //     224     5980     6196    +3.6%
            //
            // The bracket is (64, 96] and 96 is the sampled edge rather than
            // the midpoint, which is what to write down when only one side
            // was measured. Applied by FAMILY and not by core count: the
            // crossover is set by per-core matrix throughput, which the
            // family names and the count does not.
            9 => t.qmm_bn_crossover_tg = 96,
            // M2. The DENSE crossover here is eight, which is the default and
            // is not restated. The ROUTED one is twelve, and that is the
            // finding rather than an omission — at the same batches the GEMV
            // still won on every mixture measured. It is named HERE rather
            // than inherited because the default moved UNDER it: the M1
            // reversed its own routed number once the expert GEMM stopped
            // emulating a bfloat matrix unit, this machine has not been
            // re-measured since, and inheriting a default that changed for a
            // reason this device was never tested against would be a guess
            // wearing a measurement's clothes.
            8 => t.qmm_min_batch_moe = 12,
            _ => {}
        }
        t
    }

    /// The same table with a boot document's answers laid over it.
    #[must_use]
    pub fn with(mut self, over: &Overrides) -> Self {
        macro_rules! lay {
            ($($field:ident),+ $(,)?) => {
                $(if let Some(v) = over.$field { self.$field = v; })+
            };
        }
        lay!(
            qmm_min_batch,
            qmm_min_batch_moe,
            qmm_min_batch_emulated,
            qmm_bn_crossover_tg,
            moe_tile_mid_per,
            moe_tile_wide_per,
            fp16_qmm,
            sdpa_tile_min_rows_per_request,
            sdpa_mma,
            gdn_scan_lanes,
            gdn_scan_rows,
            moe_batch_min_per_expert,
        );
        self
    }

    /// The GEMV/GEMM crossover for one projection, given whether the
    /// checkpoint's FFN is routed and whether this weight's format reaches
    /// the FP16 matrix path.
    #[must_use]
    pub const fn qmm_min_batch(&self, routed: bool, fp16_gemm: bool) -> u32 {
        if !fp16_gemm {
            return self.qmm_min_batch_emulated;
        }
        if routed {
            self.qmm_min_batch_moe
        } else {
            self.qmm_min_batch
        }
    }

    /// Whether a bank of this format reaches the FP16 matrix path — the
    /// staged-input GEMM is stamped at 4 bits, group 64, and nowhere else.
    #[must_use]
    pub const fn fp16_gemm_format(&self, bits: u32, group: u32) -> bool {
        self.fp16_qmm && bits == 4 && group == 64
    }
}

/// One boot document's answers: `None` is "keep what the device's table
/// said", and every value — zero included — is a value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overrides {
    pub qmm_min_batch: Option<u32>,
    pub qmm_min_batch_moe: Option<u32>,
    pub qmm_min_batch_emulated: Option<u32>,
    pub qmm_bn_crossover_tg: Option<u32>,
    pub moe_tile_mid_per: Option<u32>,
    pub moe_tile_wide_per: Option<u32>,
    pub fp16_qmm: Option<bool>,
    pub sdpa_tile_min_rows_per_request: Option<u32>,
    pub sdpa_mma: Option<bool>,
    pub gdn_scan_lanes: Option<u32>,
    pub gdn_scan_rows: Option<u32>,
    pub moe_batch_min_per_expert: Option<u32>,
}

static DEVICE: OnceLock<DeviceInfo> = OnceLock::new();
static OVERRIDES: OnceLock<Overrides> = OnceLock::new();
static RESOLVED: OnceLock<DeviceTuning> = OnceLock::new();

/// Say what device this process is running on — the shell that binds it,
/// once.
///
/// **THE TWO INPUTS ARRIVE FROM DIFFERENT PLACES AND IN NO FIXED ORDER**, and
/// that is why they are seated separately rather than combined by the caller:
/// the boot document is read at the door and the device is bound at load, so
/// whichever arrives first would otherwise decide what the other could still
/// change. They are folded at the first [`current`] instead, and the answer is
/// frozen from then on — a table that could move under a fire in flight would
/// let two dispatches of one step disagree about which kernel they are.
///
/// Answers whether this call is the one that seated it. Not calling it at all
/// is a supported state and lands on the M1 Max measurements, which is what
/// [`DeviceTuning::of`] answers for a device it does not recognize.
pub fn describe(info: DeviceInfo) -> bool {
    DEVICE.set(info).is_ok()
}

/// Lay a boot document's answers over whatever the device's table says. See
/// [`describe`] for the ordering.
pub fn override_with(over: Overrides) -> bool {
    OVERRIDES.set(over).is_ok()
}

/// The tuning every selection here reads. `Copy`, because a call site that
/// held a borrow would be holding one for the life of the process.
#[must_use]
pub fn current() -> DeviceTuning {
    *RESOLVED.get_or_init(|| {
        let info = DEVICE.get().copied().unwrap_or_default();
        let over = OVERRIDES.get().copied().unwrap_or_default();
        DeviceTuning::of(info).with(&over)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unrecognised_device_gets_the_m1_max_measurements() {
        let m1 = DeviceTuning::default();
        assert_eq!(DeviceTuning::of(DeviceInfo::default()), m1);
        assert_eq!(DeviceTuning::of(DeviceInfo::of_name("Apple M9 Ultra")), m1);
        assert_eq!(DeviceTuning::of(DeviceInfo::of_name("Apple M1 Max")), m1);
        assert_eq!(m1.qmm_min_batch, 8);
        assert_eq!(m1.qmm_bn_crossover_tg, 160);
        assert_eq!(m1.moe_batch_min_per_expert, 1);
    }

    #[test]
    fn a_family_override_moves_one_field_and_no_others() {
        let m1 = DeviceTuning::default();
        let m4 = DeviceTuning::of(DeviceInfo::of_name("Apple M4 Pro"));
        assert_eq!(m4.qmm_bn_crossover_tg, 96);
        assert_eq!(m4.qmm_min_batch, m1.qmm_min_batch);
        assert_eq!(m4.qmm_min_batch_moe, m1.qmm_min_batch_moe);

        let m2 = DeviceTuning::of(DeviceInfo::of_name("Apple M2 Max"));
        assert_eq!(m2.qmm_min_batch_moe, 12);
        assert_eq!(m2.qmm_min_batch, 8);
        assert_eq!(m2.qmm_bn_crossover_tg, m1.qmm_bn_crossover_tg);
    }

    #[test]
    fn the_name_probe_reads_the_family_and_never_the_core_count() {
        assert_eq!(DeviceInfo::of_name("Apple M1 Max").apple_family, 7);
        assert_eq!(DeviceInfo::of_name("Apple M2").apple_family, 8);
        assert_eq!(DeviceInfo::of_name("Apple M3 Max").apple_family, 9);
        assert_eq!(DeviceInfo::of_name("Apple M4 Pro").apple_family, 9);
        assert_eq!(DeviceInfo::of_name("Apple M4 Pro").gpu_core_count, 0);
    }

    #[test]
    fn a_zero_override_is_a_value_and_not_a_parse_failure() {
        let t = DeviceTuning::default().with(&Overrides {
            moe_batch_min_per_expert: Some(0),
            ..Overrides::default()
        });
        assert_eq!(t.moe_batch_min_per_expert, 0);
        assert_eq!(t.qmm_min_batch, DeviceTuning::default().qmm_min_batch);
    }

    #[test]
    fn the_emulated_crossover_answers_for_any_format_but_g64_b4() {
        let t = DeviceTuning::default();
        assert!(t.fp16_gemm_format(4, 64));
        assert!(!t.fp16_gemm_format(4, 128));
        assert!(!t.fp16_gemm_format(8, 64));
        assert_eq!(t.qmm_min_batch(false, true), 8);
        assert_eq!(t.qmm_min_batch(false, false), 12);
        assert_eq!(t.qmm_min_batch(true, false), 12);
    }
}
