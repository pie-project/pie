//! Per-device tuned constants: the table, the family selection, and the env
//! overrides that measured them.
//!
//! Portable on purpose. The only Apple-specific part of tuning is asking the
//! device what it is (`crate::device::DeviceInfo`); everything downstream of
//! that answer is arithmetic over two integers, so it lives here where it can
//! be tested on any host. The C++ shell splits the same way, for the same
//! reason -- `device_tuning_apple.mm` asks and `device_tuning.cpp` decides.
//!
//! # Provenance
//!
//! Every constant below is a measurement, and the comment on it is the run.
//! Changing one without a run is how a tuning table becomes a table of
//! numbers nobody can defend, so the comments are the load-bearing part.
//!
//! # A note on the C++ this replaces
//!
//! `csrc/src/device_tuning.hpp` declares `qmm_min_batch_moe` twice in one
//! struct (8, then 12) and `csrc/src/device_tuning.cpp` has two `case 8:`
//! arms in one `switch`. Both are hard compile errors, so the C++ tuning
//! layer does not currently build; see [`Tuning::for_device`] for how the
//! conflict is resolved here.
//!
//! # No process-wide singleton
//!
//! The C++ reaches every constant through free functions over a function-local
//! `static const DeviceTuning`, so the table is queried once per process and
//! cannot be varied afterwards. Nothing here does that: [`Tuning::resolve`]
//! returns a value and the caller holds it. The singleton is why the C++ shell
//! needs `set_*_for_test` hooks scattered through `mtl4_context.hpp`, and a
//! table that a test can simply construct does not need any of them.

use std::env;

/// What the device is, as far as tuning cares.
///
/// Both fields are 0 when unknown, and 0 selects the defaults -- the
/// constants this driver shipped before the tuning layer existed. That is why
/// the type is `Default`-able and why nothing here branches on "did the query
/// succeed": a device that would not answer gets the M1 numbers, which is the
/// same thing every device got before there was a table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Device {
    /// `MTLGPUFamilyApple<N>`, resolved newest-first because the families are
    /// cumulative. 0 when no Metal device answered.
    pub apple_family: u32,
    /// IOKit's `gpu-core-count`, the only place the count is published --
    /// `MTLDevice` does not expose it. 0 when absent.
    pub gpu_core_count: u32,
}

/// The tuned constants, defaulted to the M1 Max measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tuning {
    /// The batch at which the ported steel GEMM overtakes the batched GEMV,
    /// for a checkpoint whose GEMM reaches the FP16 matrix path.
    ///
    /// M1 Max: 8. Every dense checkpoint measured prefers 8 over the 12 this
    /// used to read, by between 7% and 68% aggregate tok/s; the 12 came from
    /// a sweep taken while the batched GEMM still emulated a bfloat matrix
    /// unit, which it no longer does.
    pub qmm_min_batch: u32,

    /// The same crossover for a ROUTED (mixture) checkpoint.
    ///
    /// Split from [`Self::qmm_min_batch`] because the two measure differently
    /// once the expert GEMM stops emulating a matrix unit: the dense half
    /// moved to 8 and the routed half did not follow on every family.
    pub qmm_min_batch_moe: u32,

    /// The same crossover for a checkpoint whose quantization does NOT reach
    /// the FP16 matrix path (group-128, say), and so runs the emulated GEMM.
    pub qmm_min_batch_emulated: u32,

    /// The threadgroup count at which the unsplit GEMM's BN=32 tile overtakes
    /// the wide one. Set by how soon the wide tile's smaller grid fills the
    /// machine, so it moves with CORE COUNT rather than with family.
    pub qmm_bn_crossover_tg: u32,

    /// Mixture tiling: rows per mid tile.
    pub moe_tile_mid_per: u32,
    /// Mixture tiling: rows per wide tile. The default is effectively
    /// "never split", which is what `1 << 24` says.
    pub moe_tile_wide_per: u32,

    /// Whether the 4-bit dense projections stage to FP16 and use the matrix
    /// instruction the hardware has, rather than emulating it.
    pub fp16_qmm: bool,

    /// Rows per request below which SDPA does not tile.
    pub sdpa_tile_min_rows_per_request: u32,
    /// Whether SDPA uses the matrix path.
    pub sdpa_mma: bool,

    /// Minimum rows per expert before a mixture batches that expert.
    pub moe_batch_min_per_expert: u32,

    /// Gated-delta-net scan geometry: lanes.
    pub gdn_scan_lanes: u32,
    /// Gated-delta-net scan geometry: rows.
    pub gdn_scan_rows: u32,
}

impl Default for Tuning {
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
            moe_batch_min_per_expert: 1,
            gdn_scan_lanes: 32,
            gdn_scan_rows: 4,
        }
    }
}

impl Tuning {
    /// The table entry for `device`, BEFORE environment overrides.
    ///
    /// Kept separate from [`Self::resolve`] so the table can be tested
    /// without the process environment in the way -- the overrides read
    /// globals, and a test that sets them is a test that cannot run beside
    /// another one.
    ///
    /// # The Apple8 conflict
    ///
    /// The C++ has two `case 8:` arms and they disagree. The first sets
    /// `qmm_min_batch = 8` and leaves the routed crossover inherited; the
    /// second leaves the dense one to the default and sets
    /// `qmm_min_batch_moe = 12`. The duplicated FIELD resolves it: the first
    /// declaration (`= 8`) closes with "the M2 table above stands as a
    /// measurement of the binary it was taken on; nothing has re-run it since
    /// the FP16 wiring, **which is why the Apple8 entry still names its own
    /// number**", while the second declaration (`= 12`) is the pre-FP16 text,
    /// still ending "so a routed checkpoint keeps the M1 number" from when the
    /// M1 number WAS 12.
    ///
    /// So the two duplicates are one new pair and one stale pair, and the new
    /// pair is: default routed 8, Apple8 routed 12. That is what is taken
    /// here, and the first declaration's closing sentence is the C++ saying so
    /// outright rather than an inference from intent.
    #[must_use]
    pub fn for_device(device: Device) -> Self {
        let mut t = Self::default();
        match device.apple_family {
            // M3/M4 generation, measured on an M4 Pro (20 cores) with
            // gemma-4-E4B at concurrency 8: 138.90 tok/s at 12 against 144.04
            // at 8, +3.7%.
            9 => {
                // With 20 cores rather than 32 the wide tile's smaller grid
                // fills the machine sooner, so the tile crossover moves down.
                t.qmm_bn_crossover_tg = 96;
                // 8 is what this device measured and shipped with while there
                // was one crossover, and a mixture ran it too. The routed half
                // is unmeasured on this family; leaving it at 12 would be a
                // change dressed as a default.
                t.qmm_min_batch_moe = 8;
            }
            // M2 generation, measured on an M2 Max (38 cores). The dense
            // crossover is 8, which is the default and is not restated. At the
            // same batches the GEMV still won on every mixture measured --
            // Qwen3-30B by 8%, gemma-4-26B by 12%, gpt-oss-20B by nothing
            // either way -- so the routed crossover stays at 12.
            8 => t.qmm_min_batch_moe = 12,
            _ => {}
        }
        t
    }

    /// [`Self::for_device`] with the environment overrides applied.
    ///
    /// Every tuned constant gets one, and for the same reason the first one
    /// did: measuring a crossover means running the same binary twice with
    /// different answers, and a rebuild between the arms is a different
    /// binary.
    #[must_use]
    pub fn resolve(device: Device) -> Self {
        let mut t = Self::for_device(device);

        t.qmm_min_batch = env_u32("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch);
        // The dense override carries the routed and emulated ones unless they
        // are named separately. A sweep that moved only the dense number on a
        // mixture would measure a model that never changed path and read the
        // resulting flat curve as the crossover not mattering.
        let dense_override = env_u32("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch_moe);
        t.qmm_min_batch_moe = env_u32("PIE_METAL_QMM_MIN_BATCH_MOE", dense_override);
        let dense_override = env_u32("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch_emulated);
        t.qmm_min_batch_emulated = env_u32("PIE_METAL_QMM_MIN_BATCH_EMULATED", dense_override);

        t.qmm_bn_crossover_tg = env_u32("PIE_METAL_QMM_BN_CROSSOVER_TG", t.qmm_bn_crossover_tg);
        t.moe_tile_mid_per = env_u32("PIE_METAL_MOE_TILE_MID_PER", t.moe_tile_mid_per);
        t.moe_tile_wide_per = env_u32("PIE_METAL_MOE_TILE_WIDE_PER", t.moe_tile_wide_per);
        t.fp16_qmm = env_bool("PIE_METAL_FP16_QMM", t.fp16_qmm);
        t.sdpa_tile_min_rows_per_request = env_u32(
            "PIE_METAL_SDPA_TILE_MIN_ROWS",
            t.sdpa_tile_min_rows_per_request,
        );
        t.sdpa_mma = env_bool("PIE_METAL_SDPA_MMA", t.sdpa_mma);
        // A THRESHOLD, not a count: `moe_should_batch` compares `n_pairs` to
        // `n_experts * this`, so 0 means "batch every fire". The C++ reads it
        // through the same positive-only parser as the crossovers and
        // therefore SILENTLY IGNORES the one value its own documentation names
        // -- `device_tuning.hpp` cites `PIE_METAL_MOE_BATCH_MIN_PER_EXPERT=0`
        // as how the routed GEMM was forced onto every fire to clear it of a
        // numerics charge. That sweep cannot have gone through this variable.
        t.moe_batch_min_per_expert = env_threshold(
            "PIE_METAL_MOE_BATCH_MIN_PER_EXPERT",
            t.moe_batch_min_per_expert,
        );
        t.gdn_scan_lanes = env_u32("PIE_METAL_GDN_SCAN_LANES", t.gdn_scan_lanes);
        t.gdn_scan_rows = env_u32("PIE_METAL_GDN_SCAN_ROWS", t.gdn_scan_rows);

        t
    }

    /// The GEMM/GEMV crossover for a given checkpoint shape.
    #[must_use]
    pub fn qmm_min_batch_for(&self, is_moe: bool, fp16_gemm: bool) -> u32 {
        if !fp16_gemm {
            return self.qmm_min_batch_emulated;
        }
        if is_moe {
            self.qmm_min_batch_moe
        } else {
            self.qmm_min_batch
        }
    }

    /// Whether a mixture of `n_experts` batches its routed FFN for `n_pairs`.
    ///
    /// Batching reads each expert's weights once for all its rows; a matvec
    /// reads them per row. The two meet when an expert's run half fills a
    /// tile. Written against the NARROW tile because that is the cheapest way
    /// in: a batch that cannot pay for a 16-row tile cannot pay for a wider
    /// one, and [`Self::moe_tile_rows`] widens only after this says yes.
    ///
    /// `n_experts == 0` is false rather than a division: a checkpoint with no
    /// experts has no routed FFN to batch.
    #[must_use]
    pub const fn moe_should_batch(&self, n_pairs: u32, n_experts: u32) -> bool {
        if n_experts == 0 {
            return false;
        }
        // A threshold no fleet can reach refuses, rather than becoming one
        // every fleet reaches. The C++ writes this as a plain `int` multiply,
        // where passing the range is undefined behaviour and the reachable
        // outcome -- a negative product -- makes the comparison TRUE and
        // batches a mixture that should not batch.
        match n_experts.checked_mul(self.moe_batch_min_per_expert) {
            Some(threshold) => n_pairs >= threshold,
            None => false,
        }
    }

    /// Rows each expert's run is padded to, for a batch of `n_pairs`.
    ///
    /// 1 when the mixture does not batch at all, and otherwise the widest tile
    /// the rows per expert pay for. Priced off ROWS PER EXPERT because that is
    /// what decides how much of a tile a run fills; the thresholds are a table
    /// of measurements rather than a curve, and they have to be re-swept
    /// whenever the routed GEMM changes.
    #[must_use]
    pub const fn moe_tile_rows(&self, n_pairs: u32, n_experts: u32) -> u32 {
        if !self.moe_should_batch(n_pairs, n_experts) {
            return 1;
        }
        let per = n_pairs / n_experts;
        if per >= self.moe_tile_wide_per {
            return 64;
        }
        if per >= self.moe_tile_mid_per { 32 } else { 16 }
    }

    /// Whether a `bits`/`group` quantization reaches the FP16 matrix path.
    #[must_use]
    pub fn fp16_gemm_format(&self, bits: u32, group: u32) -> bool {
        self.fp16_qmm && bits == 4 && group == 64
    }
}

/// A positive integer from the environment, or `fallback`.
///
/// For a COUNT: a batch size, a threadgroup count, a lane width. Zero is
/// rejected rather than accepted, matching the C++, because 0 for any of these
/// would disable the thing being measured rather than tune it. Use
/// [`env_threshold`] where 0 is a setting.
fn env_u32(name: &str, fallback: u32) -> u32 {
    parse_count(env::var(name).ok().as_deref(), fallback)
}

/// A non-negative integer from the environment, or `fallback`.
///
/// For a THRESHOLD, where 0 means "no threshold" and is a value a sweep sets
/// on purpose.
fn env_threshold(name: &str, fallback: u32) -> u32 {
    parse_threshold(env::var(name).ok().as_deref(), fallback)
}

/// A boolean from the environment, or `fallback`.
///
/// Unlike [`env_u32`], `0` is a VALUE here and not a rejected one -- it is
/// how a sweep turns a path off. Anything else non-empty is true.
fn env_bool(name: &str, fallback: bool) -> bool {
    parse_bool(env::var(name).ok().as_deref(), fallback)
}

/// Surrounding whitespace is ignored on every override.
///
/// `strtol` skips LEADING whitespace and then rejects anything trailing, so
/// the C++ reads `" 8"` as 8 and `"8 "` as unset. That asymmetry is an
/// artifact of `strtol` rather than a decision, and a sweep script that
/// interpolates a padded value should not silently get the default on one side
/// and the value on the other.
fn trimmed(raw: Option<&str>) -> Option<&str> {
    raw.map(str::trim).filter(|s| !s.is_empty())
}

fn parse_count(raw: Option<&str>, fallback: u32) -> u32 {
    match trimmed(raw).map(str::parse::<u32>) {
        Some(Ok(v)) if v > 0 => v,
        _ => fallback,
    }
}

fn parse_threshold(raw: Option<&str>, fallback: u32) -> u32 {
    match trimmed(raw).map(str::parse::<u32>) {
        Some(Ok(v)) => v,
        _ => fallback,
    }
}

fn parse_bool(raw: Option<&str>, fallback: bool) -> bool {
    match trimmed(raw) {
        Some(v) => v != "0",
        None => fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_device_gets_the_m1_defaults() {
        assert_eq!(Tuning::for_device(Device::default()), Tuning::default());
    }

    #[test]
    fn apple9_lowers_the_tile_crossover_and_keeps_the_dense_routed_pair() {
        let t = Tuning::for_device(Device {
            apple_family: 9,
            gpu_core_count: 20,
        });
        assert_eq!(t.qmm_bn_crossover_tg, 96);
        assert_eq!(t.qmm_min_batch, 8);
        assert_eq!(t.qmm_min_batch_moe, 8);
    }

    /// The arm the duplicated C++ `case 8:` could not express. See
    /// [`Tuning::for_device`].
    #[test]
    fn apple8_names_the_routed_crossover_and_inherits_the_dense_one() {
        let t = Tuning::for_device(Device {
            apple_family: 8,
            gpu_core_count: 38,
        });
        assert_eq!(t.qmm_min_batch, 8, "dense is the default, not restated");
        assert_eq!(t.qmm_min_batch_moe, 12, "routed is named on this family");
    }

    #[test]
    fn a_future_family_falls_back_rather_than_guessing() {
        let t = Tuning::for_device(Device {
            apple_family: 10,
            gpu_core_count: 64,
        });
        assert_eq!(t, Tuning::default());
    }

    #[test]
    fn the_emulated_crossover_is_selected_by_format_not_by_routing() {
        let t = Tuning::default();
        assert_eq!(t.qmm_min_batch_for(false, false), t.qmm_min_batch_emulated);
        assert_eq!(t.qmm_min_batch_for(true, false), t.qmm_min_batch_emulated);
        assert_eq!(t.qmm_min_batch_for(false, true), t.qmm_min_batch);
        assert_eq!(t.qmm_min_batch_for(true, true), t.qmm_min_batch_moe);
    }

    #[test]
    fn only_group64_4bit_reaches_the_fp16_path() {
        let t = Tuning::default();
        assert!(t.fp16_gemm_format(4, 64));
        assert!(!t.fp16_gemm_format(4, 128));
        assert!(!t.fp16_gemm_format(8, 64));

        let off = Tuning {
            fp16_qmm: false,
            ..Tuning::default()
        };
        assert!(!off.fp16_gemm_format(4, 64));
    }

    #[test]
    fn a_count_refuses_zero_because_zero_would_disable_it() {
        assert_eq!(parse_count(Some("8"), 12), 8);
        assert_eq!(parse_count(None, 12), 12);
        assert_eq!(parse_count(Some(""), 12), 12);
        assert_eq!(parse_count(Some("0"), 12), 12, "0 is not a batch size");
        assert_eq!(parse_count(Some("-4"), 12), 12);
        assert_eq!(parse_count(Some("eight"), 12), 12);
        assert_eq!(parse_count(Some("8x"), 12), 12, "trailing garbage");
        assert_eq!(
            parse_count(Some("99999999999999999999"), 12),
            12,
            "past u32 the fallback is taken; the C++ `int(strtol(...))` \
             truncates LONG_MAX instead and hands the table a negative"
        );
    }

    /// The C++ documents `PIE_METAL_MOE_BATCH_MIN_PER_EXPERT=0` as the way it
    /// cleared the routed GEMM of a numerics charge, and then reads the
    /// variable through a parser that rejects 0.
    #[test]
    fn the_moe_threshold_accepts_the_zero_its_own_documentation_names() {
        assert_eq!(parse_threshold(Some("0"), 1), 0);
        assert_eq!(parse_threshold(Some("4"), 1), 4);
        assert_eq!(parse_threshold(None, 1), 1);
        assert_eq!(parse_threshold(Some("-1"), 1), 1);
        assert_eq!(parse_threshold(Some(""), 1), 1);
    }

    #[test]
    fn a_decode_does_not_batch_its_mixture_and_a_prefill_does() {
        let t = Tuning::default();
        // The case the constant exists for: eight pairs over 128 experts is
        // one live row in sixteen of every tile.
        assert!(!t.moe_should_batch(8, 128));
        assert_eq!(t.moe_tile_rows(8, 128), 1);

        assert!(t.moe_should_batch(128, 128));
        assert_eq!(t.moe_tile_rows(1024, 128), 16, "eight rows an expert");
        assert_eq!(t.moe_tile_rows(4096, 128), 32, "thirty-two an expert");

        // The default wide threshold is 1<<24, which is "never split".
        assert_eq!(t.moe_tile_rows(1 << 20, 128), 32);
        let wide = Tuning {
            moe_tile_wide_per: 64,
            ..t
        };
        assert_eq!(wide.moe_tile_rows(128 * 64, 128), 64);
    }

    /// The behaviour the C++ documents and its own parser makes unreachable.
    #[test]
    fn a_zero_threshold_batches_every_fire() {
        let always = Tuning {
            moe_batch_min_per_expert: 0,
            ..Tuning::default()
        };
        assert!(
            always.moe_should_batch(1, 128),
            "at a zero threshold even a one-row decode batches"
        );
        assert_eq!(always.moe_tile_rows(1, 128), 16);
        assert!(
            !always.moe_should_batch(1, 0),
            "no experts is still no routed FFN"
        );
    }

    #[test]
    fn an_unreachable_threshold_refuses_rather_than_wrapping() {
        let never = Tuning {
            moe_batch_min_per_expert: u32::MAX,
            ..Tuning::default()
        };
        assert!(
            !never.moe_should_batch(u32::MAX, 128),
            "the product passes u32; the C++ int multiply is undefined here and \
             its reachable outcome batches a fleet that should not"
        );
        assert_eq!(never.moe_tile_rows(u32::MAX, 128), 1);
    }

    #[test]
    fn a_bool_is_off_only_for_zero() {
        assert!(!parse_bool(Some("0"), true));
        assert!(parse_bool(Some("1"), false));
        assert!(parse_bool(Some("00"), false), "only a bare 0 turns it off");
        assert!(
            parse_bool(Some("false"), false),
            "matching the C++, which \
                 compares against the one character"
        );
        assert!(parse_bool(None, true));
        assert!(!parse_bool(None, false));
        assert!(parse_bool(Some(""), true), "empty is unset, not false");
    }

    #[test]
    fn padding_reads_the_same_on_both_sides() {
        assert_eq!(parse_count(Some(" 8"), 12), 8);
        assert_eq!(
            parse_count(Some("8 "), 12),
            8,
            "the C++ takes the leading space and refuses the trailing one, \
             which is strtol's shape rather than a decision"
        );
        assert_eq!(parse_count(Some("\t8\n"), 12), 8);
        assert_eq!(parse_threshold(Some(" 0 "), 1), 0);
        assert!(!parse_bool(Some(" 0 "), true));
        assert_eq!(parse_count(Some("   "), 12), 12, "blank is unset");
    }
}
