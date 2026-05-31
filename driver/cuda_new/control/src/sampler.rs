//! Sampling — replaces `driver/cuda/src/sampling_dispatch.cpp` and the
//! seed/PRNG helpers scattered in executor.cpp.
//!
//! The *policy* (temperature, top-p/top-k, per-row seeding schedule) is
//! Rust; the argmax / categorical *kernels* stay C++ behind `pie_sample`.
//! This split is why the giant sampling surface leaves executor.cpp.
//!
//! What is ported here (all pure host-side, no device calls):
//!   * `splitmix64` — the seed mixer (bit-identical to executor.cpp).
//!   * `SeedScheduler` — the per-fire / per-row fresh-seed schedule that
//!     `fresh_sampling_seed()` implements via a process-global atomic
//!     counter advanced by the golden-ratio constant.
//!   * greedy detection (`temperature == 0` → argmax fast path).
//!   * `build_soa` — turns per-row [`RowParams`] into the SoA the device
//!     `pie_sample` kernel consumes (`temperature[] top_p[] top_k[] seed[]`),
//!     replicating the per-slot fill loop in `build_sample_plan`.

use std::sync::atomic::{AtomicU64, Ordering};

/// The golden-ratio odd constant splitmix64 advances its state by; also the
/// stride `fresh_sampling_seed()` adds to the global counter per draw.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64 — ported verbatim from executor.cpp. Used to derive per-row
/// seeds. This is the standard splitmix64 finalizer over a gamma-advanced
/// state; the Rust below is bit-identical to the C++ (wrapping arithmetic
/// matches the implicit u64 overflow there).
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(SPLITMIX64_GAMMA);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fold a 64-bit mixed value into the non-zero u32 seed the kernels want.
///
/// Mirrors the tail of `fresh_sampling_seed()`:
/// `seed = (u32)(x ^ (x >> 32)); return seed == 0 ? 1 : seed;`. The kernels
/// treat seed 0 as "unseeded / pick a fresh one", so the schedule never
/// hands out 0.
fn fold_to_nonzero_u32(x: u64) -> u32 {
    let seed = (x ^ (x >> 32)) as u32;
    if seed == 0 { 1 } else { seed }
}

/// The fresh-seed schedule.
///
/// `driver/cuda` derives a row's PRNG seed lazily: when a token-sampler slot
/// has temperature > 0 but the runtime left the wire seed at 0 (meaning
/// "you pick one"), it calls `fresh_sampling_seed()`, which:
///   1. starts a process-global atomic counter at `initial_sampling_seed()`
///      (a one-time entropy mix, also splitmix64-finalized),
///   2. on each draw does `pre = counter.fetch_add(GAMMA)` and returns
///      `fold_to_nonzero_u32(splitmix64(pre))`.
///
/// So the schedule is *not* indexed by `(fire, row)` directly — it is a
/// monotone counter consumed in slot order across the whole process. Two
/// distinct draws use two distinct counter values (GAMMA apart) and thus
/// (overwhelmingly) distinct seeds; rows that already carry a nonzero wire
/// seed or are greedy don't draw at all.
///
/// In the rewrite the counter is owned by this struct rather than a function-
/// local `static atomic`, so the executor holds one [`SeedScheduler`] and it
/// can be deterministically reset for tests. `next_fresh_seed()` is the exact
/// analogue of one `fresh_sampling_seed()` call.
pub struct SeedScheduler {
    counter: AtomicU64,
}

impl SeedScheduler {
    /// Seed the schedule from an explicit base. The base plays the role of
    /// `initial_sampling_seed()`'s return value (already splitmix-finalized
    /// entropy in production). Tests use a fixed base for determinism.
    pub fn from_base(base: u64) -> Self {
        SeedScheduler {
            counter: AtomicU64::new(base),
        }
    }

    /// Production constructor: mix wall-clock + a stack address + (best
    /// effort) OS entropy, then finalize through splitmix64 — the Rust
    /// analogue of `initial_sampling_seed()`. Distinct process starts get
    /// distinct sequences. Kept allocation-/panic-free.
    pub fn new() -> Self {
        let stack_marker = 0u8;
        let mut seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        seed ^= (&stack_marker as *const u8) as usize as u64;
        // RandomState gives us per-process OS entropy without an extra
        // dependency; hashing the unit value yields a well-mixed u64.
        let entropy = {
            use std::hash::{BuildHasher, Hasher};
            let h = std::collections::hash_map::RandomState::new();
            let mut hasher = h.build_hasher();
            hasher.write_u64(seed);
            hasher.finish()
        };
        seed ^= entropy;
        SeedScheduler::from_base(splitmix64(seed))
    }

    /// One `fresh_sampling_seed()` draw: advance the counter by GAMMA and
    /// return the folded non-zero u32. Thread-safe (relaxed atomic), exactly
    /// as the C++ `counter.fetch_add(GAMMA, relaxed)`.
    pub fn next_fresh_seed(&self) -> u32 {
        let pre = self.counter.fetch_add(SPLITMIX64_GAMMA, Ordering::Relaxed);
        fold_to_nonzero_u32(splitmix64(pre))
    }

    /// Resolve a single row's effective seed under the policy:
    ///   * greedy rows (temperature == 0) never sample stochastically, so
    ///     their seed is irrelevant — passed through unchanged (the C++ also
    ///     leaves it untouched and the kernel ignores it on the argmax path);
    ///   * a stochastic row (temperature > 0) with a nonzero wire seed keeps
    ///     that caller-supplied seed (reproducible sampling);
    ///   * a stochastic row whose wire seed is 0 draws a fresh one.
    ///
    /// This is exactly the `if (T > 0.f && s == 0u) s = fresh_sampling_seed();`
    /// branch in `build_sample_plan`.
    pub fn resolve_row_seed(&self, temperature: f32, wire_seed: u32) -> u32 {
        if temperature > 0.0 && wire_seed == 0 {
            self.next_fresh_seed()
        } else {
            wire_seed
        }
    }
}

impl Default for SeedScheduler {
    fn default() -> Self {
        SeedScheduler::new()
    }
}

/// Per-row sampling knobs the runtime sends on the wire, in the row's logit
/// order. `top_k == 0` is the wire encoding for "no top-k filter" and is
/// mapped to `vocab_size` when the SoA is built (matching the C++).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RowParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    /// Caller-supplied seed; 0 means "driver picks a fresh one" for a
    /// stochastic row.
    pub wire_seed: u32,
}

impl RowParams {
    /// Greedy fast path: temperature exactly 0 → the kernel does argmax and
    /// ignores top-p/top-k/seed. Mirrors `h_temp[r] > 0.f` being false /
    /// `all_rows_greedy` in `build_sample_plan`.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Whether this row would route the batch to flashinfer's top-k/top-p
    /// kernel: a non-trivial top-k or top-p filter *and* a positive
    /// temperature. Mirrors `(Tk_raw > 0 || Tp < 1.f) && T > 0.f`.
    pub fn needs_topk_topp(&self) -> bool {
        (self.top_k > 0 || self.top_p < 1.0) && self.temperature > 0.0
    }
}

/// Struct-of-arrays the device `pie_sample` kernel consumes. One entry per
/// logit row, in row order. Owned `Vec`s so the executor can hand stable
/// host pointers to `pie_upload_inputs` (the C++ side widens `seed` u32→u64
/// internally for the flashinfer path; we keep it u32 here, matching the
/// `pi.sample_seed` layout).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SampleParamsSoA {
    pub temperature: Vec<f32>,
    pub top_p: Vec<f32>,
    pub top_k: Vec<i32>,
    pub seed: Vec<u32>,
}

impl SampleParamsSoA {
    pub fn len(&self) -> usize {
        self.temperature.len()
    }

    pub fn is_empty(&self) -> bool {
        self.temperature.is_empty()
    }
}

/// Build the per-row SoA the `pie_sample` kernel consumes from per-row knobs
/// + the fire's seed schedule.
///
/// Replicates the per-slot fill in `build_sample_plan`:
///   * `top_k == 0` (no filter) maps to `vocab_size`;
///   * the seed is resolved through the schedule (fresh only for stochastic
///     rows whose wire seed is 0);
///   * `top_p`/`temperature` pass through unchanged.
///
/// `rows` is in logit-row order and the returned arrays are aligned with it
/// (length `rows.len()`).
pub fn build_soa(
    rows: &[RowParams],
    vocab_size: i32,
    seeds: &SeedScheduler,
) -> SampleParamsSoA {
    let n = rows.len();
    let mut soa = SampleParamsSoA {
        temperature: Vec::with_capacity(n),
        top_p: Vec::with_capacity(n),
        top_k: Vec::with_capacity(n),
        seed: Vec::with_capacity(n),
    };
    for r in rows {
        let top_k = if r.top_k == 0 { vocab_size } else { r.top_k };
        let seed = seeds.resolve_row_seed(r.temperature, r.wire_seed);
        soa.temperature.push(r.temperature);
        soa.top_p.push(r.top_p);
        soa.top_k.push(top_k);
        soa.seed.push(seed);
    }
    soa
}

/// Whether the whole batch is greedy (every row temperature == 0). Selects
/// the dense argmax fast path — `all_rows_greedy` in `build_sample_plan`.
pub fn all_rows_greedy(rows: &[RowParams]) -> bool {
    rows.iter().all(RowParams::is_greedy)
}

/// Whether any row routes the batch to the flashinfer top-k/top-p kernel —
/// `any_topk_topp` in `build_sample_plan`.
pub fn any_topk_topp(rows: &[RowParams]) -> bool {
    rows.iter().any(RowParams::needs_topk_topp)
}

/// Build the device-side `PieSampleParams` SoA and dispatch the kernel.
///
/// The device sampling kernel stays C++ behind the ABI (`pie_sample`); this
/// host stub documents the call shape and is wired in phase 3 once the ABI
/// entry is live. Until then it builds the SoA (the real work this module
/// owns) and leaves the device call as a documented stub.
pub fn sample(rows: &[RowParams], vocab_size: i32, seeds: &SeedScheduler) -> SampleParamsSoA {
    let soa = build_soa(rows, vocab_size, seeds);
    // DEVICE STUB: `pie_sample(ws, &soa.temperature, &soa.top_p,
    //   &soa.top_k, &soa.seed, n, vocab_size, prng_offset, stream)`.
    // The argmax / categorical kernels live in C++ (kernels/sample_temp,
    // kernels/sample_flashinfer); this host side only assembles their inputs.
    soa
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known-answer vectors generated from the verbatim executor.cpp
    // splitmix64 (see /tmp/sampler_ref.cpp).
    #[test]
    fn splitmix64_known_answers() {
        assert_eq!(splitmix64(0x0000_0000_0000_0000), 0xe220_a839_7b1d_cdaf);
        assert_eq!(splitmix64(0x0000_0000_0000_0001), 0x910a_2dec_8902_5cc1);
        assert_eq!(splitmix64(0x9E37_79B9_7F4A_7C15), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(splitmix64(0xDEAD_BEEF_CAFE_BABE), 0x0d7d_9356_0d19_29d2);
        assert_eq!(splitmix64(0xFFFF_FFFF_FFFF_FFFF), 0xe4d9_7177_1b65_2c20);
    }

    #[test]
    fn splitmix64_deterministic() {
        for x in [0u64, 1, 42, 0xABCD_EF01_2345_6789, u64::MAX] {
            assert_eq!(splitmix64(x), splitmix64(x));
        }
    }

    // Matches the C++ reference fresh schedule with base 0x0123456789ABCDEF.
    #[test]
    fn seed_schedule_known_answers() {
        let sched = SeedScheduler::from_base(0x0123_4567_89AB_CDEF);
        let expected: [u32; 6] = [
            0xb1f5_929a, 0xe1d2_8208, 0xb6fd_7b90, 0xee92_7edf, 0x15d3_cce1, 0x3e97_1b3e,
        ];
        for &want in &expected {
            assert_eq!(sched.next_fresh_seed(), want);
        }
    }

    #[test]
    fn seed_schedule_deterministic_across_instances() {
        let a = SeedScheduler::from_base(12345);
        let b = SeedScheduler::from_base(12345);
        for _ in 0..32 {
            assert_eq!(a.next_fresh_seed(), b.next_fresh_seed());
        }
    }

    #[test]
    fn seed_schedule_distinct_draws() {
        // Distinct rows/fires (successive draws) get distinct seeds.
        let sched = SeedScheduler::from_base(0xDEAD_BEEF);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10_000 {
            let s = sched.next_fresh_seed();
            assert_ne!(s, 0, "schedule must never hand out 0");
            assert!(seen.insert(s), "draw collision (should be vanishingly rare)");
        }
    }

    #[test]
    fn distinct_bases_give_distinct_sequences() {
        let a = SeedScheduler::from_base(0);
        let b = SeedScheduler::from_base(1);
        let sa: Vec<u32> = (0..8).map(|_| a.next_fresh_seed()).collect();
        let sb: Vec<u32> = (0..8).map(|_| b.next_fresh_seed()).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn resolve_row_seed_policy() {
        let sched = SeedScheduler::from_base(0x0123_4567_89AB_CDEF);
        // Greedy row: seed passed through (here 0), no draw consumed.
        assert_eq!(sched.resolve_row_seed(0.0, 0), 0);
        // Stochastic row with explicit seed: kept verbatim, no draw consumed.
        assert_eq!(sched.resolve_row_seed(1.0, 777), 777);
        // Stochastic row with wire seed 0: first fresh draw.
        assert_eq!(sched.resolve_row_seed(0.5, 0), 0xb1f5_929a);
    }

    #[test]
    fn greedy_detection() {
        let greedy = RowParams { temperature: 0.0, top_p: 1.0, top_k: 0, wire_seed: 0 };
        let hot = RowParams { temperature: 0.7, top_p: 0.9, top_k: 40, wire_seed: 0 };
        assert!(greedy.is_greedy());
        assert!(!hot.is_greedy());

        assert!(all_rows_greedy(&[greedy, greedy]));
        assert!(!all_rows_greedy(&[greedy, hot]));
        assert!(all_rows_greedy(&[]));
    }

    #[test]
    fn topk_topp_detection() {
        // top_k filter + temp > 0 → routes to flashinfer.
        let tk = RowParams { temperature: 1.0, top_p: 1.0, top_k: 40, wire_seed: 0 };
        // top_p filter + temp > 0 → routes to flashinfer.
        let tp = RowParams { temperature: 1.0, top_p: 0.8, top_k: 0, wire_seed: 0 };
        // filter present but temp == 0 (greedy) → does NOT route.
        let greedy_filtered = RowParams { temperature: 0.0, top_p: 0.8, top_k: 40, wire_seed: 0 };
        // no filter → does NOT route.
        let plain = RowParams { temperature: 1.0, top_p: 1.0, top_k: 0, wire_seed: 0 };
        assert!(tk.needs_topk_topp());
        assert!(tp.needs_topk_topp());
        assert!(!greedy_filtered.needs_topk_topp());
        assert!(!plain.needs_topk_topp());

        assert!(any_topk_topp(&[plain, tk]));
        assert!(!any_topk_topp(&[plain, greedy_filtered]));
    }

    #[test]
    fn soa_construction_shape_and_values() {
        let vocab = 32_000;
        let sched = SeedScheduler::from_base(0x0123_4567_89AB_CDEF);
        let rows = vec![
            // greedy: seed stays 0, top_k 0 -> vocab.
            RowParams { temperature: 0.0, top_p: 1.0, top_k: 0, wire_seed: 0 },
            // stochastic, explicit seed kept, explicit top_k kept.
            RowParams { temperature: 0.8, top_p: 0.95, top_k: 50, wire_seed: 999 },
            // stochastic, wire seed 0 -> first fresh draw; top_k 0 -> vocab.
            RowParams { temperature: 1.0, top_p: 0.9, top_k: 0, wire_seed: 0 },
        ];
        let soa = build_soa(&rows, vocab, &sched);

        // Shape: every array aligned to rows.len().
        assert_eq!(soa.len(), 3);
        assert_eq!(soa.temperature.len(), 3);
        assert_eq!(soa.top_p.len(), 3);
        assert_eq!(soa.top_k.len(), 3);
        assert_eq!(soa.seed.len(), 3);

        // Values.
        assert_eq!(soa.temperature, vec![0.0, 0.8, 1.0]);
        assert_eq!(soa.top_p, vec![1.0, 0.95, 0.9]);
        assert_eq!(soa.top_k, vec![vocab, 50, vocab]); // 0 -> vocab mapping
        // Row 0 greedy: seed untouched (0). Row 1: explicit 999. Row 2:
        // first fresh draw from the known schedule.
        assert_eq!(soa.seed, vec![0, 999, 0xb1f5_929a]);
    }

    #[test]
    fn soa_empty_batch() {
        let sched = SeedScheduler::from_base(7);
        let soa = build_soa(&[], 1000, &sched);
        assert!(soa.is_empty());
        assert_eq!(soa.len(), 0);
    }

    #[test]
    fn sample_returns_soa() {
        let sched = SeedScheduler::from_base(7);
        let rows = vec![RowParams { temperature: 0.0, top_p: 1.0, top_k: 0, wire_seed: 0 }];
        let soa = sample(&rows, 100, &sched);
        assert_eq!(soa.len(), 1);
        assert_eq!(soa.top_k, vec![100]);
    }
}
