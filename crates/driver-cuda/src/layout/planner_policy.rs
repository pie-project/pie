//! The memory planner's policy layer: how the search decides what to search.
//!
//! Port of the anonymous namespace at the top of
//! `driver-cuda/csrc/src/store/memory_planner.cpp`.
//!
//! # Why this half, and why first
//!
//! `plan_cuda_memory` is 936 lines that sweep a
//! (`kv_page_size` × decode target × prefill target) lattice and keep the
//! highest-scoring layout that fits. Almost all of it is bookkeeping over
//! model-specific byte counts, and it drags in every model header in the tree.
//! The part that decides *what the lattice is* and *what "highest-scoring"
//! means* is much smaller, has no model in it at all, and reads only six
//! numbers from the outside world:
//!
//! * the memory profile string and the pinned page size, from config,
//! * the tensor-parallel size, from config,
//! * `multiProcessorCount` and `major`, from the device.
//!
//! That is the whole input surface, and it is why this layer could be lifted
//! before the planner around it.
//!
//! # It was unreachable, not merely untested
//!
//! Every function here lives in an anonymous namespace, so it has internal
//! linkage and no test in the tree can name it -- the C++ test suite cannot
//! reach `clamp_pow2_nearest` even in principle. The policy that picks page
//! sizes and occupancy targets for every deployment was, structurally,
//! untestable. It has a differential proof against the original now; see
//! `tests/planner_policy_parity.rs`.

/// The minimum KV a candidate layout must hold, whatever the request cap says.
///
/// Below this a boot would admit so few sequences that admission and eviction
/// cannot recover. Named in the C++ because the "no viable layout" diagnostic
/// reports it back to the operator.
pub const MIN_KV_TOKENS_FLOOR: u64 = 32768;

/// The profile the search falls back to when it must name one but config says
/// `auto`.
///
/// `auto` is not itself a layout, so the single-answer entry points
/// ([`kv_page_size`], [`decode_target`], [`prefill_target`]) have to pick a
/// representative. The C++ picks `throughput` at each of the three call sites,
/// spelled out each time.
const AUTO_REPRESENTATIVE: &str = "throughput";

/// Round `n` up to a multiple of `a`.
#[must_use]
pub const fn align_up(n: u64, a: u64) -> u64 {
    if a == 0 { n } else { n.div_ceil(a) * a }
}

/// Clamp into `[lo, hi]`, then snap to whichever bracketing power of two is
/// nearer.
///
/// The C++ walks `p` up from 1 while `p < value && p <= hi / 2`, then considers
/// the bracket `[max(lo, p/2), min(hi, p)]`. Two consequences worth stating,
/// because both are observable and neither is obvious:
///
/// * The clamp happens **first**, so an out-of-range input is snapped from the
///   boundary, not from itself. `clamp_pow2_nearest(0, 64, 2048)` is 64.
/// * Ties go **down**. The test is `value - lower <= upper - value`, so the
///   midpoint 96 between 64 and 128 yields 64. A rewrite that used a plain
///   "round to nearest" helper would round 96 up and change every device whose
///   SM count lands on a midpoint.
#[must_use]
pub fn clamp_pow2_nearest(value: i32, lo: i32, hi: i32) -> i32 {
    let value = lo.max(value.min(hi));
    let mut p: i32 = 1;
    while p < value && p <= hi / 2 {
        p <<= 1;
    }
    let lower = lo.max(p >> 1);
    let upper = hi.min(p);
    if upper <= lower {
        return lower;
    }
    if value - lower <= upper - value {
        lower
    } else {
        upper
    }
}

/// Is this the `auto` profile?
#[must_use]
pub fn is_auto_profile(profile: &str) -> bool {
    profile == "auto"
}

/// The concrete profiles the search evaluates for a given config setting.
///
/// `auto` expands to four families; anything else is its own single-element
/// search. Note that `balanced` and `capacity` are in the `auto` search but
/// are no longer nameable from config -- the C++ comment records that keeping
/// them here is what made narrowing the config surface free, since `auto` is
/// the default and dropping them would have changed every deployment.
#[must_use]
pub fn policy_profiles(profile: &str) -> Vec<&str> {
    if is_auto_profile(profile) {
        vec!["latency", "balanced", "throughput", "capacity"]
    } else {
        vec![profile]
    }
}

/// The KV page size a single profile wants.
///
/// FlashInfer's CUDA backends support 16- and 32-token pages.
/// `balanced`/`throughput`/`capacity` prefer fewer page refs and less
/// scheduler metadata pressure; `latency` keeps the finer 16-token
/// granularity. Any tensor-parallel deployment takes 32 regardless.
///
/// The `capacity` profile is absent from the 16-token list, so it lands on 32
/// even at `tp_size == 1` -- deliberate, and easy to lose in a transcription.
#[must_use]
pub fn kv_page_size_for_profile(profile: &str, tp_size: i32) -> i32 {
    if tp_size == 1 && matches!(profile, "latency" | "balanced" | "throughput") {
        16
    } else {
        32
    }
}

/// The page sizes the lattice will sweep, ascending and deduplicated.
///
/// A pinned `kv_page_size` collapses this to one candidate: the operator has
/// already made the choice the search exists to make. Otherwise the profile
/// families contribute their preferences and 16 and 32 are appended
/// unconditionally, so the sweep always covers both even when no family asked
/// for one of them.
#[must_use]
pub fn kv_page_size_candidates(pinned: u32, profile: &str, tp_size: i32) -> Vec<i32> {
    if pinned > 0 {
        return vec![pinned as i32];
    }
    let tp = tp_size.max(1);
    let mut xs: Vec<i32> = policy_profiles(profile)
        .iter()
        .map(|p| kv_page_size_for_profile(p, tp))
        .collect();
    xs.push(16);
    xs.push(32);
    xs.sort_unstable();
    xs.dedup();
    xs
}

/// The page size to use when a single answer is needed rather than a sweep.
///
/// The parity harness sizes its own KV cache through this; the planner proper
/// sweeps [`kv_page_size_candidates`].
#[must_use]
pub fn kv_page_size(pinned_profile: &str, tp_size: i32) -> i32 {
    let profile = if is_auto_profile(pinned_profile) {
        AUTO_REPRESENTATIVE
    } else {
        pinned_profile
    };
    kv_page_size_for_profile(profile, tp_size.max(1))
}

/// `log2(value / target)`, with both floored at 1.
///
/// Signed: negative below target, positive above, zero at it. This is the
/// "how far off are we, in doublings" term of the objective.
#[must_use]
pub fn log2_ratio(value: i32, target: i32) -> f64 {
    let v = f64::from(value.max(1));
    let t = f64::from(target.max(1));
    (v / t).log2()
}

/// How saturated `value` is against `target`, on a log curve in `[0, 1]`.
///
/// `value` is capped at `target` first, so exceeding the target scores exactly
/// the same as hitting it -- this term rewards approaching a knee and is
/// deliberately flat past it, which is what stops the search buying occupancy
/// it cannot use. The `+ 1.0` on both sides keeps it defined at `value == 0`
/// and makes `target == 1` score 1 rather than dividing by zero.
#[must_use]
pub fn target_saturation_score(value: i32, target: i32) -> f64 {
    let capped = f64::from(value.max(1).min(target.max(1)));
    let t = f64::from(target.max(1));
    (capped + 1.0).log2() / (t + 1.0).log2()
}

/// Decode rows to aim for, snapped to a power of two in `[64, 2048]`.
///
/// Decode throughput has a knee that tracks SM count: below it the decode
/// matmuls leave SMs underfed, above it KV and attention pressure grow without
/// buying occupancy. `latency` and `capacity` use a smaller multiplier than
/// the other two.
///
/// The C++ signature also takes the whole `Config` and never reads it. Dropped
/// here for the same reason `activation_dtype` was dropped from the KV format
/// parser: a parameter that appears to select something and does not is worse
/// than no parameter.
#[must_use]
pub fn decode_target_for_profile(profile: &str, sm_count: i32) -> i32 {
    let sm_factor = if matches!(profile, "latency" | "capacity") {
        4
    } else {
        6
    };
    clamp_pow2_nearest(sm_count * sm_factor, 64, 2048)
}

/// [`decode_target_for_profile`] with `auto` resolved.
#[must_use]
pub fn decode_target(profile: &str, sm_count: i32) -> i32 {
    let profile = if is_auto_profile(profile) {
        AUTO_REPRESENTATIVE
    } else {
        profile
    };
    decode_target_for_profile(profile, sm_count)
}

/// Prefill tokens to aim for, snapped to a power of two.
///
/// Three profile-dependent quantities, and they do not vary together:
///
/// * the SM multiplier: 64 or 32 for `throughput` depending on the device, 16
///   otherwise;
/// * the ceiling: 8192 or 4096 for `throughput` depending on the device, 8192
///   otherwise -- so a *narrow* device caps `throughput` **lower** than every
///   other profile, which looks like a typo and is load-bearing;
/// * a post-clamp halving for `latency` and `capacity`, floored at 512.
///
/// `major >= 12` is the "wide prefill device" test. Tensor parallelism
/// contributes a factor of `min(tp, 2)`, so it stops helping past two ranks.
#[must_use]
pub fn prefill_target_for_profile(profile: &str, sm_count: i32, major: i32, tp_size: i32) -> i32 {
    let tp_factor = tp_size.clamp(1, 2);
    let wide = major >= 12;
    let is_throughput = profile == "throughput";
    let sm_factor = if is_throughput {
        if wide { 64 } else { 32 }
    } else {
        16
    };
    let max_target = if is_throughput {
        if wide { 8192 } else { 4096 }
    } else {
        8192
    };
    let target = clamp_pow2_nearest(sm_count * sm_factor * tp_factor, 512, max_target);
    if matches!(profile, "latency" | "capacity") {
        512.max(target / 2)
    } else {
        target
    }
}

/// [`prefill_target_for_profile`] with `auto` resolved.
#[must_use]
pub fn prefill_target(profile: &str, sm_count: i32, major: i32, tp_size: i32) -> i32 {
    let profile = if is_auto_profile(profile) {
        AUTO_REPRESENTATIVE
    } else {
        profile
    };
    prefill_target_for_profile(profile, sm_count, major, tp_size)
}

/// The largest prefill candidate the device is allowed to consider.
#[must_use]
pub const fn prefill_candidate_cap(major: i32) -> i32 {
    if major >= 12 { 16384 } else { 8192 }
}

/// Clamp every entry into `[1, cap]`, deduplicate, and order descending.
///
/// Descending because the lattice is searched largest-first: the first
/// candidate that fits is the most generous one, so the search can stop early.
#[must_use]
pub fn uniq_clip_desc(xs: &[i32], cap: i32) -> Vec<i32> {
    let mut out: Vec<i32> = xs.iter().map(|&x| x.min(cap).max(1)).collect();
    out.sort_unstable();
    out.dedup();
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // The exhaustive agreement with the C++ is in
    // `tests/planner_policy_parity.rs`. What is here is the behaviour worth
    // naming: the edges a reader would otherwise have to derive from the
    // arithmetic.

    #[test]
    fn clamping_happens_before_snapping_so_out_of_range_lands_on_the_bound() {
        assert_eq!(clamp_pow2_nearest(0, 64, 2048), 64);
        assert_eq!(clamp_pow2_nearest(-100, 64, 2048), 64);
        assert_eq!(clamp_pow2_nearest(999_999, 64, 2048), 2048);
    }

    #[test]
    fn ties_snap_downward() {
        // 96 is equidistant from 64 and 128. The C++ test is `<=`, so it goes
        // down -- and an SM count landing on a midpoint is not exotic.
        assert_eq!(clamp_pow2_nearest(96, 64, 2048), 64);
        assert_eq!(clamp_pow2_nearest(192, 64, 2048), 128);
        assert_eq!(
            clamp_pow2_nearest(97, 64, 2048),
            128,
            "one past the tie goes up"
        );
        assert_eq!(clamp_pow2_nearest(95, 64, 2048), 64);
    }

    #[test]
    fn snapping_lands_on_an_exact_power_of_two_within_the_range() {
        for v in -10..5000 {
            let got = clamp_pow2_nearest(v, 64, 2048);
            assert!((64..=2048).contains(&got), "{v} -> {got} left the range");
            assert!(got.count_ones() == 1, "{v} -> {got} is not a power of two");
        }
    }

    #[test]
    fn auto_expands_to_four_families_and_anything_else_to_itself() {
        assert_eq!(
            policy_profiles("auto"),
            ["latency", "balanced", "throughput", "capacity"]
        );
        assert_eq!(policy_profiles("latency"), ["latency"]);
        // An unknown profile is searched as itself rather than rejected, which
        // is how it ends up scoring on the `else` branch of every predicate.
        assert_eq!(policy_profiles("nonsense"), ["nonsense"]);
        assert_eq!(policy_profiles(""), [""]);
    }

    #[test]
    fn capacity_takes_thirty_two_byte_pages_even_on_a_single_device() {
        // The one asymmetry in the profile list, and the easiest to lose.
        assert_eq!(kv_page_size_for_profile("latency", 1), 16);
        assert_eq!(kv_page_size_for_profile("balanced", 1), 16);
        assert_eq!(kv_page_size_for_profile("throughput", 1), 16);
        assert_eq!(kv_page_size_for_profile("capacity", 1), 32);
    }

    #[test]
    fn any_tensor_parallel_deployment_takes_the_coarser_page() {
        for profile in ["latency", "balanced", "throughput", "capacity"] {
            for tp in [0, 2, 3, 8] {
                assert_eq!(
                    kv_page_size_for_profile(profile, tp),
                    32,
                    "{profile} at tp={tp}"
                );
            }
        }
    }

    #[test]
    fn a_pinned_page_size_collapses_the_lattice_to_one_candidate() {
        assert_eq!(kv_page_size_candidates(64, "auto", 1), [64]);
        // Even a pinned value no family would ever propose.
        assert_eq!(kv_page_size_candidates(1, "auto", 8), [1]);
    }

    #[test]
    fn an_unpinned_sweep_always_covers_both_supported_page_sizes() {
        // 16 and 32 are appended unconditionally, so the sweep does not depend
        // on a family having asked for them.
        assert_eq!(kv_page_size_candidates(0, "auto", 1), [16, 32]);
        assert_eq!(kv_page_size_candidates(0, "capacity", 1), [16, 32]);
        assert_eq!(kv_page_size_candidates(0, "latency", 8), [16, 32]);
    }

    #[test]
    fn saturation_is_flat_once_the_target_is_reached() {
        // The property that stops the search paying for occupancy it cannot
        // use: past the knee, more scores exactly the same.
        let at = target_saturation_score(256, 256);
        assert!(
            (at - 1.0).abs() < 1e-12,
            "hitting the target scores 1, got {at}"
        );
        assert_eq!(target_saturation_score(512, 256), at);
        assert_eq!(target_saturation_score(i32::MAX, 256), at);
    }

    #[test]
    fn saturation_is_monotone_below_the_target_and_bounded_in_zero_to_one() {
        let mut previous = f64::NEG_INFINITY;
        for v in 0..=256 {
            let s = target_saturation_score(v, 256);
            assert!((0.0..=1.0).contains(&s), "{v} scored {s}");
            assert!(s >= previous, "score fell going from {} to {v}", v - 1);
            previous = s;
        }
    }

    #[test]
    fn a_target_of_one_scores_one_rather_than_dividing_by_zero() {
        // `log2(1 + 1)` is the denominator, so the `+ 1.0` is what keeps this
        // finite. Worth pinning: dropping it is an easy "simplification".
        assert_eq!(target_saturation_score(0, 1), 1.0);
        assert_eq!(target_saturation_score(-5, 1), 1.0);
        assert!(target_saturation_score(10, 0).is_finite());
    }

    #[test]
    fn log2_ratio_is_signed_around_the_target() {
        assert_eq!(log2_ratio(256, 256), 0.0);
        assert_eq!(log2_ratio(512, 256), 1.0);
        assert_eq!(log2_ratio(128, 256), -1.0);
        // Both sides floor at 1, so nothing here can reach a log of zero.
        assert_eq!(log2_ratio(0, 1), 0.0);
        assert_eq!(log2_ratio(-9, 1), 0.0);
    }

    #[test]
    fn a_narrow_device_caps_throughput_prefill_below_every_other_profile() {
        // Reads like a typo in the original and is not: `throughput` on a
        // pre-12 device is capped at 4096 while `balanced` is capped at 8192.
        let narrow = 11;
        assert_eq!(
            prefill_target_for_profile("throughput", 512, narrow, 1),
            4096
        );
        assert_eq!(prefill_target_for_profile("balanced", 512, narrow, 1), 8192);
        // On a wide device the cap rises and the ordering inverts.
        assert_eq!(prefill_target_for_profile("throughput", 512, 12, 1), 8192);
    }

    #[test]
    fn latency_and_capacity_halve_the_prefill_target_with_a_floor() {
        for profile in ["latency", "capacity"] {
            let full = prefill_target_for_profile("balanced", 132, 9, 1);
            let halved = prefill_target_for_profile(profile, 132, 9, 1);
            assert_eq!(halved, 512.max(full / 2), "{profile}");
        }
        // The floor bites on a small device rather than going below 512.
        assert_eq!(prefill_target_for_profile("latency", 1, 9, 1), 512);
    }

    #[test]
    fn tensor_parallelism_stops_helping_prefill_past_two_ranks() {
        let at2 = prefill_target_for_profile("balanced", 16, 9, 2);
        for tp in [2, 3, 4, 8, 64] {
            assert_eq!(
                prefill_target_for_profile("balanced", 16, 9, tp),
                at2,
                "tp={tp}"
            );
        }
        // tp of 0 is treated as 1, not as 0.
        assert_eq!(
            prefill_target_for_profile("balanced", 16, 9, 0),
            prefill_target_for_profile("balanced", 16, 9, 1)
        );
    }

    #[test]
    fn uniq_clip_desc_clamps_deduplicates_and_orders_largest_first() {
        assert_eq!(
            uniq_clip_desc(&[4000, 2048, 1024, 1024, 1, 0, -7], 8192),
            [4000, 2048, 1024, 1]
        );
        // The clamp collapses everything above the cap into one entry.
        assert_eq!(uniq_clip_desc(&[8192, 16384, 300], 512), [512, 300]);
        assert_eq!(uniq_clip_desc(&[], 512), [] as [i32; 0]);
        assert_eq!(uniq_clip_desc(&[5, 5, 5], 512), [5]);
        // Descending is what lets the search stop at the first fit.
        let out = uniq_clip_desc(&[1, 9, 3, 7], 512);
        assert!(
            out.windows(2).all(|w| w[0] > w[1]),
            "{out:?} is not strictly descending"
        );
    }

    #[test]
    fn align_up_rounds_up_and_tolerates_a_zero_alignment() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        // The C++ would divide by zero here; there is no caller that does it,
        // but a total function is cheaper than a precondition nobody checks.
        assert_eq!(align_up(7, 0), 7);
    }
}
