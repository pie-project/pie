//! **The wired-residency statute, on unified memory** — Metal's twin of
//! `engine_cuda::store::Accounting`, so the two shells tell one story.
//!
//! # Why a ceiling exists here at all, and why it is `recommendedMaxWorkingSetSize`
//!
//! On a discrete card the CUDA sibling asks "does `card x utilization` hold the
//! weights, the elastic pool and the driver floor". Apple Silicon has no
//! separate card, but it has the same failure and needs the same statute — and
//! the measurement in `.wiki/alto/streaming.md` ("mmap residency measurement,
//! M1 Max") is why. A `StorageModeShared` MTLBuffer's pages become **WIRED**
//! (non-evictable) the moment the GPU touches them, and the pager never
//! reclaims a wired page to relieve pressure: free collapses toward zero
//! instead. So a load whose weights plus kv pool exceed what the device will
//! hold does not page — it watchdog-resets the box. The one number the device
//! publishes for "what I will hold resident without paging" is
//! [`recommendedMaxWorkingSetSize`](crate::device::Context::working_set), and
//! that is this statute's `card`.
//!
//! # The sentence, stated once
//!
//! `working_set x utilization` is pie's whole allowance. Out of it come the
//! resident weight tier (the slab plus every dense plane) and the driver safety
//! floor; what is left must still cover the kv pool the deployment reserves. If
//! it does not, the load is refused — [`Fault::Residency`], reaching the
//! contract as `Error::Impossible`, because nothing another process frees
//! changes a wired ceiling. The refusal names all six numbers a CUDA operator
//! already reads: resident weights, the stated budget, the kv pool, the floor,
//! the ceiling and the utilization fraction.
//!
//! # What this is NOT
//!
//! It is not the elastic-pool accounting the CUDA plane runs at every
//! allocation — this shell's pools are one fixed reservation (`store::Pools`),
//! so there is one admission at load and no per-fire ledger. And it is not a
//! second budget the operator states: `device_weight_budget` is still the lever
//! that shrinks the weight tier, and this statute's job is to tighten that
//! lever automatically when the operator's number (or full residency) would
//! walk the box off the wired cliff.

use crate::error::{Fault, Result};

/// The default fraction of the device's recommended working set pie may hold
/// resident — weights, kv pool and scratch. Mirrors the CUDA plane's 0.90
/// (`engine_cuda::DEFAULT_GPU_MEM_UTILIZATION`), and is what
/// [`DeviceBoot`](crate::DeviceBoot) answers when a boot document states no
/// `[metal] gpu_mem_utilization`.
pub const DEFAULT_GPU_MEM_UTILIZATION: f64 = 0.90;

/// **The bytes held back for the driver**: `min(128 MiB, working_set / 10)`,
/// the same shape as `engine_cuda::device::elastic::safety_floor_bytes`.
///
/// A fixed slice off the top so a small machine is not driven to its exact
/// recommended ceiling and a large one does not reserve a wasteful tenth.
#[must_use]
pub fn safety_floor_bytes(working_set: u64) -> u64 {
    const CAP: u64 = 128 * 1024 * 1024;
    CAP.min(working_set / 10)
}

/// **The wired-residency statute for one load** — the six numbers a refusal
/// names, and the arithmetic that decides it.
///
/// Pure: [`Accounting::of`] takes the working set, the utilization fraction and
/// two demands, so a test spells the sentence with no device bound. Mirrors
/// `engine_cuda::store::Accounting` field for field, with `card` read as the
/// device's recommended working set rather than its total VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    /// What the device will hold resident without paging —
    /// `recommendedMaxWorkingSetSize`. The CUDA sibling's `card`.
    pub working_set: u64,
    /// `working_set x utilization` — pie's whole wired allowance.
    pub ceiling: u64,
    /// The resident weight tier's bytes: the slab plus every dense plane
    /// ([`Plan::device_demand`](crate::experts::Plan::device_demand)).
    pub weights: u64,
    /// `min(128 MiB, working_set / 10)`, held back for the driver.
    pub floor: u64,
    /// `ceiling - weights - floor`: what is left for the kv pool.
    pub pool: u64,
    /// The kv pool this load reserves at the declared context, across every
    /// cache row ([`pool_demand`](crate::store::pool_demand)).
    pub minimum: u64,
}

impl Accounting {
    /// Write the sentence down, from the working set, the fraction and the two
    /// demands.
    #[must_use]
    pub fn of(working_set: u64, utilization: f64, weights: u64, minimum: u64) -> Accounting {
        let fraction = if utilization.is_finite() {
            utilization.clamp(0.0, 1.0)
        } else {
            1.0
        };
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a byte count of unified memory is far inside f64's exact integer \
                      range, and the product is floored back into u64 deliberately"
        )]
        let ceiling = (working_set as f64 * fraction) as u64;
        let floor = safety_floor_bytes(working_set);
        Accounting {
            working_set,
            ceiling,
            weights,
            floor,
            pool: ceiling.saturating_sub(weights).saturating_sub(floor),
            minimum,
        }
    }

    /// **The most a resident weight tier may occupy** and still leave the kv
    /// pool its minimum under the ceiling: `ceiling - floor - minimum`.
    ///
    /// This is the effective `device_weight_budget` the slab is shrunk to when
    /// the operator's stated budget — or full residency — would over-commit the
    /// wired ceiling. Saturating to zero for a ceiling the floor and pool
    /// already exhaust, in which case even an empty weight tier does not fit and
    /// [`Accounting::admit`] refuses.
    #[must_use]
    pub fn weight_headroom(&self) -> u64 {
        self.ceiling
            .saturating_sub(self.floor)
            .saturating_sub(self.minimum)
    }

    /// **Does the device hold this deployment?** One refusal, naming every term.
    ///
    /// `budget` is the operator's stated `device_weight_budget` (or `None` for
    /// uncapped), carried only so the sentence can name it; `utilization` the
    /// same, for the fraction. Neither changes the arithmetic — the decision is
    /// `pool >= minimum`.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] — `Error::Impossible` — when the ceiling, after the
    /// resident weight tier and the driver floor, does not leave the kv pool
    /// its bytes. `Residency` and not a device fault because a wired ceiling is
    /// a statute: nothing another process frees moves it.
    pub fn admit(&self, budget: Option<u64>, utilization: f64) -> Result<()> {
        if self.pool >= self.minimum {
            return Ok(());
        }
        let budget = match budget {
            Some(bytes) => format!("{bytes} bytes"),
            None => "uncapped (the whole table resident)".to_string(),
        };
        Err(Fault::Residency(format!(
            "the device does not hold this deployment: recommendedMaxWorkingSetSize is \
             {working_set} bytes, of which `[metal] gpu_mem_utilization` = {utilization} \
             allows pie {ceiling}; this load's resident weight tier takes {weights} \
             (`device_weight_budget` {budget}) and the driver's safety floor holds back \
             {floor}, leaving {pool} bytes for the kv pool — and this model's pool at the \
             declared context needs {minimum} resident. On Apple Silicon a GPU-touched \
             Shared page is WIRED and the pager never evicts it (.wiki/alto/streaming.md), \
             so this is a hard bound and not a hint: lower `[model] max_context` or `[model] \
             slots`, raise `[metal] gpu_mem_utilization`, or state a smaller `[model] \
             device_weight_budget` to stream the weight tier down.",
            working_set = self.working_set,
            ceiling = self.ceiling,
            weights = self.weights,
            floor = self.floor,
            pool = self.pool,
            minimum = self.minimum,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{Accounting, DEFAULT_GPU_MEM_UTILIZATION, safety_floor_bytes};

    const GIB: u64 = 1 << 30;

    #[test]
    fn the_floor_is_the_smaller_of_a_tenth_and_128_mib() {
        // A big box: the 128 MiB cap bites, not the tenth.
        assert_eq!(safety_floor_bytes(32 * GIB), 128 * 1024 * 1024);
        // A tiny one: the tenth is smaller than the cap.
        assert_eq!(safety_floor_bytes(GIB), GIB / 10);
    }

    #[test]
    fn a_load_under_the_ceiling_is_admitted_and_one_over_it_refuses() {
        // 21.8 GiB working set is this box's number; 0.90 leaves ~19.6 GiB.
        let ws = 21_800 * (GIB / 1000);
        let util = DEFAULT_GPU_MEM_UTILIZATION;
        // 11 GiB weights + 4 GiB kv fits comfortably.
        let ok = Accounting::of(ws, util, 11 * GIB, 4 * GIB);
        assert!(ok.admit(Some(11 * GIB), util).is_ok(), "11 + 4 GiB fits under ~19.6");
        // 18 GiB weights + 4 GiB kv does not.
        let over = Accounting::of(ws, util, 18 * GIB, 4 * GIB);
        let why = over
            .admit(Some(18 * GIB), util)
            .expect_err("18 + 4 GiB over the ceiling");
        let said = format!("{why}");
        // The six numbers a CUDA operator already reads.
        for needle in ["recommendedMaxWorkingSetSize", "gpu_mem_utilization", "WIRED", "device_weight_budget"] {
            assert!(said.contains(needle), "the refusal names {needle}: {said}");
        }
    }

    #[test]
    fn the_headroom_is_what_the_ceiling_leaves_for_weights() {
        let ws = 20 * GIB;
        let util = 1.0; // ceiling == working_set
        let acct = Accounting::of(ws, util, 0, 4 * GIB);
        // ceiling(20) - floor(128 MiB) - kv(4) == headroom.
        assert_eq!(
            acct.weight_headroom(),
            20 * GIB - super::safety_floor_bytes(ws) - 4 * GIB
        );
        // A weight tier at exactly the headroom admits; one byte over does not.
        let fit = Accounting::of(ws, util, acct.weight_headroom(), 4 * GIB);
        assert!(fit.admit(None, util).is_ok());
        let bust = Accounting::of(ws, util, acct.weight_headroom() + 1, 4 * GIB);
        assert!(bust.admit(None, util).is_err());
    }

    #[test]
    fn a_non_finite_fraction_is_read_as_the_whole_working_set() {
        let acct = Accounting::of(20 * GIB, f64::NAN, GIB, GIB);
        assert_eq!(acct.ceiling, 20 * GIB, "NaN utilization means no fraction");
    }
}
