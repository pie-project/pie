//! Wired-residency accounting on unified memory. A GPU-touched
//! `StorageModeShared` page is wired and the pager never reclaims it, so a
//! load whose weights plus kv pool exceed
//! `recommendedMaxWorkingSetSize x utilization` does not page but
//! watchdog-resets the box. Refuses with [`Fault::Residency`] when the
//! ceiling, minus resident weights and the driver floor, can't cover the kv
//! pool.

use crate::error::{Fault, Result};

/// Default fraction of the device's recommended working set pie may hold
/// resident. Used when a boot document states no `[metal] gpu_mem_utilization`.
pub const DEFAULT_GPU_MEM_UTILIZATION: f64 = 0.90;

/// Bytes held back for the driver: `min(128 MiB, working_set / 10)`.
#[must_use]
pub fn safety_floor_bytes(working_set: u64) -> u64 {
    const CAP: u64 = 128 * 1024 * 1024;
    CAP.min(working_set / 10)
}

/// The numbers a residency refusal names, and the arithmetic that decides
/// it. Pure, so a test can build it with no device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    /// What the device holds resident without paging
    /// (`recommendedMaxWorkingSetSize`).
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
    /// Builds the accounting from the working set, the fraction, and the two
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

    /// The most a resident weight tier may occupy and still leave the kv
    /// pool its minimum: `ceiling - floor - minimum`, saturating to zero.
    #[must_use]
    pub fn weight_headroom(&self) -> u64 {
        self.ceiling
            .saturating_sub(self.floor)
            .saturating_sub(self.minimum)
    }

    /// Does the device hold this deployment? Decision is `pool >= minimum`;
    /// `budget` and `utilization` are carried only so the refusal can name them.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] when the ceiling, after weights and the floor,
    /// doesn't leave the kv pool its bytes.
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
    use super::{Accounting, DEFAULT_GPU_MEM_UTILIZATION};

    const GIB: u64 = 1 << 30;

    #[test]
    fn a_load_under_the_ceiling_is_admitted_and_one_over_it_refuses() {
        // 21.8 GiB working set; 0.90 leaves ~19.6 GiB.
        let ws = 21_800 * (GIB / 1000);
        let util = DEFAULT_GPU_MEM_UTILIZATION;
        let ok = Accounting::of(ws, util, 11 * GIB, 4 * GIB);
        assert!(ok.admit(Some(11 * GIB), util).is_ok(), "11 + 4 GiB fits under ~19.6");
        let over = Accounting::of(ws, util, 18 * GIB, 4 * GIB);
        let why = over
            .admit(Some(18 * GIB), util)
            .expect_err("18 + 4 GiB over the ceiling");
        let said = format!("{why}");
        for needle in ["recommendedMaxWorkingSetSize", "gpu_mem_utilization", "WIRED", "device_weight_budget"] {
            assert!(said.contains(needle), "the refusal names {needle}: {said}");
        }
    }

    #[test]
    fn a_non_finite_fraction_is_read_as_the_whole_working_set() {
        let acct = Accounting::of(20 * GIB, f64::NAN, GIB, GIB);
        assert_eq!(acct.ceiling, 20 * GIB, "NaN utilization means no fraction");
    }
}
