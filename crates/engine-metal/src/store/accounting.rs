use crate::error::{Fault, Result};

pub const DEFAULT_GPU_MEM_UTILIZATION: f64 = 0.90;

#[must_use]
pub fn safety_floor_bytes(working_set: u64) -> u64 {
    const CAP: u64 = 128 * 1024 * 1024;
    CAP.min(working_set / 10)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    pub working_set: u64,
    pub ceiling: u64,
    pub weights: u64,
    pub scratch: u64,
    pub floor: u64,
    pub pool: u64,
    pub minimum: u64,
}

impl Accounting {
    #[must_use]
    pub fn of(working_set: u64, utilization: f64, weights: u64, minimum: u64) -> Accounting {
        Accounting::with_scratch(working_set, utilization, weights, 0, minimum)
    }

    #[must_use]
    pub fn with_scratch(
        working_set: u64,
        utilization: f64,
        weights: u64,
        scratch: u64,
        minimum: u64,
    ) -> Accounting {
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
            scratch,
            floor,
            pool: ceiling
                .saturating_sub(weights)
                .saturating_sub(scratch)
                .saturating_sub(floor),
            minimum,
        }
    }

    #[must_use]
    pub fn weight_headroom(&self) -> u64 {
        self.ceiling
            .saturating_sub(self.floor)
            .saturating_sub(self.minimum)
    }

    pub fn admit(&self, budget: Option<u64>, utilization: f64) -> Result<()> {
        if self.pool >= self.minimum {
            return Ok(());
        }
        let budget = match budget {
            Some(bytes) => format!("{bytes} bytes"),
            None => "uncapped (the whole table resident)".to_string(),
        };
        let scratch = if self.scratch > 0 {
            format!(
                ", the arena scratch the compiled axes reserve takes {} (sized by `[engine] \
                 max_forward_tokens`)",
                self.scratch
            )
        } else {
            String::new()
        };
        Err(Fault::Residency(format!(
            "the device does not hold this deployment: recommendedMaxWorkingSetSize is \
             {working_set} bytes, of which `[metal] gpu_mem_utilization` = {utilization} \
             allows pie {ceiling}; this load's resident weight tier takes {weights} \
             (`device_weight_budget` {budget}){scratch} and the driver's safety floor holds \
             back {floor}, leaving {pool} bytes for the kv pool — and this model's pool at \
             the declared context needs {minimum} resident. On Apple Silicon a GPU-touched \
             Shared page is WIRED and the pager never evicts it (.wiki/alto/streaming.md), \
             so this is a hard bound and not a hint: lower `[model] max_context` or `[model] \
             slots`, lower `[engine] max_forward_tokens`, raise `[metal] \
             gpu_mem_utilization`, or state a smaller `[model] device_weight_budget` to \
             stream the weight tier down.",
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
    fn accounting_every_case() {
        the_arena_scratch_counts_against_the_ceiling();
        a_load_under_the_ceiling_is_admitted_and_one_over_it_refuses();
        a_non_finite_fraction_is_read_as_the_whole_working_set();
    }

    fn the_arena_scratch_counts_against_the_ceiling() {
        let ws = 21_800 * (GIB / 1000);
        let util = DEFAULT_GPU_MEM_UTILIZATION;
        let without = Accounting::of(ws, util, 11 * GIB, 4 * GIB);
        assert!(without.admit(Some(11 * GIB), util).is_ok());
        let with = Accounting::with_scratch(ws, util, 11 * GIB, 6 * GIB, 4 * GIB);
        let why = with
            .admit(Some(11 * GIB), util)
            .expect_err("11 + 6 + 4 GiB is over ~19.6");
        let said = format!("{why}");
        assert!(
            said.contains("max_forward_tokens"),
            "the refusal names the scratch's knob: {said}"
        );
        assert_eq!(with.pool, without.pool - 6 * GIB);
    }

    fn a_load_under_the_ceiling_is_admitted_and_one_over_it_refuses() {
        let ws = 21_800 * (GIB / 1000);
        let util = DEFAULT_GPU_MEM_UTILIZATION;
        let ok = Accounting::of(ws, util, 11 * GIB, 4 * GIB);
        assert!(
            ok.admit(Some(11 * GIB), util).is_ok(),
            "11 + 4 GiB fits under ~19.6"
        );
        let over = Accounting::of(ws, util, 18 * GIB, 4 * GIB);
        let why = over
            .admit(Some(18 * GIB), util)
            .expect_err("18 + 4 GiB over the ceiling");
        let said = format!("{why}");
        for needle in [
            "recommendedMaxWorkingSetSize",
            "gpu_mem_utilization",
            "WIRED",
            "device_weight_budget",
        ] {
            assert!(said.contains(needle), "the refusal names {needle}: {said}");
        }
    }

    fn a_non_finite_fraction_is_read_as_the_whole_working_set() {
        let acct = Accounting::of(20 * GIB, f64::NAN, GIB, GIB);
        assert_eq!(acct.ceiling, 20 * GIB, "NaN utilization means no fraction");
    }
}
