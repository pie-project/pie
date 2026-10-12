use crate::error::{Fault, Result};

#[must_use]
pub fn safety_floor_bytes(working_set: u64) -> u64 {
    const CAP: u64 = 128 * 1024 * 1024;
    CAP.min(working_set / 10)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    pub working_set: u64,
    /// The working set other processes held when the load began.
    pub elsewhere: u64,
    pub ceiling: u64,
    pub weights: u64,
    pub scratch: u64,
    pub floor: u64,
    pub pool: u64,
    pub minimum: u64,
}

impl Accounting {
    #[must_use]
    pub fn of(
        working_set: u64,
        elsewhere: u64,
        utilization: f64,
        weights: u64,
        minimum: u64,
    ) -> Accounting {
        Accounting::with_scratch(working_set, elsewhere, utilization, weights, 0, minimum)
    }

    /// `utilization` of `working_set`, less what `elsewhere` holds, is this
    /// load's ceiling: the weights, the arena scratch and the driver's floor
    /// come off it first, and the kv pool has the rest.
    #[must_use]
    pub fn with_scratch(
        working_set: u64,
        elsewhere: u64,
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
        let ceiling = ((working_set as f64 * fraction) as u64).saturating_sub(elsewhere);
        let floor = safety_floor_bytes(working_set);
        Accounting {
            working_set,
            elsewhere,
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
             {working_set} bytes, other processes held {elsewhere} of it when this load began, \
             and `[model] device_memory_share` = {utilization} allows pie {ceiling}; this load's resident weight tier takes {weights} \
             (`device_weight_budget` {budget}){scratch} and the driver's safety floor holds \
             back {floor}, leaving {pool} bytes for the kv pool — and the pool needs \
             {minimum} resident (`[model] device_kv_budget`, or one sequence at the \
             declared context when that is omitted). On Apple Silicon a GPU-touched \
             Shared page is WIRED and the pager never evicts it (.wiki/alto/streaming.md), \
             so this is a hard bound and not a hint: lower `[model] max_context` or `[model] \
             slots`, lower `[engine] max_forward_tokens`, raise `[model] \
             device_memory_share`, or state a smaller `[model] device_weight_budget` to \
             stream the weight tier down.",
            working_set = self.working_set,
            elsewhere = self.elsewhere,
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
    use super::Accounting;
    use engine::load::DEFAULT_DEVICE_MEMORY_SHARE;

    const GIB: u64 = 1 << 30;

    #[test]
    fn accounting_every_case() {
        the_arena_scratch_counts_against_the_ceiling();
        a_load_under_the_ceiling_is_admitted_and_one_over_it_refuses();
        a_non_finite_fraction_is_read_as_the_whole_working_set();
        what_other_processes_hold_comes_off_the_ceiling();
    }

    fn the_arena_scratch_counts_against_the_ceiling() {
        let ws = 21_800 * (GIB / 1000);
        let util = DEFAULT_DEVICE_MEMORY_SHARE;
        let without = Accounting::of(ws, 0, util, 11 * GIB, 4 * GIB);
        assert!(without.admit(Some(11 * GIB), util).is_ok());
        let with = Accounting::with_scratch(ws, 0, util, 11 * GIB, 6 * GIB, 4 * GIB);
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
        let util = DEFAULT_DEVICE_MEMORY_SHARE;
        let ok = Accounting::of(ws, 0, util, 11 * GIB, 4 * GIB);
        assert!(
            ok.admit(Some(11 * GIB), util).is_ok(),
            "11 + 4 GiB fits under ~19.6"
        );
        let over = Accounting::of(ws, 0, util, 18 * GIB, 4 * GIB);
        let why = over
            .admit(Some(18 * GIB), util)
            .expect_err("18 + 4 GiB over the ceiling");
        let said = format!("{why}");
        for needle in [
            "recommendedMaxWorkingSetSize",
            "device_memory_share",
            "WIRED",
            "device_weight_budget",
        ] {
            assert!(said.contains(needle), "the refusal names {needle}: {said}");
        }
    }

    fn a_non_finite_fraction_is_read_as_the_whole_working_set() {
        let acct = Accounting::of(20 * GIB, 0, f64::NAN, GIB, GIB);
        assert_eq!(acct.ceiling, 20 * GIB, "NaN utilization means no fraction");
    }

    fn what_other_processes_hold_comes_off_the_ceiling() {
        let alone = Accounting::of(40 * GIB, 0, 0.9, 16 * GIB, GIB);
        let shared = Accounting::of(40 * GIB, 12 * GIB, 0.9, 16 * GIB, GIB);
        assert_eq!(shared.ceiling, alone.ceiling - 12 * GIB);
        assert_eq!(shared.pool, alone.pool - 12 * GIB);
    }
}
