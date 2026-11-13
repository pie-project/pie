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
    /// The streamable weights' whole bytes (routed experts, token-id
    /// tables), which `weights` counts at their floor only: they leave the
    /// device through the weight cache.
    pub streamable: u64,
}

impl Accounting {
    #[must_use]
    pub fn of(
        working_set: u64,
        elsewhere: u64,
        allowance: engine::Allowance,
        weights: u64,
        minimum: u64,
    ) -> Accounting {
        Accounting::with_scratch(working_set, elsewhere, allowance, weights, 0, minimum)
    }

    /// What `allowance` gives of `working_set`, `elsewhere` held already, is
    /// this load's ceiling: the weights, the arena scratch and the driver's
    /// floor come off it first, and the kv pool has the rest.
    #[must_use]
    pub fn with_scratch(
        working_set: u64,
        elsewhere: u64,
        allowance: engine::Allowance,
        weights: u64,
        scratch: u64,
        minimum: u64,
    ) -> Accounting {
        let ceiling = allowance.room(working_set, elsewhere);
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
            streamable: 0,
        }
    }

    /// This accounting naming `bytes` of streamable weights the weights it
    /// charges count at their floor.
    #[must_use]
    pub fn with_streamable(mut self, bytes: u64) -> Accounting {
        self.streamable = bytes;
        self
    }

    /// Refuses a load whose dense weights, arena scratch and the driver's
    /// floor leave the kv pool less than it needs. Dense weights always live
    /// in device memory, so the refusal names their bytes and the remedies.
    pub fn admit(&self) -> Result<()> {
        if self.pool >= self.minimum {
            return Ok(());
        }
        let streamable = if self.streamable > 0 {
            format!(
                ", the {} bytes of streamable experts and tables counted at the floor the \
                 weight cache keeps of them",
                self.streamable
            )
        } else {
            String::new()
        };
        let scratch = if self.scratch > 0 {
            format!(
                ", the arena scratch the compiled axes reserve takes {} (sized by the widest \
                 fire)",
                self.scratch
            )
        } else {
            String::new()
        };
        Err(Fault::Residency(format!(
            "the device does not hold this deployment: recommendedMaxWorkingSetSize is \
             {working_set} bytes, other processes held {elsewhere} of it when this load began, \
             and the load's allowance (`[engine.memory] budget_device`, or the whole working \
             set) gives pie {ceiling}; the weights that stay in device memory take {weights} \
             (every dense plane{streamable}){scratch} and the driver's safety floor holds back \
             {floor}, leaving {pool} bytes for the kv pool, which needs {minimum}, one \
             sequence at the declared context. On Apple Silicon a GPU-touched Shared page is \
             WIRED and the pager never evicts it, so this is a hard bound: serve a quantized \
             deployment of the model (its dense weights in fewer bytes), lower `[engine] \
             max_context`, or raise `[engine.memory] budget_device`.",
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
    use engine::Allowance;

    const GIB: u64 = 1 << 30;

    #[test]
    fn accounting_every_case() {
        the_arena_scratch_counts_against_the_ceiling();
        a_load_under_the_ceiling_is_admitted_and_one_over_it_refuses();
        a_budget_caps_the_ceiling();
        what_other_processes_hold_comes_off_the_ceiling();
    }

    fn the_arena_scratch_counts_against_the_ceiling() {
        let ws = 21_800 * (GIB / 1000);
        let whole = Allowance::WHOLE;
        let without = Accounting::of(ws, 0, whole, 11 * GIB, 4 * GIB);
        assert!(without.admit().is_ok());
        let with = Accounting::with_scratch(ws, 0, whole, 11 * GIB, 6 * GIB, 4 * GIB);
        let why = with.admit().expect_err("11 + 6 + 4 GiB is over ~19.6");
        let said = format!("{why}");
        assert!(
            said.contains("arena scratch"),
            "the refusal names the scratch: {said}"
        );
        assert_eq!(with.pool, without.pool - 6 * GIB);
    }

    fn a_load_under_the_ceiling_is_admitted_and_one_over_it_refuses() {
        let ws = 21_800 * (GIB / 1000);
        let whole = Allowance::WHOLE;
        let ok = Accounting::of(ws, 0, whole, 11 * GIB, 4 * GIB);
        assert!(ok.admit().is_ok(), "11 + 4 GiB fits under ~19.6");
        let over = Accounting::of(ws, 0, whole, 18 * GIB, 4 * GIB);
        let why = over.admit().expect_err("18 + 4 GiB over the ceiling");
        let said = format!("{why}");
        for needle in [
            "recommendedMaxWorkingSetSize",
            "budget_device",
            "WIRED",
            "quantized deployment",
        ] {
            assert!(said.contains(needle), "the refusal names {needle}: {said}");
        }
    }

    fn a_budget_caps_the_ceiling() {
        let budget = Allowance {
            budget: Some(8 * GIB),
        };
        let acct = Accounting::of(20 * GIB, 0, budget, GIB, GIB);
        assert_eq!(
            acct.ceiling,
            8 * GIB,
            "under the share, the budget is the ceiling"
        );
    }

    fn what_other_processes_hold_comes_off_the_ceiling() {
        let alone = Accounting::of(40 * GIB, 0, Allowance::WHOLE, 16 * GIB, GIB);
        let shared = Accounting::of(40 * GIB, 12 * GIB, Allowance::WHOLE, 16 * GIB, GIB);
        assert_eq!(shared.ceiling, alone.ceiling - 12 * GIB);
        assert_eq!(shared.pool, alone.pool - 12 * GIB);
    }
}
