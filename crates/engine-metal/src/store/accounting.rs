use crate::error::{Fault, Result};

/// What a load may place on the device: the working set Metal recommends
/// for this process, less what the device already held for it when the
/// load began (the context, queues and libraries: `outside`), capped by
/// `[engine.memory] budget_device` where the operator states one. Measured,
/// not a share of the device.
#[must_use]
pub fn capacity(working_set: u64, outside: u64, budget: Option<u64>) -> u64 {
    let measured = working_set.saturating_sub(outside);
    budget.map_or(measured, |budget| measured.min(budget))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    /// The working set Metal recommends for this process.
    pub working_set: u64,
    /// What the device held for this process before the load placed
    /// anything.
    pub outside: u64,
    /// `[engine.memory] budget_device`, where stated.
    pub budget: Option<u64>,
    pub capacity: u64,
    pub weights: u64,
    /// The fire's bytes the load places beside the weights: the arena and,
    /// once they are reserved, the staged inputs, readout and kernel scratch.
    pub fire: u64,
    pub pool: u64,
    pub minimum: u64,
    /// The streamable weights' whole bytes (routed experts, token-id
    /// tables), which `weights` counts at their floor only: they leave the
    /// device through the weight cache.
    pub streamable: u64,
}

impl Accounting {
    /// `capacity` less the weights and the fire is what the kv pool has.
    #[must_use]
    pub fn of(
        working_set: u64,
        outside: u64,
        budget: Option<u64>,
        weights: u64,
        fire: u64,
        minimum: u64,
    ) -> Accounting {
        let capacity = capacity(working_set, outside, budget);
        Accounting {
            working_set,
            outside,
            budget,
            capacity,
            weights,
            fire,
            pool: capacity.saturating_sub(weights).saturating_sub(fire),
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

    /// Refuses a load whose dense weights and fire leave the kv pool less
    /// than it needs. Dense weights always live in device memory, so the
    /// refusal names their bytes and the remedies.
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
        let budget = self.budget.map_or_else(String::new, |budget| {
            format!(", and `[engine.memory] budget_device` caps it at {budget}")
        });
        Err(Fault::Residency(format!(
            "the device does not hold this deployment: recommendedMaxWorkingSetSize is \
             {working_set} bytes and the device held {outside} of it for this process before \
             the load{budget}, leaving {capacity}; the weights that stay in device memory take \
             {weights} (every dense plane{streamable}) and one fire's activations, staged \
             inputs, readout and kernel scratch take {fire} (sized by the widest fire), leaving \
             {pool} bytes for the kv pool, which needs {minimum}, one sequence at the declared \
             context. On Apple Silicon a GPU-touched Shared page is WIRED and the pager never \
             evicts it, so this is a hard bound: serve a quantized deployment of the model (its \
             dense weights in fewer bytes), lower `[engine] max_context`, or raise \
             `[engine.memory] budget_device` where it is stated.",
            working_set = self.working_set,
            outside = self.outside,
            capacity = self.capacity,
            weights = self.weights,
            fire = self.fire,
            pool = self.pool,
            minimum = self.minimum,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::Accounting;

    const GIB: u64 = 1 << 30;

    #[test]
    fn accounting_every_case() {
        the_fire_counts_against_the_capacity();
        a_load_under_the_capacity_is_admitted_and_one_over_it_refuses();
        what_the_device_held_before_the_load_comes_off_the_capacity();
        a_budget_caps_the_capacity();
    }

    fn the_fire_counts_against_the_capacity() {
        let ws = 21_800 * (GIB / 1000);
        let without = Accounting::of(ws, 0, None, 11 * GIB, 0, 4 * GIB);
        assert!(without.admit().is_ok());
        let with = Accounting::of(ws, 0, None, 11 * GIB, 7 * GIB, 4 * GIB);
        let why = with.admit().expect_err("11 + 7 + 4 GiB is over ~21.3");
        let said = format!("{why}");
        assert!(
            said.contains("widest fire"),
            "the refusal names the fire: {said}"
        );
        assert_eq!(with.pool, without.pool - 7 * GIB);
    }

    fn a_load_under_the_capacity_is_admitted_and_one_over_it_refuses() {
        let ws = 21_800 * (GIB / 1000);
        let ok = Accounting::of(ws, 0, None, 11 * GIB, 0, 4 * GIB);
        assert!(ok.admit().is_ok(), "11 + 4 GiB fits under ~21.3");
        let over = Accounting::of(ws, 0, None, 18 * GIB, 0, 4 * GIB);
        let why = over.admit().expect_err("18 + 4 GiB over the capacity");
        let said = format!("{why}");
        for needle in [
            "recommendedMaxWorkingSetSize",
            "WIRED",
            "quantized deployment",
            "[engine] max_context",
            "budget_device",
        ] {
            assert!(said.contains(needle), "the refusal names {needle}: {said}");
        }
        for refused in [
            "device_memory_share",
            "max_forward_tokens",
            "device_kv_budget",
        ] {
            assert!(
                !said.contains(refused),
                "a refused key is not named: {said}"
            );
        }
    }

    fn what_the_device_held_before_the_load_comes_off_the_capacity() {
        let alone = Accounting::of(40 * GIB, 0, None, 16 * GIB, 0, GIB);
        let held = Accounting::of(40 * GIB, GIB, None, 16 * GIB, 0, GIB);
        assert_eq!(held.capacity, alone.capacity - GIB);
        assert_eq!(held.pool, alone.pool - GIB);
    }

    fn a_budget_caps_the_capacity() {
        let whole = Accounting::of(40 * GIB, GIB, None, 16 * GIB, 0, GIB);
        assert_eq!(whole.capacity, 39 * GIB, "no share of the device is taken");
        let capped = Accounting::of(40 * GIB, GIB, Some(20 * GIB), 16 * GIB, 0, GIB);
        assert_eq!(capped.capacity, 20 * GIB);
        assert_eq!(capped.pool, 4 * GIB);
        let roomy = Accounting::of(40 * GIB, GIB, Some(100 * GIB), 16 * GIB, 0, GIB);
        assert_eq!(
            roomy.capacity,
            39 * GIB,
            "a budget past the device caps nothing"
        );
        let over = Accounting::of(40 * GIB, 0, Some(17 * GIB), 16 * GIB, 0, 2 * GIB);
        let said = format!(
            "{}",
            over.admit().expect_err("16 + 2 GiB over a 17 GiB budget")
        );
        assert!(
            said.contains("caps it at"),
            "the refusal names the budget: {said}"
        );
    }
}
