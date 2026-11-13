use crate::error::{Fault, Result};

/// What a load may place on the device: the working set Metal recommends
/// for this process, less what the device already held for it when the
/// load began (the context, queues and libraries: `outside`). Measured, not
/// a configured share of the device.
#[must_use]
pub fn capacity(working_set: u64, outside: u64) -> u64 {
    working_set.saturating_sub(outside)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    /// The working set Metal recommends for this process.
    pub working_set: u64,
    /// What the device held for this process before the load placed
    /// anything.
    pub outside: u64,
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
    pub fn of(working_set: u64, outside: u64, weights: u64, fire: u64, minimum: u64) -> Accounting {
        let capacity = capacity(working_set, outside);
        Accounting {
            working_set,
            outside,
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
        Err(Fault::Residency(format!(
            "the device does not hold this deployment: recommendedMaxWorkingSetSize is \
             {working_set} bytes and the device held {outside} of it for this process before \
             the load, leaving {capacity}; the weights that stay in device memory take \
             {weights} (every dense plane{streamable}) and one fire's activations, staged \
             inputs, readout and kernel scratch take {fire} (sized by `[engine] \
             max_forward_tokens`), leaving {pool} bytes for the kv pool, which needs {minimum} \
             (`[model] device_kv_budget`, or one sequence at the declared context when that is \
             omitted). On Apple Silicon a GPU-touched Shared page is WIRED and the pager never \
             evicts it, so this is a hard bound: serve a quantized deployment of the model \
             (its dense weights in fewer bytes), or lower `[model] max_context` or `[engine] \
             max_forward_tokens`.",
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
    }

    fn the_fire_counts_against_the_capacity() {
        let ws = 21_800 * (GIB / 1000);
        let without = Accounting::of(ws, 0, 11 * GIB, 0, 4 * GIB);
        assert!(without.admit().is_ok());
        let with = Accounting::of(ws, 0, 11 * GIB, 7 * GIB, 4 * GIB);
        let why = with.admit().expect_err("11 + 7 + 4 GiB is over ~21.3");
        let said = format!("{why}");
        assert!(
            said.contains("max_forward_tokens"),
            "the refusal names the fire's knob: {said}"
        );
        assert_eq!(with.pool, without.pool - 7 * GIB);
    }

    fn a_load_under_the_capacity_is_admitted_and_one_over_it_refuses() {
        let ws = 21_800 * (GIB / 1000);
        let ok = Accounting::of(ws, 0, 11 * GIB, 0, 4 * GIB);
        assert!(ok.admit().is_ok(), "11 + 4 GiB fits under ~21.3");
        let over = Accounting::of(ws, 0, 18 * GIB, 0, 4 * GIB);
        let why = over.admit().expect_err("18 + 4 GiB over the capacity");
        let said = format!("{why}");
        for needle in [
            "recommendedMaxWorkingSetSize",
            "WIRED",
            "quantized deployment",
        ] {
            assert!(said.contains(needle), "the refusal names {needle}: {said}");
        }
        assert!(
            !said.contains("device_memory_share"),
            "no share is read: {said}"
        );
    }

    fn what_the_device_held_before_the_load_comes_off_the_capacity() {
        let alone = Accounting::of(40 * GIB, 0, 16 * GIB, 0, GIB);
        let held = Accounting::of(40 * GIB, GIB, 16 * GIB, 0, GIB);
        assert_eq!(held.capacity, alone.capacity - GIB);
        assert_eq!(held.pool, alone.pool - GIB);
    }
}
