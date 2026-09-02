//! The three-phase step, typed.
//!
//! ```text
//!   StepView ──prepare──▶ Prepared ──enqueue──▶ Enqueued ──settle──▶ Settled
//!             host only              stream only            device answered
//!             no stream              no alloc               pinned reads only
//! ```
//!
//! `prepare`: host only, never touches a stream. `enqueue`: stream only,
//! allocates nothing (every device address was fixed at load or bake).
//! `settle`: after the device answers — sync, readback, error attribution,
//! staging release, and bookkeeping the next step's prepare reads.
//!
//! `submit` runs the three back to back per step today, at the same cost as
//! the old inline `fire`; the seam lets later work move `settle` off the
//! critical path or interleave steps without reworking this contract.

/// What a shell's per-step host state must be able to answer — its demand,
/// so the union across steps can be taken before anything runs.
///
/// A `Prepared` must hold no stream handle (enforced by the shell's own
/// field list and by review, not by this trait).
pub trait Prepared {
    /// What this step will take from supply, if it is admitted.
    fn demand(&self) -> Demand;
}

/// What a step that is on the stream can still be asked, before the device
/// has answered — a receipt for launches already in flight; everything else
/// is [`Shell::settle`]'s to read once the device is done.
pub trait Enqueued {
    /// How many launches this step put on the stream. Zero is legal — a
    /// fire every window was empty for is still a fire.
    fn launches(&self) -> u32;
}

/// The device half of a load, cut at its three phases. One implementation
/// per backend; the associated lifetimes let a `Prepared` borrow the
/// submission rather than copy it.
///
/// `enqueue` takes a `Prepared` by value and `settle` takes an `Enqueued` by
/// value, so a step cannot be enqueued twice, settled unenqueued, or settled
/// twice.
pub trait Shell {
    /// The submission a step is prepared from, as the shell reads it.
    type Step<'a>
    where
        Self: 'a;
    /// This shell's host-side per-step state. Holds no stream handle.
    type Prepared<'a>: Prepared
    where
        Self: 'a;
    /// This shell's in-flight per-step state.
    type Enqueued<'a>: Enqueued
    where
        Self: 'a;
    /// What a settled step answers its caller.
    type Settled;
    /// This shell's fault type.
    type Error;

    /// Every host decision this step needs, made now. `prev` is the step
    /// before this one in the same frame, still prepared (wave-order
    /// effects read it); `None` for the first step of a frame.
    ///
    /// # Errors
    ///
    /// The shell's fault, for a step it cannot compose, seat or plan. Nothing
    /// has launched, so a refusal here is free.
    fn prepare<'a>(
        &mut self,
        step: Self::Step<'a>,
        prev: Option<&Self::Prepared<'a>>,
    ) -> std::result::Result<Self::Prepared<'a>, Self::Error>
    where
        Self: 'a;

    /// Everything this step puts on the stream, and nothing else: no
    /// allocation, no synchronize, no host read of device state.
    ///
    /// # Errors
    ///
    /// The shell's fault, for a launch the backend refused at enqueue time.
    fn enqueue<'a>(
        &mut self,
        prepared: Self::Prepared<'a>,
    ) -> std::result::Result<Self::Enqueued<'a>, Self::Error>
    where
        Self: 'a;

    /// The obligations the sync guards: readback, error attribution, staging
    /// lifetime, eviction/teardown, and bookkeeping order.
    ///
    /// # Errors
    ///
    /// The shell's fault, carrying this step's name.
    fn settle<'a>(
        &mut self,
        enqueued: Self::Enqueued<'a>,
    ) -> std::result::Result<Self::Settled, Self::Error>
    where
        Self: 'a;
}

/// What a frame will take from the engine's supply. A frame's demand is the
/// union of its steps', not the sum: steps run one after another on one
/// device, so a page a step frees is a page the next may reuse. Committed
/// once, atomically across arenas, before any stream work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Demand {
    /// KV pages this step addresses.
    pub kv_pages: u32,
    /// Recurrent-state slots it writes.
    pub state_slots: u32,
    /// Scratch bytes it needs live at once.
    pub workspace: u64,
}

impl Demand {
    /// The demand of a frame with no steps.
    pub const ZERO: Demand = Demand {
        kv_pages: 0,
        state_slots: 0,
        workspace: 0,
    };

    /// The union of two demands, arena by arena. `max`, not `+`: two steps
    /// of one frame never hold their KV pages at the same instant, so
    /// summing would refuse frames that fit.
    #[must_use]
    pub const fn union(self, other: Demand) -> Demand {
        Demand {
            kv_pages: if self.kv_pages > other.kv_pages {
                self.kv_pages
            } else {
                other.kv_pages
            },
            state_slots: if self.state_slots > other.state_slots {
                self.state_slots
            } else {
                other.state_slots
            },
            workspace: if self.workspace > other.workspace {
                self.workspace
            } else {
                other.workspace
            },
        }
    }
}

/// The engine's half of memory: physical commit and trim. The runtime owns
/// page ids, CoW, the prefix cache and the eviction choice. [`Supply::commit`]
/// is the frame admission gate: atomic across arenas, `Exhausted` with zero
/// side effects, and past it the stream work is success-only.
pub trait Supply {
    /// What a refusal is spelled as; the contract layer above translates it
    /// to `Exhausted`/`Impossible`, not the arena code itself.
    type Error;

    /// Commit a frame's union demand, atomically.
    ///
    /// # Errors
    ///
    /// The shell's refusal when the budget will not cover it. Nothing is
    /// committed in that case, so the caller may retry the identical frame.
    fn commit(&mut self, demand: Demand) -> std::result::Result<(), Self::Error>;

    /// Give back what the load no longer needs. Background, best-effort,
    /// never on the fire path's critical section. The hint is a residency
    /// statement, not a frame's demand: a shell unmaps exactly what it is
    /// told, above the committed line, only while idle.
    fn trim(&mut self, hint: Demand) {
        let _ = hint;
    }
}

#[cfg(test)]
mod tests {
    use super::Demand;

    /// A frame's demand is the union of its steps', arena by arena — the
    /// steps are sequential on one device, so nothing sums.
    #[test]
    fn a_frames_demand_is_the_union_and_not_the_sum() {
        let a = Demand {
            kv_pages: 8,
            state_slots: 1,
            workspace: 4096,
        };
        let b = Demand {
            kv_pages: 3,
            state_slots: 5,
            workspace: 1024,
        };
        let union = a.union(b);
        assert_eq!(union.kv_pages, 8);
        assert_eq!(union.state_slots, 5);
        assert_eq!(union.workspace, 4096);
        assert_eq!(Demand::ZERO.union(a), a);
        assert_eq!(a.union(b), b.union(a));
    }
}
