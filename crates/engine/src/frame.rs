//! **The three-phase step, typed** (alto design §3 and §4; articles 2, 4, 7).
//!
//! ```text
//!   StepView ──prepare──▶ Prepared ──enqueue──▶ Enqueued ──settle──▶ Settled
//!             host only              stream only            device answered
//!             no stream              no alloc               pinned reads only
//! ```
//!
//! # Why this is three functions and not one
//!
//! `fire_captured` is one function today, and the seam this module names is
//! already drawn inside it — by a comment. The sync at the end of that
//! function carries a list, written by the person who tried to move it, of the
//! five things it guards: the readback, error attribution, staging lifetime,
//! eviction/teardown, and bookkeeping order. Every one of those is
//! **settle's** work; everything above the sync that touches a stream is
//! **enqueue's**; everything above the first stream touch is **prepare's**.
//! The comment was the design. This module is the same design, in types the
//! compiler checks.
//!
//! # What each phase may do, and what it may not
//!
//! * **`prepare` — host only.** Composition, descriptor arithmetic, page
//!   geometry, window resolution, mask expansion, plan building, validation.
//!   Article 2 says no host read, decision, synchronize or memcpy may gate the
//!   transition between consecutive waves; the enforcement is that
//!   [`Prepared`] is a value that **cannot reach a stream** — a shell's
//!   `Prepared` holds ring indices and plan objects, never a stream handle,
//!   so hoisting all k steps' host work to frame entry is structurally
//!   possible rather than a discipline someone maintains.
//! * **`enqueue` — stream only, and it allocates nothing.** Article 7: every
//!   device address captured work reads was fixed at load or bake, so the fire
//!   path allocates nothing. What crosses here is bytes and table entries.
//! * **`settle` — the device has answered.** The sync, the readback, error
//!   attribution, staging release, and the bookkeeping the NEXT step's prepare
//!   will read. In F1 it is synchronous and runs immediately; in F2 it is what
//!   the completion broker drives.
//!
//! # F1 lands the seams, not the saturation
//!
//! `submit` runs `prepare → enqueue → settle` back to back, per step, exactly
//! where `fire` ran them inline — the same launches in the same order at the
//! same cost. What changed is that the three are now functions with types
//! between them, so wave F2 can move `settle` off the critical path and F3 can
//! interleave `prepare(W+1)` with `enqueue(W)` without inventing the seam
//! first.

/// **What a shell's per-step host state must be able to answer** — its demand,
/// so that article 4's union can be taken before anything runs.
///
/// The type itself is the shell's: a `Prepared` holds a composition, a
/// descriptor, staged slot INDICES, a fold decision, an attention plan. What
/// this trait fixes is the one question the neutral admission gate must ask of
/// it.
///
/// **THE CONSTITUTIONAL PROPERTY IS A NEGATIVE ONE AND NO TRAIT CAN STATE IT:**
/// a `Prepared` must hold no stream handle. It is enforced by the shell's own
/// field list and by review, and it is why the phase exists at all.
pub trait Prepared {
    /// What this step will take from supply, if it is admitted.
    fn demand(&self) -> Demand;
}

/// **What a step that is on the stream can still be asked**, before the device
/// has answered.
///
/// Deliberately almost nothing. An `Enqueued` is a receipt for launches that
/// are already in flight: the shell may count them, and everything else is
/// [`Shell::settle`]'s to read once the device is done.
pub trait Enqueued {
    /// How many launches this step put on the stream. Zero is legal — a fire
    /// every window was empty for is still a fire (article 5's zero-row
    /// always-launch is about a KERNEL launching, not about a step having to).
    fn launches(&self) -> u32;
}

/// **The device half of a load, cut at its three phases.**
///
/// One implementation per backend. The associated lifetimes are what let a
/// `Prepared` borrow the submission it was prepared from rather than copying
/// it: a frame's steps outlive the frame's `submit`, so nothing here needs to
/// own what the caller already owns.
///
/// # The phases are consumed in order and the types say so
///
/// `enqueue` takes a `Prepared` **by value** and `settle` takes an `Enqueued`
/// by value, so a step cannot be enqueued twice, settled without being
/// enqueued, or settled twice. Wave F2 wants `enqueue(&Prepared)` instead —
/// its ping-pong needs step W-1's prepared state alive while W is being
/// prepared — and that is a widening of this signature, not a rework of it:
/// `prepare` already takes `prev` for exactly that reason and F1 always passes
/// `None`.
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

    /// **Every host decision this step needs, made now** (articles 2 and 5).
    ///
    /// `prev` is the step before this one in the same frame, still prepared —
    /// wave-order effects read it (channel sequence tickets apply in wave
    /// order; the fold's ping-pong rebinds W's idle exec while W-1 runs).
    /// `None` for the first step of a frame, and for every step in F1.
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

    /// **Everything this step puts on the stream, and nothing else**
    /// (articles 1 and 7).
    ///
    /// No allocation, no synchronize, no host read of device state.
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

    /// **The five obligations the sync guards**: the readback, error
    /// attribution, staging lifetime, eviction/teardown, and the bookkeeping
    /// order the next step's prepare depends on.
    ///
    /// # Errors
    ///
    /// The shell's fault, carrying THIS step's name — which is the second of
    /// the five obligations and the reason the sync is here rather than
    /// wherever the next blocking call happens to be.
    fn settle<'a>(
        &mut self,
        enqueued: Self::Enqueued<'a>,
    ) -> std::result::Result<Self::Settled, Self::Error>
    where
        Self: 'a;
}

/// **What a frame will take from the engine's supply** (alto design §8;
/// article 8: the runtime owns policy, the engine owns supply).
///
/// A frame's demand is the UNION of its steps' — not the sum — because the
/// steps run one after another on one device and a page a step frees is a page
/// the next may have. Article 4 commits it once, atomically across arenas,
/// before any stream work.
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

    /// **The union of two demands, arena by arena.**
    ///
    /// `max`, not `+`. Two steps of one frame do not hold their KV pages at
    /// the same instant — a frame is sequential on one device — so summing
    /// would refuse frames that fit.
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

/// **The engine's half of memory: physical commit and trim** (design §8).
///
/// The runtime owns page ids, CoW, the prefix cache and the eviction choice;
/// this is the other side of that line, and the line is crossed once in one
/// spelling (article 8). [`Supply::commit`] is the frame admission gate:
/// atomic across arenas, `Exhausted` with **zero side effects**, and past it
/// the stream work is success-only.
///
/// **F1 IS THE RESERVATION MODEL, HONESTLY NAMED.** The shells here carve
/// fixed pools at load and grow nothing, so `commit` is the ceiling check that
/// already existed and `trim` does nothing. Wave C makes it dev's elastic
/// shape — budgeted physical pool, VMM arenas under fixed virtual ranges — and
/// the 10-second resize poll dies, because elasticity becomes a side effect of
/// admission rather than a thing somebody polls for.
pub trait Supply {
    /// What a refusal is spelled as. The shells speak their own fault type and
    /// the contract layer above them turns it into `Exhausted`/`Impossible`;
    /// putting `engine_api::Error` here would push that translation down into
    /// the arena code, which is the one place that genuinely knows only about
    /// bytes.
    type Error;

    /// Commit a frame's union demand, atomically.
    ///
    /// # Errors
    ///
    /// The shell's refusal when the budget will not cover it. Nothing is
    /// committed in that case: article 4's zero side effects are this method's
    /// promise, and it is why the caller may retry the identical frame.
    fn commit(&mut self, demand: Demand) -> std::result::Result<(), Self::Error>;

    /// Give back what a frame no longer needs. Background, best-effort, and
    /// never on the fire path's critical section.
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
