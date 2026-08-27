//! [`Sink`]: the structure half of the walk, and [`EagerSink`], the mode that
//! does nothing with it.
//!
//! **THE WHOLE POLYMORPHISM OF THE MODEL PLANE IS TWO TRAITS** — `Dispatch`
//! for one op, `Sink` for the structure around the ops — and there is no
//! `trait Backend` (decision #13). A shell is a thin call-order crate that
//! supplies both and calls [`walk()`](fn@crate::fire::walk); the seam between the
//! shared substrate and a device is DIRECTIONAL, not polymorphic.
//!
//! # The two modes, and why they are the same walk
//!
//! ```text
//! EagerSink       every method a no-op; the structure IS the walk's control
//!                 flow, and `Dispatch::exec` runs the op now. The golden
//!                 path, and the Metal shell (encoding per fire is eager).
//! a graph sink    the shell's, at record time: region begins open a capture
//!                 scope, cond events emit conditional nodes, fork/join emit
//!                 event nodes, and `Dispatch::exec` enqueues INTO the
//!                 capture instead of onto a stream.
//! ```
//!
//! Captured is eager by construction (decision #11): one loop decides what
//! runs and in what order, and the two modes differ only in what "runs" means.
//! That is the verification strategy — eager as the golden, replay as the
//! subject — built into the architecture rather than asserted by a test that
//! could drift.
//!
//! # Why every method is required
//!
//! No default bodies. A sink that silently ignored `cond_begin` would record a
//! conditional region's body unconditionally, and the graph would be WRONG in
//! exactly the way nothing downstream can detect — it computes. So the trait
//! makes an implementor say, once, what it does about each event; `EagerSink`
//! says "nothing", and says why.

use model_compiler::{Lowering, Region};

/// A recorded synchronization point: what a fork signals and a join waits on.
///
/// P6'S CURRENCY, TYPED NOW AND ISSUED LATER. The dep DAG over capture regions
/// is what hands these out, and v1 runs one stream, so no event is ever
/// created — but the trait that will carry them is the trait shells implement
/// today, and adding a method to it later is a change every shell pays for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u32);

/// The structure events of one walk, in the order they happen.
///
/// The ops themselves do not come through here — they go to `Dispatch` — so an
/// implementor sees the SHAPE of a fire and nothing about its arithmetic. That
/// division is what lets one `Run` serve both modes: the dispatch impl is the
/// same object in an eager fire and a recorded one.
pub trait Sink {
    /// A region is about to run. Regions arrive in `Baked::template` order,
    /// every one of them, whether or not this fire has rows for it — the
    /// structure is composition-independent, which is the property that makes
    /// one recorded graph serve every composition.
    fn region_begin(&mut self, region: &Region);

    /// That region is done.
    fn region_end(&mut self, region: &Region);

    /// A conditional region is being entered — `Lowering::Switch` over a
    /// merge's arms, or `Lowering::If` over a device mask bit (design §4).
    ///
    /// NOT CALLED IN v1: P3 constructs `AlwaysLaunch` for every region,
    /// because conditionals are an optimization and zero-row always-launch is
    /// the correctness mechanism (decision #3).
    fn cond_begin(&mut self, lowering: &Lowering);

    /// One arm of the conditional. Arms are exclusive by construction — they
    /// ARE a `Def::Merge`'s arms (decision #4) — so exactly one runs.
    fn cond_arm(&mut self, arm: u8);

    /// The conditional region is closed.
    fn cond_end(&mut self);

    /// Signal `event`: work after this point may proceed beside what follows
    /// on another stream.
    fn fork(&mut self, event: EventId);

    /// Wait on `event`.
    fn join(&mut self, event: EventId);
}

/// The eager mode: no structure to record, because the structure already
/// happened.
///
/// EVERY METHOD IS A NO-OP AND THAT IS THE CORRECT IMPLEMENTATION, not a stub.
/// In an eager fire the walk's own control flow IS the structure — a region
/// begins when the loop reaches it, a conditional arm is taken by the `if`
/// that decides whether to dispatch — so there is nothing left for a sink to
/// carry. What the type buys is that the walk which produced a graph and the
/// walk which produced the golden numbers are the same function, called twice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EagerSink;

impl Sink for EagerSink {
    fn region_begin(&mut self, _region: &Region) {}
    fn region_end(&mut self, _region: &Region) {}
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, _event: EventId) {}
    fn join(&mut self, _event: EventId) {}
}
