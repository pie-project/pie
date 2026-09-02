//! [`Sink`]: the structure half of the walk, and [`EagerSink`], the mode
//! that does nothing with it. The model plane's polymorphism is two
//! traits — `Dispatch` for one op, `Sink` for the structure around the
//! ops. A shell supplies both and calls [`walk()`](fn@crate::fire::walk).
//!
//! Two modes: `EagerSink` makes every method a no-op (the walk's own
//! control flow is the structure, and `Dispatch::exec` runs the op now —
//! the golden path); a recording sink opens a capture scope at region
//! begins, emits conditional/event nodes for cond/fork/join, and enqueues
//! into the capture instead of a stream. Captured is eager by
//! construction: one loop decides what runs and in what order, so the two
//! modes differ only in what "runs" means — the verification strategy
//! (eager as golden, replay as subject) is built into the architecture.
//!
//! No default bodies: a sink that silently ignored `cond_begin` would
//! record a conditional region's body unconditionally, and the graph
//! would be wrong in a way nothing downstream can detect. `EagerSink`
//! says "nothing" for each event, explicitly.

use model_compiler::{Lowering, Region};

/// A recorded synchronization point: what a fork records and a join waits
/// on. The compiler mints these (`model_compiler::stream`) and the
/// artifact names them on the regions themselves, so this is a re-export
/// rather than a second numbering.
pub use model_compiler::EventId;

/// The structure events of one walk, in the order they happen. The ops
/// themselves go to `Dispatch`, not here, so an implementor sees the shape
/// of a fire and nothing about its arithmetic.
pub trait Sink {
    /// A region is about to run. Regions arrive in `CompiledModel::template`
    /// order, every one of them, whether or not this fire has rows for it —
    /// the structure is composition-independent.
    fn region_begin(&mut self, region: &Region);

    /// That region is done.
    fn region_end(&mut self, region: &Region);

    /// One run of the region's window is about to be dispatched: `run` of
    /// `runs`, ascending in row order. A region whose class set can't be
    /// covered by one interval of the row order runs `Fallback::Split { r }`
    /// times, once per interval; the walk announces which one is about to
    /// dispatch, and the shell's window table resolves against that
    /// interval rather than the union. Called for every region, always —
    /// the consecutive case is `run = 0` of `runs = 1`, same as an empty
    /// window — so a sink never has to ask whether the fallback is in play.
    fn run(&mut self, run: u32, runs: u32);

    /// A conditional region is being entered — `Lowering::Switch` over a
    /// merge's arms, or `Lowering::If` over one independently-present
    /// body. A recording sink must honour it; an eager one need not,
    /// since the walk's zero-row rule already decides the same thing.
    /// Rare: on today's whole catalog only one region of one SKU uses it.
    fn cond_begin(&mut self, lowering: &Lowering);

    /// One arm of a switch, between [`cond_begin`](Sink::cond_begin) and
    /// [`cond_end`](Sink::cond_end), in `Def::Merge` arm order. Not called
    /// for an `If`, which has one body and no arm to name.
    fn cond_arm(&mut self, arm: u8);

    /// The conditional region is closed. For a switch group this arrives
    /// after the last arm's region, not after each one.
    fn cond_end(&mut self);

    /// Record `event` on the stream this region is on, here; anything
    /// waiting on it may proceed from this point. Called at the top of a
    /// fork group's main arm (the fork, arms wait on it) and at the end
    /// of an arm on the arm's stream (the join, the region after the
    /// group waits on it). A recording sink turns this into
    /// `cudaEventRecord` inside the capture.
    fn fork(&mut self, event: EventId);

    /// Wait for `event` on the stream this region is on, here, before the
    /// region's first node.
    fn join(&mut self, event: EventId);
}

/// The eager mode: no structure to record, because the structure already
/// happened. Every method is a no-op, and that is the correct
/// implementation, not a stub — the walk's own control flow is the
/// structure, and a conditional's body is taken or skipped by the same
/// zero-row rule that would have decided it anyway.
///
/// [`fork`](Sink::fork) and [`join`](Sink::join) are no-ops here, and
/// every region runs on one stream in program order, which is a
/// topological order of the dependency DAG (every edge runs from a lower
/// region index to a higher one). So the recorded graph and the eager
/// walk are two schedules of one DAG: every value written by one region
/// and read by another is separated by an edge in both, and pairs with no
/// edge write disjoint values and disjoint arena bytes. A fire's numbers
/// cannot depend on which schedule ran it — eager is the golden, replay
/// is the subject.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EagerSink;

impl Sink for EagerSink {
    fn region_begin(&mut self, _region: &Region) {}
    fn region_end(&mut self, _region: &Region) {}
    /// A no-op only because this sink resolves no operands; a sink that
    /// does resolve operands must record this number, or every run after
    /// the first reads the first one's rows.
    fn run(&mut self, _run: u32, _runs: u32) {}
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, _event: EventId) {}
    fn join(&mut self, _event: EventId) {}
}
