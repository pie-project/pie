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

/// A recorded synchronization point: what a fork records and a join waits on.
///
/// P6'S CURRENCY. The compiler mints these — see `model_compiler::stream` —
/// and the artifact names them on the regions themselves, so this is a
/// re-export rather than a second numbering.
pub use model_compiler::EventId;

/// The structure events of one walk, in the order they happen.
///
/// The ops themselves do not come through here — they go to `Dispatch` — so an
/// implementor sees the SHAPE of a fire and nothing about its arithmetic. That
/// division is what lets one `Run` serve both modes: the dispatch impl is the
/// same object in an eager fire and a recorded one.
pub trait Sink {
    /// A region is about to run. Regions arrive in `CompiledModel::template` order,
    /// every one of them, whether or not this fire has rows for it — the
    /// structure is composition-independent, which is the property that makes
    /// one recorded graph serve every composition.
    fn region_begin(&mut self, region: &Region);

    /// That region is done.
    fn region_end(&mut self, region: &Region);

    /// One RUN of the region's window is about to be dispatched: `run` of
    /// `runs`, ascending in row order.
    ///
    /// **P4'S FALLBACK, AT THE SEAM WHERE IT IS PAID** (design §3). A region
    /// whose class set P4 could not make an interval of the row order covers
    /// several row intervals rather than one, and `Fallback::Split { r }` is
    /// the compiler saying so: the nodes run `r` times, once per interval,
    /// each over its own pointer and extent. The walk announces which one it
    /// is about to dispatch, and the shell's window table answers the
    /// resolution against THAT interval rather than against the union — which
    /// is the whole of what a sink has to carry for the slow path to be
    /// correct rather than merely not to fault.
    ///
    /// **CALLED FOR EVERY REGION, ALWAYS**, exactly as `region_begin` is: the
    /// consecutive case that P4 exists to produce is `run = 0` of `runs = 1`,
    /// and an empty window is `run = 0` of `runs = 1` as well, because a
    /// region with no rows still announces itself and still dispatches its
    /// collectives. So a sink never has to ask whether the fallback is in
    /// play; it reads one number either way.
    fn run(&mut self, run: u32, runs: u32);

    /// A conditional region is being entered — `Lowering::Switch` over a
    /// merge's arms, or `Lowering::If` over one independently-present body
    /// (design §4).
    ///
    /// **A RECORDING SINK MUST HONOUR IT; AN EAGER ONE MUST NOT HAVE TO.** The
    /// walk's zero-row rule decides exactly what a conditional decides — is
    /// this behavior in this composition — so an eager sink that no-ops this
    /// runs the same nodes over the same rows. A recorded graph outlives the
    /// fire that recorded it, so there the decision has to become a node.
    ///
    /// Rare, and deliberately so: on today's whole catalog one region of one
    /// SKU is called with this (qwen36-27b's MTP head), because a conditional
    /// is only worth its evaluation point around a body of layer granularity
    /// or coarser.
    fn cond_begin(&mut self, lowering: &Lowering);

    /// One arm of a SWITCH, between [`cond_begin`](Sink::cond_begin) and
    /// [`cond_end`](Sink::cond_end), in `Def::Merge` arm order.
    ///
    /// **NOT CALLED FOR AN `If`**, which has one body and no arm to name. And
    /// the exclusivity here is stronger than decision #4's: `resolve_classes`
    /// proves no LANE demands two arms, and a switch node needs no FIRE to
    /// have rows for two — `model_compiler::lowering` is where that second
    /// proof is discharged, and it is why a batching deployment gets `If` per
    /// arm instead.
    fn cond_arm(&mut self, arm: u8);

    /// The conditional region is closed. For a SWITCH group this arrives after
    /// the LAST arm's region, not after each one: the group is a run of
    /// consecutive regions and the bracket is around the run.
    fn cond_end(&mut self);

    /// **RECORD `event` ON THE STREAM THIS REGION IS ON, HERE.** Anything
    /// waiting on it may proceed from this point.
    ///
    /// The walk calls this at two instants and they are the two halves of
    /// design §6's pair, decomposed so that one verb serves both:
    ///
    /// ```text
    /// Region::open   at the TOP of a fork group's main arm, on the main
    ///                stream — the fork. The arms wait on it.
    /// Region::close  at the END of an arm, on the arm's stream — the join.
    ///                The region after the group waits on it.
    /// ```
    ///
    /// A recording sink turns this into `cudaEventRecord` inside the capture,
    /// which is the exact shape `.wiki/tart/evidence/green_contexts.md`
    /// Finding 3 measured: one graph, several streams, joined by events.
    fn fork(&mut self, event: EventId);

    /// **WAIT FOR `event` ON THE STREAM THIS REGION IS ON, HERE**, before the
    /// region's first node. `Region::wait`, in order.
    fn join(&mut self, event: EventId);
}

/// The eager mode: no structure to record, because the structure already
/// happened.
///
/// EVERY METHOD IS A NO-OP AND THAT IS THE CORRECT IMPLEMENTATION, not a stub.
/// In an eager fire the walk's own control flow IS the structure — a region
/// begins when the loop reaches it, and a conditional's body is taken or
/// skipped by the zero-row rule that would have decided it anyway — so there
/// is nothing left for a sink to carry. That equivalence is design §4's whole
/// claim: conditionals are an optimization and zero-row always-launch is the
/// correctness mechanism, which is only true if ignoring one is correct.
///
/// What the type buys is that the walk which produced a graph and the walk
/// which produced the golden numbers are the same function, called twice.
///
/// # Eager IS the serialization of the DAG, and that is why the tokens cannot
/// move
///
/// [`fork`](Sink::fork) and [`join`](Sink::join) are no-ops here, and every
/// region runs on one stream in program order. That is not "streams turned
/// off and hope": program order is a TOPOLOGICAL ORDER of P6's dependency
/// DAG, because every edge the DAG has runs from a lower region index to a
/// higher one — the pass builds it that way, over regions that are already in
/// program order. Running a DAG's nodes in a topological order, one at a
/// time, is what its edges mean.
///
/// So the recorded graph and the eager walk are two schedules of one DAG.
/// Every value written by one region and read by another is separated by an
/// edge in both; the only pairs whose relative order differs are pairs with no
/// edge between them, which write disjoint values and disjoint arena bytes
/// (`model_compiler::stream`'s safety argument). **A fire's numbers cannot
/// depend on which schedule ran it**, which is why the byte-identical-tokens
/// gate is a real gate rather than a hopeful one: eager is the golden, replay
/// is the subject, and P6 changed only the subject.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EagerSink;

impl Sink for EagerSink {
    fn region_begin(&mut self, _region: &Region) {}
    fn region_end(&mut self, _region: &Region) {}
    /// Nothing, and correctly so — but for a different reason from the
    /// conditional below, and the difference matters. A no-op `cond_begin` is
    /// correct because an eager walk decides the same thing the conditional
    /// would; a no-op `run` is correct only because THIS sink resolves no
    /// operands. The walk still dispatches the region's nodes once per run
    /// and still hands each launch its own row count, so a golden path
    /// carried by an eager sink runs a fragmented window as several launches
    /// exactly as a shell does. A sink that DOES resolve operands
    /// (`driver_cuda::window::Cursor`) must record this number, or every run
    /// after the first reads the first one's rows.
    fn run(&mut self, _run: u32, _runs: u32) {}
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, _event: EventId) {}
    fn join(&mut self, _event: EventId) {}
}
