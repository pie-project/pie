//! The model plane's fire substrate: lane words in, dispatched ops out.
//!
//! A **fire** is one forward pass over the batch the engine assembled, and
//! this module is everything the host does about one that is not a device
//! call. Three steps, in order (palo design §5's fire path, host half):
//!
//! ```text
//! 1. per lane: word = Classify::of(request)        engine, model code
//! 2. compose  lanes -> classes, order, prefix sums -> the window table
//! 3. descriptor  the window table, flat, versioned -> the one mutable
//!                channel into a recorded graph
//! 4. walk     the artifact's regions -> Dispatch calls + Sink events
//! ```
//!
//! # Why the walk is written once, here
//!
//! [`walk()`] is generic over two traits and nothing else: [`kernels::Dispatch`] for one
//! op and [`Sink`] for structure. An eager driver hands it a `Run` and an
//! [`EagerSink`] and the ops happen; a recording driver hands it the same
//! `Run` and a graph sink and the ops are captured instead. **Captured is
//! eager by construction** — the same loop, the same order, the same skip
//! decisions — which is the verification strategy built into the shape rather
//! than asserted by a test (decision #11). It also fixes the dependency
//! direction the menlo stack had backwards: the walk needs `Baked`, so it
//! cannot live in `kernels`, which must not know what a compiler is
//! (decision #12).
//!
//! # What varies per fire, and what does not
//!
//! Nothing here compiles, allocates or captures. [`compose()`] is arithmetic
//! over a `Vec` of lane words; [`FireDescriptor`] is that arithmetic laid out
//! as bytes; [`walk()`] is a straight iteration over a table baked once per
//! load. **Composition** — which windows a fire has rows for — is runtime
//! data absorbed by zero-row always-launch (decision #3), not by picking a
//! different script: an empty window means the walk does not call
//! [`kernels::Dispatch::exec`] at all in eager mode, and means a captured kernel reads
//! a zero count and returns in about a microsecond in recorded mode.
//!
//! **Except for collectives.** A `Collective`-family node runs even when its
//! window is empty (decision #5): NCCL matches calls by order, so a rank that
//! elides one deadlocks the ranks that did not — or, worse, silently pairs it
//! with a later collective. Zero-count participation is the only correct
//! reading, and [`walk()`] is where it is spelled.
//!
//! # Zero device
//!
//! This is the shared substrate: it computes offsets, counts and orders, and
//! it hands them to somebody else. Nothing in it names a stream, a graph or a
//! kernel symbol — `unsafe_code = "forbid"` is the mechanical proof, and the
//! reason the whole fire path is testable on a laptop with no GPU in it.

use std::fmt;

#[cfg(test)]
pub(crate) mod fixture;

pub mod compose;
pub mod descriptor;
pub mod fallback;
pub mod sink;
pub mod walk;

pub use compose::{
    ClassWindow, Composition, Lane, LaneRow, MaskSpan, RowSpan, WindowTable, compose,
};
pub use descriptor::{ABI_VERSION, CLASS_BYTES, FireDescriptor, HEADER_BYTES, LANE_BYTES, MAGIC};
pub use fallback::{Serve, answers as fallback_answers, fragmentable, max_runs};
pub use sink::{EagerSink, EventId, Sink};
pub use walk::{Phases, walk, walk_phases};

/// What the fire substrate refuses, and why.
///
/// **A DRIVER-INTEGRITY ERROR, NEVER A BACKEND ONE.** The split is the
/// kernels doctrine kept from the other side: a
/// [`KernelError`](kernels::KernelError) is always about the backend — no
/// kernel for this op, no kernel for this dtype, the launch would not enqueue
/// — and never about the plan. Everything here is the other kind: a fire the
/// artifact cannot describe, or a template the walk cannot execute. A lane
/// whose word matches no class is not a kernel that is missing; it is an
/// engine and a `Baked` that disagree about what model is loaded, and saying
/// so in the backend's error type would send the operator hunting for a
/// kernel that was never the problem.
///
/// Every variant carries the numbers, because every one of these presents at
/// three in the morning as "the batch with the odd mix in it".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// A lane's fact word matches no class of this artifact.
    ///
    /// The class sweep is total over the `2^F` words its own guards reach, and
    /// `compose` masks every lane's word down to those bits first — so a bit
    /// the plan never splits on is not this, and a well-formed `Baked` cannot
    /// reach here at all. What can is a `Classes` whose class list does not
    /// cover its own mask: an artifact that was not baked by the compiler this
    /// walk was written against.
    UnknownWord {
        /// The lane's index in the submitted slice.
        lane: u32,
        /// The word it carried.
        word: u64,
    },
    /// A lane carrying no token rows.
    ///
    /// A lane IS its rows: the compiler refuses a budget admitting more lanes
    /// than rows for exactly this reason, and a zero-row lane would take a
    /// seat in the class's lane count while contributing nothing to its
    /// window.
    EmptyLane {
        /// The lane's index in the submitted slice.
        lane: u32,
    },
    /// More lanes than the artifact was baked for.
    TooManyLanes {
        /// How many were submitted.
        lanes: usize,
        /// `Budgets::max_lanes`, the number every `Dim::Lanes` column was cut
        /// at.
        max: u32,
    },
    /// More token rows than the artifact was baked for.
    ///
    /// The arena's rectangles are reserved at `max_tokens` rows, so a fire
    /// past it does not run slowly — it writes past the end of every
    /// `Dim::Tokens` column it touches.
    TooManyRows {
        /// The rows the submitted lanes add up to.
        rows: u64,
        /// `Budgets::max_tokens`.
        max: u32,
    },
    /// The fire's rows are above every bucket in the lattice.
    ///
    /// One immutable graph per bucket (design §5), so a fire that rounds up
    /// to nothing has no graph to launch. A lattice that stops short of
    /// `max_tokens` is the usual cause.
    NoBucket {
        /// The rows this fire carries.
        rows: u32,
        /// The largest bucket the budget lists.
        top: u32,
    },
    /// A descriptor whose class table is not this artifact's.
    ///
    /// A region's mask indexes the class table by position, so a descriptor
    /// of the wrong width does not fail to find a class — it finds the wrong
    /// one, and the fire runs the wrong windows.
    ClassTable {
        /// Classes the descriptor carries.
        descriptor: usize,
        /// Classes the artifact has.
        baked: usize,
    },
    /// A prepare region stands after a capture region in the template.
    ///
    /// Prepare is host work that writes descriptor slots the graph then reads
    /// (design §5); running one after the launch has begun means the slot is
    /// read before it is written. The walk REFUSES rather than reorders: the
    /// order is P2's output and a walk that quietly repaired it would hide a
    /// compiler bug behind a fire that mostly works.
    PrepareAfterCapture {
        /// The offending region's index in `Baked::template`.
        region: u32,
    },
    /// The template names a node the plan does not have.
    ///
    /// `Baked` carries regions as ranges of `Plan::nodes`, so the two are
    /// only meaningful together — this is a plan and an artifact that were
    /// not baked from each other.
    NoSuchNode {
        /// The node index the region asked for.
        node: u32,
        /// How many nodes the plan has.
        nodes: usize,
    },
    /// The packed bytes are not a descriptor this build can read.
    Descriptor {
        /// What was wrong with them, as a phrase.
        what: &'static str,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownWord { lane, word } => write!(
                f,
                "lane {lane} carries fact word {word:#b}, which is no class of \
                 this model — the engine and the artifact disagree about what \
                 is loaded"
            ),
            Self::EmptyLane { lane } => write!(
                f,
                "lane {lane} carries no token rows, and a lane is its rows"
            ),
            Self::TooManyLanes { lanes, max } => write!(
                f,
                "this fire assembles {lanes} lanes and the artifact was baked \
                 for {max}"
            ),
            Self::TooManyRows { rows, max } => write!(
                f,
                "this fire carries {rows} token rows and every column was cut \
                 at {max}"
            ),
            Self::NoBucket { rows, top } => write!(
                f,
                "this fire carries {rows} token rows and the largest bucket is \
                 {top} — there is no graph to launch it in"
            ),
            Self::ClassTable { descriptor, baked } => write!(
                f,
                "the descriptor carries {descriptor} classes and the artifact \
                 has {baked} — a region's mask would index the wrong window"
            ),
            Self::PrepareAfterCapture { region } => write!(
                f,
                "region {region} is host prepare work standing after the graph \
                 body, so it would write a descriptor slot the launch already \
                 read"
            ),
            Self::NoSuchNode { node, nodes } => write!(
                f,
                "the template runs node {node} of a plan that has {nodes} — \
                 this artifact was baked from another plan"
            ),
            Self::Descriptor { what } => {
                write!(f, "these bytes are not a fire descriptor: {what}")
            }
        }
    }
}

impl std::error::Error for Fault {}
