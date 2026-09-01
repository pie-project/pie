//! The model plane's fire substrate: lane words in, dispatched ops out.
//!
//! A **fire** is one forward pass over the batch the runtime assembled, and
//! this module is everything the host does about one that is not a device
//! call. Three steps, in order (palo design §5's fire path, host half):
//!
//! ```text
//! 1. per lane: word = Classify::of(request)        runtime, model code
//! 2. compose  lanes -> classes, order, prefix sums -> the window table
//! 3. descriptor  the window table, flat, versioned -> the one mutable
//!                channel into a recorded graph
//! 4. walk     the artifact's regions -> Dispatch calls + Sink events
//! ```
//!
//! # Why the walk is written once, here
//!
//! [`walk()`] is generic over two traits and nothing else:
//! [`Dispatch`](crate::dispatch::Dispatch) for one op and [`Sink`] for
//! structure. An eager engine hands it a `Run` and an
//! [`EagerSink`] and the ops happen; a recording engine hands it the same
//! `Run` and a graph sink and the ops are captured instead. **Captured is
//! eager by construction** — the same loop, the same order, the same skip
//! decisions — which is the verification strategy built into the shape rather
//! than asserted by a test (decision #11). It also fixes the dependency
//! direction the menlo stack had backwards: the walk needs `CompiledModel`, so
//! it could not live in the contract crate, which must not know what a
//! compiler is (decision #12). That crate is gone and the contract is
//! [`crate::dispatch`] now — beside the walk rather than under it, which is a
//! different question from the one #12 answered and is argued there.
//!
//! # What varies per fire, and what does not
//!
//! Nothing here compiles, allocates or captures. [`compose()`] is arithmetic
//! over a `Vec` of lane words; [`FireDescriptor`] is that arithmetic laid out
//! as bytes; [`walk()`] is a straight iteration over a table baked once per
//! load. **Composition** — which windows a fire has rows for — is runtime
//! data absorbed by zero-row always-launch (decision #3), not by picking a
//! different script: an empty window means the walk does not call
//! [`Dispatch::exec`](crate::dispatch::Dispatch::exec) at all in eager mode,
//! and means a captured kernel reads a zero count and returns in about a
//! microsecond in recorded mode.
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
    AxisComposition, ClassWindow, Composition, Lane, LaneRow, MaskSpan, WindowTable, compose,
    compose_axes, rung_of,
};
pub use descriptor::{
    ABI_VERSION, CLASS_BYTES, FireDescriptor, HEADER_BYTES, LANE_BYTES, MAGIC, PATCH_LANE_BYTES,
};
pub use fallback::{Serve, answers as fallback_answers, fragmentable, max_runs};
pub use sink::{EagerSink, EventId, Sink};
pub use walk::{Phases, Regions, Units, walk, walk_phases, walk_regions, walk_units};

/// What the fire substrate refuses, and why.
///
/// **AN ENGINE-INTEGRITY ERROR, NEVER A BACKEND ONE.** The split is the
/// dispatch contract's, kept from the other side: a
/// [`KernelError`](crate::KernelError) is always about the backend — no
/// kernel for this op, no kernel for this dtype, the launch would not enqueue
/// — and never about the plan. Everything here is the other kind: a fire the
/// artifact cannot describe, or a template the walk cannot execute. A lane
/// whose word matches no class is not a kernel that is missing; it is a
/// runtime and a `CompiledModel` that disagree about what model is loaded, and saying
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
    /// the plan never splits on is not this, and a well-formed `CompiledModel` cannot
    /// reach here at all. What can is a `ClassTable` whose class list does not
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
        /// `Budget::max_lanes`, the number every `Dim::Lanes` column was cut
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
        /// `Budget::max_tokens`.
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
        compiled: usize,
    },
    /// A prepare region stands after a capture region in the template.
    ///
    /// Prepare is host work that writes descriptor slots the graph then reads
    /// (design §5); running one after the launch has begun means the slot is
    /// read before it is written. The walk REFUSES rather than reorders: the
    /// order is P2's output and a walk that quietly repaired it would hide a
    /// compiler bug behind a fire that mostly works.
    PrepareAfterCapture {
        /// The offending region's index in `CompiledModel::template`.
        region: u32,
    },
    /// The template names a node the plan does not have.
    ///
    /// `CompiledModel` carries regions as ranges of `Trace::nodes`, so the two are
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
    /// The packed bytes carry a descriptor ABI this build does not speak.
    ///
    /// **REFUSED AND NEVER REGENERATED, WHICH IS A DECISION** (multimodal
    /// M3). A descriptor is built and thrown away once per fire, so no
    /// artifact and no checkpoint holds one: the only way older bytes reach a
    /// newer unpack is a shell and an engine from two builds, and re-deriving
    /// the missing fields would be this side pretending it knows what the
    /// other side meant by the words it did not write. The layout is checked,
    /// not negotiated, and the refusal names both numbers so the mismatch is
    /// legible without a debugger.
    DescriptorAbi {
        /// The version the bytes claim.
        saw: u32,
        /// The version this build packs and unpacks.
        speaks: u32,
    },
    /// More patch rows than the artifact was baked for — the second row
    /// axis's [`TooManyRows`](Fault::TooManyRows), on its own ceiling
    /// (`PatchLadder::max_patches`).
    ///
    /// Refusal (iii) of multimodal M-1e, and it is the same danger the token
    /// ceiling names: every `Dim::Patches` column is reserved at the patch
    /// ceiling, so a fire past it writes past the end of the tower's
    /// rectangles rather than running slowly.
    TooManyPatches {
        /// The patch rows the submitted lanes add up to.
        patches: u64,
        /// `PatchLadder::max_patches`.
        max: u32,
    },
    /// More images than the artifact was baked for — the patch axis's
    /// [`TooManyLanes`](Fault::TooManyLanes).
    TooManyImages {
        /// How many images were submitted, over every lane.
        images: u64,
        /// `PatchLadder::max_images`.
        max: u32,
    },
    /// The fire's patch rows are above every rung of the PATCH lattice.
    ///
    /// One immutable graph per rung per unit, so a tower fire that rounds up
    /// to nothing has no exec to launch — and it is its own refusal because
    /// the ladder is its own ladder (multimodal §5.5).
    NoPatchBucket {
        /// The patch rows this fire carries.
        patches: u32,
        /// The largest rung the patch ladder lists.
        top: u32,
    },
    /// A lane submitted images to an artifact that declares no patch axis.
    ///
    /// **REFUSAL (ii) OF MULTIMODAL M-1e**, and the honest one: a text with
    /// no tower has nowhere for a patch row to go, no exec to run it in and
    /// no rectangle to land it in. Accepting the payload and dropping it
    /// would answer the caller's image with the continuation of their text,
    /// which is a wrong answer wearing the shape of a right one.
    Towerless {
        /// The lane's index in the submitted slice.
        lane: u32,
    },
    /// A lane whose declared image geometry and patch payload disagree.
    ///
    /// **REFUSAL (i) OF MULTIMODAL M-1e, HOST HALF.** An image is at least
    /// one patch row and a patch row belongs to some image, so neither count
    /// can be zero while the other is not; a lane that says otherwise has a
    /// submission whose two halves were written by two different beliefs
    /// about what it carries. The device half of the same refusal is
    /// [`PatchRoute`](Fault::PatchRoute) — checked at the shell, because the
    /// scatter kernel cannot see it.
    PatchGeometry {
        /// The lane's index in the submitted slice.
        lane: u32,
        /// The images it declared.
        images: u32,
        /// The patch rows it declared.
        patches: u32,
    },
    /// A patch route pointing outside this fire's token rectangle.
    ///
    /// **REFUSAL (i)'S DEVICE HALF, AND THE ONE NOTHING ELSE CATCHES.**
    /// `layout.scatter_rows` is a copy with an index and no arithmetic: it
    /// reads `routes[r]` and writes that row of a rectangle it was handed,
    /// so an entry past the rectangle is an out-of-bounds DEVICE WRITE. The
    /// kernel cannot check it (it does not know the fire's row count as a
    /// bound, only as a launch shape) and the arena does not fault on it (the
    /// address stays inside one `cudaMalloc`). So the fire path checks it,
    /// BEFORE the launch, which is the only instant at which it is checkable.
    PatchRoute {
        /// Which entry of the route vector it was.
        at: u32,
        /// What it said.
        route: i32,
        /// The token rows this fire carries.
        rows: u32,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownWord { lane, word } => write!(
                f,
                "lane {lane} carries fact word {word:#b}, which is no class of \
                 this model — the runtime and the artifact disagree about what \
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
            Self::ClassTable { descriptor, compiled } => write!(
                f,
                "the descriptor carries {descriptor} classes and the artifact \
                 has {compiled} — a region's mask would index the wrong window"
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
                 this artifact was compiled from another plan"
            ),
            Self::Descriptor { what } => {
                write!(f, "these bytes are not a fire descriptor: {what}")
            }
            Self::DescriptorAbi { saw, speaks } => write!(
                f,
                "these bytes carry descriptor ABI {saw} and this build speaks {speaks} — \
                 the layout is checked and never negotiated, so the two halves are from \
                 two builds"
            ),
            Self::TooManyPatches { patches, max } => write!(
                f,
                "this fire carries {patches} patch rows and every tower column was cut \
                 at {max}"
            ),
            Self::TooManyImages { images, max } => write!(
                f,
                "this fire assembles {images} images and the artifact was baked for {max}"
            ),
            Self::NoPatchBucket { patches, top } => write!(
                f,
                "this fire carries {patches} patch rows and the largest patch rung is \
                 {top} — there is no tower exec to launch it in"
            ),
            Self::Towerless { lane } => write!(
                f,
                "lane {lane} submitted images and this artifact declares no patch axis — \
                 there is no vision tower in it for them to go through"
            ),
            Self::PatchGeometry {
                lane,
                images,
                patches,
            } => write!(
                f,
                "lane {lane} declares {images} images and {patches} patch rows, and an \
                 image is at least one patch row — its geometry and its payload disagree"
            ),
            Self::PatchRoute { at, route, rows } => write!(
                f,
                "patch route {at} lands tower output at token row {route} of a fire with \
                 {rows} rows, and the scatter would write outside the rectangle"
            ),
        }
    }
}

impl std::error::Error for Fault {}
