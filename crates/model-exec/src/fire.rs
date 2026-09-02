//! The model plane's fire substrate: lane words in, dispatched ops out. A
//! fire is one forward pass over the runtime's batch; this module is
//! everything the host does about it that is not a device call:
//!
//! ```text
//! 1. per lane: word = Classify::of(request)        runtime, model code
//! 2. compose  lanes -> classes, order, prefix sums -> the window table
//! 3. descriptor  the window table, flat, versioned -> the one mutable
//!                channel into a recorded graph
//! 4. walk     the artifact's regions -> Dispatch calls + Sink events
//! ```
//!
//! [`walk()`] is generic over [`Dispatch`](crate::dispatch::Dispatch) (one
//! op) and [`Sink`] (structure): an eager engine hands it an [`EagerSink`]
//! and the ops happen; a recording engine hands it a graph sink and the ops
//! are captured instead — same loop, same order, same skip decisions.
//!
//! An empty window is absorbed by zero-row always-launch, not by branching
//! to a different script: the walk skips `Dispatch::exec` in eager mode, and
//! a captured kernel reads a zero count and returns in recorded mode.
//! Collectives are the exception — they run even on an empty window, since
//! NCCL matches calls by order and a rank that elides one deadlocks the rest.
//!
//! Nothing here names a stream, a graph, or a kernel symbol
//! (`unsafe_code = "forbid"`), which is why the fire path is testable with
//! no GPU.

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
    compose_axes, rung_of, chunk_spans, pass_spans};
pub use descriptor::{
    ABI_VERSION, CLASS_BYTES, FireDescriptor, HEADER_BYTES, LANE_BYTES, MAGIC, PATCH_LANE_BYTES,
};
pub use fallback::{Serve, answers as fallback_answers, fragmentable, max_runs};
pub use sink::{EagerSink, EventId, Sink};
pub use walk::{Filter, Phases, Regions, Units, walk, walk_phases, walk_regions};

/// What the fire substrate refuses, and why. Always an engine-integrity
/// error (a fire the artifact cannot describe), never a backend one (that's
/// [`KernelError`](crate::KernelError)) — so an operator isn't sent hunting
/// for a missing kernel that was never the problem. Every variant carries
/// the numbers for diagnosis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// A lane's fact word matches no class of this artifact. A well-formed
    /// `CompiledModel` cannot reach here; only a `ClassTable` whose class
    /// list doesn't cover its own mask can — an artifact baked by a
    /// different compiler than this walk.
    UnknownWord {
        /// The lane's index in the submitted slice.
        lane: u32,
        /// The word it carried.
        word: u64,
    },
    /// A lane carrying no token rows. A lane is its rows: a zero-row lane
    /// would take a seat in the class's lane count while contributing
    /// nothing to its window.
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
    /// More token rows than the artifact was baked for. The arena's
    /// rectangles are reserved at `max_tokens` rows, so a fire past it
    /// writes past the end of every `Dim::Tokens` column it touches.
    TooManyRows {
        /// The rows the submitted lanes add up to.
        rows: u64,
        /// `Budget::max_tokens`.
        max: u32,
    },
    /// The fire's rows are above every bucket in the lattice: one immutable
    /// graph per bucket, so a fire that rounds up to nothing has no graph
    /// to launch. A lattice that stops short of `max_tokens` is the usual cause.
    NoBucket {
        /// The rows this fire carries.
        rows: u32,
        /// The largest bucket the budget lists.
        top: u32,
    },
    /// A descriptor whose class table is not this artifact's. A region's
    /// mask indexes the class table by position, so a wrong-width
    /// descriptor finds the wrong class rather than none — the fire runs
    /// the wrong windows.
    ClassTable {
        /// Classes the descriptor carries.
        descriptor: usize,
        /// Classes the artifact has.
        compiled: usize,
    },
    /// A prepare region stands after a capture region in the template.
    /// Prepare is host work that writes descriptor slots the graph then
    /// reads, so running one after launch means the slot is read before
    /// it's written. The walk refuses rather than reorders, so a compiler
    /// bug can't hide behind a fire that mostly works.
    PrepareAfterCapture {
        /// The offending region's index in `CompiledModel::template`.
        region: u32,
    },
    /// The template names a node the plan does not have: a plan and an
    /// artifact that were not baked from each other.
    NoSuchNode {
        /// The node index the region asked for.
        node: u32,
        /// How many nodes the plan has.
        nodes: usize,
    },
    /// Fewer bytes than a descriptor header.
    DescriptorShort {
        /// How many bytes arrived.
        bytes: usize,
    },
    /// The first four bytes are not `"FIRE"`.
    DescriptorMagic {
        /// The word that stood there instead.
        saw: u32,
    },
    /// A length its own header disagrees with, so a record would be read
    /// half out of the next one.
    DescriptorLength {
        /// How many bytes arrived.
        bytes: usize,
        /// How many the header's counts call for.
        want: u64,
    },
    /// Class windows that do not add up to the header's row count — the
    /// corruption a device does not fault on, it computes.
    DescriptorRows {
        /// What the windows add up to.
        counted: u64,
        /// What the header claims.
        header: u32,
    },
    /// The same, on the patch axis.
    DescriptorPatchRows {
        /// What the patch windows add up to.
        counted: u64,
        /// What the header claims.
        header: u32,
    },
    /// The packed bytes carry a descriptor ABI this build does not speak.
    /// Refused, never regenerated: a descriptor is built and thrown away
    /// once per fire, so mismatched bytes mean a shell and an engine from
    /// two builds — re-deriving the missing fields would be guessing.
    DescriptorAbi {
        /// The version the bytes claim.
        saw: u32,
        /// The version this build packs and unpacks.
        speaks: u32,
    },
    /// More patch rows than the artifact was baked for — the patch axis's
    /// [`TooManyRows`](Fault::TooManyRows), on `PatchLadder::max_patches`.
    /// A fire past it writes past the end of the tower's rectangles.
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
    /// The fire's patch rows are above every rung of the patch lattice: one
    /// immutable graph per rung per unit, so a tower fire that rounds up to
    /// nothing has no exec to launch.
    NoPatchBucket {
        /// The patch rows this fire carries.
        patches: u32,
        /// The largest rung the patch ladder lists.
        top: u32,
    },
    /// A lane submitted images to an artifact that declares no patch axis: a
    /// text with no tower has nowhere for a patch row to go. Silently
    /// dropping the payload would answer the image with a text continuation.
    Towerless {
        /// The lane's index in the submitted slice.
        lane: u32,
    },
    /// A lane with images against a deployment that stated no patch ladder.
    NoPatchLadder {
        /// The lane's index in the submitted slice.
        lane: u32,
    },
    /// A windowed region fragmented past what the bake promised or bounded.
    Fragmented {
        /// The region, as the template numbers it.
        region: u32,
        /// How many runs this fire's order split it into.
        runs: u32,
        /// The most the bake allows.
        bound: u32,
        /// Whether the bake promised this region a single interval.
        promised: bool,
    },
    /// A lane whose declared image geometry and patch payload disagree: an
    /// image is at least one patch row, so neither count can be zero while
    /// the other is not. The device half of this refusal is
    /// [`PatchRoute`](Fault::PatchRoute), checked at the shell since the
    /// scatter kernel cannot see it.
    PatchGeometry {
        /// The lane's index in the submitted slice.
        lane: u32,
        /// The images it declared.
        images: u32,
        /// The patch rows it declared.
        patches: u32,
    },
    /// A patch route pointing outside this fire's token rectangle. The
    /// scatter kernel can't check this itself (no bound, only a launch
    /// shape) and the arena won't fault on it (still inside one
    /// `cudaMalloc`), so the fire path checks it before the launch — an
    /// out-of-range entry would otherwise be an out-of-bounds device write.
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
            Self::DescriptorShort { bytes } => write!(
                f,
                "these bytes are not a fire descriptor: {bytes} of them, shorter \
                 than a header"
            ),
            Self::DescriptorMagic { saw } => write!(
                f,
                "these bytes are not a fire descriptor: {saw:#010x} where the FIRE \
                 magic should stand"
            ),
            Self::DescriptorLength { bytes, want } => write!(
                f,
                "this fire descriptor carries {bytes} bytes and its own header calls \
                 for {want}, so a record would be read half out of the next one"
            ),
            Self::DescriptorRows { counted, header } => write!(
                f,
                "this fire descriptor's class windows add up to {counted} rows and \
                 its header claims {header}"
            ),
            Self::DescriptorPatchRows { counted, header } => write!(
                f,
                "this fire descriptor's patch windows add up to {counted} patch rows \
                 and its header claims {header}"
            ),
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
            Self::NoPatchLadder { lane } => write!(
                f,
                "lane {lane} submitted images and this deployment stated no patch ladder"
            ),
            Self::Fragmented {
                region,
                runs,
                bound,
                promised,
            } => write!(
                f,
                "region {region} runs as {runs} launches; the bake {} — a mismatched \
                 artifact and class table",
                if *promised {
                    "promised it one interval".to_string()
                } else {
                    format!("bounds it at {bound}")
                }
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
