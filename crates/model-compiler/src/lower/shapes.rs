//! The lowered form a driver executes.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fire {
    /// When set, the device split is authoritative for peel rows.
    pub captures_across_splits: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    pub at_op: u32,
    pub rows: Range<u32>,
}

/// One stated host preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prep {
    /// Which op stated it.
    pub at_op: u32,
    /// What to raise.
    pub kind: model_ir::trace::PrepKind,
    /// The value it publishes — `Op::outputs[0]`.
    ///
    /// Carried here so a driver raising plans can answer BY VALUE. The key
    /// alone cannot: two prefill schedules at different head dims are two
    /// objects and one word, so a resolver keyed on the string can only ever
    /// hand back whichever was raised last. This is what a statement names.
    pub value: model_ir::trace::ValueId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arg {
    /// An activation arena offset and per-row width.
    Arena {
        at: usize,
        width: u32,
        /// Bytes per element is needed to derive row stride.
        bytes: u32,
    },
    /// A value the backend binds by name.
    Named {
        value: ValueId,
        width: u32,
        /// Bytes per element is needed to derive row stride.
        bytes: u32,
    },
    Weight(String),
    /// ONE OBJECT THE FIRE RAISED, by the word its `raise!` declared.
    ///
    /// Not [`Arg::Named`], though both are "the backend binds this". A named
    /// value is a TENSOR the backend holds — it has a row width and an element
    /// width, and both are on that variant because a launch measures its
    /// rectangle in them. A raise has neither: it is a host aggregate a body
    /// reads to fill the block a kernel takes, and the only thing the binder
    /// needs to resolve it is which raise it is.
    ///
    /// Carries the KEY and not just the value id, so the driver's arm matches
    /// on the same string `kernels-cuda`'s `raise!` wrote rather than
    /// re-deriving it from the op that produced the value.
    Raised { value: ValueId, key: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub kernel: u16,
    pub rows: Range<u32>,
    pub layers: Range<u16>,
    pub op: u32,
    /// Arg range order is inputs, outputs, then weights.
    pub args: Range<u32>,
    pub params: Range<u32>,
    pub peel: Option<PeelRegion>,
    /// Union mode preserves guard arms as condition regions.
    pub cond: u32,
}

impl Launch {
    pub const NO_COND: u32 = u32::MAX;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CondRegion {
    pub parent: u32,
    /// The predicate's wire slot, also the driver's predicate-word index.
    pub slot: u32,
    /// The predicate's parameter: `TokensLE(k)`'s `k`, or zero.
    pub param: u32,
    /// True for the body taken when the predicate holds; false for the else side.
    pub on_true: bool,
    /// The other arm of this conditional.
    pub sibling: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuardMode {
    #[default]
    Resolve,
    Union,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeelRegion {
    pub axis: PeelWindow,
    /// The tail region uses absolute offsets into the full window.
    pub tail: bool,
    /// The host row window is advisory when `rows_device` is set.
    pub rows_device: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Uncovered {
    Rows { at_op: usize, rows: Range<u32> },
    /// `whole` kernels may only cover the full fire.
    WholeKernelSplit {
        at_op: usize,
        kernel: String,
        rows: Range<u32>,
    },
    /// Non-contiguous partitions mean the row order and trace disagree.
    Discontiguous { at_op: usize, axis: &'static str },
    UnknownBackend(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unlowered {
    pub at_op: usize,
    pub kind: &'static str,
    pub why: &'static str,
}

#[derive(Debug, Clone)]
pub struct Lowered {
    pub launches: Vec<Launch>,
    pub kernels: Vec<String>,
    pub rectangles: usize,
    pub arena_bytes: usize,
    /// Where each traced value lives: arena offset or [`Buffers::NAMED`].
    pub value_offset: Vec<usize>,
    /// For each value, the value that owns the bytes it lives in.
    pub value_owner: Vec<ValueId>,
    pub epilogue_gather: usize,
    pub epilogue_norm: usize,
    pub args: Vec<Arg>,
    /// Each arg's OWN row count, parallel to [`Self::args`].
    ///
    /// A launch's rectangle is not always its operands' row space. The one
    /// that proves it is the epilogue gather: it reads the whole token stream
    /// and writes one row per SAMPLED row, so the launch covers `n_requests`
    /// rows while its input spans `n_tokens`. A backend that measures every
    /// operand by `launch.rows` binds that input one row long, and a shader
    /// bounds-clamps the rest to zero -- a plausible tensor, and the reason
    /// `driver-wgpu` and `driver-vulkan` both force `samples: true` on every
    /// row and pay a prefill's lm head over every token.
    ///
    /// Zero means "no opinion": a weight, a raise, or a value with no rows.
    /// A backend is free to ignore this and keep measuring by the launch.
    pub arg_rows: Vec<u32>,
    pub structural: Vec<Site>,
    /// Host preparations this fire needs raised before its launches run, in
    /// stated order.
    ///
    /// Separate from [`Self::launches`] because a prep enters no graph: it is
    /// CPU work over the batch's geometry, and a captured replay runs the
    /// launches again without it. A backend raises these where it already
    /// raised them, and reads WHICH from here instead of inferring it from
    /// [`Self::kernels`].
    pub preps: Vec<Prep>,
    /// Residue must be empty before launches are the whole execution.
    pub residue: Vec<Unlowered>,
    pub params: Vec<u32>,
    /// Rows the fire samples; this is the `Dim::Requests` count.
    pub n_requests: u32,
    pub conds: Vec<CondRegion>,
    pub readout: Option<Readout>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Readout {
    pub at: usize,
    pub rows: u32,
    pub vocab: u32,
    /// Metal readout bytes may be bf16, not f32.
    pub bytes: u32,
}

impl Launch {
    /// Whether this rectangle names `symbol` in its lowering's kernel table.
    pub fn kernel_is(&self, lowered: &Lowered, symbol: &str) -> bool {
        lowered
            .kernels
            .get(self.kernel as usize)
            .is_some_and(|k| k == symbol)
    }
}

impl Lowered {
    pub fn coverage(&self) -> f64 {
        let covered = self.launches.len();
        let total = covered + self.residue.len();
        if total == 0 {
            1.0
        } else {
            covered as f64 / total as f64
        }
    }
}
