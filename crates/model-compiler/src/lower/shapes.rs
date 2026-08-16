//! THE LOWERED FORM — the types a driver reads: the fire it is lowering
//! for, the launch list, the operands, and what a lowering could not cover.

use super::*;

/// How the fire will be EXECUTED, where that changes what runs.
///
/// Not row facts and not a guard: one thing the driver decides about the
/// fire as a whole, which the lowering has to know because it changes
/// the launch list rather than the launch arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fire {
    /// The fire is captured once and replayed across DIFFERENT row
    /// splits, so a peel's regions cannot bake their row counts: both
    /// regions launch and an empty one early-outs on a device word.
    ///
    /// Set, a peel emits BOTH regions even when one is empty here, and
    /// their launches carry [`PeelRegion::rows_device`]. Clear, the host's
    /// counts are the truth and an empty region emits nothing.
    pub captures_across_splits: bool,
}

/// A STRUCTURAL statement and the rows it brackets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    pub at_op: u32,
    /// The rows live where it sits — the SAME window its neighbouring
    /// launches take.
    ///
    /// A site observes rows, so it needs a row count for the same
    /// reason a launch needs a grid: an observation program is handed
    /// `rows` rows of the query buffer, and past the live count those
    /// rows are frozen at whatever the last layer that owned them left
    /// behind. Carrying only the statement index (what this list did
    /// when sites first joined it) makes every site a fire-wide one,
    /// which is right for exactly the fires that are not truncated.
    pub rows: Range<u32>,
}

/// One operand a launch binds.
///
/// This is what makes the flat list FAMILY-INDEPENDENT. An executor that
/// walks the traced ops has to answer, per op and per family, "which
/// workspace field is this operand?" — which is why today's four
/// `declared_forward.cpp` hard-code `ws.norm_x`, `ws.q`, `la.mixed_qkv`
/// and cannot be shared. A launch that carries its operands answers it
/// once, in the lowering, for every family at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arg {
    /// An activation: a byte offset into the frame's arena, and the
    /// operand's WIDTH — elements per row.
    ///
    /// The width is here because an arm needs it and the alternative is
    /// worse. Today's per-family executors track it as `cur_d`, `cur_hk`
    /// and friends — per-layer bookkeeping the walk maintains — and a
    /// driver that had to re-derive it would be reading the plan again,
    /// which is exactly what the flat list exists to avoid. The rows come
    /// from [`Launch::rows`]; together they are the rectangle the kernel
    /// addresses.
    Arena {
        at: usize,
        width: u32,
        /// Bytes per ELEMENT of this operand.
        ///
        /// Stated because a driver that windows a rectangle needs the row
        /// STRIDE, and `width` alone is elements. Every hand windowing in
        /// the CUDA executor used to multiply by two — true of activations
        /// and an assumption rather than a fact the trace carried, which
        /// is the shape of defect §4 of the retirement plan keeps finding.
        /// The lowering knows the dtype; now it says it.
        bytes: u32,
    },
    /// A value the BACKEND binds by name — the values a seam exposes
    /// (the observed query, the logits). `Buffers::NAMED` says which.
    Named {
        value: ValueId,
        width: u32,
        /// Bytes per ELEMENT, for the same reason [`Arg::Arena::bytes`]
        /// states it — and this variant needed it more.
        ///
        /// `slot` has always computed this and threw it away here, so a
        /// backend that wanted the rectangle in BYTES had to guess the
        /// width. Two is the guess that looks safe, and it is wrong in
        /// the one place a driver has actually measured: `driver-vulkan`
        /// records a four-row `row_gather` over a one-entry `u32`
        /// sampling table as *"a sixteen-byte read of a four-byte
        /// buffer"*, and notes it can see it nowhere — `Arg::Named` has
        /// no extent to check, the descriptor is bound whole, and the
        /// validation layer does not report storage-buffer overruns
        /// even with GPU-AV on. A two-byte guess would have called that
        /// rectangle eight bytes and still not caught it.
        bytes: u32,
    },
    /// A weight, by the name the trace states (`layer.3.q_proj`). The
    /// driver resolves it against its own tensor store, which is the one
    /// thing that stays per-family and is a MAP rather than a switch.
    Weight(String),
}

/// One flat launch: a kernel over a rectangle of (rows × layers).
///
/// `args` is an index into the frame's argument slots — the driver binds
/// operands from there, which is why no buffer appears in this struct.
///
/// `rows` is read in the OP'S OWN row space, which its output shape
/// names. That is [`Dim::Tokens`] for the body — where it is the fire's
/// rows and every window is a run of them — and [`Dim::Requests`] for
/// the epilogue, whose statements run over the SAMPLED rows after the
/// gather has collected them. A gather is not a window: its source rows
/// are an index list, which is an operand (hence `args`), while the
/// rectangle it fills is contiguous like every other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub kernel: u16,
    pub rows: Range<u32>,
    pub layers: Range<u16>,
    /// Which traced op produced this rectangle. Kept beside the
    /// operands because it answers a different question: `args` is what
    /// a driver BINDS, `op` is where a refusal or a shadow comparison
    /// points. They shared a field until the operands existed, which
    /// read as one number meaning two things.
    pub op: u32,
    /// This launch's operands, as a run of [`Lowered::args`]. Inputs in
    /// operand order, then outputs, then the weights the statement
    /// names — the order the trace states them, so nothing here is a
    /// convention a reader has to learn twice.
    pub args: Range<u32>,
    /// The scalar arguments the statement states, as a run of
    /// [`Lowered::params`].
    ///
    /// A kernel takes numbers no operand shape gives — a QKV split's two
    /// widths, a strided kernel's row pitch. `OpKind::Launch::params` is the
    /// channel the trace carries them on, and its own doc says why it exists:
    /// *"a scalar that has nowhere to ride is a scalar the DRIVER re-derives
    /// from its config. That is the thing this arc removes."* Dropping them
    /// here would have put the derivation straight back.
    ///
    /// Empty for every statement that states none, which is most of them.
    pub params: Range<u32>,
    /// Which peel region this rectangle sits in, when it sits in one.
    ///
    /// The executing arms read exactly four things about where they
    /// are: the row count, the layer, which side of a row split they
    /// serve, and which prepared plan to use. The first two are `rows`
    /// and `layers`; the third is this; and the fourth stops being a
    /// question — a prepared plan is found by the rectangle's ROW
    /// COUNT, which is why the driver's band index, and its three-band
    /// ceiling, has nothing left to index.
    pub peel: Option<PeelRegion>,
    /// Which CONDITIONAL region this rectangle sits in, as an index into
    /// [`Lowered::conds`], or [`Launch::NO_COND`] for the root.
    ///
    /// Always `NO_COND` under [`GuardMode::Resolve`], which is the mode
    /// that answers every guard at lowering time. Under
    /// [`GuardMode::Union`] the guards are NOT answered — every arm
    /// lowers, and this says which arm's body a rectangle belongs to, so
    /// a driver can turn the tree back into conditional graph nodes.
    pub cond: u32,
}

impl Launch {
    /// [`Launch::cond`] for a rectangle that sits under no conditional.
    pub const NO_COND: u32 = u32::MAX;
}

/// One arm of a guard chain, as a node in the tree
/// [`GuardMode::Union`] preserves.
///
/// A guard chain of N arms plus an else is N NESTED conditionals — the
/// second arm runs when the first predicate did not hold and the second
/// did — so the tree is binary and every node carries exactly one
/// predicate. That is also the shape a CUDA conditional node has, which
/// is not a coincidence: the C++ emitter nests chains into else bodies
/// for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CondRegion {
    /// The enclosing region, or [`Launch::NO_COND`] at the root.
    pub parent: u32,
    /// The predicate's wire slot — `GuardPred::wire().0`, which is also
    /// the driver's device predicate-word index.
    pub slot: u32,
    /// The predicate's parameter — `TokensLE(k)`'s `k`; zero for the
    /// predicates that take none.
    pub param: u32,
    /// True for the body taken when the predicate HOLDS; false for the
    /// else side, under which the rest of the chain nests.
    pub on_true: bool,
    /// The OTHER arm of this conditional.
    ///
    /// Stated rather than derived, and the difference is not cosmetic. A
    /// family states the same guard once per layer, so `(parent, slot,
    /// param)` identifies twenty-eight conditionals in a
    /// twenty-eight-layer model, not one — a driver pairing arms by those
    /// three fields matches the wrong node and captures an arm into some
    /// other layer's else body. The pairing is a fact the lowering knows
    /// when it emits the pair, so it says it.
    pub sibling: u32,
}

/// Whether the lowering ANSWERS a guard or keeps it.
///
/// This is the axis the supergraph turns on, and it is worth being
/// precise about what each mode produces:
///
/// * [`GuardMode::Resolve`] reads the fire's rows, decides each
///   predicate, and emits only the arm that wins. The flat list is then
///   already specialised to one fire's variant bits — which is what
///   makes the eager executor simple, and what makes a union impossible.
/// * [`GuardMode::Union`] decides nothing. Every arm lowers, tagged with
///   its place in the tree, and the arena is sized for all of them. One
///   lowering then covers every structurally-distinct program in the
///   bucket, and the predicates move to device memory where a conditional
///   node reads them per launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuardMode {
    /// Answer every guard at lowering time. The historical behaviour.
    #[default]
    Resolve,
    /// Keep every arm and tag it — the unionized supergraph's input.
    Union,
}

/// A rectangle's place inside a row partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeelRegion {
    pub axis: PeelWindow,
    /// The SUFFIX region (hook-visible rows, masked rows) rather than
    /// the prefix — what the executor calls its mask region, and what
    /// decides whether a statement addresses rows at absolute offsets.
    pub tail: bool,
    /// `rows` is the host's BELIEF and the executing form must read the
    /// fire's runtime split instead.
    ///
    /// This is the one place a rectangle is not a pair of numbers, and
    /// deliberately the only one: a captured fire replays across splits,
    /// so both regions launch and each early-outs on a device word.
    /// Everything else stays plain counts — "mostly numbers, two of them
    /// runtime" is a list you can still read, which "any of these might
    /// be runtime" would not be.
    pub rows_device: bool,
}

/// Why a fire cannot be lowered against this trace.
///
/// Not an error to recover from at fire time — an ADMISSION answer. See
/// the module doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Uncovered {
    /// Some rows match no arm of a partition, so nothing would run over
    /// them.
    Rows { at_op: usize, rows: Range<u32> },
    /// A `whole` kernel was asked to cover a strict subset of the fire's
    /// rows. Its addressing (a fire-wide prepare, a padded staging
    /// buffer) cannot honour a row window.
    WholeKernelSplit {
        at_op: usize,
        kernel: String,
        rows: Range<u32>,
    },
    /// A partition's arms do not select a CONTIGUOUS run of rows. The
    /// engine's seriation guarantees contiguity per axis; a violation
    /// means this row order and this trace disagree, and the honest
    /// answer is that the group should not have been formed.
    Discontiguous { at_op: usize, axis: &'static str },
    /// The trace states kernels whose backend the family name does not
    /// name.
    UnknownBackend(String),
}

/// A statement the flat list does not carry yet: it runs on the device,
/// but which kernel it runs is not derivable from the trace.
///
/// This is the honest name for what used to be a silent `_ => i += 1`. A
/// launch list that omits an executed statement is worse than one that
/// refuses, because the omission reads as coverage — so every kind is
/// either a rectangle, structural, or listed here.
///
/// It is NOT an [`Uncovered`]: that answers "this group cannot be
/// served" and goes to admission, while this answers "this trace is not
/// finished migrating" and goes to whoever is finishing it. The cutover
/// gate is [`Lowered::residue`] being empty, and until it is, the list
/// says exactly which statements still owe a declaration and what they
/// would have to say. `why` is that sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unlowered {
    pub at_op: usize,
    pub kind: &'static str,
    pub why: &'static str,
}

/// What a lowering produced.
#[derive(Debug, Clone)]
pub struct Lowered {
    pub launches: Vec<Launch>,
    /// Distinct kernel symbols, in first-launch order — the driver's
    /// `KERNELS` table for this frame, and what `Launch::kernel` indexes.
    pub kernels: Vec<String>,
    /// What this ROW ORDER cost, in rectangles. Feedback for the
    /// seriation key; `lower` reports it and does not act on it.
    pub rectangles: usize,
    /// Peak activation bytes the frame needs ([`Buffers`]).
    pub arena_bytes: usize,
    /// Where each traced value lives, by value id: a byte offset into
    /// the frame's arena, or [`Buffers::NAMED`] for one the backend
    /// binds.
    ///
    /// [`Launch::args`] already carries this for every operand a
    /// rectangle names, and a driver walking rectangles wants nothing
    /// else. This is here for the walk that still exists: the per-family
    /// executors step ops and ask for a value BY ID, so without a table
    /// they could not move onto host-assigned buffers until they had
    /// been rewritten to walk rectangles — two migrations chained where
    /// one will do.
    pub value_offset: Vec<usize>,
    /// For each value, the value that OWNS the bytes it lives in.
    ///
    /// Most values own their own; the exceptions are the constructs
    /// whose meaning is that the output does not get memory of its own,
    /// and they CHAIN — a residual stream is a run of in-place adds, all
    /// one owner. A driver reading `value_offset` alone cannot tell two
    /// chains that reuse a slot at different times apart from one chain;
    /// this says which values must move together, and is what makes a
    /// per-chain question askable at all.
    pub value_owner: Vec<ValueId>,
    /// The epilogue's two intermediates — see [`Buffers::epilogue_gather`].
    /// `usize::MAX` when this fire needs neither.
    pub epilogue_gather: usize,
    pub epilogue_norm: usize,
    /// Every launch's operands, concatenated; [`Launch::args`] indexes
    /// it. Flat rather than per-launch so the whole frame is two arrays
    /// and a table — which is the shape a driver can walk without
    /// knowing whose model it is.
    pub args: Vec<Arg>,
    /// The STRUCTURAL statements inside live regions, in walk order.
    ///
    /// A site launches no table kernel, so it has no rectangle — but it
    /// runs guest programs and brackets a layer's sideband, so a form
    /// driven by this list has to run it, and only when the region
    /// holding it is live. A site inside an arm the guards did not take
    /// must not fire, and `launches` alone cannot say which those are.
    ///
    /// So the list is what a fire DOES: rectangles for what it launches,
    /// these for what it brackets.
    pub structural: Vec<Site>,
    /// Statements that still run on the device without a rectangle —
    /// see [`Unlowered`]. Empty is the cutover gate: only then is
    /// `launches` the WHOLE of what a fire executes, and only then can
    /// the driver stop walking.
    pub residue: Vec<Unlowered>,
    /// Every launch's scalar arguments, concatenated; [`Launch::params`]
    /// indexes it. Flat for the same reason [`Lowered::args`] is: the whole
    /// frame stays two arrays and a table.
    pub params: Vec<u32>,
    /// Rows the fire SAMPLES — one distribution per request, and the number
    /// `Dim::Requests` already sizes every epilogue value by.
    ///
    /// Published because a driver needs it and cannot recompute it: it is
    /// `rows` filtered two ways and maxed, not `rows.len()`, and a text cannot
    /// state it at all. `Source::RequestCount` is how a row asks.
    pub n_requests: u32,
    /// The guard tree, when the lowering kept it ([`GuardMode::Union`]).
    ///
    /// Empty under [`GuardMode::Resolve`], where every guard was
    /// answered and no rectangle sits under a condition.
    pub conds: Vec<CondRegion>,
    /// Where this fire's read-out lands, when the text states an exit seam.
    ///
    /// Derived, not guessed: the `out` seam names the logits value, the plan
    /// gives its shape and dtype, and buffer assignment gives its offset. A
    /// driver that wants the distribution reads exactly this and needs to know
    /// nothing about whose model it is.
    ///
    /// `None` for a text that states no exit — which is a fire that computes
    /// something other than a distribution, not an error.
    pub readout: Option<Readout>,
}

/// Where a fire's logits are, in the frame's own arena.
///
/// The four numbers a read-out needs and no more: a byte offset, the two
/// extents, and the width of one element. Not a slice, because the arena is
/// device memory and the lowering never held it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Readout {
    /// Byte offset into the frame arena.
    pub at: usize,
    /// Distributions produced — one per request, so this is
    /// [`Lowered::n_requests`] whenever the exit value is `Requests`-shaped.
    pub rows: u32,
    /// The vocabulary: elements per row, and the row stride.
    pub vocab: u32,
    /// Bytes per element. **Four is not a given** — the metal read-out is
    /// bf16, because `affine_qmv_fast` writes bf16 and the text's declared
    /// dtype does not change what the kernel does. A reader that assumed f32
    /// got a vocabulary exactly half zeros, which looks like a dead half of a
    /// tensor and is really two elements read as one.
    pub bytes: u32,
}

impl Launch {
    /// Whether this rectangle names `symbol`, against the lowering it
    /// came from — the kernel table is per-`Lowered`, not global.
    pub fn kernel_is(&self, lowered: &Lowered, symbol: &str) -> bool {
        lowered
            .kernels
            .get(self.kernel as usize)
            .is_some_and(|k| k == symbol)
    }
}

impl Lowered {
    /// The fraction of executed statements the flat list carries. What
    /// the cutover is measured against; `1.0` is the gate.
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
