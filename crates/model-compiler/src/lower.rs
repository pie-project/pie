//! LOWERING — the traced form to a flat launch list
//! (`.wiki/tart/dsl.md` "What one fire lowers to", migration step 6).
//!
//! ```text
//! per statement: compute extent (rows × layers)
//! match arms    → partition the extent into blocks
//! one Launch per rectangle
//! ```
//!
//! The target the doc states for the driver is a loop with no vocabulary
//! in it at all:
//!
//! ```cpp
//! for (const Launch& L : frame.launches)
//!     KERNELS[L.kernel](args + L.args, L.rows.lo, L.rows.hi,
//!                       L.layers.lo, L.layers.hi, stream);
//! ```
//!
//! **This is what a declared fire runs.** `declared_forward.cpp` builds
//! the rows, calls `pie_forward_lower`, and executes the result; its
//! walk over the region IR was deleted in the cutover's step 3, so there
//! is no second form and no switch between them. The one remaining
//! consumer of the traced form that does NOT come through here is the
//! generated `.inc` — an ahead-of-time emission of the same declaration
//! that also carries the unionized supergraph build.
//!
//! # Three decisions this module makes, from the doc's amendments
//!
//! **Row order is the ENGINE's.** `lower` takes the rows as the
//! scheduler's seriation already ordered them
//! (`crates/engine/src/scheduler/fire_plan.rs`) and does not choose a
//! permutation. Two independent permutation choosers would drift, and
//! the engine's is the one coupled to admission, framing and wave
//! discipline. What `lower` may do is REPORT what an order costs
//! ([`Lowered::rectangles`]), which is useful feedback for the seriation
//! key.
//!
//! **`Uncovered` is an ADMISSION answer, not a runtime fire split.** The
//! doc's sketch routed it to "the scheduler splits the fire", which
//! changes scheduling behaviour, and this project's standing constraint
//! is that runtime scheduling does not change — tart is a driver
//! feature. So [`Uncovered`] is what a group that cannot be served looks
//! like BEFORE it is formed: the engine's `LaunchGrouping::accepts`
//! already refuses unservable combinations, and this is the same answer
//! computed from the trace instead of from a hand-written rule.
//!
//! **`lower` assigns the buffers.** The DSL is pure SSA and carries no
//! buffer notion, so choosing one is a backend job — and it was the job
//! both CUDA executors did as FAMILY CONVENTION ("the normed activation
//! is `ws.norm_y`" in one, `ws.norm_x` in the other), which is what made
//! the executor two files. [`Buffers`] does it once, from the values'
//! own extents and liveness.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::kernels::{self, Backend};
use crate::trace::{DType, Dim, ForwardPlan, GuardPred, Op, OpKind, PeelWindow, ValueId};

/// One row of a fire, as the engine's seriation ordered them.
///
/// These are exactly the axes the seriation key sorts on
/// (`(devgeo, mask, truncated, Reverse(k), hook, !multi_token,
/// arrival)`), so a run of rows sharing any one of them is contiguous by
/// construction — the sentinel this project promoted from a diagnostic
/// to a guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Row {
    pub multi_token: bool,
    pub custom_mask: bool,
    pub hooked: bool,
    pub lora: bool,
    /// Truncated at layer `k`, or `None` for full depth.
    pub depth_k: Option<u32>,
    /// The fire steers a graph replay, so the KV write takes explicit
    /// descriptors. A fire-wide fact today; a row field here because
    /// that is what it will become.
    pub write_desc: bool,
    /// The fire's attached programs read attention scores.
    pub wants_scores: bool,
    /// This row's logits are read — it is one of the fire's SAMPLED
    /// rows. A pure-decode fire samples every row; a prefill fire
    /// samples the last row of each request and gathers them, which is
    /// what makes the epilogue's row space [`Dim::Requests`] rather than
    /// [`Dim::Tokens`], and what the driver spells `logit_row_indices`.
    pub samples: bool,
}

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
    /// their launches carry [`Launch::rows_device`]. Clear, the host's
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
    Arena { at: usize, width: u32 },
    /// A value the BACKEND binds by name — the values a seam exposes
    /// (the observed query, the logits). `Buffers::NAMED` says which.
    Named { value: ValueId, width: u32 },
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

/// Lower `plan` over `rows`, in the order the engine seriated them.
pub fn lower(plan: &ForwardPlan, rows: &[Row], fire: Fire) -> Result<Lowered, Uncovered> {
    let backend = Backend::of_family(&plan.family);
    let n = rows.len() as u32;
    let mut out = Lowerer {
        plan,
        rows,
        backend,
        launches: Vec::new(),
        kernels: Vec::new(),
        kernel_ids: BTreeMap::new(),
        peel_tail: false,
        residue: Vec::new(),
        fire,
        structural: Vec::new(),
        args: Vec::new(),
        buffers: Buffers::assign(plan, rows),
        peel_region: None,
    };
    let arena_bytes = out.buffers.bytes;
    let value_offset = out.buffers.offset.clone();
    let epilogue_gather = out.buffers.epilogue_gather;
    let epilogue_norm = out.buffers.epilogue_norm;
    let value_owner = alias_owners(plan);
    out.region(0..plan.ops.len(), 0..n)?;
    Ok(Lowered {
        rectangles: out.launches.len(),
        launches: out.launches,
        kernels: out.kernels,
        arena_bytes,
        value_offset,
        value_owner,
        epilogue_gather,
        epilogue_norm,
        args: out.args,
        structural: out.structural,
        residue: out.residue,
    })
}

struct Lowerer<'a> {
    plan: &'a ForwardPlan,
    rows: &'a [Row],
    backend: Option<Backend>,
    launches: Vec<Launch>,
    kernels: Vec<String>,
    kernel_ids: BTreeMap<String, u16>,
    /// Inside a peel's TAIL region, where the rows a statement serves sit
    /// at absolute offsets in a full-N buffer. Two llama_like statements
    /// take a different kernel there (`_devwin`), which is the region
    /// asking, not the driver choosing — so the lowering knows it.
    peel_tail: bool,
    residue: Vec<Unlowered>,
    fire: Fire,
    structural: Vec<Site>,
    /// The operand slots emitted so far.
    args: Vec<Arg>,
    /// The arena, so an operand can be resolved to an offset as it is
    /// emitted rather than in a second pass that would have to re-walk
    /// the regions to know which launches exist.
    buffers: Buffers,
    /// The peel region the launches being emitted sit in.
    peel_region: Option<PeelRegion>,
}

impl Lowerer<'_> {
    /// Lower the ops in `span` over `window`, the rows currently live.
    fn region(&mut self, span: Range<usize>, window: Range<u32>) -> Result<(), Uncovered> {
        let mut i = span.start;
        while i < span.end {
            let op = &self.plan.ops[i];
            match &op.kind {
                OpKind::Guard { arms, else_ops } => {
                    // A fire-level chain: the first arm whose predicate
                    // holds runs, over the SAME rows. In the flattened
                    // world these are row predicates, and an arm's rows
                    // are the subset of `window` satisfying it — which
                    // is what this computes. An arm selecting nobody
                    // emits nothing, which is the "vanishes" behaviour
                    // an argument-driven site already has.
                    let mut at = i + 1;
                    let mut remaining = window.clone();
                    for arm in arms {
                        let taken = self.select(&remaining, arm.pred);
                        let body = at..at + arm.ops as usize;
                        if !taken.is_empty() {
                            self.region(body, taken.clone())?;
                            remaining = subtract(&remaining, &taken, i)?;
                        }
                        at += arm.ops as usize;
                    }
                    let else_body = at..at + *else_ops as usize;
                    if !remaining.is_empty() {
                        self.region(else_body, remaining)?;
                    }
                    i = at + *else_ops as usize;
                }
                OpKind::Peel {
                    prefix_ops,
                    tail_ops,
                    window: axis,
                } => {
                    // BOTH regions run, over complementary row ranges.
                    let split = self.split_at(&window, *axis, i)?;
                    let prefix = window.start..split;
                    let tail = split..window.end;
                    let p = i + 1..i + 1 + *prefix_ops as usize;
                    let t = p.end..p.end + *tail_ops as usize;
                    // A captured fire replays across splits, so BOTH
                    // regions launch whatever this fire's split is and
                    // each early-outs on the device word. Skipping an
                    // empty one here would describe a graph that cannot
                    // serve the next fire.
                    //
                    // And each region's rows are then the WHOLE window,
                    // not its half: the launches are full-window grids
                    // and the split is a device word they read, which is
                    // the entire point of capturing across splits. A
                    // rectangle that named the half would be describing
                    // a grid nobody launches, and an executor that
                    // believed it would freeze THIS fire's split into
                    // the graph — a wrong answer no byte-parity run can
                    // see, because it is only wrong on the REPLAY.
                    let device = self.fire.captures_across_splits;
                    let outer_region = self.peel_region;
                    let axis = *axis;
                    let grid = window.clone();
                    let run = |me: &mut Self, span: Range<usize>, w: Range<u32>, tail_side: bool| {
                        if !device && w.is_empty() {
                            return Ok(());
                        }
                        let w = if device { grid.clone() } else { w };
                        me.peel_region = Some(PeelRegion {
                            axis,
                            tail: tail_side,
                            rows_device: device,
                        });
                        let outer = std::mem::replace(&mut me.peel_tail, tail_side);
                        let r = me.region(span, w);
                        me.peel_tail = outer;
                        r
                    };
                    let next = t.end;
                    run(self, p, prefix, false)?;
                    run(self, t, tail, true)?;
                    self.peel_region = outer_region;
                    i = next;
                }
                OpKind::Launch { kernel, .. } => {
                    let live = self.depth_window(op, &window, i)?;
                    self.emit(i, kernel, op, &live)?;
                    i += 1;
                }
                OpKind::LmHead { .. } => {
                    self.epilogue(i, op, &window)?;
                    i += 1;
                }
                // Everything else is a SEMANTIC statement, and it still
                // runs on the device. The flat list is the whole of what
                // the driver launches or it is not a replacement for the
                // walk, so each kind either names its kernels, declares
                // itself structural, or refuses — never falls through.
                kind => {
                    match semantic(kind, self.peel_tail) {
                        Semantic::Structural => {
                            // A site is layer-tagged like any other
                            // statement, so a RETIRED layer's bracket
                            // does not fire: the rows it would observe
                            // are gone. Same window the launches take,
                            // and skipping it here is what the walk
                            // does by refusing to enter the op at all.
                            let live = self.depth_window(op, &window, i)?;
                            if !live.is_empty() {
                                self.structural.push(Site {
                                    at_op: i as u32,
                                    rows: live,
                                });
                            }
                        }
                        Semantic::Kernels(symbols) => {
                            let live = self.depth_window(op, &window, i)?;
                            for symbol in symbols {
                                self.emit(i, symbol, op, &live)?;
                            }
                        }
                        Semantic::Unlowered(why) => self.residue.push(Unlowered {
                            at_op: i,
                            kind: kind_name(kind),
                            why,
                        }),
                    }
                    i += 1;
                }
            }
        }
        Ok(())
    }

    fn emit(
        &mut self,
        at: usize,
        kernel: &str,
        op: &Op,
        window: &Range<u32>,
    ) -> Result<(), Uncovered> {
        if window.is_empty() && !self.peel_region.is_some_and(|r| r.rows_device) {
            return Ok(());
        }
        let backend = self
            .backend
            .ok_or_else(|| Uncovered::UnknownBackend(self.plan.family.clone()))?;
        // ② `whole`, finally CONSUMED rather than declared: the kernel
        // refuses a row window, so it may only be emitted over the whole
        // fire. This is the same rule `kernels::check_plan` enforces
        // statically against Peel regions; here it also catches the
        // dynamic case, where an arm happens to select a subset.
        if let Some(sig) = kernels::sig_in(backend, kernel) {
            if sig.whole && (window.start != 0 || window.end != self.rows.len() as u32) {
                return Err(Uncovered::WholeKernelSplit {
                    at_op: at,
                    kernel: kernel.to_string(),
                    rows: window.clone(),
                });
            }
        }
        let id = match self.kernel_ids.get(kernel) {
            Some(&id) => id,
            None => {
                let id = self.kernels.len() as u16;
                self.kernels.push(kernel.to_string());
                self.kernel_ids.insert(kernel.to_string(), id);
                id
            }
        };
        // The trace is layer-unrolled, so a statement's layer extent is
        // one layer. `Launch::layers` is a range because a ROLLED trace
        // states a layer span; both spellings reach the same driver loop.
        let layer = op.layer.unwrap_or(0) as u16;
        // The operands, resolved HERE — inputs, then outputs, then the
        // weight names the statement carries. A driver reading this run
        // needs nothing else about the op, which is the whole point.
        let first = self.args.len() as u32;
        for &v in op.inputs.iter().chain(op.outputs.iter()) {
            self.args.push(self.slot(v));
        }
        if let OpKind::Launch { weights, .. } = &op.kind {
            for name in weights {
                self.args.push(Arg::Weight(name.clone()));
            }
        }
        self.launches.push(Launch {
            kernel: id,
            rows: window.clone(),
            layers: layer..layer + 1,
            op: at as u32,
            args: first..self.args.len() as u32,
            peel: self.peel_region,
        });
        Ok(())
    }

    /// Where a value's bytes are, and how wide a row of it is.
    fn slot(&self, v: ValueId) -> Arg {
        let width = self.row_width(v);
        match self.buffers.offset.get(v as usize) {
            Some(&Buffers::NAMED) | None => Arg::Named { value: v, width },
            Some(&at) => Arg::Arena { at, width },
        }
    }

    /// Elements per row: the product of every dim but the leading one.
    ///
    /// The leading dim is the row axis (`Tokens` for the body,
    /// `Requests` for the epilogue), which [`Launch::rows`] already
    /// names. A symbolic dim after the first would make the width a
    /// runtime number, and no statement in the tree has one — so this
    /// takes the constants and says zero if that ever stops being true,
    /// which a driver reads as "this operand has no fixed width".
    fn row_width(&self, v: ValueId) -> u32 {
        let Some(info) = self.plan.values.get(v as usize) else {
            return 0;
        };
        let mut w: u32 = 1;
        for dim in info.shape.0.iter().skip(1) {
            match dim {
                Dim::Const(k) => w = w.saturating_mul(*k),
                _ => return 0,
            }
        }
        w
    }

    /// THE EPILOGUE, as rectangles rather than as a branch.
    ///
    /// The executor reads two runtime inputs here and picks between
    /// three shapes: nothing at all when the fire samples no rows
    /// (`emit_logits` is `num_sampling > 0`, per fire), gather → norm →
    /// project when it samples fewer rows than it carries, and
    /// norm → project when every row is sampled. That reads like a
    /// two-level branch, and in the driver it is one.
    ///
    /// It is not one here, because all three shapes are the same three
    /// statements over a ROW COUNT:
    ///
    /// * the norm and the projection run over the sampled rows — the
    ///   epilogue's row space, which the trace already names
    ///   [`Dim::Requests`];
    /// * the gather EXISTS only when there are unsampled rows to skip
    ///   past, and an empty rectangle emits nothing, which is how "no
    ///   gather" is spelled without a branch;
    /// * a fire that samples nothing produces zero rectangles, which is
    ///   how `emit_logits == false` is spelled.
    ///
    /// So the branch survives only as long as the driver walks; when it
    /// consumes this list instead, it disappears — which is the same
    /// thing that happened to the swiglu's binding `if`, one layer up.
    ///
    /// The gather's SOURCE rows are deliberately not a rectangle. They
    /// are an index list (`logit_row_indices`) and therefore an operand;
    /// a prefill fire samples the last row of each request, which is not
    /// a contiguous run and was never going to be one.
    fn epilogue(&mut self, at: usize, op: &Op, window: &Range<u32>) -> Result<(), Uncovered> {
        let sampled = window
            .clone()
            .filter(|&i| self.rows[i as usize].samples)
            .count() as u32;
        if sampled == 0 {
            return Ok(());
        }
        let out = 0..sampled;
        if sampled < window.len() as u32 {
            self.emit(at, "layout::gather_bf16_rows", op, &out)?;
        }
        self.emit(at, "norm::rmsnorm_bf16", op, &out)?;
        self.emit(at, "gemm::act_x_w", op, &out)?;
        Ok(())
    }

    /// DEPTH, with no syntax (`.wiki/tart/dsl.md` ③): a statement
    /// tagged with layer `l` covers the rows still live at that depth,
    /// and nothing states it — membership is the layer tag.
    ///
    /// This is where the driver's BAND FORMATION goes away. Today the
    /// driver derives up to three bands from the region table and
    /// refuses a fourth (`derive_depth_bands`'s `if (count == 3) return
    /// 0`), because its walk carries per-band plans. Here a layer's live
    /// row count is just a number, so a fire with four distinct
    /// truncations lowers exactly like one with two — the ceiling is not
    /// raised, it has nowhere to live.
    ///
    /// The seriation orders truncated rows deepest-first after the
    /// full-depth ones, so the live rows at any layer are a PREFIX of
    /// the window. That is checked, not assumed: an order that breaks it
    /// is `Uncovered`, which is an admission answer.
    fn depth_window(
        &self,
        op: &Op,
        window: &Range<u32>,
        at: usize,
    ) -> Result<Range<u32>, Uncovered> {
        // A declaration that does not state the axis cannot window (the
        // XQA and padded-head deployments), and an untagged op is
        // prologue/epilogue.
        if !self.plan.depth_windowed(op) {
            return Ok(window.clone());
        }
        let layer = op.layer.unwrap_or(0);
        let alive = |r: &Row| r.depth_k.is_none_or(|k| layer < k);
        let mut end = window.start;
        for i in window.clone() {
            if alive(&self.rows[i as usize]) {
                if end != i {
                    return Err(Uncovered::Discontiguous {
                        at_op: at,
                        axis: "depth",
                    });
                }
                end = i + 1;
            }
        }
        Ok(window.start..end)
    }

    /// Whether `pred` holds for THE FIRE — so the arm covers `window`
    /// whole, or not at all.
    ///
    /// Every `GuardPred` is a fire fact. The vocabulary says so in each
    /// variant's own words ("the fire carries…") and the taxonomy says
    /// it in one line: a guard chain is *per-fire runtime input, kernel
    /// choice within one op list*, while the *per-fire ROW SPLIT* is the
    /// [`OpKind::Peel`]. Two constructs, two jobs.
    ///
    /// This function used to select the SUBSET of `window` whose rows
    /// carried the mark — inventing row semantics for a fire-level
    /// construct, and quietly giving the DSL two row-partitioning
    /// mechanisms where it had deliberately built one. The live shadow
    /// comparison is what surfaced it (`.wiki/tart/dsl.md`): the walk
    /// entered the `HasCustomMask` arm with the whole fire while the
    /// lowering handed the unmasked rows to the *else* arm — the
    /// fused/lora path, not the causal decode those rows want.
    ///
    /// So the fix is a DELETION. `Row` still carries the marks, because
    /// the peel reads them and because moving an axis from a guard to a
    /// row predicate is how it becomes per-row — a deliberate change,
    /// stated in the text, not a reinterpretation the backend performs
    /// on its own.
    fn select(&self, window: &Range<u32>, pred: GuardPred) -> Range<u32> {
        let holds = match pred {
            GuardPred::HasCustomMask => self.rows.iter().any(|r| r.custom_mask),
            GuardPred::HasLora => self.rows.iter().any(|r| r.lora),
            GuardPred::HasStageHooks => self.rows.iter().any(|r| r.hooked),
            GuardPred::WantsAttnScore => self.rows.iter().any(|r| r.wants_scores),
            GuardPred::HasWriteDesc => self.rows.iter().any(|r| r.write_desc),
            // The token thresholds read the fire's N, which is the same
            // question asked of a count instead of a flag.
            GuardPred::TokensLE(k) => self.rows.len() as u32 <= k,
            GuardPred::TokensGT(k) => self.rows.len() as u32 > k,
        };
        if holds {
            window.clone()
        } else {
            window.start..window.start
        }
    }

    /// Where a peel's axis splits `window` — the prefix is the rows that
    /// do NOT carry the axis's mark (hook-free, unmasked), which is the
    /// order the seriation produces.
    fn split_at(
        &self,
        window: &Range<u32>,
        axis: PeelWindow,
        at: usize,
    ) -> Result<u32, Uncovered> {
        let (name, marked): (&'static str, fn(&Row) -> bool) = match axis {
            PeelWindow::HookFreePrefix => ("hook", |r| r.hooked),
            PeelWindow::UnmaskedPrefix => ("mask", |r| r.custom_mask),
        };
        let tail = contiguous(self.rows, window, marked, name, at)?;
        // The marked rows are the SUFFIX; anything else means this order
        // and this trace disagree.
        if !tail.is_empty() && tail.end != window.end {
            return Err(Uncovered::Discontiguous { at_op: at, axis: name });
        }
        Ok(if tail.is_empty() { window.end } else { tail.start })
    }
}

// ── The semantic statements ────────────────────────────────────────────

/// What a statement that does NOT state its kernel lowers to.
enum Semantic {
    /// Launches nothing from the kernel table: a structural marker.
    Structural,
    /// The kernels it launches, in order. Usually one — the kinds whose
    /// own doc comments say "one op because it is one launch".
    Kernels(&'static [&'static str]),
    /// It runs on the device, but the trace does not say what it runs.
    /// The payload is what the trace would have to state.
    Unlowered(&'static str),
}

/// The kernels a semantic statement launches, read off the executor that
/// launches them today (`crates/driver-cuda/csrc/src/model/llama_like/
/// declared_forward.cpp`), not guessed.
///
/// Most of these arms are the ones the doc's Amendment A diagnosed: they
/// branch, but on which BUFFER (`ws.norm_x` vs `ws.y` vs the value slot),
/// never on which kernel. Strip the buffer question — [`Buffers`] owns it
/// — and what is left is 1:1, which is why the flat list can carry them.
///
/// Where an arm genuinely picks a kernel, the pick is either a REGION
/// question the lowering already knows (`peel_tail`) or a fact the trace
/// does not carry, and the second is [`Semantic::Unlowered`] rather than
/// a guess.
fn semantic(kind: &OpKind, peel_tail: bool) -> Semantic {
    use OpKind::*;
    match kind {
        // The sites are argument no-ops with nothing attached, and with
        // programs attached what they run is guest sideband plus the
        // bracket machinery (page-view reset, score staging) — never a
        // table kernel. Stating that bracket is what `seam!` is for.
        HookSite { .. } => Semantic::Structural,

        Embed { .. } => Semantic::Kernels(&["layout::embed_bf16"]),
        AddBias { .. } => Semantic::Kernels(&["norm::add_bias_bf16"]),
        ResidualAdd => Semantic::Kernels(&["norm::residual_add_bf16"]),

        // The GDN and full-attention kinds. Each is ONE kernel with no
        // branch — no fact to read, no variant to dispatch on, nothing
        // chosen per fire. They were residue only because the rule was
        // never written: the qwen3_5 executor walks, so nothing ever
        // asked the lowering what they were.
        //
        // Their operand plumbing (the per-layer `la.*` scratch, the fp32
        // parameter banks) is the EMITTER's, exactly as it is for the
        // kinds above — naming the symbol is what the lowering owes.
        GdnPrep { .. } => Semantic::Kernels(&["ssm::qwen_gdn_post_conv_prep_bf16"]),
        RmsnormGated { .. } => Semantic::Kernels(&["norm::rmsnorm_gated_fp32_in_bf16"]),
        SplitQGate { .. } => Semantic::Kernels(&["layout::split_q_gate_bf16"]),
        SigmoidGateMul => Semantic::Kernels(&["mlp::sigmoid_gate_inplace_bf16"]),

        // Gemma folds `(1 + w)` — different arithmetic, so a different
        // kernel, but the same signature and the same row space. The
        // variant is already on the wire (`param0`), so naming the
        // symbol is the whole of what the lowering owes; the executor
        // reads the same field to pick.
        //
        // The per-head kind differs only in its ROW COUNT (`N * heads`
        // rows of `head_dim` rather than `N` of `hidden`), which the
        // executor derives from the weight's geometry either way — so
        // both kinds fan onto the same pair.
        Rmsnorm { variant, .. } | RmsnormPerHead { variant, .. } => {
            Semantic::Kernels(if variant.is_plain() {
                &["norm::rmsnorm_bf16"]
            } else {
                &["norm::rmsnorm_gemma_bf16"]
            })
        }

        // Inside a peel's tail the split serves absolute row offsets in a
        // full-N buffer, which is a different kernel — and the REGION is
        // what asks for it, so the lowering states it rather than the
        // driver deriving it from a window pointer.
        SplitQkv { .. } => Semantic::Kernels(if peel_tail {
            &["attn::split_qkv_bf16_devwin"]
        } else {
            &["attn::split_qkv_bf16"]
        }),

        // Partial rope IS a different kernel, and the trace already says
        // which: the rotary width crosses as `param1`, zero for the full
        // rotation. So the lowering names the pair the same way it names
        // the norm's, and the width the executor needs is the width the
        // declaration already carried.
        Rope { kind, partial } => {
            if !matches!(kind, crate::trace::RopeKind::Standard) {
                Semantic::Unlowered("only standard rope is emitted")
            } else if partial.is_some() {
                Semantic::Kernels(&["rope::rope_partial_bf16"])
            } else {
                Semantic::Kernels(&["rope::rope_bf16"])
            }
        }

        // A selector makes the weight per-token, and grouped GEMM is that
        // op's lowering — a different call with a different argument
        // shape, chosen per fire.
        Matmul { selector, .. } => {
            if selector.is_none() {
                Semantic::Kernels(&["gemm::act_x_w"])
            } else {
                // A selector makes the weight per-token, and the grouped
                // GEMM is that op's lowering. It used to be a refusal
                // because no text stated the kernel; `moe_mlp_body_cuda`'s
                // general leg does now.
                Semantic::Kernels(&["moe::moe_grouped_gemm_bf16"])
            }
        }

        // Both of these THROW when a class trace reaches this executor
        // with them — a lowered trace states its KV write and its
        // attention as stated-kernel launches. So they cannot appear, and
        // if they do the honest answer is the same refusal.
        KvAppend { .. } => {
            Semantic::Unlowered("a lowered trace states the KV write as a launch")
        }
        Attention { .. } => {
            Semantic::Unlowered("a lowered trace states its attention kernel as a launch")
        }

        // The packed-bank form when the checkpoint materialised a fused
        // gate_up. That is a BINDING fact, which the taxonomy puts in the
        // facts and erases at trace time — but no fact carries it today,
        // so the executor reads the workspace and picks. The trace has to
        // state it before this statement can be a rectangle.
        Swiglu { .. } => {
            Semantic::Unlowered("the fused-gate_up binding fact is not in the facts")
        }

        // The MoE branch's three statements, each refused BY NAME until
        // `moe_mlp_body_cuda` states its kernel. They are grouped here
        // because they share one cause, and a residue ledger that says
        // "no lowering rule for this kind" three times would read as
        // three gaps instead of one missing text.
        // The router. One launch, and the semantic reading takes the
        // softmax form -- a text that wants the sigmoid or sqrt-softplus
        // router states it as a `Launch` instead.
        TopK { .. } => Semantic::Kernels(&["moe::topk_softmax_bf16"]),
        // The combine, in its TOKEN-BATCHED form. The two other forms --
        // the per-expert scatter-add and the fused +residual -- are what a
        // CUDA text states as launches when its binding takes them; this
        // is the reading a SEMANTIC trace gets, the same way `Swiglu`'s
        // unpacked form is.
        WeightedSum { .. } => Semantic::Kernels(&["moe::token_batched_weighted_sum_bf16"]),
        // The shared expert's landing: `sigmoid(x·g)` scaling the shared
        // output onto the routed sum, one launch.
        SigmoidGateAdd => Semantic::Kernels(&["mlp::sigmoid_dot_scalar_gate_add_bf16"]),

        // Handled by `Lowerer::epilogue`, which needs the row counts and
        // so cannot answer from the kind alone.
        LmHead { .. } => Semantic::Structural,

        // A window, not a launch: `Buffers` gives its value an offset
        // into its operand's, and there is no rectangle to emit. That is
        // the whole of what `Select` means.
        Select { .. } => Semantic::Structural,
        _ => Semantic::Unlowered("no lowering rule for this kind"),
    }
}

/// The kind's name, for a refusal a human reads.
fn kind_name(kind: &OpKind) -> &'static str {
    use OpKind::*;
    match kind {
        Embed { .. } => "Embed",
        Matmul { .. } => "Matmul",
        Select { .. } => "Select",
        Rmsnorm { .. } => "Rmsnorm",
        AddBias { .. } => "AddBias",
        RmsnormPerHead { .. } => "RmsnormPerHead",
        SplitQkv { .. } => "SplitQkv",
        Rope { .. } => "Rope",
        KvAppend { .. } => "KvAppend",
        Attention { .. } => "Attention",
        Swiglu { .. } => "Swiglu",
        LmHead { .. } => "LmHead",
        ResidualAdd => "ResidualAdd",
        TopK { .. } => "TopK",
        WeightedSum { .. } => "WeightedSum",
        SigmoidGateAdd => "SigmoidGateAdd",
        SplitGdn { .. } => "SplitGdn",
        CausalConv1d { .. } => "CausalConv1d",
        GdnPrep { .. } => "GdnPrep",
        GatedDelta { .. } => "GatedDelta",
        RmsnormGated { .. } => "RmsnormGated",
        SplitQGate { .. } => "SplitQGate",
        SigmoidGateMul => "SigmoidGateMul",
        Launch { .. } => "Launch",
        Guard { .. } => "Guard",
        HookSite { .. } => "HookSite",
        Peel { .. } => "Peel",
    }
}

/// The rows of `window` satisfying `holds`, refusing a non-contiguous
/// answer — the seriation's guarantee, checked rather than assumed.
fn contiguous(
    rows: &[Row],
    window: &Range<u32>,
    holds: fn(&Row) -> bool,
    axis: &'static str,
    at: usize,
) -> Result<Range<u32>, Uncovered> {
    let mut start = None;
    let mut end = window.start;
    for i in window.clone() {
        if holds(&rows[i as usize]) {
            if start.is_none() {
                start = Some(i);
            } else if end != i {
                return Err(Uncovered::Discontiguous { at_op: at, axis });
            }
            end = i + 1;
        }
    }
    Ok(match start {
        Some(s) => s..end,
        None => window.start..window.start,
    })
}

/// `window` minus `taken`, which must leave a contiguous remainder.
fn subtract(window: &Range<u32>, taken: &Range<u32>, at: usize) -> Result<Range<u32>, Uncovered> {
    if taken.start == window.start {
        Ok(taken.end..window.end)
    } else if taken.end == window.end {
        Ok(window.start..taken.start)
    } else {
        Err(Uncovered::Discontiguous {
            at_op: at,
            axis: "arm",
        })
    }
}

// ── Buffer assignment ──────────────────────────────────────────────────

/// Where each SSA value's bytes live.
///
/// A PINNED value is not allocated here at all: its bytes ARE the named
/// buffer's, and which buffer that is is the backend's binding. So
/// `offset[v] == NAMED` says "ask the backend", and such values are
/// excluded from [`Buffers::bytes`].
///
/// The DSL carries no buffer notion — `rmsnorm(x: &Val) -> Val` — so
/// choosing one is a backend job, and it is the job both CUDA executors
/// did as family convention. Doing it here, once, from the values' own
/// extents and liveness, is what lets an arm ask by value id and stay
/// family-blind.
///
/// A layer-unrolled plan names 28 distinct "normed activation" values
/// whose live ranges never overlap, so liveness reuse keeps the whole
/// frame inside a handful of buffers' worth of arena.
///
/// PINS are the exception: values that machinery OUTSIDE the traced ops
/// reaches by name — the query a hook observes, the normed activation an
/// adapter's host setup captures, the logits the sampler reads. The seam
/// signatures declare exactly that set (`sees`), so pins are derivable
/// rather than a per-family table. One empirical warning, paid for once:
/// a pin must be declared BY CONSUMER, not producer. A lowered trace may
/// state its attention as a stated-kernel `Launch` rather than a
/// semantic `Attention` op, so "the value o_proj reads lives in the
/// attention output buffer" is the sentence that holds under both
/// spellings.
#[derive(Debug, Clone)]
pub struct Buffers {
    /// Byte offset into the frame's activation arena, per value id, or
    /// [`Buffers::NAMED`] for a pinned value the backend binds by name.
    pub offset: Vec<usize>,
    /// Peak bytes.
    pub bytes: usize,
    /// Value ids a seam statement exposes, which therefore may not be
    /// recycled under a name outside machinery cannot follow.
    pub pinned: Vec<ValueId>,
    /// The epilogue's two intermediates, as byte offsets into the same
    /// arena — [`Buffers::NAMED`] when this fire needs neither.
    ///
    /// These are the ONLY buffers here that belong to no traced value,
    /// and the reason is that they belong to no traced STATEMENT either.
    /// One `LmHead` lowers to a row gather, a norm and a GEMM, and
    /// whether the gather runs at all is a fact about the FIRE's rows
    /// (`Row::samples`), not about the text — so the text cannot name
    /// what sits between them, and the lowering has to.
    ///
    /// Every CUDA executor reached for a workspace field here
    /// (`ws.norm_y`, `ws.norm_x`), each with its own apologetic comment,
    /// because the flat list handed all three rectangles the same
    /// operand run: `(activation, logits)`, which is true of the GEMM
    /// and of neither of the others.
    pub epilogue_gather: usize,
    pub epilogue_norm: usize,
}

impl Buffers {
    /// `offset[v]` for a value whose bytes are a named buffer's.
    pub const NAMED: usize = usize::MAX;

    pub fn assign(plan: &ForwardPlan, rows: &[Row]) -> Buffers {
        let n_tokens = rows.len();
        // `Dim::Requests` sizes the epilogue's values, so it must bound
        // the SAMPLED rows too: a multi-token fire whose extra rows are
        // sampled (MTP verify) has more logit rows than the
        // one-row-per-request count admits, and under-sizing the logits
        // is not a defect the arena would report.
        let n_requests = rows
            .iter()
            .filter(|r| !r.multi_token)
            .count()
            .max(rows.iter().filter(|r| r.samples).count())
            .max(1);

        // The values a seam exposes: read off the seam statements, not a
        // per-family table, and now off the statement's OWN value list
        // rather than the operands of whatever op it points at.
        //
        // The probe that did the guessing is kept as the fallback for a
        // record written before seams carried their values, and it was
        // wrong in both directions. It took the neighbouring op's
        // INPUTS, so `attn.qv` -- which names q and v -- pinned q, k and
        // v, costing reuse; and no exposed value that is an OUTPUT was
        // ever pinned at all. That second one is not a cost, it is a
        // wrong answer: the sampler reads the logit softcap's RESULT,
        // which the arena was placing while the driver read `ws.logits`.
        let mut pinned: Vec<ValueId> = Vec::new();
        for stmt in &plan.seams {
            if !stmt.values.is_empty() {
                pinned.extend(stmt.values.iter().copied());
                continue;
            }
            let Some(at) = stmt.op else { continue };
            for probe in [at as usize, at as usize + 1] {
                if let Some(op) = plan.ops.get(probe) {
                    if matches!(op.kind, OpKind::HookSite { .. } | OpKind::Launch { .. }) {
                        pinned.extend(op.inputs.iter().copied());
                        break;
                    }
                }
            }
        }
        // A seam names the value at the point it is STATED, and later
        // statements may write over those bytes in place -- the logit
        // softcap accumulates into the logits it was handed. Everything
        // sharing an exposed value's buffer is therefore exposed too, or
        // the arena places the final contents somewhere the reader is
        // not looking.
        {
            let owner = alias_owners(plan);
            let roots: std::collections::BTreeSet<ValueId> =
                pinned.iter().map(|&v| owner[v as usize]).collect();
            for v in 0..plan.values.len() {
                if roots.contains(&owner[v]) {
                    pinned.push(v as ValueId);
                }
            }
        }
        pinned.sort_unstable();
        pinned.dedup();

        // Values that SHARE bytes by construction, and the one of each
        // set that owns the allocation. Two ops mean this: a `Select`
        // output is a window of its operand, and an in-place launcher's
        // output is the operand it accumulates into.
        let owner = alias_owners(plan);

        // Last use, in one op pass — then folded onto the OWNER, which
        // is the correction that makes sharing safe.
        //
        // Read per value id, a shared buffer frees at the last use of
        // whichever member the op happened to name. That is not when the
        // bytes stop being read: a residual stream is a chain of in-place
        // adds, so the first link's id is dead after one op while the
        // bytes stay live for the whole network, and the freed block gets
        // handed to the next value that fits. The window case is the same
        // shape and was previously reasoned away in a comment here ("the
        // window's readers are the source's readers by dataflow") — they
        // are not, because a reader names the WINDOW's id, not the
        // source's.
        let mut last_use = vec![0usize; plan.values.len()];
        for (i, op) in plan.ops.iter().enumerate() {
            for &v in op.inputs.iter().chain(op.outputs.iter()) {
                let Some(&own) = owner.get(v as usize) else {
                    continue;
                };
                if let Some(slot) = last_use.get_mut(own as usize) {
                    *slot = (*slot).max(i);
                }
            }
        }
        for v in 0..plan.values.len() {
            last_use[v] = last_use[owner[v] as usize];
        }

        let mut offset = vec![Self::NAMED; plan.values.len()];
        let mut size = vec![0usize; plan.values.len()];
        let mut free: Vec<(usize, usize)> = Vec::new();
        let mut used = 0usize;
        let mut live: Vec<ValueId> = Vec::new();

        for (i, op) in plan.ops.iter().enumerate() {
            // Free what nobody reads any more. A pinned value never
            // returns to the pool: its bytes are reachable by name.
            live.retain(|&v| {
                if last_use[v as usize] >= i {
                    return true;
                }
                insert_free(&mut free, (offset[v as usize], size[v as usize]));
                false
            });
            // A `Select` allocates nothing: its value IS a window of its
            // operand's bytes, which is the whole of what the op means.
            // It joins the operand's alias set rather than entering
            // `live`, so those bytes return to the pool ONCE, at the last
            // use of the set — see the `last_use` fold above.
            if let OpKind::Select { index } = op.kind {
                let src = op.inputs[0];
                let out = op.outputs[0];
                let want = value_bytes(plan, out, n_tokens, n_requests);
                if offset[src as usize] == Self::NAMED {
                    // A window of a NAMED buffer is still the backend's
                    // to bind; the arena has no address to offset from.
                    offset[out as usize] = Self::NAMED;
                } else {
                    offset[out as usize] =
                        offset[src as usize] + index as usize * want;
                }
                size[out as usize] = want;
                continue;
            }
            // An IN-PLACE op writes over an operand, so its output is
            // that operand's bytes. Giving it an allocation of its own
            // would be a copy
            // the model does not make, and for a text that accumulates
            // into a `select` window it would be worse than wasteful:
            // the window would keep its pre-update value and the streams
            // would silently never see the add.
            //
            // Read from the SAME two tables `alias_owners` reads —
            // the `kernel!` row for a stated symbol, the kind itself for
            // a semantic one. They were not the same for a while: the
            // owner table joined a semantic rope's operand and result
            // while this loop, which only knew about `Launch`, handed
            // the result a block of its own. Liveness then freed one
            // buffer for what placement had made two, and the rotated k
            // was written to an address nothing read.
            {
                let pairs = match &op.kind {
                    OpKind::Launch { kernel, .. } => {
                        crate::kernels::in_place_pairs(plan, kernel)
                    }
                    other => crate::kernels::semantic_in_place(other),
                };
                let mut aliased = false;
                for &(o, i) in pairs {
                    // A pair outside this statement's arity is not an
                    // error: one symbol serves a q-only site and a q/k
                    // pair, and the row states the widest form.
                    if let (Some(&src), Some(&out)) =
                        (op.inputs.get(i as usize), op.outputs.get(o as usize))
                    {
                        offset[out as usize] = offset[src as usize];
                        size[out as usize] =
                            value_bytes(plan, out, n_tokens, n_requests);
                        aliased = true;
                    }
                }
                if aliased {
                    // Outputs this kernel does NOT write in place still
                    // need buffers of their own.
                    for (o, &v) in op.outputs.iter().enumerate() {
                        if pairs.iter().any(|&(oi, _)| oi as usize == o) {
                            continue;
                        }
                        if pinned.binary_search(&v).is_ok() {
                            offset[v as usize] = Self::NAMED;
                            continue;
                        }
                        let want = value_bytes(plan, v, n_tokens, n_requests);
                        let at = take_block(&mut free, &mut used, want);
                        offset[v as usize] = at;
                        size[v as usize] = want;
                        live.push(v);
                    }
                    continue;
                }
            }
            for &v in &op.outputs {
                if pinned.binary_search(&v).is_ok() {
                    // Reachable by name from outside the trace — the
                    // query a hook observes, the logits the sampler
                    // reads. The backend binds it; the arena does not.
                    offset[v as usize] = Self::NAMED;
                    continue;
                }
                let want = value_bytes(plan, v, n_tokens, n_requests);
                let at = take_block(&mut free, &mut used, want);
                offset[v as usize] = at;
                size[v as usize] = want;
                live.push(v);
            }
        }
        // The epilogue's scratch, sized from the statement it serves.
        // Allocated LAST and never freed: it is live across the three
        // rectangles that make up one statement, and nothing else in
        // the fire runs between them.
        let mut epilogue_gather = Self::NAMED;
        let mut epilogue_norm = Self::NAMED;
        for op in &plan.ops {
            if !matches!(op.kind, OpKind::LmHead { .. }) {
                continue;
            }
            let Some(&input) = op.inputs.first() else { continue };
            let width = value_bytes(plan, input, 1, 1);
            let sampled = rows.iter().filter(|r| r.samples).count().max(1);
            let want = width * sampled;
            if want == 0 {
                continue;
            }
            epilogue_gather = take_block(&mut free, &mut used, want);
            epilogue_norm = take_block(&mut free, &mut used, want);
            break;
        }

        Buffers {
            offset,
            bytes: used,
            pinned,
            epilogue_gather,
            epilogue_norm,
        }
    }
}

/// Take `want` bytes from the pool, or bump.
///
/// BEST fit, and SPLIT the remainder back. First-fit-and-keep-the-whole-
/// block was costing 4-15x at the fire shape that sizes the driver's
/// activation block (`arena_soundness.rs` prices it per family): a freed
/// logits-sized block satisfying a one-row norm retired the rest of
/// itself, so the walk bump-allocated almost everything. It read as
/// cheap because the ratio had been measured on an eight-row
/// all-sampled fire, where the logits dominate the arena AND the floor
/// and the loss hides inside both.
fn take_block(free: &mut Vec<(usize, usize)>, used: &mut usize, want: usize) -> usize {
    match free
        .iter()
        .enumerate()
        .filter(|(_, block)| block.1 >= want)
        .min_by_key(|(_, block)| block.1)
        .map(|(i, _)| i)
    {
        Some(f) => {
            let (off, size_of) = free.remove(f);
            // The tail keeps the block's alignment, so a split never
            // hands out an address the bump path would not have.
            let tail = (off + want).div_ceil(256) * 256;
            if tail < off + size_of {
                insert_free(free, (tail, off + size_of - tail));
            }
            off
        }
        None => {
            // 256-byte alignment, and BUMP only: a decode body runs
            // inside a capture, so the same plan must land the same
            // value at the same address on every fire.
            let at = used.div_ceil(256) * 256;
            *used = at + want;
            at
        }
    }
}

/// Return a block to the pool, MERGED with any neighbour it touches.
///
/// The pool is kept sorted by offset so this is one scan. Without it,
/// splitting makes fragmentation worse rather than better: a block cut
/// into pieces to serve small values never becomes whole again, so a
/// later large value bump-allocates past a run of adjacent free bytes
/// that would have held it.
fn insert_free(free: &mut Vec<(usize, usize)>, block: (usize, usize)) {
    let (at, len) = block;
    if len == 0 {
        return;
    }
    let i = free.partition_point(|&(off, _)| off < at);
    free.insert(i, (at, len));
    // Merge forward first, then back, so a block filling a hole between
    // two free neighbours coalesces all three.
    if i + 1 < free.len() && free[i].0 + free[i].1 == free[i + 1].0 {
        let (_, next_len) = free.remove(i + 1);
        free[i].1 += next_len;
    }
    if i > 0 && free[i - 1].0 + free[i - 1].1 == free[i].0 {
        let (_, this_len) = free.remove(i);
        free[i - 1].1 += this_len;
    }
}

/// For each value, the value that OWNS the bytes it lives in.
///
/// Most values own their own. The exceptions are the three constructs
/// whose meaning is that the output does not get memory of its own: a
/// [`OpKind::Select`] output is a window of its operand; a launcher the
/// `kernel!` table marks in-place writes over the operand it
/// accumulates into; and a semantic kind that rewrites its operand says
/// so through [`crate::kernels::semantic_in_place`]. All chain — a
/// residual stream is a run of in-place adds — so this is a union-find,
/// and the owner is always the EARLIER value, i.e. the one whose
/// allocation the rest inherit.
///
/// Buffer assignment needs this in two places: the live range of a
/// shared buffer is the union's, not any one member's, and only the
/// owner may return those bytes to the free pool.
pub(crate) fn alias_owners(plan: &ForwardPlan) -> Vec<ValueId> {
    let mut owner: Vec<ValueId> = (0..plan.values.len() as ValueId).collect();

    fn find(owner: &mut [ValueId], v: ValueId) -> ValueId {
        let mut v = v;
        while owner[v as usize] != v {
            let up = owner[v as usize];
            owner[v as usize] = owner[up as usize];
            v = owner[v as usize];
        }
        v
    }

    for op in &plan.ops {
        let joined: Vec<(ValueId, ValueId)> = match &op.kind {
            OpKind::Select { .. } => match (op.inputs.first(), op.outputs.first()) {
                (Some(&src), Some(&out)) => vec![(src, out)],
                _ => Vec::new(),
            },
            OpKind::Launch { kernel, .. } => crate::kernels::in_place_pairs(plan, kernel)
                .iter()
                .filter_map(|&(o, i)| {
                    Some((*op.inputs.get(i as usize)?, *op.outputs.get(o as usize)?))
                })
                .collect(),
            // The kinds that name no kernel but still write over their
            // operand — see `kernels::semantic_in_place`. Read the same
            // way as the table's, because it is the same fact.
            other => crate::kernels::semantic_in_place(other)
                .iter()
                .filter_map(|&(o, i)| {
                    Some((*op.inputs.get(i as usize)?, *op.outputs.get(o as usize)?))
                })
                .collect(),
        };
        for (src, out) in joined {
            if src as usize >= owner.len() || out as usize >= owner.len() {
                continue;
            }
            let (a, b) = (find(&mut owner, src), find(&mut owner, out));
            if a != b {
                // The earlier value keeps the allocation; SSA numbering
                // makes "earlier" and "smaller id" the same thing, and
                // the ops are walked in order anyway.
                let (keep, drop) = if a <= b { (a, b) } else { (b, a) };
                owner[drop as usize] = keep;
            }
        }
    }
    for v in 0..owner.len() {
        owner[v] = find(&mut owner, v as ValueId);
    }
    owner
}

pub fn value_bytes(plan: &ForwardPlan, v: ValueId, n_tokens: usize, n_requests: usize) -> usize {
    let Some(info) = plan.values.get(v as usize) else {
        return 0;
    };
    let mut elements = 1usize;
    for dim in &info.shape.0 {
        elements *= match dim {
            Dim::Tokens => n_tokens,
            Dim::Requests => n_requests,
            Dim::Const(c) => *c as usize,
            // The padded route count, which is a function of the fire's
            // tokens and three load-time numbers -- so a residue ledger
            // sizing this value gets the real footprint, not an estimate.
            Dim::MoeAlignedRoutes {
                top_k,
                experts,
                block,
            } => Dim::moe_aligned_rows(n_tokens as u32, *top_k, *experts, *block) as usize,
        };
    }
    elements
        * match info.dtype {
            DType::BF16 | DType::F16 => 2,
            DType::F32 | DType::I32 => 4,
        }
}
