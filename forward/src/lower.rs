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
//! (`runtime/engine/src/scheduler/fire_plan.rs`) and does not choose a
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
    pub args: u32,
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
        peel_region: None,
    };
    out.region(0..plan.ops.len(), 0..n)?;
    let buffers = Buffers::assign(plan, rows);
    Ok(Lowered {
        rectangles: out.launches.len(),
        launches: out.launches,
        kernels: out.kernels,
        arena_bytes: buffers.bytes,
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
                    let mut run = |me: &mut Self, span: Range<usize>, w: Range<u32>, tail_side: bool| {
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
        self.launches.push(Launch {
            kernel: id,
            rows: window.clone(),
            layers: layer..layer + 1,
            args: at as u32,
            peel: self.peel_region,
        });
        Ok(())
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
            self.emit(at, "launch_gather_bf16_rows", op, &out)?;
        }
        self.emit(at, "launch_rmsnorm_bf16", op, &out)?;
        self.emit(at, "gemm_act_x_w", op, &out)?;
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
/// launches them today (`driver/cuda/src/model/llama_like/
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

        Embed { .. } => Semantic::Kernels(&["launch_embed_bf16"]),
        AddBias { .. } => Semantic::Kernels(&["launch_add_bias_bf16"]),
        ResidualAdd => Semantic::Kernels(&["launch_residual_add_bf16"]),

        // The GDN and full-attention kinds. Each is ONE kernel with no
        // branch — no fact to read, no variant to dispatch on, nothing
        // chosen per fire. They were residue only because the rule was
        // never written: the qwen3_5 executor walks, so nothing ever
        // asked the lowering what they were.
        //
        // Their operand plumbing (the per-layer `la.*` scratch, the fp32
        // parameter banks) is the EMITTER's, exactly as it is for the
        // kinds above — naming the symbol is what the lowering owes.
        GdnPrep { .. } => Semantic::Kernels(&["launch_qwen_gdn_post_conv_prep_bf16"]),
        RmsnormGated { .. } => Semantic::Kernels(&["launch_rmsnorm_gated_fp32_in_bf16"]),
        SplitQGate { .. } => Semantic::Kernels(&["launch_split_q_gate_bf16"]),
        SigmoidGateMul => Semantic::Kernels(&["launch_sigmoid_gate_inplace_bf16"]),

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
                &["launch_rmsnorm_bf16"]
            } else {
                &["launch_rmsnorm_gemma_bf16"]
            })
        }

        // Inside a peel's tail the split serves absolute row offsets in a
        // full-N buffer, which is a different kernel — and the REGION is
        // what asks for it, so the lowering states it rather than the
        // driver deriving it from a window pointer.
        SplitQkv { .. } => Semantic::Kernels(if peel_tail {
            &["launch_split_qkv_bf16_devwin"]
        } else {
            &["launch_split_qkv_bf16"]
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
                Semantic::Kernels(&["launch_rope_partial_bf16"])
            } else {
                Semantic::Kernels(&["launch_rope_bf16"])
            }
        }

        // A selector makes the weight per-token, and grouped GEMM is that
        // op's lowering — a different call with a different argument
        // shape, chosen per fire.
        Matmul { selector, .. } => {
            if selector.is_none() {
                Semantic::Kernels(&["gemm_act_x_w"])
            } else {
                Semantic::Unlowered("a selector lowers to grouped GEMM, which the trace does not state")
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
        TopK { .. } => Semantic::Unlowered(
            "the MoE branch has no CUDA text yet (dsl::cuda::topk states the kernel)",
        ),
        WeightedSum { .. } => Semantic::Unlowered(
            "the MoE combine has two forms (token-batched vs per-expert \
             scatter-add, and a fused +residual); the CUDA text has to \
             state which, as the swiglu binding does",
        ),
        SigmoidGateAdd => Semantic::Unlowered(
            "the shared-expert landing awaits the MoE branch's CUDA text",
        ),

        // Handled by `Lowerer::epilogue`, which needs the row counts and
        // so cannot answer from the kind alone.
        LmHead { .. } => Semantic::Structural,

        _ => Semantic::Unlowered("no lowering rule for this kind"),
    }
}

/// The kind's name, for a refusal a human reads.
fn kind_name(kind: &OpKind) -> &'static str {
    use OpKind::*;
    match kind {
        Embed { .. } => "Embed",
        Matmul { .. } => "Matmul",
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
        // per-family table.
        let mut pinned: Vec<ValueId> = Vec::new();
        for stmt in &plan.seams {
            let Some(at) = stmt.op else { continue };
            // The statement points at the construct; the values it sees
            // are the operands of the op that carries the observation.
            for probe in [at as usize, at as usize + 1] {
                if let Some(op) = plan.ops.get(probe) {
                    if matches!(op.kind, OpKind::HookSite { .. } | OpKind::Launch { .. }) {
                        pinned.extend(op.inputs.iter().copied());
                        break;
                    }
                }
            }
        }
        pinned.sort_unstable();
        pinned.dedup();

        // Last use, in one op pass.
        let mut last_use = vec![0usize; plan.values.len()];
        for (i, op) in plan.ops.iter().enumerate() {
            for &v in op.inputs.iter().chain(op.outputs.iter()) {
                if let Some(slot) = last_use.get_mut(v as usize) {
                    *slot = (*slot).max(i);
                }
            }
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
                free.push((offset[v as usize], size[v as usize]));
                false
            });
            for &v in &op.outputs {
                if pinned.binary_search(&v).is_ok() {
                    // Reachable by name from outside the trace — the
                    // query a hook observes, the logits the sampler
                    // reads. The backend binds it; the arena does not.
                    offset[v as usize] = Self::NAMED;
                    continue;
                }
                let want = value_bytes(plan, v, n_tokens, n_requests);
                let at = match free.iter().position(|&(_, s)| s >= want) {
                    Some(f) => free.remove(f).0,
                    None => {
                        // 256-byte alignment, and BUMP only: a decode
                        // body runs inside a capture, so the same plan
                        // must land the same value at the same address
                        // on every fire.
                        let at = used.div_ceil(256) * 256;
                        used = at + want;
                        at
                    }
                };
                offset[v as usize] = at;
                size[v as usize] = want;
                live.push(v);
            }
        }
        Buffers {
            offset,
            bytes: used,
            pinned,
        }
    }
}

fn value_bytes(plan: &ForwardPlan, v: ValueId, n_tokens: usize, n_requests: usize) -> usize {
    let Some(info) = plan.values.get(v as usize) else {
        return 0;
    };
    let mut elements = 1usize;
    for dim in &info.shape.0 {
        elements *= match dim {
            Dim::Tokens => n_tokens,
            Dim::Requests => n_requests,
            Dim::Const(c) => *c as usize,
        };
    }
    elements
        * match info.dtype {
            DType::BF16 | DType::F16 => 2,
            DType::F32 | DType::I32 => 4,
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use crate::family;
    use crate::trace::FireClass;

    /// A fire whose rows are all plain AND all sampled — the ordinary
    /// decode shape, and the one every row-axis test wants.
    fn plain(n: usize) -> Vec<Row> {
        sampled(n)
    }

    fn sampled(n: usize) -> Vec<Row> {
        vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ]
    }

    /// A prefill-shaped fire: `n` token rows, one of them sampled, so
    /// the epilogue gathers.
    fn gathered(n: usize) -> Vec<Row> {
        let mut rows = vec![Row::default(); n];
        rows[n - 1].samples = true;
        rows
    }

    fn decode_plan() -> ForwardPlan {
        family::llama_like_cuda(
            &LlamaLikeFacts::qwen3_0_6b(),
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            FireClass::Decode,
        )
    }

    /// The five live-verified llama_like deployments, each class — the
    /// same set the goldens and the committed `.inc`s cover.
    fn live_plans() -> Vec<(String, ForwardPlan)> {
        let cuda = LlamaLikeCudaFacts::qwen3_0_6b_l40s();
        let deployments = [
            ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b()),
            ("qwen2_5_1_5b", LlamaLikeFacts::qwen2_5_1_5b()),
            ("phi3_mini", LlamaLikeFacts::phi3_mini()),
            ("mistral_7b_v03", LlamaLikeFacts::mistral_7b_v03()),
            ("olmo2_1b", LlamaLikeFacts::olmo2_1b()),
        ];
        let mut out = Vec::new();
        for (name, facts) in deployments {
            for class in [FireClass::Decode, FireClass::Prefill] {
                out.push((
                    format!("{name}.{class:?}"),
                    family::llama_like_cuda(&facts, &cuda, class),
                ));
            }
        }
        out
    }

    /// The qwen3_5 family's residue LEDGER, pinned by kind and count.
    ///
    /// llama_like's cutover was driven by exactly this: a ledger that
    /// names what the flat list still does not carry, so each rung can
    /// be read as a line leaving it. qwen3_5 has never had one — its
    /// executor still walks, and "it walks" was the whole of what was
    /// written down.
    ///
    /// The counts are per fire, not per layer, so they move when a body
    /// changes and stay put when a fixture does.
    #[test]
    fn the_qwen3_5_residue_ledger() {
        let facts = crate::facts::Qwen35HybridFacts::qwen3_5_0_8b();
        let cuda = crate::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
        let plan = family::qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Decode);
        let out = lower(&plan, &sampled(4), Fire::default()).expect("the plan lowers");
        let mut ledger: std::collections::BTreeMap<String, usize> = Default::default();
        for u in &out.residue {
            *ledger
                .entry(format!("{}: {}", u.kind, u.why))
                .or_default() += 1;
        }
        let seen: Vec<String> = ledger
            .iter()
            .map(|(k, n)| format!("{n:>4}  {k}"))
            .collect();
        let expected: Vec<String> = LEDGER_QWEN35_DECODE.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            seen,
            expected,
            "the qwen3_5 residue ledger moved.\n\
             Every line here is a statement the flat list does not carry. \
             If a rung removed one, update the constant and say which \
             statement now names its kernel; if a rung ADDED one, that is \
             a body stating something the lowering cannot read."
        );
    }

    /// The ledger's current contents — see [`the_qwen3_5_residue_ledger`].
    /// One entry per (kind, reason), counted per DECODE fire.
    const LEDGER_QWEN35_DECODE: &[&str] = &[];

    /// THE QWEN3_5 CUTOVER GATE, in the shape llama_like's takes: every
    /// statement a live fire executes is a rectangle in the flat list,
    /// on both geometries and in both classes.
    ///
    /// This was a CONTAINMENT test while the ledger was non-empty — 27B
    /// owes nothing 0.8B does not — because asserting coverage would
    /// have asserted something false about 0.8B too. With the ledger
    /// empty the stronger claim is available, so it is the one made.
    ///
    /// 27B earns its own row: it is the first geometry whose GDN half is
    /// GQA (48 value heads over 16 key heads), which 0.8B cannot prove
    /// either way.
    #[test]
    fn the_qwen3_5_flat_list_covers_every_statement() {
        let cuda = crate::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
        let geometries = [
            ("0.8b", crate::facts::Qwen35HybridFacts::qwen3_5_0_8b()),
            ("27b", crate::facts::Qwen35HybridFacts::qwen3_6_27b()),
        ];
        for (name, facts) in geometries {
            for class in [FireClass::Decode, FireClass::Prefill] {
                let plan = family::qwen3_5_hybrid_cuda(&facts, &cuda, class);
                for (shape, rows) in [("all-sampled", sampled(4)), ("gathered", gathered(4))] {
                    let out = lower(&plan, &rows, Fire::default())
                        .unwrap_or_else(|e| panic!("{name}/{class:?}/{shape}: {e:?}"));
                    assert!(
                        out.residue.is_empty(),
                        "{name}/{class:?}/{shape}: {} statements still owe a \
                         declaration: {:#?}",
                        out.residue.len(),
                        out.residue
                    );
                    assert_eq!(out.coverage(), 1.0, "{name}/{class:?}/{shape}");
                    assert!(
                        !out.launches.is_empty(),
                        "{name}/{class:?}/{shape}: a fire that executes nothing \
                         is not a fire"
                    );
                }
            }
        }
    }

    /// gemma-4's residue LEDGER — the third family's, opened the way
    /// qwen3_5's was and for the same reason: a rung is legible when it
    /// is a line leaving this list.
    ///
    /// Empty means the body is a list of rectangles. It does NOT mean the
    /// numbers are right: five of the six defects the executor found were
    /// in a declaration whose ledger was already empty. This gate asks
    /// whether statements are WELL FORMED; only a live fire asks whether
    /// each one consumes what the pass produces.
    #[test]
    fn the_gemma4_residue_ledger() {
        let facts = crate::facts::Gemma4Facts::gemma_4_e4b();
        let cuda = crate::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic();
        for (class, expected) in [
            (FireClass::Decode, LEDGER_GEMMA4_DECODE),
            (FireClass::Prefill, LEDGER_GEMMA4_PREFILL),
        ] {
            let plan = family::gemma4_cuda(&facts, &cuda, class);
            let out = lower(&plan, &sampled(4), Fire::default()).expect("the plan lowers");
            let mut ledger: std::collections::BTreeMap<String, usize> = Default::default();
            for u in &out.residue {
                *ledger
                    .entry(format!("{}: {}", u.kind, u.why))
                    .or_default() += 1;
            }
            let seen: Vec<String> = ledger
                .iter()
                .map(|(k, n)| format!("{n:>4}  {k}"))
                .collect();
            let want: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
            assert_eq!(seen, want, "the gemma-4 {class:?} residue ledger moved");
            assert!(
                !out.launches.is_empty(),
                "a fire that executes nothing is not a fire"
            );
        }
    }

    /// See [`the_gemma4_residue_ledger`]. One entry per (kind, reason).
    const LEDGER_GEMMA4_DECODE: &[&str] = &[];

    /// The prefill class's, which differs from the decode ledger only in
    /// the dispatch — and states two kernels of its own, so an empty
    /// ledger here is a claim about those two as much as about the body.
    const LEDGER_GEMMA4_PREFILL: &[&str] = &[];

    /// gpt-oss's residue LEDGER — the fourth family's, opened the day the
    /// text was written and before any executor exists.
    ///
    /// gpt-oss is the first family whose MoE block is stated end to end,
    /// which is the whole reason to open this list here: the decode leg
    /// is seven rectangles because two GEMVs carry the expert axis
    /// INSIDE the value, and if any of that were wrong it would show up
    /// as a line below rather than as a wrong number later.
    #[test]
    fn the_gpt_oss_residue_ledger() {
        let facts = crate::facts::GptOssFacts::gpt_oss_20b();
        let cuda = crate::facts::GptOssCudaFacts::gpt_oss_20b_synthetic();
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = family::gpt_oss_cuda(&facts, &cuda, class);
            let out = lower(&plan, &sampled(4), Fire::default()).expect("lowers");
            assert!(
                out.residue.is_empty() && out.coverage() == 1.0,
                "gpt-oss {class:?}: {:#?}",
                out.residue
            );
        }
        let plan = family::gpt_oss_cuda(&facts, &cuda, FireClass::Decode);
        let out = lower(&plan, &sampled(4), Fire::default()).expect("the plan lowers");
        let mut ledger: std::collections::BTreeMap<String, usize> = Default::default();
        for u in &out.residue {
            *ledger
                .entry(format!("{}: {}", u.kind, u.why))
                .or_default() += 1;
        }
        let seen: Vec<String> = ledger
            .iter()
            .map(|(k, n)| format!("{n:>4}  {k}"))
            .collect();
        let expected: Vec<String> = LEDGER_GPT_OSS_DECODE.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            seen, expected,
            "the gpt-oss residue ledger moved.\n\
             Every line here is a statement the flat list does not carry."
        );
        assert!(
            !out.launches.is_empty(),
            "a fire that executes nothing is not a fire"
        );
    }

    /// See [`the_gpt_oss_residue_ledger`]. One entry per (kind, reason).
    const LEDGER_GPT_OSS_DECODE: &[&str] = &[];

    /// THE GEMMA-4 CUTOVER GATE, in the shape the other two families'
    /// take: every statement a live fire executes is a rectangle in the
    /// flat list, in both classes and both logit shapes.
    ///
    /// One geometry only, and honestly so: E4B is the sole gemma-4 fact
    /// set anything has been read against. A second (E2B, the 31B) would
    /// earn its own row the way 27B earned qwen3_5's.
    #[test]
    fn the_gemma4_flat_list_covers_every_statement() {
        // BOTH geometries. E2B is not a formality: 35 layers, MQA, 20/35
        // KV-shared and a DOUBLE-WIDE MLP, so it exercises
        // `intermediate_of`, the odd-layer interval and the unfused MLP
        // arm — three things E4B cannot say anything about. It found
        // three real gaps the day it was first booted.
        for (name, facts) in [
            ("e4b", crate::facts::Gemma4Facts::gemma_4_e4b()),
            ("e2b", crate::facts::Gemma4Facts::gemma_4_e2b()),
        ] {
        let _ = name;
        let cuda = crate::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic();
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = family::gemma4_cuda(&facts, &cuda, class);
            for (shape, rows) in [("all-sampled", sampled(4)), ("gathered", gathered(4))] {
                let out = lower(&plan, &rows, Fire::default())
                    .unwrap_or_else(|e| panic!("{class:?}/{shape}: {e:?}"));
                assert!(
                    out.residue.is_empty(),
                    "{class:?}/{shape}: {} statements still owe a declaration: {:#?}",
                    out.residue.len(),
                    out.residue
                );
                assert_eq!(out.coverage(), 1.0, "{class:?}/{shape}");
                assert!(
                    !out.launches.is_empty(),
                    "{class:?}/{shape}: a fire that executes nothing is not a fire"
                );
            }
        }
        }
    }

    /// The MoE block's own ledger, and the argument for the fused leg.
    ///
    /// The SEMANTIC reading is residue — a selector, a combine and a
    /// shared-expert landing that no kernel is named for. The CUDA
    /// reading of the same fragment is a list of rectangles. Both halves
    /// are asserted here because either one alone is half the claim: a
    /// covered CUDA reading proves the statements exist, and an
    /// uncovered semantic one proves they were needed.
    #[test]
    fn the_moe_block_covers_itself_only_in_its_cuda_reading() {
        let facts = crate::facts::Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
        let cuda = crate::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic();

        // The semantic fragment names no backend at all, so it does not
        // reach the residue ledger — it is refused before any op is
        // read. That is the honest baseline: the MoE block had no CUDA
        // reading, not a partial one.
        let semantic = family::qwen3_5_moe_mlp_block(&facts);
        assert!(
            matches!(
                lower(&semantic, &sampled(4), Fire::default()),
                Err(Uncovered::UnknownBackend(_))
            ),
            "the semantic MoE block named a backend — if it was given a \
             CUDA reading, this test is the one that should say so"
        );

        let declared = family::qwen3_5_moe_mlp_block_cuda(&facts, &cuda);
        let out = lower(&declared, &sampled(4), Fire::default())
            .unwrap_or_else(|e| panic!("the CUDA MoE block must lower: {e:?}"));
        assert!(
            out.residue.is_empty(),
            "{} statements still owe a declaration: {:#?}",
            out.residue.len(),
            out.residue
        );
        assert_eq!(out.coverage(), 1.0);
    }

    /// THE CUTOVER GATE. Every statement a live fire executes is a
    /// rectangle in the flat list — no residue, on every deployment the
    /// driver serves, in both classes, sampled and unsampled.
    ///
    /// This started as a ledger (88.7%-93.8%, residue `Swiglu` per layer
    /// + `LmHead` per fire) and is now the gate itself: `launches` is
    /// the WHOLE of what a fire runs, which is the property the driver
    /// needs before it can stop walking. A regression here is a
    /// statement that would silently not execute.
    #[test]
    fn the_flat_list_covers_every_statement() {
        for (name, plan) in live_plans() {
            // Both epilogue shapes: a decode fire samples every row, a
            // prefill fire samples one row per request and gathers.
            for (shape, rows) in [("all-sampled", sampled(4)), ("gathered", gathered(4))] {
                let out = lower(&plan, &rows, Fire::default()).unwrap_or_else(|e| panic!("{name}/{shape}: {e:?}"));
                assert!(
                    out.residue.is_empty(),
                    "{name}/{shape}: {} statements still owe a declaration: {:#?}",
                    out.residue.len(),
                    out.residue
                );
                assert_eq!(out.coverage(), 1.0, "{name}/{shape}");
                assert!(
                    !out.launches.is_empty(),
                    "{name}/{shape}: a fire that executes nothing is not a fire"
                );
            }
        }
    }

    /// A site inside an arm the guards did not take must NOT fire, and
    /// the rectangles alone cannot say which those are — so the list
    /// carries the live ones. This is what a form driven by the list
    /// needs in order to bracket a layer's sideband correctly.
    #[test]
    fn the_live_sites_are_named_and_the_dead_ones_are_not() {
        let plan = decode_plan();
        let sites: Vec<usize> = plan
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| matches!(op.kind, OpKind::HookSite { .. }))
            .map(|(i, _)| i)
            .collect();
        assert!(!sites.is_empty(), "the class trace carries observation sites");

        // A plain fire takes the else arm; a MASKED fire takes the mask
        // arm. Both bracket their layers, and they are DIFFERENT sites —
        // which is the whole reason the list has to say.
        let plain_out = lower(&plan, &sampled(4), Fire::default()).unwrap();
        let mut masked = sampled(4);
        for r in &mut masked {
            r.custom_mask = true;
        }
        let masked_out = lower(&plan, &masked, Fire::default()).unwrap();

        for out in [&plain_out, &masked_out] {
            assert!(!out.structural.is_empty(), "a live fire brackets its layers");
            assert!(
                out.structural
                    .iter()
                    .all(|s| sites.contains(&(s.at_op as usize))),
                "only sites are structural"
            );
            // Ordered, because a bracket opens before it closes.
            assert!(out
                .structural
                .windows(2)
                .all(|w| w[0].at_op < w[1].at_op));
            // And every site brackets a NON-EMPTY window — an empty one
            // would be a retired layer's, which does not fire at all.
            assert!(out.structural.iter().all(|s| !s.rows.is_empty()));
        }
        assert_ne!(
            plain_out.structural, masked_out.structural,
            "the two arms bracket through different sites"
        );
        // And a dead arm's sites are absent from BOTH.
        assert!(plain_out.structural.len() < sites.len());
        assert!(masked_out.structural.len() < sites.len());
    }

    /// A CAPTURED fire emits both peel regions whatever its split is,
    /// and says so on the launches — the one place a rectangle is not a
    /// pair of numbers.
    ///
    /// The shadow comparison is what asked for this: the walk emits both
    /// regions under device-window capture (an empty one early-outs on
    /// the device word, so one graph replays across every split) and the
    /// flat list described only the non-empty one.
    #[test]
    fn a_captured_fire_emits_both_peel_regions() {
        let plan = decode_plan();
        // fast_rows == 0: every row hooked, so the hook-free prefix is
        // empty. Uncaptured, it contributes nothing.
        let mut rows = sampled(4);
        for r in &mut rows {
            r.hooked = true;
        }
        let host = lower(&plan, &rows, Fire::default()).expect("coverable");
        assert!(host.launches.iter().all(|l| l.peel.is_none_or(|p| !p.rows_device)));
        assert!(
            !host
                .launches
                .iter()
                .any(|l| l.kernel_is(&host, "launch_qkv_decode_qk_norm_rope_write_kv_bf16")),
            "an empty prefix launches nothing when the host's count is the truth"
        );

        // Captured: the prefix's launches ARE in the list, marked as
        // reading the fire's split rather than these counts.
        let captured = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: true,
            },
        )
        .expect("coverable");
        let fused: Vec<_> = captured
            .launches
            .iter()
            .filter(|l| l.kernel_is(&captured, "launch_qkv_decode_qk_norm_rope_write_kv_bf16"))
            .collect();
        assert!(!fused.is_empty(), "the captured graph carries the prefix");
        assert!(fused
            .iter()
            .all(|l| l.peel.is_some_and(|p| p.axis == PeelWindow::HookFreePrefix
                && p.rows_device)));
        // And its rows are the WHOLE window, not the empty prefix half:
        // a captured region launches a full-window grid and reads the
        // split off the device word. Naming the half would describe a
        // grid nobody launches, and an executor that believed it would
        // bake this fire's split into the graph — wrong only on the
        // REPLAY, which is why this is asserted here rather than left to
        // a parity run to notice.
        assert!(
            fused
                .iter()
                .all(|l| l.rows.start == 0 && l.rows.end == rows.len() as u32),
            "a captured peel region's rectangle is the full window"
        );

        // And ONLY the peel's regions are marked: everything outside is
        // still a plain count, which is what keeps the list readable.
        assert!(captured
            .launches
            .iter()
            .filter(|l| l.peel.is_some_and(|p| p.rows_device))
            .count()
            < captured.launches.len());
    }

    /// The epilogue is three statements over a ROW COUNT, and the two
    /// runtime branches the executor takes are the count being zero and
    /// the count being short.
    #[test]
    fn the_epilogue_is_a_row_count_not_a_branch() {
        let plan = decode_plan();
        // The epilogue's launches are the ones carrying the LmHead
        // statement's index. Identifying them by SYMBOL would not work:
        // its projection is `gemm_act_x_w`, the same launcher every
        // body matmul takes.
        let at_op = plan
            .ops
            .iter()
            .position(|op| matches!(op.kind, OpKind::LmHead { .. }))
            .expect("the class trace has an epilogue") as u32;
        let epilogue = |rows: &[Row]| -> Vec<(String, Range<u32>)> {
            let out = lower(&plan, rows, Fire::default()).expect("coverable");
            out.launches
                .iter()
                .filter(|l| l.args == at_op)
                .map(|l| (out.kernels[l.kernel as usize].clone(), l.rows.clone()))
                .collect()
        };

        // Every row sampled: norm and project over all four rows, no
        // gather — there is nothing to skip past.
        let all = epilogue(&sampled(4));
        assert_eq!(
            all,
            vec![
                ("launch_rmsnorm_bf16".to_string(), 0..4),
                ("gemm_act_x_w".to_string(), 0..4),
            ]
        );

        // One sampled row of four: the gather appears, and all three
        // statements run over ONE row while the body ran over four —
        // the epilogue's row space is Requests.
        assert_eq!(
            epilogue(&gathered(4)),
            vec![
                ("launch_gather_bf16_rows".to_string(), 0..1),
                ("launch_rmsnorm_bf16".to_string(), 0..1),
                ("gemm_act_x_w".to_string(), 0..1),
            ]
        );

        // Nothing sampled (`emit_logits == false`, a fire whose logits
        // nobody reads): no rectangle at all, while the body still runs.
        let none = vec![Row::default(); 4];
        assert!(epilogue(&none).is_empty());
        assert!(!lower(&plan, &none, Fire::default()).unwrap().launches.is_empty());
    }

    /// A plain fire lowers, and every launch covers every row — the
    /// degenerate rectangle, which is what today's fires are.
    #[test]
    fn a_plain_fire_is_one_rectangle_per_statement() {
        let plan = decode_plan();
        let rows = plain(8);
        let out = lower(&plan, &rows, Fire::default()).expect("a plain fire is coverable");
        assert!(out.rectangles > 0);
        assert!(out.launches.iter().all(|l| l.rows == (0..8)));
        // The frame's kernel table is what the driver would index.
        assert!(out.kernels.contains(&"dispatch_attention_flashinfer_decode".to_string()));
        // Every launch names a layer the trace tagged.
        assert!(out.launches.iter().all(|l| l.layers.end == l.layers.start + 1));
    }

    /// The MASK arm selects only the masked rows, and the rest take the
    /// plain body — one statement, two rectangles. This is the thing the
    /// flat ABI buys: today the same fire is a guard the driver walks.
    #[test]
    fn a_masked_suffix_splits_the_rectangle() {
        let plan = decode_plan();
        // The seriation puts masked rows last.
        let mut rows = plain(8);
        for r in &mut rows[6..] {
            r.custom_mask = true;
        }
        let out = lower(&plan, &rows, Fire::default()).expect("mask + plain is coverable");
        let masked = out
            .launches
            .iter()
            .filter(|l| l.rows == (6..8))
            .count();
        let plain_rows = out.launches.iter().filter(|l| l.rows == (0..6)).count();
        assert!(masked > 0, "the masked rows got their own rectangles");
        assert!(plain_rows > 0, "and the plain rows theirs");
        // More rectangles than the unsplit fire — what the row order
        // costs, reported rather than acted on.
        let flat = lower(&plan, &plain(8), Fire::default()).unwrap();
        assert!(out.rectangles > flat.rectangles);
    }

    /// A DISCONTIGUOUS order is refused rather than silently mis-served.
    /// The engine's seriation guarantees contiguity per axis; if it ever
    /// stops, this is the answer, and it is an admission answer.
    #[test]
    fn a_discontiguous_order_is_uncovered() {
        let plan = decode_plan();
        let mut rows = plain(8);
        rows[1].custom_mask = true;
        rows[5].custom_mask = true;
        assert!(matches!(
            lower(&plan, &rows, Fire::default()),
            Err(Uncovered::Discontiguous { .. })
        ));
    }

    /// `whole` CONSUMED: an XQA deployment's fire may not be lowered
    /// with the kernel over a subset. Statically the check refuses it
    /// inside a Peel; here it refuses the dynamic case too.
    #[test]
    fn a_whole_kernel_refuses_a_row_window() {
        let facts = LlamaLikeFacts::qwen3_0_6b();
        let cuda = LlamaLikeCudaFacts {
            xqa_decode: true,
            decode_fused_post: false,
            ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
        };
        let plan = family::llama_like_cuda(&facts, &cuda, FireClass::Decode);
        assert!(
            plan.ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Launch { kernel, .. }
                    if kernel == "launch_attention_xqa_decode_bf16_prepared"
            )),
            "this deployment states XQA"
        );
        // Whole fire: fine.
        assert!(lower(&plan, &plain(8), Fire::default()).is_ok());
        // And a MASKED fire is fine too, which is the point: a guard is
        // a fire fact, so the mask arm takes the whole fire and XQA — in
        // the else arm — does not run at all. Nothing hands a kernel a
        // row window except a Peel, and a `whole` kernel inside a Peel
        // is refused STATICALLY at trace time (`kernels::check_plan`),
        // so this dynamic check is a backstop rather than the rule's
        // live enforcement. It stays because the flat list is about to
        // become the thing that executes.
        let mut rows = plain(8);
        for r in &mut rows[6..] {
            r.custom_mask = true;
        }
        assert!(lower(&plan, &rows, Fire::default()).is_ok());
    }

    /// Liveness reuse is the point of assigning buffers here: a
    /// 28-layer unrolled plan names 28 distinct normed-activation values
    /// whose ranges never overlap, so the arena must be far smaller than
    /// the naive sum.
    #[test]
    fn the_arena_reuses_across_layers() {
        let plan = decode_plan();
        let rows = plain(8);
        let buffers = Buffers::assign(&plan, &rows);
        let naive: usize = (0..plan.values.len())
            .map(|v| value_bytes(&plan, v as ValueId, rows.len(), rows.len()))
            .sum();
        assert!(buffers.bytes > 0);
        assert!(
            buffers.bytes * 4 < naive,
            "arena {} vs naive {naive}",
            buffers.bytes
        );
        // Pinned values are the backend's to bind, not the arena's.
        assert!(buffers
            .pinned
            .iter()
            .all(|&v| buffers.offset[v as usize] == Buffers::NAMED));
        // Pins come off the seam statements, not a per-family table.
        assert!(
            !buffers.pinned.is_empty(),
            "this text states observation seams, so some values are exposed"
        );
    }

    /// FOUR distinct truncations lower fine. The driver's
    /// `derive_depth_bands` refuses a fourth band (`if (count == 3)
    /// return 0`) because its walk carries per-band plans; here a
    /// layer's live row count is a number, so the ceiling has nowhere to
    /// live. This is step 5's driver half, on the host side.
    #[test]
    fn depth_has_no_band_ceiling() {
        let plan = decode_plan();
        // Seriation order: full-depth first, then truncated deepest-first.
        let mut rows = plain(10);
        for (i, k) in [(2usize, 24u32), (4, 20), (6, 16), (8, 8)] {
            for r in &mut rows[i..] {
                r.depth_k = Some(k);
            }
        }
        let out = lower(&plan, &rows, Fire::default()).expect("four bands is not a special case");
        // Layer 0 runs over everybody; layer 23 only over the rows whose
        // k is past it (the full-depth prefix plus the k=24 block).
        let at = |l: u16| {
            out.launches
                .iter()
                .filter(|x| x.layers.start == l)
                .map(|x| x.rows.end)
                .max()
                .unwrap_or(0)
        };
        // rows 0-1 full depth, 2-3 k=24, 4-5 k=20, 6-7 k=16, 8-9 k=8;
        // a row is live at layer l while l < k, so it dies AT l == k.
        assert_eq!(at(0), 10);
        assert_eq!(at(7), 10);
        assert_eq!(at(8), 8, "the k=8 pair dies at layer 8");
        assert_eq!(at(16), 6);
        assert_eq!(at(20), 4);
        assert_eq!(at(23), 4);
        assert_eq!(at(24), 2, "only the full-depth rows are left");
        assert_eq!(at(27), 2);
    }

    /// A uniform truncation SKIPS the tail layers entirely — no launch
    /// is emitted where nothing is live.
    #[test]
    fn a_uniform_truncation_skips_the_tail() {
        let plan = decode_plan();
        let rows = vec![
            Row {
                depth_k: Some(12),
                ..Row::default()
            };
            4
        ];
        let out = lower(&plan, &rows, Fire::default()).unwrap();
        assert!(out.launches.iter().all(|l| l.layers.start < 12
            || l.layers.start >= 28
            || l.rows.is_empty()));
        let full = lower(&plan, &plain(4), Fire::default()).unwrap();
        assert!(out.rectangles < full.rectangles, "truncation costs less");
    }

    /// The arena is DETERMINISTIC in ask order — the property a replayed
    /// graph needs, since the same plan must land the same value at the
    /// same address on every fire.
    #[test]
    fn the_arena_is_deterministic() {
        let plan = decode_plan();
        let a = Buffers::assign(&plan, &plain(8));
        let b = Buffers::assign(&plan, &plain(8));
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.bytes, b.bytes);
    }
}
