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
//! This module is the host half of that, and ONLY the host half. The
//! driver still executes `declared_forward.cpp` and the generated
//! `.inc`s; switching it over is a separate change whose gate is the
//! killer soak and the declared==hand A/B, not a byte comparison — see
//! `.wiki/tart/macos.md`'s sibling note in `dsl.md` step 6. Nothing here
//! is on any execution path yet.
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
}

/// One flat launch: a kernel over a rectangle of (rows × layers).
///
/// `args` is an index into the frame's argument slots — the driver binds
/// operands from there, which is why no buffer appears in this struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub kernel: u16,
    pub rows: Range<u32>,
    pub layers: Range<u16>,
    pub args: u32,
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
}

/// Lower `plan` over `rows`, in the order the engine seriated them.
pub fn lower(plan: &ForwardPlan, rows: &[Row]) -> Result<Lowered, Uncovered> {
    let backend = Backend::of_family(&plan.family);
    let n = rows.len() as u32;
    let mut out = Lowerer {
        plan,
        rows,
        backend,
        launches: Vec::new(),
        kernels: Vec::new(),
        kernel_ids: BTreeMap::new(),
    };
    out.region(0..plan.ops.len(), 0..n)?;
    let buffers = Buffers::assign(plan, rows);
    Ok(Lowered {
        rectangles: out.launches.len(),
        launches: out.launches,
        kernels: out.kernels,
        arena_bytes: buffers.bytes,
    })
}

struct Lowerer<'a> {
    plan: &'a ForwardPlan,
    rows: &'a [Row],
    backend: Option<Backend>,
    launches: Vec<Launch>,
    kernels: Vec<String>,
    kernel_ids: BTreeMap<String, u16>,
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
                        let taken = self.select(&remaining, arm.pred, i)?;
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
                    if !prefix.is_empty() {
                        self.region(p, prefix)?;
                    }
                    let next = t.end;
                    if !tail.is_empty() {
                        self.region(t, tail)?;
                    }
                    i = next;
                }
                OpKind::Launch { kernel, .. } => {
                    let live = self.depth_window(op, &window, i)?;
                    self.emit(i, kernel, op, &live)?;
                    i += 1;
                }
                // A semantic op is a statement the backend has not
                // lowered. It has no kernel to launch, so it produces no
                // rectangle; a trace that is meant to execute states
                // kernels (that is what `lower`'s backend is FOR).
                _ => i += 1,
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
        if window.is_empty() {
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
        });
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

    /// The rows in `window` satisfying `pred`, as a contiguous range.
    fn select(
        &self,
        window: &Range<u32>,
        pred: GuardPred,
        at: usize,
    ) -> Result<Range<u32>, Uncovered> {
        let (axis, holds): (&'static str, fn(&Row) -> bool) = match pred {
            GuardPred::HasCustomMask => ("mask", |r| r.custom_mask),
            GuardPred::HasLora => ("lora", |r| r.lora),
            GuardPred::HasStageHooks => ("hook", |r| r.hooked),
            GuardPred::WantsAttnScore => ("scores", |r| r.wants_scores),
            GuardPred::HasWriteDesc => ("write_desc", |r| r.write_desc),
            // Token-count predicates are FIRE-wide, not per row: they
            // read the fire's N. Every row is in or out together.
            GuardPred::TokensLE(k) => {
                return Ok(if self.rows.len() as u32 <= k {
                    window.clone()
                } else {
                    window.start..window.start
                });
            }
            GuardPred::TokensGT(k) => {
                return Ok(if self.rows.len() as u32 > k {
                    window.clone()
                } else {
                    window.start..window.start
                });
            }
        };
        contiguous(self.rows, window, holds, axis, at)
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
        let n_requests = rows.iter().filter(|r| !r.multi_token).count().max(1);

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
            DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use crate::family;
    use crate::trace::FireClass;

    fn plain(n: usize) -> Vec<Row> {
        vec![Row::default(); n]
    }

    fn decode_plan() -> ForwardPlan {
        family::llama_like_cuda(
            &LlamaLikeFacts::qwen3_0_6b(),
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            FireClass::Decode,
        )
    }

    /// A plain fire lowers, and every launch covers every row — the
    /// degenerate rectangle, which is what today's fires are.
    #[test]
    fn a_plain_fire_is_one_rectangle_per_statement() {
        let plan = decode_plan();
        let rows = plain(8);
        let out = lower(&plan, &rows).expect("a plain fire is coverable");
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
        let out = lower(&plan, &rows).expect("mask + plain is coverable");
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
        let flat = lower(&plan, &plain(8)).unwrap();
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
            lower(&plan, &rows),
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
        assert!(lower(&plan, &plain(8)).is_ok());
        // A masked suffix would hand XQA the unmasked prefix only.
        let mut rows = plain(8);
        for r in &mut rows[6..] {
            r.custom_mask = true;
        }
        assert!(matches!(
            lower(&plan, &rows),
            Err(Uncovered::WholeKernelSplit { kernel, .. }) if kernel.contains("xqa")
        ));
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
        let out = lower(&plan, &rows).expect("four bands is not a special case");
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
        let out = lower(&plan, &rows).unwrap();
        assert!(out.launches.iter().all(|l| l.layers.start < 12
            || l.layers.start >= 28
            || l.rows.is_empty()));
        let full = lower(&plan, &plain(4)).unwrap();
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
