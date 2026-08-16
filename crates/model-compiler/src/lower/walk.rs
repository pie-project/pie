//! THE WALK — `lower()` and the `Lowerer` that answers it.
//!
//! The entry points sit WITH the walker rather than in the parent, and
//! that is load-bearing rather than tidy: `lower_with` builds a `Lowerer`
//! field by field and reads six of its vectors back out at the end, so
//! splitting them apart would have meant `pub(super)` on the whole
//! struct. The seam that costs nothing is the one drawn here.

use super::*;
use super::semantics::{Semantic, contiguous, kind_name, semantic, subtract};

/// Lower `plan` over `rows`, in the order the engine seriated them.
///
/// Guards are ANSWERED — see [`GuardMode`] for the other mode and why it
/// exists.
pub fn lower(plan: &ForwardPlan, rows: &[Row], fire: Fire) -> Result<Lowered, Uncovered> {
    lower_with(plan, rows, fire, GuardMode::Resolve)
}

/// Lower `plan` over `rows`, choosing whether guards are answered.
///
/// [`GuardMode::Union`] is the unionized supergraph's input: one lowering
/// that covers every structurally-distinct program in a bucket, with the
/// guard tree preserved in [`Lowered::conds`] so a driver can rebuild it
/// as conditional graph nodes.
pub fn lower_with(
    plan: &ForwardPlan,
    rows: &[Row],
    fire: Fire,
    guards: GuardMode,
) -> Result<Lowered, Uncovered> {
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
        params: Vec::new(),
        buffers: Buffers::assign(plan, rows),
        value_owner: alias_owners(plan),
        peel_region: None,
        guards,
        cond: Launch::NO_COND,
        region_outs: Vec::new(),
        conds: Vec::new(),
    };
    let arena_bytes = out.buffers.bytes;
    let value_offset = out.buffers.offset.clone();
    let epilogue_gather = out.buffers.epilogue_gather;
    let n_requests = out.buffers.n_requests;
    let epilogue_norm = out.buffers.epilogue_norm;
    let value_owner = out.value_owner.clone();
    // The exit, resolved before the walk so it needs nothing the walk builds.
    // The seam names the value; the plan gives its shape and dtype; buffer
    // assignment already placed it.
    let readout = plan
        .seams
        .iter()
        .find(|s| s.seam == model_ir::seam::OUT.name)
        .and_then(|s| s.values.first().copied())
        .and_then(|id| {
            let info = plan.values.get(id as usize)?;
            let at = *value_offset.get(id as usize)?;
            if at == Buffers::NAMED {
                return None;
            }
            // `[rows, vocab]`, and the last dim is the row stride. A vocabulary
            // is a constant of the deployment; the row count is the FIRE's, so
            // a `Requests` axis reads the count the fire computed rather than a
            // number the text could not have known.
            let vocab = match info.shape.0.last()? {
                Dim::Const(v) => *v,
                _ => return None,
            };
            let rows = match info.shape.0.first()? {
                Dim::Requests => n_requests,
                Dim::Tokens => n,
                Dim::Const(v) => *v,
                _ => return None,
            };
            Some(Readout {
                at,
                rows,
                vocab,
                bytes: dtype_bytes(info.dtype),
            })
        });
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
        n_requests,
        args: out.args,
        params: out.params,
        structural: out.structural,
        residue: out.residue,
        conds: out.conds,
        readout,
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
    params: Vec<u32>,
    /// The arena, so an operand can be resolved to an offset as it is
    /// emitted rather than in a second pass that would have to re-walk
    /// the regions to know which launches exist.
    buffers: Buffers,
    /// The alias groups, so a NAMED operand can be emitted under the
    /// value that owns its bytes. [`Lowerer::slot`] is the only reader
    /// and says why an arena operand needs no such step.
    value_owner: Vec<ValueId>,
    /// The peel region the launches being emitted sit in.
    peel_region: Option<PeelRegion>,
    /// Whether guards are answered or kept.
    guards: GuardMode,
    /// The conditional region the launches being emitted sit in, and the
    /// tree they index into. Both stay empty under
    /// [`GuardMode::Resolve`].
    cond: u32,
    /// The enclosing value-producing region's outputs, when there is one.
    ///
    /// A guard or peel that produces a value owns it, and its arms' launches
    /// are LOWERINGS of that value rather than producers of their own --
    /// `model_dsl::guarded_value`'s doc states it exactly: *"each region's launches
    /// are their lowerings, binding the same output buffer and recording no
    /// SSA outputs of their own."* The tape half of that has existed since
    /// `seam::attn_at` (`TraceBuilder::inside_value_region`); this is the
    /// lowering half.
    ///
    /// Without it an arm's launch reaches `dispatch::reorder` with no result
    /// operand at all, and the row's `Out(0)` resolves to the launch's only
    /// widthed operand -- its INPUT. Measured on a real checkpoint: every
    /// projection in the arm wrote zeros over the value it had just read.
    region_outs: Vec<ValueId>,
    conds: Vec<CondRegion>,
}

impl Lowerer<'_> {
    /// Lower the ops in `span` over `window`, the rows currently live.
    fn region(&mut self, span: Range<usize>, window: Range<u32>) -> Result<(), Uncovered> {
        let mut i = span.start;
        'ops: while i < span.end {
            let op = &self.plan.ops[i];
            match &op.kind {
                OpKind::Guard { arms, else_ops } => {
                    // A guard that produces a value owns it; its arms bind it.
                    // Saved and restored so a nested guard does not leak its
                    // outputs to an enclosing one's arms.
                    let outer_outs = std::mem::take(&mut self.region_outs);
                    self.region_outs = op.outputs.clone();
                    // A fire-level chain: the first arm whose predicate
                    // holds runs, over the SAME rows. In the flattened
                    // world these are row predicates, and an arm's rows
                    // are the subset of `window` satisfying it — which
                    // is what this computes. An arm selecting nobody
                    // emits nothing, which is the "vanishes" behaviour
                    // an argument-driven site already has.
                    let mut at = i + 1;
                    if self.guards == GuardMode::Union {
                        // NOT EVERY PREDICATE FOLDS, and the north star's
                        // own list says which: "hook attachment, mask
                        // kind, correction arm, depth, LoRA rank". Scores
                        // are not on it, and the reason shows up the
                        // moment you try — a score-capturing dispatch
                        // needs a plan prepared for it, buffers laid out
                        // for it, and an observation window, none of which
                        // a fire that wants no scores builds. An arm whose
                        // PREPARED STATE the fire declines to build cannot
                        // be recorded, only refused.
                        //
                        // So a non-foldable predicate is answered here,
                        // exactly as `Resolve` answers it, and the axis
                        // belongs in the bucket KEY instead — the same
                        // conclusion `BucketKey::lora_shape` reached from
                        // the other end.
                        fn folds(p: GuardPred) -> bool {
                            !matches!(p, GuardPred::WantsAttnScore)
                        }
                        // UNION: answer nothing. Every arm lowers over
                        // the whole window, tagged with its place in the
                        // tree, and the chain nests — arm k runs when
                        // predicates 0..k did not hold and k did, which
                        // is a run of else bodies and therefore a run of
                        // nested conditional nodes.
                        let outer = self.cond;
                        let mut parent = outer;
                        for arm in arms {
                            let body = at..at + arm.ops as usize;
                            at += arm.ops as usize;
                            if !folds(arm.pred) {
                                // Answered, not folded. An arm that does
                                // not hold vanishes; one that does takes
                                // the whole window and ends the chain,
                                // because a resolved arm is the choice.
                                if !self.select(&window, arm.pred).is_empty() {
                                    self.cond = parent;
                                    self.region(body, window.clone())?;
                                    self.cond = outer;
                                    self.region_outs = outer_outs;
                                    i = at + *else_ops as usize;
                                    continue 'ops;
                                }
                                continue;
                            }
                            let (slot, param) = arm.pred.wire();
                            let (if_arm, else_arm) = self.push_cond_pair(parent, slot, param);
                            self.cond = if_arm;
                            self.region(body, window.clone())?;
                            parent = else_arm;
                        }
                        self.cond = parent;
                        self.region(at..at + *else_ops as usize, window.clone())?;
                        self.cond = outer;
                        self.region_outs = outer_outs;
                        i = at + *else_ops as usize;
                        continue;
                    }
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
                    self.region_outs = outer_outs;
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
                    let run =
                        |me: &mut Self, span: Range<usize>, w: Range<u32>, tail_side: bool| {
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
        self.emit_bound(at, kernel, op, window, None, None)
    }

    /// [`Self::emit`], with the statement's OUTPUT replaced.
    ///
    /// The epilogue's row gather is the only caller: one `LmHead` lowers
    /// to two launches that pass a value between them, and the trace has
    /// no name for it. See [`Self::epilogue`].
    fn emit_with_out(
        &mut self,
        at: usize,
        kernel: &str,
        op: &Op,
        window: &Range<u32>,
        out: Option<Arg>,
    ) -> Result<(), Uncovered> {
        self.emit_bound(at, kernel, op, window, None, out)
    }

    /// [`Self::emit`], with the statement's INPUT replaced — the other
    /// half of the gather's hand-off.
    fn emit_with_in(
        &mut self,
        at: usize,
        kernel: &str,
        op: &Op,
        window: &Range<u32>,
        input: Option<Arg>,
    ) -> Result<(), Uncovered> {
        self.emit_bound(at, kernel, op, window, input, None)
    }

    fn emit_bound(
        &mut self,
        at: usize,
        kernel: &str,
        op: &Op,
        window: &Range<u32>,
        in_override: Option<Arg>,
        out_override: Option<Arg>,
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
        //
        // `stated_in` and not `sig_in`: the row is one of two planes a
        // backend can state this from, and Metal's is the routine registry
        // now. A table lookup answers `None` for every Metal symbol, which
        // reads as "nothing said it was whole" and lets exactly the split
        // this refuses through.
        if let Some(sig) = kernels::stated_in(backend, kernel)
            && sig.whole
            && (window.start != 0 || window.end != self.rows.len() as u32)
        {
            return Err(Uncovered::WholeKernelSplit {
                at_op: at,
                kernel: kernel.to_string(),
                rows: window.clone(),
            });
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
        // A statement inside a value-producing region states no result of its
        // own and binds the REGION's -- see `Self::region_outs`.
        //
        // "States no result" is not the same as "has none to state", and
        // the difference is STATE. A statement that names a `StateRef`
        // writes the KV pages or the recurrent slabs, which outlive the
        // fire and have no SSA value; producing nothing is what it IS,
        // not evidence that a region owns its output.
        //
        // `attn::dequant_kv_cache_layer_to_bf16_active` is the one that
        // found this. It stages a quantized cache before a prefill
        // dispatch, names `kv_state` and declares no result -- and inside
        // the attention's value-producing guard it was handed the guard's
        // output as a fourth operand, so its row's arity check refused
        // every fire that took the quantized path.
        let outs: &[ValueId] = if op.outputs.is_empty()
            && op.kind.state_ref().is_none()
            && !self.region_outs.is_empty()
        {
            &self.region_outs
        } else {
            &op.outputs
        };
        for (i, &v) in op.inputs.iter().enumerate() {
            match (i, in_override.clone()) {
                (0, Some(a)) => self.args.push(a),
                _ => self.args.push(self.slot(v)),
            }
        }
        for (i, &v) in outs.iter().enumerate() {
            match (i, out_override.clone()) {
                (0, Some(a)) => self.args.push(a),
                _ => self.args.push(self.slot(v)),
            }
        }
        let first_param = self.params.len() as u32;
        if let OpKind::Launch {
            weights,
            params,
            param_extents,
            ..
        } = &op.kind
        {
            for name in weights {
                self.args.push(Arg::Weight(name.clone()));
            }
            self.params.extend_from_slice(params);
            // The scalars that are extents. Written over the constants the
            // text left, because the fire is what decides them and only the
            // lowering knows the fire.
            for (i, shape) in param_extents {
                let at = first_param as usize + *i as usize;
                if let Some(slot) = self.params.get_mut(at) {
                    *slot = u32::try_from(shape_elements(
                        shape,
                        self.rows.len(),
                        self.buffers.n_requests as usize,
                    ))
                    .unwrap_or(u32::MAX);
                }
            }
        }
        self.launches.push(Launch {
            kernel: id,
            rows: self.rectangle_rows(op, window),
            layers: layer..layer + 1,
            op: at as u32,
            args: first..self.args.len() as u32,
            params: first_param..self.params.len() as u32,
            peel: self.peel_region,
            cond: self.cond,
        });
        Ok(())
    }

    /// THE RECTANGLE'S ROW AXIS, which is the row space of the value the
    /// statement WRITES and not the fire's stream.
    ///
    /// [`Launch::rows`]' own doc already says this — "the leading dim is the
    /// row axis (`Tokens` for the body, `Requests` for the epilogue)" — and
    /// for every statement in the body the two are the same range, so nothing
    /// had ever forced the distinction. A statement writing a
    /// `[Requests, ..]` value is where they differ, and the cost of not
    /// distinguishing them was measured rather than argued.
    ///
    /// Metal's llama-like text spells its own readout: `sample_rows` compacts
    /// the sampled rows and `lm_head` projects them. Both are ordinary
    /// `OpKind::Launch` statements, so both took the fire's window — and the
    /// head's launch rule reads that window as its M. On a 2048-token prefill
    /// of Llama-3.2-1B that made the head a 2048-row matvec against a 128256
    /// vocabulary when ONE row is sampled: 904 ms of a 2184 ms prefill, 41%
    /// of the whole fire, computing 2047 distributions nobody reads. Correct
    /// output — row 0 is the one sampling takes — and 2048x the work.
    ///
    /// Narrowed only from the FULL window, so a caller that has already
    /// picked a subrange keeps it: `epilogue` passes `0..sampled`, which is
    /// the same count arrived at from the row flags rather than from the
    /// buffer plan, and a rule that overwrote it would replace a measured
    /// number with an equal one for no reason and differ if they ever parted.
    fn rectangle_rows(&self, op: &Op, window: &Range<u32>) -> Range<u32> {
        let full = window.start == 0 && window.end == self.rows.len() as u32;
        let requests_shaped = op
            .outputs
            .first()
            .and_then(|&v| self.plan.values.get(v as usize))
            .and_then(|info| info.shape.0.first())
            .is_some_and(|dim| matches!(dim, Dim::Requests));
        if full && requests_shaped {
            return 0..self.buffers.n_requests;
        }
        window.clone()
    }

    /// Add a conditional's TWO arms to the guard tree, paired, and return
    /// `(if_arm, else_arm)`.
    ///
    /// Always in pairs: an arm without its sibling is a node the driver
    /// cannot open a two-bodied conditional for.
    fn push_cond_pair(&mut self, parent: u32, slot: u32, param: u32) -> (u32, u32) {
        let t = self.conds.len() as u32;
        let f = t + 1;
        self.conds.push(CondRegion {
            parent,
            slot,
            param,
            on_true: true,
            sibling: f,
        });
        self.conds.push(CondRegion {
            parent,
            slot,
            param,
            on_true: false,
            sibling: t,
        });
        (t, f)
    }

    /// Where a value's bytes are, how wide a row of it is, and how many
    /// bytes one element takes.
    ///
    /// # A NAMED operand is emitted under its alias OWNER
    ///
    /// An in-place kernel is one pointer read and written, and
    /// [`alias_owners`] is the table that says which values therefore have
    /// to be the same bytes. The arena already obeys it — `Buffers::assign`
    /// gives every member of a group one offset, so two aliased arena
    /// operands come out of the match below as the same `at` and nothing
    /// downstream has to know a group existed.
    ///
    /// `Buffers::NAMED` is not an offset, it is a SENTINEL meaning "the
    /// backend binds this one", and it carried the value id through
    /// unchanged — so the alias survived for arena values and was **lost
    /// for named ones**. A backend then did the obvious correct thing with
    /// what it was given: one buffer per distinct id. `qk_rmsnorm_rope`
    /// reads and writes a single `q` pointer, its in-place pair says output
    /// 0 IS input 0, and it was handed two different buffers — so it normed
    /// the fresh zeroed output over itself and wrote zeros back, every
    /// layer, and attention ran on a zero query. That is what this line
    /// fixes, and it is the whole of it.
    ///
    /// The WIDTH stays this value's own. The owner says which bytes an
    /// operand lives in; it does not say how this statement reads them, and
    /// an in-place pair whose two ends have different widths is reading one
    /// buffer two ways on purpose.
    fn slot(&self, v: ValueId) -> Arg {
        let width = self.row_width(v);
        let bytes = self
            .plan
            .values
            .get(v as usize)
            .map_or(2, |info| dtype_bytes(info.dtype));
        match self.buffers.offset.get(v as usize) {
            Some(&Buffers::NAMED) | None => Arg::Named {
                value: self.value_owner.get(v as usize).copied().unwrap_or(v),
                width,
                bytes,
            },
            Some(&at) => Arg::Arena { at, width, bytes },
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
            // THE GATHER NOW HAS SOMEWHERE TO WRITE.
            //
            // It used to be emitted through the plain `emit`, which binds
            // the OP's operands -- and `OpKind::LmHead` states
            // `inputs=[hidden] outputs=[logits]`. So the gather, whose job
            // is to compact `[sampled, hidden]` for the head to read, was
            // handed the LOGITS buffer as its destination: the wrong
            // width, and the head then read what it had overwritten. It
            // produced all-zero logits on gemma-4 and the hybrid, which is
            // why the shell forces `samples: true` on every row and pays a
            // prefill's head over every token rather than one per request.
            //
            // The scratch has existed all along -- `Buffers::epilogue_gather`
            // is sized from this very statement and carried on `Lowered`.
            // Nothing ever bound it. So the gather is emitted with its
            // output REPLACED by that block, and the head's input with it,
            // which is the whole of what "the epilogue names its temp"
            // means.
            //
            // A fire whose planner refused the block (`NAMED`) keeps the
            // old binding rather than inventing an address: that is a
            // fire the caller must not ask a gather of, and `samples` is
            // the shell's to over-claim.
            let temp = (self.buffers.epilogue_gather != Buffers::NAMED).then(|| Arg::Arena {
                at: self.buffers.epilogue_gather,
                width: self.row_width(op.inputs.first().copied().unwrap_or(0)),
                bytes: op
                    .inputs
                    .first()
                    .and_then(|&v| self.plan.values.get(v as usize))
                    .map_or(2, |i| dtype_bytes(i.dtype)),
            });
            self.emit_with_out(at, "layout::gather_bf16_rows", op, &out, temp.clone())?;
            // The head reads the compacted rows, not the raw stream.
            self.emit_with_in(at, "gemm::act_x_w", op, &out, temp)?;
            return Ok(());
        }
        // NO FINAL NORM HERE, and its absence is the correction rather
        // than an omission.
        //
        // This used to emit `norm::rmsnorm_bf16` between the two, and it
        // was dead on every fire of every family. Each text applies the
        // final norm ITSELF and hands `logits()` the normed value —
        // `m.logits(&model_dsl::cuda::rmsnorm(&y, &m.final_norm()))` in
        // llama_like, `lm_head_tied(t, &normed, ..)` in gemma-4,
        // `lm_head(rmsnorm(y, final_norm))` in qwen3.5. So the epilogue's
        // own norm read an already-normed input and wrote `rows x hidden`
        // bf16 into the LOGITS buffer, which the projection below
        // overwrites on the next launch (beta is 0 here — the accumulate
        // form needs three operands and this op has two).
        //
        // It survived because the epilogue's test asserted the three
        // SYMBOLS and their row ranges, and those were right; nothing
        // read the arguments. It is residue from before the texts stated
        // their own norm.
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
            // `k == 0` is false rather than a division: a guard that named a
            // zero tile would otherwise trap here, and a tile of no rows is a
            // statement error rather than a fire property.
            GuardPred::TokensMultipleOf(k) => k != 0 && (self.rows.len() as u32).is_multiple_of(k),
            // The window CLASS, as a row property. `FireClass::Decode`
            // meant exactly this and could not say it; a mixed fire
            // answers false and takes the ragged arm, which serves a
            // one-token request as its degenerate case.
            GuardPred::WindowOne => !self.rows.iter().any(|r| r.multi_token),
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
    fn split_at(&self, window: &Range<u32>, axis: PeelWindow, at: usize) -> Result<u32, Uncovered> {
        let (name, marked): (&'static str, fn(&Row) -> bool) = match axis {
            PeelWindow::HookFreePrefix => ("hook", |r| r.hooked),
            PeelWindow::UnmaskedPrefix => ("mask", |r| r.custom_mask),
        };
        let tail = contiguous(self.rows, window, marked, name, at)?;
        // The marked rows are the SUFFIX; anything else means this order
        // and this trace disagree.
        if !tail.is_empty() && tail.end != window.end {
            return Err(Uncovered::Discontiguous {
                at_op: at,
                axis: name,
            });
        }
        Ok(if tail.is_empty() {
            window.end
        } else {
            tail.start
        })
    }
}
