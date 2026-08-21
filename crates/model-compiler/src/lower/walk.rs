//! The lowering walk: row windows to launch rectangles.

use super::semantics::{Semantic, contiguous, kind_name, semantic, subtract};
use super::*;

/// Lower `plan` over `rows`, in the order the engine seriated them.
pub fn lower(plan: &ForwardPlan, rows: &[Row], fire: Fire) -> Result<Lowered, Uncovered> {
    lower_with(plan, rows, fire, GuardMode::Resolve)
}

/// Union mode preserves guard arms as condition regions.
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
        preps: Vec::new(),
        args: Vec::new(),
        arg_rows: Vec::new(),
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
        arg_rows: out.arg_rows,
        params: out.params,
        structural: out.structural,
        preps: out.preps,
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
    /// Inside a peel tail region, where statements address full-window rows.
    peel_tail: bool,
    residue: Vec<Unlowered>,
    fire: Fire,
    structural: Vec<Site>,
    preps: Vec<Prep>,
    /// The operand slots emitted so far.
    args: Vec<Arg>,
    arg_rows: Vec<u32>,
    params: Vec<u32>,
    /// Arena offsets for operands as they are emitted.
    buffers: Buffers,
    /// Alias groups for named operands; arena aliases already share offsets.
    value_owner: Vec<ValueId>,
    /// The peel region the launches being emitted sit in.
    peel_region: Option<PeelRegion>,
    /// Whether guards are answered or kept.
    guards: GuardMode,
    /// Current conditional region and the tree it indexes.
    cond: u32,
    /// The enclosing value-producing region's outputs, when there is one.
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
                    let outer_outs = std::mem::take(&mut self.region_outs);
                    self.region_outs = op.outputs.clone();
                    // A guard arm takes the whole window when its fire predicate holds.
                    let mut at = i + 1;
                    if self.guards == GuardMode::Union {
                        // Score predicates need prepared state, so union mode answers them.
                        fn folds(p: GuardPred) -> bool {
                            !matches!(p, GuardPred::WantsAttnScore)
                        }
                        // In union mode, foldable arms lower over the whole window and carry condition ids.
                        let outer = self.cond;
                        let mut parent = outer;
                        for arm in arms {
                            let body = at..at + arm.ops as usize;
                            at += arm.ops as usize;
                            if !folds(arm.pred) {
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
                    let split = self.split_at(&window, *axis, i)?;
                    let prefix = window.start..split;
                    let tail = split..window.end;
                    let p = i + 1..i + 1 + *prefix_ops as usize;
                    let t = p.end..p.end + *tail_ops as usize;
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
                    // The one fire-dependent symbol swap the DSL cannot
                    // state: a peel tail's split reads through the device
                    // window. Trace-time cannot know which region a fire
                    // peels, so the lowering keeps this single remap.
                    let kernel = if self.peel_tail && kernel == "attn::split_qkv_bf16" {
                        "attn::split_qkv_bf16_devwin"
                    } else {
                        kernel.as_str()
                    };
                    self.emit(i, kernel, op, &live)?;
                    i += 1;
                }
                OpKind::LmHead { .. } => {
                    self.epilogue(i, op, &window)?;
                    i += 1;
                }
                kind => {
                    match semantic(kind, self.peel_tail) {
                        // A prep is stated once for the fire, so it takes
                        // no row window: an empty rectangle would drop the
                        // schedule the launches below are planned against.
                        Semantic::Prep(kind) => {
                            // `outputs[0]` or nothing: a prep publishes exactly
                            // one value, and a trace whose prep publishes none
                            // is one built before this existed.
                            let Some(&value) = op.outputs.first() else {
                                return Err(Uncovered::UnknownBackend(format!(
                                    "a prep at op {i} that publishes no value"
                                )));
                            };
                            self.preps.push(Prep {
                                at_op: i as u32,
                                kind,
                                value,
                            });
                        }
                        Semantic::Structural => {
                            // Structural sites use the same depth window as launches.
                            let live = self.depth_window(op, &window, i)?;
                            if !live.is_empty() {
                                self.structural.push(Site {
                                    at_op: i as u32,
                                    rows: live,
                                });
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
        self.emit_bound(at, kernel, op, window, None, None, None)
    }

    /// The epilogue emits only sampled rows; gather uses scratch when sampling a subset.
    fn emit_with_out(
        &mut self,
        at: usize,
        kernel: &str,
        op: &Op,
        window: &Range<u32>,
        out: Option<Arg>,
    ) -> Result<(), Uncovered> {
        self.emit_bound(at, kernel, op, window, None, out, None)
    }

    // `emit_with_in` STOOD HERE. Its one caller was the epilogue's split leg,
    // which now also has to CAP the statement's inputs at one, so it calls
    // `emit_bound` directly; a helper that takes two of the three overrides is
    // no shorter than the call it wraps. RETIRED.

    fn emit_bound(
        &mut self,
        at: usize,
        kernel: &str,
        op: &Op,
        window: &Range<u32>,
        in_override: Option<Arg>,
        out_override: Option<Arg>,
        in_cap: Option<usize>,
    ) -> Result<(), Uncovered> {
        if window.is_empty() && !self.peel_region.is_some_and(|r| r.rows_device) {
            return Ok(());
        }
        let backend = self
            .backend
            .ok_or_else(|| Uncovered::UnknownBackend(self.plan.family.clone()))?;
        // `whole` kernels may only cover the full fire.
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
        // Unrolled trace ops cover one layer; rolled drivers still read a range.
        let layer = op.layer.unwrap_or(0) as u16;
        // Operand order is inputs, outputs, then weights.
        let first = self.args.len() as u32;
        // Value-producing regions bind their arms to the region outputs.
        let outs: &[ValueId] = if op.outputs.is_empty() && !op.dest.is_empty() {
            &op.dest
        } else {
            &op.outputs
        };
        let ins = &op.inputs[..in_cap.unwrap_or(op.inputs.len()).min(op.inputs.len())];
        for (i, &v) in ins.iter().enumerate() {
            match (i, in_override.clone()) {
                (0, Some(a)) => self.args.push(a),
                _ => self.args.push(self.slot(v)),
            }
            self.arg_rows.push(self.value_rows(v));
        }
        for (i, &v) in outs.iter().enumerate() {
            match (i, out_override.clone()) {
                (0, Some(a)) => self.args.push(a),
                _ => self.args.push(self.slot(v)),
            }
            self.arg_rows.push(self.value_rows(v));
        }
        let first_param = self.params.len() as u32;
        if let OpKind::Launch {
            weights,
            params,
            param_extents,
            peel_slots,
            ..
        } = &op.kind
        {
            for name in weights {
                self.args.push(Arg::Weight(name.clone()));
                self.arg_rows.push(0);
            }
            self.params.extend_from_slice(params);
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
            // THE REMAPPED SPLIT'S RUN. `attn::split_qkv_bf16_devwin` is
            // reached by the remap above, so its statement is the PLAIN
            // split's — no params at all — while the routine reads
            // `[n_max, win_start, win_len]`. The walk supplies the trio the
            // way it supplies every peel window: from its own knowledge.
            if kernel == "attn::split_qkv_bf16_devwin" && self.params.len() == first_param as usize
            {
                let w = self.rectangle_rows(op, window);
                self.params.push(self.rows.len() as u32);
                self.params.push(w.start);
                self.params.push(w.end.saturating_sub(w.start));
            }
            // THE PEEL WINDOW, WHICH ONLY THIS WALK CAN SAY. The launch's
            // own rectangle IS the region window it runs under — the tail's
            // split inside a peel, `(0, N)` on an unpeeled fire — and the
            // statement carried zeros at these slots because no statement
            // can state a fire's split.
            if let Some((s_at, l_at)) = peel_slots {
                let window = self.rectangle_rows(op, window);
                if let Some(slot) = self.params.get_mut(first_param as usize + *s_at as usize) {
                    *slot = window.start;
                }
                if let Some(slot) = self.params.get_mut(first_param as usize + *l_at as usize) {
                    *slot = window.end.saturating_sub(window.start);
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

    /// One value's OWN row count, which is not always the launch's.
    ///
    /// The same three cases [`Self::rectangle_rows`] answers, asked of any
    /// operand rather than only the first output and without the "the launch
    /// covers the whole window" precondition -- an operand's row space is a
    /// property of the value, and a partial window does not shrink it.
    fn value_rows(&self, v: ValueId) -> u32 {
        let n = u32::try_from(self.rows.len()).unwrap_or(u32::MAX);
        let Some(info) = self.plan.values.get(v as usize) else {
            return 0;
        };
        match info.shape.0.first() {
            Some(Dim::Requests) => self.buffers.n_requests,
            Some(Dim::Tokens) => n,
            Some(&Dim::MoeAlignedRoutes {
                top_k,
                experts,
                block,
            }) => Dim::moe_aligned_rows(n, top_k, experts, block),
            Some(Dim::Const(v)) => *v,
            _ => 0,
        }
    }

    /// Rows are in the written value's row space, not always the token stream.
    fn rectangle_rows(&self, op: &Op, window: &Range<u32>) -> Range<u32> {
        let full = window.start == 0 && window.end == self.rows.len() as u32;
        let first = op
            .outputs
            .first()
            .and_then(|&v| self.plan.values.get(v as usize))
            .and_then(|info| info.shape.0.first());
        if full {
            match first {
                Some(Dim::Requests) => return 0..self.buffers.n_requests,
                Some(&Dim::MoeAlignedRoutes {
                    top_k,
                    experts,
                    block,
                }) => {
                    return 0..Dim::moe_aligned_rows(
                        self.rows.len().try_into().unwrap_or(u32::MAX),
                        top_k,
                        experts,
                        block,
                    );
                }
                _ => {}
            }
        }
        window.clone()
    }

    /// Condition regions are pushed in if/else pairs so drivers can rebuild a two-arm node.
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

    /// Named values are emitted under their alias owner; arena aliases already share an offset.
    fn slot(&self, v: ValueId) -> Arg {
        // A RAISE IS NOT A RECTANGLE, so it is answered before anything
        // measures one. `row_width` would fold an empty shape to 1 and
        // `dtype_bytes` would read the arbitrary dtype stored beside it --
        // two numbers that mean nothing, on an argument that has no rows.
        if let Some(key) = self
            .plan
            .values
            .get(v as usize)
            .and_then(|info| info.raised.as_deref())
        {
            return Arg::Raised {
                value: v,
                key: key.to_string(),
            };
        }
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

    /// Row width is the product after the leading row axis; symbolic widths return zero.
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

    /// The epilogue as rectangles rather than a branch.
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
            self.emit_bound(at, "gemm::act_x_w", op, &out, temp, None, Some(1))?;
            return Ok(());
        }
        // No final norm: texts already hand `LmHead` a normed value.
        //
        // ONE INPUT, NOT TWO. `LmHead` states a second input for the gather's
        // sake (the row list the split leg collects); the projection's own
        // routine has no mark for it, and the driver answers the name with
        // null when a fire samples every row -- so binding it here would
        // refuse the launch that needs it least.
        self.emit_bound(at, "gemm::act_x_w", op, &out, None, None, Some(1))?;
        Ok(())
    }

    /// Depth-windowed rows must form a prefix of the current window.
    fn depth_window(
        &self,
        op: &Op,
        window: &Range<u32>,
        at: usize,
    ) -> Result<Range<u32>, Uncovered> {
        // Non-windowed or unlayered ops keep the current window.
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

    /// Guard predicates are fire facts: true means the whole window, false means empty.
    fn select(&self, window: &Range<u32>, pred: GuardPred) -> Range<u32> {
        let holds = match pred {
            GuardPred::HasCustomMask => self.rows.iter().any(|r| r.custom_mask),
            GuardPred::HasLora => self.rows.iter().any(|r| r.lora),
            GuardPred::HasStageHooks => self.rows.iter().any(|r| r.hooked),
            GuardPred::WantsAttnScore => self.rows.iter().any(|r| r.wants_scores),
            GuardPred::HasWriteDesc => self.rows.iter().any(|r| r.write_desc),
            // Token thresholds read the fire row count.
            GuardPred::TokensLE(k) => self.rows.len() as u32 <= k,
            GuardPred::TokensGT(k) => self.rows.len() as u32 > k,
            // A zero multiple predicate is false rather than a division.
            GuardPred::TokensMultipleOf(k) => k != 0 && (self.rows.len() as u32).is_multiple_of(k),
            GuardPred::WindowOne => !self.rows.iter().any(|r| r.multi_token),
        };
        if holds {
            window.clone()
        } else {
            window.start..window.start
        }
    }

    /// Where a peel axis splits `window`; unmarked rows are the prefix.
    fn split_at(&self, window: &Range<u32>, axis: PeelWindow, at: usize) -> Result<u32, Uncovered> {
        let (name, marked): (&'static str, fn(&Row) -> bool) = match axis {
            PeelWindow::HookFreePrefix => ("hook", |r| r.hooked),
            PeelWindow::UnmaskedPrefix => ("mask", |r| r.custom_mask),
        };
        let tail = contiguous(self.rows, window, marked, name, at)?;
        // Peel-marked rows must be the suffix of the window.
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
