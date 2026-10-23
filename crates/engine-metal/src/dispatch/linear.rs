use kernels_metal::linear;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Fused, Linear, Operands};

use crate::run::{Run, TiledFold};

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    pub(super) fn fused_linear(&mut self, op: &Fused) -> Result<(), kernels_metal::Error> {
        // The shared expert rides along on the routed register tiles: its rows
        // are tile 0 of their down, which the gated add reads.
        if let Fused::MatmulSwigluMatmul { y, .. } = op
            && self.tiles_shared(*y)
        {
            return Ok(());
        }
        match op {
            Fused::MatmulSwigluMatmul {
                act,
                gate_up,
                down,
                intermediate,
                packed,
                h,
                y,
            } => match self
                .ane()
                .and_then(|ane| {
                    let head = ane.head_rows(self.tensor(*act).rows);
                    Some((ane, ane.plan(*gate_up, head)?))
                })
                .filter(|(_, plan)| self.split_fits(plan, *act, *h, *y))
            {
                Some((ane, plan)) => self.mlp_split(
                    ane,
                    &plan,
                    *act,
                    (*gate_up, *down, *intermediate),
                    *packed,
                    *h,
                    *y,
                ),
                None => {
                    // On the register-fed tile the swiglu runs in the gate and up's launch.
                    if self.coop_swiglu(*act, *gate_up, *intermediate, *h)? {
                        return self.linear(&Linear::Matmul {
                            act: *h,
                            w: *down,
                            y: *y,
                            input_scales: None,
                            scales: None,
                            quant: None,
                            logits: false,
                        });
                    }
                    self.linear(&Linear::Matmul {
                        act: *act,
                        w: *gate_up,
                        y: *packed,
                        input_scales: None,
                        scales: None,
                        quant: None,
                        logits: false,
                    })?;
                    // The down on the register tile reads its input table as
                    // the swiglu lays it.
                    let laid = self
                        .register_counters(*h, *down, self.tensor(*y).width)
                        .and_then(|counters| self.laid_table(*h, counters));
                    match laid {
                        Some((table, counters)) => linear::mlp::swiglu_table(
                            self.ctx(),
                            self.tensor(*packed),
                            *intermediate,
                            self.tensor(*h),
                            table,
                            counters,
                        )?,
                        None => self.linear(&Linear::Glu {
                            packed: Some(*packed),
                            intermediate: *intermediate,
                            y: *h,
                            gate: None,
                            weights: None,
                            limit: None,
                            act: poem_ir::GluAct::Swiglu,
                            up: None,
                        })?,
                    }
                    self.linear(&Linear::Matmul {
                        act: *h,
                        w: *down,
                        y: *y,
                        input_scales: None,
                        scales: None,
                        quant: None,
                        logits: false,
                    })
                }
            },
            _ => Err(kernels_metal::Error::Unsupported { op: op.name() }),
        }
    }

    /// The gate and up of `act` and their swiglu into `h` as one launch, where
    /// the gate-up matmul takes the register-fed tile whole and runs on the GPU
    /// alone: `h` lands as the matmul and the swiglu land it, and the packed
    /// plane between them is left unwritten (nothing past the MLP reads it).
    /// False where they run as their own launches.
    fn coop_swiglu(
        &self,
        act: poem_ir::ValueId,
        gate_up: poem_ir::ValueId,
        intermediate: u32,
        h: poem_ir::ValueId,
    ) -> Result<bool, kernels_metal::Error> {
        let Some(bank) = self.banked(gate_up) else {
            return Ok(false);
        };
        let x = self.tensor(act);
        if self
            .ane()
            .is_some_and(|ane| ane.plan_columns(gate_up, x.rows).is_some())
        {
            return Ok(false);
        }
        linear::qmm::staged::coop_swiglu(
            self.ctx(),
            self.tuning(),
            x,
            bank,
            intermediate,
            self.tensor(h),
            self.capacity(act).min(self.capacity(h)),
        )
    }

    /// Whether the plan's rows, padded to the matmul tile, fit the planes
    /// this call writes and the precast the GPU's share needs.
    fn split_fits(
        &self,
        plan: &crate::ane::Plan,
        act: poem_ir::ValueId,
        h: poem_ir::ValueId,
        y: poem_ir::ValueId,
    ) -> bool {
        let padded = plan.rows.div_ceil(32) * 32;
        padded
            <= self
                .capacity(act)
                .min(self.capacity(h))
                .min(self.capacity(y))
            && self.precast(padded, plan.split.keep).is_some()
    }

    /// Whether a column plan's rows, padded to the matmul tile, fit the
    /// planes this call reads and writes.
    fn columns_fit(
        &self,
        plan: &crate::ane::Columns,
        act: poem_ir::ValueId,
        y: poem_ir::ValueId,
    ) -> bool {
        plan.rows.div_ceil(32) * 32 <= self.capacity(act).min(self.capacity(y))
    }

    /// The GPU's columns of a split projection, between the hand-off and
    /// the join: the leading `keep` landed in place, the rest the Neural
    /// Engine's.
    fn matmul_split(
        &self,
        ane: &crate::ane::Ane,
        plan: &crate::ane::Columns,
        act: poem_ir::ValueId,
        y: poem_ir::ValueId,
    ) -> Result<(), kernels_metal::Error> {
        let x = self.tensor(act);
        let out = self.tensor(y);
        ane.before_columns(self.ctx(), plan, x)?;
        linear::qmm::matmul_columns(
            self.ctx(),
            self.tuning(),
            x,
            plan.view,
            out,
            plan.keep,
            linear::qmm::Scratch {
                precast: &|rows, contraction| self.precast(rows, contraction),
                partials: &|rows, width| self.partials(rows, width),
                table: None,
            },
            self.capacity(act).min(self.capacity(y)),
        )?;
        ane.after_columns(self.ctx(), plan, out)
    }

    /// The whole MLP on the GPU over the rows of `act` past the `head` the
    /// Neural Engine shares, landed in the same planes' rows while the
    /// Neural Engine runs the head.
    #[allow(clippy::too_many_arguments)]
    fn mlp_tail(
        &self,
        head: u32,
        act: poem_ir::ValueId,
        gate_up: poem_ir::ValueId,
        down: poem_ir::ValueId,
        intermediate: u32,
        packed: poem_ir::ValueId,
        h: poem_ir::ValueId,
        y: poem_ir::ValueId,
    ) -> Result<(), kernels_metal::Error> {
        let (Some(gate_up), Some(down)) = (self.banked(gate_up), self.banked(down)) else {
            return Err(kernels_metal::Error::Unsupported {
                op: "fused.matmul_swiglu_matmul",
            });
        };
        let tail = self.tensor(act).rows - head;
        let at = |id: poem_ir::ValueId| self.slice(self.tensor(id), head, tail);
        let room = |id: poem_ir::ValueId| self.capacity(id).saturating_sub(head);
        let precast = |rows, contraction| self.precast(rows, contraction);
        let partials = |rows, width| self.partials(rows, width);
        let scratch = || linear::qmm::Scratch {
            precast: &precast,
            partials: &partials,
            table: None,
        };
        let (x, gate, hidden, out) = (at(act), at(packed), at(h), at(y));
        linear::qmm::matmul(
            self.ctx(),
            self.tuning(),
            x,
            gate_up,
            gate,
            scratch(),
            room(act).min(room(packed)),
        )?;
        linear::mlp::swiglu(self.ctx(), gate, intermediate, hidden)?;
        linear::qmm::matmul(
            self.ctx(),
            self.tuning(),
            hidden,
            down,
            out,
            scratch(),
            room(h).min(room(y)),
        )
    }

    /// The GPU's share of a split MLP, between the hand-off and the join:
    /// gate and up over its leading `keep` channels, written compactly into
    /// the packed and staging planes, swiglu, and down over those channels
    /// into `y`, which the join then adds the Neural Engine's partial to.
    /// Rows past the plan's run the whole MLP on the GPU meanwhile.
    #[allow(clippy::too_many_arguments)]
    fn mlp_split(
        &self,
        ane: &crate::ane::Ane,
        plan: &crate::ane::Plan,
        act: poem_ir::ValueId,
        (gate_up, down, intermediate): (poem_ir::ValueId, poem_ir::ValueId, u32),
        packed: poem_ir::ValueId,
        h: poem_ir::ValueId,
        y: poem_ir::ValueId,
    ) -> Result<(), kernels_metal::Error> {
        let keep = plan.split.keep;
        let x = kernels_metal::Tensor {
            rows: plan.rows,
            ..self.tensor(act)
        };
        let out = kernels_metal::Tensor {
            rows: plan.rows,
            ..self.tensor(y)
        };
        let gate = kernels_metal::Tensor {
            width: keep,
            rows: plan.rows,
            ..self.tensor(packed)
        };
        let h_gpu = kernels_metal::Tensor {
            width: keep,
            rows: plan.rows,
            ..self.tensor(h)
        };
        ane.before(self.ctx(), plan, x)?;
        let precast = |rows, contraction| self.precast(rows, contraction);
        let partials = |rows, width| self.partials(rows, width);
        let scratch = || linear::qmm::Scratch {
            precast: &precast,
            partials: &partials,
            table: None,
        };
        let rows = self.capacity(act);
        linear::qmm::matmul(
            self.ctx(),
            self.tuning(),
            x,
            plan.split.gate,
            gate,
            scratch(),
            rows.min(self.capacity(packed)),
        )?;
        linear::qmm::matmul(
            self.ctx(),
            self.tuning(),
            x,
            plan.split.up,
            plan.up_rows,
            scratch(),
            rows,
        )?;
        linear::mlp::swiglu_clamp_split(self.ctx(), gate, plan.up_rows, f32::MAX, h_gpu)?;
        linear::qmm::matmul(
            self.ctx(),
            self.tuning(),
            h_gpu,
            plan.split.down,
            out,
            scratch(),
            self.capacity(h).min(self.capacity(y)),
        )?;
        if plan.rows < self.tensor(act).rows {
            self.mlp_tail(plan.rows, act, gate_up, down, intermediate, packed, h, y)?;
        }
        ane.after(self.ctx(), plan, out)
    }

    /// The matmuls of `act` the trace places right after the one by `w` into
    /// `y` (a GDN layer's in-projection then its gates; a query, key and
    /// value, the query's split into query and gate between the query's and
    /// the key's) in one register-tile launch with it, when every one of them
    /// runs on the register tile, all in the region the walk is in, and their
    /// outputs stand clear of each other and of `act`: each lands as its own
    /// launch would land it, bit for bit, and the later ones' nodes find it
    /// done ([`Run::landed_ahead`]). A projection whose only reader is a split
    /// of its rows in two (the GDN in-projection's convolved half and its
    /// gate) lands its halves where the split would copy them, as two
    /// segments of its bank at its own split, and the split finds them done.
    /// False where they run as their own launches.
    fn matmul_segments(
        &self,
        act: poem_ir::ValueId,
        w: poem_ir::ValueId,
        y: poem_ir::ValueId,
    ) -> Result<bool, kernels_metal::Error> {
        use linear::qmm::sg::{Segment, register_split};
        use poem_ir::Operation;
        let Some((nodes, one_pass)) = self.region_nodes() else {
            return Ok(false);
        };
        let op = |at: u32| self.nodes().get(at as usize).map(|n| &n.op);
        let matmul = |at: u32| match op(at) {
            Some(Operation::Linear(Linear::Matmul {
                act: a,
                w,
                y,
                quant: None,
                logits: false,
                ..
            })) if *a == act => Some((*w, *y)),
            _ => None,
        };
        let Some(first) = self
            .readers_of(act)
            .iter()
            .copied()
            .find(|&at| matmul(at) == Some((w, y)))
        else {
            return Ok(false);
        };
        let mut members = vec![(w, y)];
        // What a node between the matmuls writes: a query's split into query
        // and gate, between the query's projection and the key's.
        let mut between: Vec<poem_ir::ValueId> = Vec::new();
        let (mut at, mut last) = (first + 1, first);
        while members.len() < 3 && nodes.contains(&at) {
            if let Some(next) = matmul(at) {
                members.push(next);
                last = at;
            } else {
                match op(at) {
                    Some(Operation::Gather(poem_ir::Gather::SplitQGate {
                        packed,
                        q,
                        gate,
                        ..
                    })) if between.is_empty() && members.iter().any(|m| m.1 == *packed) => {
                        between.extend([*q, *gate]);
                    }
                    _ => break,
                }
            }
            at += 1;
        }
        if members.len() < 2 || !nodes.contains(&first) {
            return Ok(false);
        }
        let x = self.tensor(act);
        let (m, k) = (x.rows, x.width);
        // Where the staged rung takes the widest member and the members fill the device
        // together, the register-fed tile runs them all whole.
        let widths: Vec<u32> = members.iter().map(|&(_, y)| self.tensor(y).width).collect();
        let coop = x.dtype == poem_ir::Dtype::Bf16
            && members.iter().all(|&(w, _)| {
                self.banked(w)
                    .is_some_and(|b| b.bits == 4 && b.group == 64 && b.biases.is_some())
            })
            && linear::qmm::staged::coop_segments_fit(self.tuning(), m, &widths, k);
        let mut segments = Vec::with_capacity(3);
        // The left half of the split whose halves a segment pair lands.
        let mut halves = None;
        for (i, &(w, y)) in members.iter().enumerate() {
            let out = self.tensor(y);
            let fits = (if coop {
                out.width.is_multiple_of(32)
            } else {
                self.register_counters(act, w, out.width).is_some()
            }) && out.rows == m
                && self.ane().and_then(|ane| ane.plan_columns(w, m)).is_none();
            let Some(bank) = self.banked(w).filter(|b| fits && b.ld % 256 == 0) else {
                return Ok(false);
            };
            let split = register_split(
                i32::try_from(out.width).unwrap_or(0),
                i32::try_from(k).unwrap_or(0),
                self.tuning().cores(),
            );
            // The split of this output's rows, its only reader, in this
            // region after the matmuls, into halves of whole column tiles.
            let split_rows = self
                .readers_of(y)
                .iter()
                .copied()
                .find_map(|at| match op(at) {
                    Some(Operation::Gather(poem_ir::Gather::SplitRows {
                        x,
                        width,
                        left,
                        right,
                    })) if *x == y
                        && self.readers_of(y).len() == 1
                        && nodes.contains(&at)
                        && at > last
                        && width % 64 == 0
                        && *width < out.width =>
                    {
                        Some((*width, *left, *right))
                    }
                    _ => None,
                });
            match split_rows {
                Some((width, left, right))
                    if one_pass && halves.is_none() && members.len() < 3 && i == 0 =>
                {
                    let (l, r) = (self.tensor(left), self.tensor(right));
                    if l.width != width
                        || r.width != out.width - width
                        || l.rows != m
                        || r.rows != m
                    {
                        return Ok(false);
                    }
                    segments.push(Segment {
                        w: bank,
                        first: 0,
                        y: l,
                        split,
                    });
                    segments.push(Segment {
                        w: bank,
                        first: width,
                        y: r,
                        split,
                    });
                    halves = Some(left);
                }
                _ => segments.push(Segment {
                    w: bank,
                    first: 0,
                    y: out,
                    split,
                }),
            }
        }
        if segments.len() > 3 {
            return Ok(false);
        }
        // The narrow segments' threadgroups first, so they run in the first
        // wave rather than trail the wide ones'.
        segments.sort_by_key(|s| s.y.width);
        // Every output its own bytes, apart from each other and the activation.
        let span = |t: kernels_metal::Tensor| {
            self.address(t.buf)
                .map(|(slab, at)| (slab, at, at + u64::from(t.rows) * u64::from(t.width) * 2))
        };
        let mut spans = vec![span(x)];
        spans.extend(segments.iter().map(|s| span(s.y)));
        let Some(spans) = spans.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(false);
        };
        let apart =
            |a: &(u64, u64, u64), b: &(u64, u64, u64)| a.0 != b.0 || a.2 <= b.1 || b.2 <= a.1;
        for (i, a) in spans.iter().enumerate() {
            if spans[i + 1..].iter().any(|b| !apart(a, b)) {
                return Ok(false);
            }
        }
        // ... and from what a node between them writes, which runs after the
        // launch landed the later matmuls' outputs (it reads the first's).
        let first_out = span(self.tensor(y));
        for &v in &between {
            let Some(b) = span(self.tensor(v)) else {
                return Ok(false);
            };
            if spans[1..]
                .iter()
                .any(|a| Some(*a) != first_out && !apart(a, &b))
            {
                return Ok(false);
            }
        }
        if coop {
            linear::qmm::staged::coop_segments(
                self.ctx(),
                &segments,
                x,
                i32::try_from(m).unwrap_or(0),
                i32::try_from(k).unwrap_or(0),
            )?;
            self.land_ahead([
                members.get(1).map(|s| s.1),
                members.get(2).map(|s| s.1),
                halves,
            ]);
            return Ok(true);
        }
        let (counters, floats) = linear::qmm::sg::segments_scratch(&segments, m);
        // Each split segment sums as its own launch would, or they run alone.
        let partials = if floats > 0 {
            match self.partials(1, floats) {
                Some(partials) => Some(partials),
                None => return Ok(false),
            }
        } else {
            None
        };
        let (m, k, counters) = (
            i32::try_from(m).unwrap_or(0),
            i32::try_from(k).unwrap_or(0),
            i32::try_from(counters).unwrap_or(0),
        );
        let rows = (17 * x.width + 4 * counters.unsigned_abs()).div_ceil(2 * x.width);
        let Some((table, laid)) = self.register_table(act, rows, x.width, counters.unsigned_abs())
        else {
            return Ok(false);
        };
        if !laid {
            linear::qmm::sg::register_table(self.ctx(), x, table, m, k, counters)?;
        }
        linear::qmm::sg::register_segments(self.ctx(), &segments, table, m, k, partials)?;
        self.land_ahead([
            members.get(1).map(|s| s.1),
            members.get(2).map(|s| s.1),
            halves,
        ]);
        Ok(true)
    }

    fn linear(&mut self, op: &Linear) -> Result<(), kernels_metal::Error> {
        // The register expert tiles' grouping landed the riding shared
        // expert's gate.
        if let Linear::Matmul { y, .. } = op
            && self.tiles_gate(*y)
        {
            return Ok(());
        }
        match op {
            Linear::Relu2 { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } if self.tensor(*act).dtype == poem_ir::Dtype::F32 && self.banked(*w).is_none() => {
                linear::lane_gemm::act_x_wt(
                    self.ctx(),
                    "linear.matmul",
                    self.tensor(*act),
                    self.tensor(*w),
                    self.tensor(*y),
                )
            }
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } => match self.banked(*w) {
                Some(bank) => {
                    // A segment launch of an earlier matmul of this activation landed it.
                    if self.landed_ahead(*y) {
                        return Ok(());
                    }
                    let x = self.tensor(*act);
                    if self.matmul_segments(*act, *w, *y)? {
                        return Ok(());
                    }
                    if let Some((ane, plan)) = self
                        .ane()
                        .and_then(|ane| Some((ane, ane.plan_columns(*w, x.rows)?)))
                        .filter(|(_, plan)| self.columns_fit(plan, *act, *y))
                    {
                        return self.matmul_split(ane, &plan, *act, *y);
                    }
                    // The matmuls of one activation (a layer's projections of
                    // its normed rows) share the register tile's input table.
                    linear::qmm::matmul(
                        self.ctx(),
                        self.tuning(),
                        x,
                        bank,
                        self.tensor(*y),
                        linear::qmm::Scratch {
                            precast: &|rows, contraction| self.precast(rows, contraction),
                            partials: &|rows, width| self.partials(rows, width),
                            table: Some(&|rows, contraction, counters| {
                                self.register_table(*act, rows, contraction, counters)
                            }),
                        },
                        self.capacity(*act).min(self.capacity(*y)),
                    )
                }
                None => linear::gemm::matmul(
                    self.ctx(),
                    self.tensor(*act),
                    self.tensor(*w),
                    self.tensor(*y),
                ),
            },
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: true,
                ..
            } => match self.banked(*w) {
                Some(bank) => linear::qmm::lm_head(
                    self.ctx(),
                    self.tuning(),
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                    linear::qmm::Scratch {
                        precast: &|rows, contraction| self.precast(rows, contraction),
                        partials: &|rows, width| self.partials(rows, width),
                        table: Some(&|rows, contraction, counters| {
                            self.register_table(*act, rows, contraction, counters)
                        }),
                    },
                    self.capacity(*act).min(self.capacity(*y)),
                ),
                None => linear::gemm::lm_head(
                    self.ctx(),
                    self.tensor(*act),
                    self.tensor(*w),
                    self.tensor(*y),
                ),
            },

            // The register chain's gate-up landed its SwiGLU already.
            Linear::Glu {
                packed: Some(packed),
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } if self.tiles_swiglu(*packed, *y) => Ok(()),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => match self.chain_swiglu(*packed, *y) {
                // The chain's gate-up rows sit sorted in the routed stack's
                // output: the swiglu runs there and lands in its input.
                Some((sorted_packed, sorted_h, perm)) => linear::mlp::swiglu_sorted(
                    self.ctx(),
                    sorted_packed,
                    *intermediate,
                    perm,
                    sorted_h,
                ),
                None => {
                    // The down projection on the register tile reads its
                    // input table as the activation lays it.
                    match self.producer_table(*y) {
                        Some((table, counters)) => linear::mlp::swiglu_table(
                            self.ctx(),
                            self.tensor(*packed),
                            *intermediate,
                            self.tensor(*y),
                            table,
                            counters,
                        ),
                        None => linear::mlp::swiglu(
                            self.ctx(),
                            self.tensor(*packed),
                            *intermediate,
                            self.tensor(*y),
                        ),
                    }
                }
            },
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => linear::mlp::swiglu_clamp(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *limit,
                self.tensor(*y),
            ),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
            } => match self.chain_swiglu(*packed, *y) {
                Some((sorted_packed, sorted_h, perm)) => linear::mlp::swiglu_clamp_alpha_sorted(
                    self.ctx(),
                    sorted_packed,
                    *intermediate,
                    *limit,
                    *alpha,
                    perm,
                    sorted_h,
                ),
                None => linear::mlp::swiglu_clamp_alpha(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    *limit,
                    *alpha,
                    self.tensor(*y),
                ),
            },
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                limit: Some(limit),
                y,
                packed: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => linear::mlp::swiglu_clamp_split(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                *limit,
                self.tensor(*y),
            ),
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                y,
                packed: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => linear::mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                self.tensor(*y),
            ),
            Linear::GeluTanh { x, y } => {
                linear::mlp::gelu_tanh(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            // A product that quantizes its activations runs only on CUDA's
            // tcgen05 GEMMs.
            Linear::RelBias { .. }
            | Linear::MoeRoute {
                score: poem_ir::RouteScore::SigmoidSink { .. },
                ..
            }
            | Linear::Matmul {
                quant: Some(_),
                logits: false,
                ..
            }
            | Linear::MoeSelect {
                bias: None,
                quantized: true,
                quant: Some(_),
                ..
            }
            | Linear::Glu {
                packed: None,
                gate: Some(_),
                weights: Some(_),
                limit: Some(_),
                act: poem_ir::GluAct::Swiglu,
                ..
            }
            | Linear::Glu {
                packed: Some(_),
                gate: None,
                weights: Some(_),
                limit: Some(_),
                act: poem_ir::GluAct::Swiglu,
                ..
            }
            | Linear::MoeCombine {
                weights: None,
                shared: Some(_),
                ..
            } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            // The forms no builder makes.
            Linear::Matmul { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => match self.chain_swiglu(*packed, *y) {
                Some((sorted_packed, sorted_h, perm)) => match self.chain_sums() {
                    Some((sums, tile)) => linear::mlp::geglu_tanh_sorted_sums(
                        self.ctx(),
                        sorted_packed,
                        *intermediate,
                        perm,
                        sorted_h,
                        sums,
                        tile,
                    ),
                    None => linear::mlp::geglu_tanh_sorted(
                        self.ctx(),
                        sorted_packed,
                        *intermediate,
                        perm,
                        sorted_h,
                    ),
                },
                None => linear::mlp::geglu_tanh_packed(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    self.tensor(*y),
                ),
            },
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Situ { beta, up_cap, .. },
                ..
            } => linear::mlp::situ(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *beta,
                *up_cap,
                self.tensor(*y),
            ),

            // The forms no builder makes.
            Linear::Glu { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Linear::MoeRoute {
                logits,
                experts,
                top_k,
                routes,
                weights,
                scale: None,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => linear::moe::topk_softmax(
                self.ctx(),
                self.tensor(*logits),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::MoeRoute {
                logits,
                scale: Some(scale),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => linear::moe::topk_softmax_scaled(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*scale),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::MoeRoute {
                logits,
                bias,
                experts,
                top_k,
                routes,
                weights,
                hint: _,
                score:
                    poem_ir::RouteScore::Sigmoid {
                        renormalize,
                        scaling,
                        ..
                    },
                ..
            } => match bias {
                Some(bias) => linear::moe::topk_sigmoid_biased(
                    self.ctx(),
                    self.tensor(*logits),
                    self.tensor(*bias),
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    self.tensor(*routes),
                    self.tensor(*weights),
                ),
                None => linear::moe::topk_sigmoid(
                    self.ctx(),
                    self.tensor(*logits),
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    self.tensor(*routes),
                    self.tensor(*weights),
                ),
            },
            Linear::MoeRoute {
                logits,
                bias: Some(bias),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Predict,
                ..
            } => linear::moe::predict_route(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*bias),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::MoeRoute {
                logits,
                bias: Some(bias),
                experts,
                top_k,
                hint: _,
                routes,
                weights,
                score:
                    poem_ir::RouteScore::SqrtSoftplus {
                        renormalize,
                        scaling,
                        ..
                    },
                ..
            } => linear::moe::topk_sqrt_softplus(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*bias),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::MoeRoute {
                ids: Some(ids),
                table: Some(tid2eid),
                logits,
                experts: _,
                top_k,
                routes,
                weights,
                score:
                    poem_ir::RouteScore::Hash {
                        vocab,
                        renormalize,
                        scaling,
                        ..
                    },
                ..
            } => linear::moe::hash_route(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*tid2eid),
                self.tensor(*logits),
                *vocab,
                *top_k,
                *renormalize,
                *scaling,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            // The forms no builder makes.
            Linear::MoeRoute { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Linear::GroupRoutes { groups, routes } => {
                linear::moe::group_routes(self.ctx(), *groups, self.tensor(*routes))
            }
            Linear::MatmulGrouped {
                x,
                w,
                routes,
                groups,
                y,
            } => linear::moe::matmul_grouped(
                self.ctx(),
                self.tensor(*x),
                match self.banked(*w) {
                    Some(bank) => linear::moe::GroupedPlane::Bank(bank),
                    None => linear::moe::GroupedPlane::Dense(self.tensor(*w)),
                },
                self.tensor(*routes),
                *groups,
                self.tensor(*y),
            ),
            Linear::MoeSelect {
                x,
                bank,
                routes,
                y,
                bias: None,
                quantized: false,
                quant: None,
                ..
            } => linear::moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                self.tensor(*y),
            ),
            Linear::MoeSelect {
                x,
                bank,
                bias: Some(bias),
                routes,
                y,
                quantized: false,
                quant: None,
                ..
            } => {
                const OP: &str = "linear.moe_select";
                if let Some(scratch) = self.routed_rooms() {
                    let sorted = self.chain_select(*x, *bank, *routes, *y);
                    if linear::moe::matmul_select_batched(
                        self.ctx(),
                        OP,
                        self.tensor(*x),
                        self.planes(*bank),
                        Some(self.tensor(*bias)),
                        self.tensor(*routes),
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                        self.tuning(),
                        sorted,
                    )? {
                        // The batched pass ran over the routed rooms, which
                        // the precast room overlaps.
                        self.knock_table();
                        return Ok(());
                    }
                    debug_assert!(
                        !sorted.input && !sorted.output,
                        "a chain starts only where the batched matmul runs"
                    );
                    self.chain_clear();
                }
                linear::moe::matmul_select_bias(
                    self.ctx(),
                    self.tensor(*x),
                    self.planes(*bank),
                    self.tensor(*bias),
                    self.tensor(*routes),
                    self.tensor(*y),
                )
            }
            Linear::MoeSelect {
                x,
                bank,
                routes,
                y,
                bias: None,
                quantized: true,
                quant: None,
                ..
            } => {
                const OP: &str = "linear.moe_select";
                // A routed MLP of a verify's few tokens on the register expert tiles.
                if self.tiles_down(*x, *bank, *routes, *y)?
                    || self.tiles_start(*x, *bank, *routes, *y)?
                {
                    return Ok(());
                }
                if let Some(scratch) = self.routed_rooms() {
                    let sorted = self.chain_select(*x, *bank, *routes, *y);
                    if linear::moe::matmul_select_batched(
                        self.ctx(),
                        OP,
                        self.tensor(*x),
                        self.planes(*bank),
                        None,
                        self.tensor(*routes),
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                        self.tuning(),
                        sorted,
                    )? {
                        // The batched pass ran over the routed rooms, which
                        // the precast room overlaps.
                        self.knock_table();
                        return Ok(());
                    }
                    debug_assert!(
                        !sorted.input && !sorted.output,
                        "a chain starts only where the batched matmul runs"
                    );
                    self.chain_clear();
                }
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    self.planes(*bank),
                    self.tensor(*routes),
                    self.tensor(*y),
                )
            }
            // The forms no builder makes.
            Linear::MoeSelect { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            } => linear::lora::correct(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank_a),
                self.tensor(*bank_b),
                self.tensor(*routes),
                self.tensor(*y_out),
            ),
            Linear::MoeCombine {
                routed,
                weights: Some(weights),
                y,
                shared: None,
                ..
            } => match match self.tiles_fold(*routed) {
                // The gated add lands the register chain's sum.
                Some(TiledFold::Deferred) => return Ok(()),
                Some(TiledFold::Sorted(sorted, inv, top_k)) => Some((sorted, inv, top_k)),
                None => self.chain_combine(*routed),
            } {
                Some((sorted, inv, top_k)) => linear::moe::weighted_sum_sorted(
                    self.ctx(),
                    sorted,
                    inv,
                    self.tensor(*weights),
                    top_k,
                    self.tensor(*y),
                ),
                None => linear::moe::weighted_sum(
                    self.ctx(),
                    self.tensor(*routed),
                    self.tensor(*weights),
                    self.tensor(*y),
                ),
            },
            // The forms no builder makes.
            Linear::MoeCombine { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Linear::MoeBiasSum {
                x,
                bias,
                routes,
                weights,
                y,
            } => linear::moe::bias_sum(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bias),
                self.tensor(*routes),
                self.tensor(*weights),
                self.tensor(*y),
            ),
            // The register chain's rooms hold the routes' rows, the shared
            // expert's and its gate.
            Linear::MoeSigmoidGateAdd { y, .. } if self.tiles_gate_add(*y)? => Ok(()),
            Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y,
            } => linear::moe::sigmoid_gate_add(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*shared),
                self.tensor(*gate),
                self.tensor(*y),
            ),
        }
    }
}
