use crate::backend::{Backend, Buffer as _, Tactics as _};
use crate::run::Run;
use kernels_slang::ssm::{StatePool, conv1d, gdn_prep, gdn_recurrent, kda};
use poem_exec::{DispatchSsm, KernelError};
use poem_ir::{Operands, Ssm, ValueId};

impl<B: Backend> DispatchSsm for Run<'_, B> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        self.ssm(op).map_err(crate::error::kernel)
    }
}

impl<B: Backend> Run<'_, B> {
    fn state_pool(&self, id: ValueId) -> StatePool {
        let pool = self.recurrent(id);
        StatePool {
            bank: pool.state,
            slots: pool.slots,
            pool: pool.pool,
            origin: pool.origin,
            slot_stride: pool.slot_stride,
            pieces: pool.pieces,
            piece_first: pool.piece_first,
            piece_atoms: pool.piece_atoms,
            snapshots: self.rs_snapshots(),
        }
    }

    /// A committed scan's dispatches over the window's lanes, as many lanes
    /// each as the work plane holds: each one's CSR from its first lane,
    /// its lanes' tables and snapshots from it, and the extended row its
    /// first lane's rows start at (the rows the lanes before it replay
    /// ahead of theirs).
    fn committed_batches(
        &self,
        seat: &crate::rs::Seat,
        state: ValueId,
    ) -> Vec<(
        kernels_slang::ssm::Csr,
        kernels_slang::ssm::Committed,
        StatePool,
        u32,
    )> {
        let csr = self.csr();
        let committed = self.rs_committed(seat);
        let batch = seat.work.rows.max(1);
        let mut out = Vec::new();
        let (mut first, mut row0) = (0u32, 0u32);
        while first < csr.lanes {
            let lanes = batch.min(csr.lanes - first);
            out.push((
                kernels_slang::ssm::Csr {
                    at: csr.at + first,
                    lanes,
                    ..csr
                },
                kernels_slang::ssm::Committed {
                    lane0: committed.lane0 + first,
                    ..committed
                },
                {
                    let pool = self.state_pool(state);
                    StatePool {
                        snapshots: pool.snapshots.map(|snaps| kernels_slang::ssm::Snapshots {
                            lane0: snaps.lane0 + first,
                            ..snaps
                        }),
                        ..pool
                    }
                },
                row0,
            ));
            for lane in first..first + lanes {
                row0 += seat
                    .lanes
                    .get((committed.lane0 + lane) as usize)
                    .map_or(0, |plan| plan.replay);
            }
            first += lanes;
        }
        out
    }

    fn slang_of(&self, id: ValueId) -> kernels_slang::Tensor {
        self.tensor(id)
    }

    /// The seat of a buffering fire whose window carries its lanes' buffered
    /// rows in its decode steps' own kernels.
    fn rs_carrying(&self) -> Option<std::sync::Arc<crate::rs::Seat>> {
        self.rs_seat()
            .filter(|seat| seat.carries(self.window().span.lane_offset))
    }

    /// The seat of a buffering fire whose window replays its lanes'
    /// buffered rows through the extended run.
    fn rs_extending(&self) -> Option<std::sync::Arc<crate::rs::Seat>> {
        self.rs_seat()
            .filter(|seat| !seat.carries(self.window().span.lane_offset))
    }

    /// The state cache row value `id` reads.
    fn state_cache(&self, op: &'static str, id: ValueId) -> Result<u32, kernels_slang::Error> {
        match self.values().get(id.0 as usize).map(|decl| &decl.def) {
            Some(poem_ir::Def::Cache(row)) => Ok(*row),
            _ => Err(kernels_slang::Error::Backend {
                op,
                detail: format!("value {} is read as a recurrent state and is no cache row", id.0),
            }),
        }
    }

    /// The staged rs table and plane `role` of the recurrence keeping
    /// `state`, as this window's decode step reads them.
    fn carried(
        &self,
        op: &'static str,
        seat: &crate::rs::Seat,
        state: ValueId,
        role: crate::rs::Role,
    ) -> Result<(kernels_slang::ssm::Carry, kernels_slang::ssm::CarryPlane), kernels_slang::Error>
    {
        let fault = |fault: crate::error::Fault| kernels_slang::Error::Backend {
            op,
            detail: fault.to_string(),
        };
        let pool = self.state_pool(state);
        // One pool, whatever handles each was bound under.
        let same = |a: u32, b: u32| {
            a == b
                || match (self.handles().get(a), self.handles().get(b)) {
                    (Some(a), Some(b)) => {
                        a.buffer().slab_id() == b.buffer().slab_id() && a.offset() == b.offset()
                    }
                    _ => false,
                }
        };
        if pool.pool.words != seat.pool.words
            || !pool.pool.bufs.iter().zip(&seat.pool.bufs).all(|(&a, &b)| same(a, b))
        {
            return Err(kernels_slang::Error::Backend {
                op,
                detail: "the state and the rs list lie in different pools, and a carried step \
                         addresses both by the state's"
                    .to_string(),
            });
        }
        let carry = seat.carry(self.window().span.lane_offset).map_err(fault)?;
        let plane = seat
            .carry_plane(self.state_cache(op, state)?, role)
            .map_err(fault)?;
        Ok((carry, plane))
    }

    /// A beam group's folds (`BEAM-PAGE-LISTS.md` §11.3) of the recurrence
    /// a carried step of `op` reads, ahead of it: `fold(count, carry)` over
    /// the fire's folds where it stages any. A naming walk names the fold
    /// twin as a way the host may go before the step.
    fn group_folds(
        &self,
        op: &'static str,
        seat: &crate::rs::Seat,
        fold: impl FnOnce(u32, &kernels_slang::ssm::Carry) -> Result<(), kernels_slang::Error>,
    ) -> Result<(), kernels_slang::Error> {
        let fault = |fault: crate::error::Fault| kernels_slang::Error::Backend {
            op,
            detail: fault.to_string(),
        };
        if let Some(naming) = self.ctx.naming() {
            // Either way: no fold, or one before the step.
            let carry = seat.carry(self.window().span.lane_offset).map_err(fault)?;
            naming.fork();
            naming.or();
            let named = fold(1, &carry);
            naming.join();
            return named;
        }
        let Some(folds) = seat.folds else {
            return Ok(());
        };
        // A window runs the folds of the groups whose first lane it holds,
        // ahead of its step: each group's once a fire.
        let span = self.window().span;
        let window = span.lane_offset..span.lane_offset + span.lanes;
        let from = seat.fold_lanes.partition_point(|&lane| lane < window.start);
        let to = seat.fold_lanes.partition_point(|&lane| lane < window.end);
        if to <= from {
            return Ok(());
        }
        let _ = folds.count;
        let carry = seat
            .fold_carry(&folds, u32::try_from(from).unwrap_or(u32::MAX))
            .map_err(fault)?;
        fold(u32::try_from(to - from).unwrap_or(u32::MAX), &carry)
    }

    /// The gate prep `gates` comes from: its `[b | a]`, `dt_bias` and
    /// `A_log`.
    fn gdn_gated(
        &self,
        op: &'static str,
        gates: ValueId,
    ) -> Result<gdn_recurrent::Gated, kernels_slang::Error> {
        let node = match self.values().get(gates.0 as usize).map(|decl| &decl.def) {
            Some(poem_ir::Def::Op(node)) => self.nodes().get(*node as usize),
            _ => None,
        };
        match node.map(|node| &node.op) {
            Some(poem_ir::Operation::Ssm(Ssm::GdnGates {
                ba, dt_bias, a_log, ..
            })) => Ok(gdn_recurrent::Gated {
                ba: self.attended(*ba),
                dt_bias: self.slang_of(*dt_bias),
                a_log: self.slang_of(*a_log),
            }),
            _ => Err(kernels_slang::Error::Backend {
                op,
                detail: format!(
                    "the gates value {} no gate prep forms, which a carried step forms its \
                     buffered rows' gates as",
                    gates.0
                ),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn kda_gates(
        &self,
        f: kernels_slang::Tensor,
        b: kernels_slang::Tensor,
        dt_bias: ValueId,
        a_log: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        gate_floor: f32,
    ) -> kda::Gates {
        kda::Gates {
            f,
            b,
            dt_bias: self.slang_of(dt_bias),
            a_log: self.slang_of(a_log),
            heads,
            head_dim,
            norm_eps,
            gate_floor,
        }
    }

    fn ssm(&mut self, op: &Ssm) -> Result<(), kernels_slang::Error> {
        let ctx = self.ctx;
        match op {
            Ssm::Mamba2 { .. }
            | Ssm::Mamba2Chunked { .. }
            | Ssm::DynConv { .. }
            | Ssm::ShortConv { .. }
            | Ssm::ShortConvChunked { .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } if self
                .rs_carrying()
                .is_some_and(|seat| seat.layout.out_of.contains_key(&y.0)) =>
            {
                // A decode conv the rs buffer covers, in a window that
                // carries: the own row's window from the committed history
                // and the lane's last buffered rows where they lie.
                const OP: &str = "ssm.causal_conv1d_carried";
                let seat = self.rs_carrying().expect("guarded");
                let (carry, plane) = self.carried(OP, &seat, *state, crate::rs::Role::ConvX)?;
                // A beam group's fold (§11.3), ahead of the step that reads
                // the history: the window's first launch names it whether a
                // fire folds or not (a naming walk), a fire runs it where it
                // stages one.
                self.group_folds(OP, &seat, |folds, fold_carry| {
                    conv1d::conv1d_carried_fold(
                        ctx,
                        self.attended(*x),
                        self.slang_of(*weight),
                        &self.state_pool(*state),
                        fold_carry,
                        plane,
                        *conv_width,
                        *dilation,
                        self.attended(*y),
                        folds,
                    )
                })?;
                conv1d::conv1d_carried(
                    ctx,
                    self.attended(*x),
                    self.slang_of(*weight),
                    &self.state_pool(*state),
                    &carry,
                    plane,
                    *conv_width,
                    *dilation,
                    self.attended(*y),
                    self.row_span(),
                )
            }
            Ssm::CausalConv1dChunked { .. }
            | Ssm::GatedDeltaChunked { .. }
            | Ssm::KdaChunked { .. }
                if self.rs_carrying().is_some() =>
            {
                Err(kernels_slang::Error::Backend {
                    op: op.name(),
                    detail: "a window whose decode steps carry its lanes' buffered rows runs a \
                             chunked recurrence, which reads them extended"
                        .to_string(),
                })
            }
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            }
            | Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } if self
                .rs_extending()
                .is_some_and(|seat| seat.layout.out_of.contains_key(&y.0)) =>
            {
                // A conv the rs buffer covers (`poem_compiler::rs::
                // buffered_convs`), whether it feeds a recurrence or none
                // (Qwen4's PLE conv): its input rows extend by the lane's
                // buffered ones, the kernel walks them over the state's
                // `(w - 1) * dilation + 1` rows and commits as of the fold.
                const OP: &str = "ssm.causal_conv1d_committed";
                let seat = self.rs_extending().expect("guarded");
                let ext_x = self.rs_extend(OP, &seat, *x)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                conv1d::conv1d_committed(
                    ctx,
                    ext_x,
                    self.csr(),
                    &self.rs_committed(&seat),
                    self.slang_of(*weight),
                    &self.state_pool(*state),
                    *conv_width,
                    *dilation,
                    ext_y,
                )?;
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } if self.rs_extending().is_some() => {
                const OP: &str = "ssm.gdn_gates";
                let seat = self.rs_extending().expect("guarded");
                let ext_ba = self.rs_extend(OP, &seat, *ba)?;
                let ext_gates = self.rs_out(OP, &seat, *gates)?;
                gdn_prep::gdn_prep(
                    ctx,
                    ext_ba,
                    self.slang_of(*dt_bias),
                    self.slang_of(*a_log),
                    ext_gates,
                    kernels_slang::ssm::RowSpan::whole(ext_ba),
                )?;
                self.rs_land(OP, &seat, ext_gates, *gates)
            }
            Ssm::GatedDelta {
                qkv,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
                ..
            } if self.rs_carrying().is_some() => {
                // A decode step in a window that carries: the lane's
                // buffered rows read where they lie, as a linear attention
                // from its committed state.
                const OP: &str = "ssm.gated_delta_carried";
                let seat = self.rs_carrying().expect("guarded");
                let (carry, ba) = self.carried(OP, &seat, *state, crate::rs::Role::Ba)?;
                let (_, keys) = self.carried(OP, &seat, *state, crate::rs::Role::Keys)?;
                let (_, u) = self.carried(OP, &seat, *state, crate::rs::Role::U)?;
                self.group_folds(OP, &seat, |folds, fold_carry| {
                    gdn_recurrent::gated_delta_carried_fold(
                        ctx,
                        self.tuning().recurrent(),
                        self.attended(*qkv),
                        self.gdn_gated(OP, *gates)?,
                        &self.state_pool(*state),
                        fold_carry,
                        [ba, keys, u],
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        self.attended(*y),
                        folds,
                    )
                })?;
                gdn_recurrent::gated_delta_carried(
                    ctx,
                    self.tuning().recurrent(),
                    self.attended(*qkv),
                    self.gdn_gated(OP, *gates)?,
                    &self.state_pool(*state),
                    &carry,
                    [ba, keys, u],
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    self.attended(*y),
                    self.row_span(),
                )
            }
            Ssm::GatedDelta {
                qkv,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
                ..
            }
            | Ssm::GatedDeltaChunked {
                qkv,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
                ..
            } if self.rs_extending().is_some() => {
                const OP: &str = "ssm.gated_delta_committed";
                let seat = self.rs_extending().expect("guarded");
                let (carry, keys) = self.carried(OP, &seat, *state, crate::rs::Role::Keys)?;
                let (_, u) = self.carried(OP, &seat, *state, crate::rs::Role::U)?;
                let ext_qkv = self.rs_ext_of(OP, &seat, *qkv)?;
                let ext_gates = self.rs_ext_of(OP, &seat, *gates)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                for (csr, committed, pool, row0) in self.committed_batches(&seat, *state) {
                    gdn_recurrent::gated_delta_committed(
                        ctx,
                        self.tuning().recurrent(),
                        ext_qkv,
                        csr,
                        &committed,
                        ext_gates,
                        &pool,
                        seat.work,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        ext_y,
                        row0,
                        &carry,
                        [keys, u],
                    )?;
                }
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Ssm::Kda {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            }
            | Ssm::KdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } if self.rs_extending().is_some() => {
                const OP: &str = "ssm.kda_committed";
                let seat = self.rs_extending().expect("guarded");
                let ext_mixed = self.rs_ext_of(OP, &seat, *mixed)?;
                let ext_f = self.rs_extend(OP, &seat, *f)?;
                let ext_b = self.rs_extend(OP, &seat, *b)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                let gates = self.kda_gates(
                    ext_f,
                    ext_b,
                    *dt_bias,
                    *a_log,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                );
                for (csr, committed, pool, row0) in self.committed_batches(&seat, *state) {
                    kda::kda_committed(
                        ctx,
                        self.tuning().recurrent().tier,
                        ext_mixed,
                        csr,
                        &committed,
                        &gates,
                        &pool,
                        seat.work,
                        ext_y,
                        row0,
                    )?;
                }
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } => conv1d::conv1d(
                ctx,
                self.attended(*x),
                self.slang_of(*weight),
                &self.state_pool(*state),
                *conv_width,
                *dilation,
                self.attended(*y),
                self.row_span(),
            ),
            Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } => conv1d::conv1d_chunked(
                ctx,
                self.attended(*x),
                self.csr(),
                self.slang_of(*weight),
                &self.state_pool(*state),
                *conv_width,
                *dilation,
                self.attended(*y),
                self.row0(),
            ),
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } => gdn_prep::gdn_prep(
                ctx,
                self.attended(*ba),
                self.slang_of(*dt_bias),
                self.slang_of(*a_log),
                self.attended(*gates),
                self.row_span(),
            ),
            Ssm::GatedDelta {
                qkv,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
                ..
            } => gdn_recurrent::gated_delta(
                ctx,
                self.tuning().recurrent(),
                self.attended(*qkv),
                self.attended(*gates),
                &self.state_pool(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.attended(*y),
                self.row_span(),
            ),
            Ssm::GatedDeltaChunked {
                qkv,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
                ..
            } => gdn_recurrent::gated_delta_chunked(
                ctx,
                self.tuning().recurrent(),
                self.attended(*qkv),
                self.csr(),
                self.attended(*gates),
                &self.state_pool(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.attended(*y),
                self.row_span(),
            ),
            Ssm::Kda {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } => {
                let gates = self.kda_gates(
                    self.slang_of(*f),
                    self.slang_of(*b),
                    *dt_bias,
                    *a_log,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                );
                kda::kda(
                    ctx,
                    self.tuning().recurrent().tier,
                    self.slang_of(*mixed),
                    &gates,
                    &self.state_pool(*state),
                    self.slang_of(*y),
                )
            }
            Ssm::KdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } => {
                let gates = self.kda_gates(
                    self.slang_of(*f),
                    self.slang_of(*b),
                    *dt_bias,
                    *a_log,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                );
                kda::kda_chunked(
                    ctx,
                    self.tuning().recurrent().tier,
                    self.slang_of(*mixed),
                    self.csr(),
                    &gates,
                    &self.state_pool(*state),
                    self.slang_of(*y),
                )
            }
        }
    }
}
