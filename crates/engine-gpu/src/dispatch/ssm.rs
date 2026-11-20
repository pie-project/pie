use crate::backend::{Backend, Tactics as _};
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
                .rs_seat()
                .is_some_and(|seat| seat.layout.out_of.contains_key(&y.0)) =>
            {
                // A conv the rs buffer covers (`poem_compiler::rs::
                // buffered_convs`), whether it feeds a recurrence or none
                // (Qwen4's PLE conv): its input rows extend by the lane's
                // buffered ones, the kernel walks them over the state's
                // `(w - 1) * dilation + 1` rows and commits as of the fold.
                const OP: &str = "ssm.causal_conv1d_committed";
                let seat = self.rs_seat().expect("guarded");
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
            } if self.rs_seat().is_some() => {
                const OP: &str = "ssm.gdn_gates";
                let seat = self.rs_seat().expect("guarded");
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
            } if self.rs_seat().is_some() => {
                const OP: &str = "ssm.gated_delta_committed";
                let seat = self.rs_seat().expect("guarded");
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
            } if self.rs_seat().is_some() => {
                const OP: &str = "ssm.kda_committed";
                let seat = self.rs_seat().expect("guarded");
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
