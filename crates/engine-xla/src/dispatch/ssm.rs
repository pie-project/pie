use crate::lower::{Lower, lower};
use crate::run::Run;
use kernels_xla::lanes::Committed;
use kernels_xla::ssm;
use kernels_xla::tensor::{Operand, RaggedTensor, RecurrentPool, Tensor};
use poem_exec::{DispatchSsm, KernelError};
use poem_ir::{Operands, Ssm};

impl DispatchSsm for Run<'_> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.ssm(op).map_err(crate::error::kernel)
    }
}

/// The conv kernels roll the conv bank in place: a pool naming two banks
/// for it is refused.
fn conv_bank(op: &'static str, pool: &RecurrentPool) -> Result<(), kernels_xla::Error> {
    if pool.conv_state.buf != pool.new_conv_state.buf {
        return Err(kernels_xla::Error::Backend {
            op,
            detail: "this plane rolls the conv state in place, and the pool names two banks"
                .to_string(),
        });
    }
    Ok(())
}

/// A rollback seat's tables, read replay, commit, slots.
fn read_seat(
    l: &mut Lower<'_, '_>,
    seat: &Committed,
) -> Result<Committed<Operand>, kernels_xla::Error> {
    Ok(Committed {
        replay: l.read(seat.replay)?,
        commit: l.read(seat.commit)?,
        slots: l.read(seat.slots)?,
        lane0: seat.lane0,
    })
}

impl Run<'_> {
    fn ssm(&mut self, op: &Ssm) -> Result<(), kernels_xla::Error> {
        match op {
            Ssm::Mamba2 { .. } | Ssm::Mamba2Chunked { .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
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
            } if self.rs_seat().is_some() => {
                const OP: &str = "ssm.causal_conv1d_committed";
                let seat = self.rs_seat().expect("guarded");
                let ext_x = self.rs_extend(OP, &seat, *x)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                let pool = self.recurrent(*state);
                conv_bank(OP, &pool)?;
                let committed = self.rs_committed(&seat);
                lower(self.ctx(), |l| {
                    let xv = l.read(ext_x)?;
                    let w = l.read(self.tensor(*weight))?;
                    let bank = l.read(pool.conv_state)?;
                    let indptr = l.read(self.qo_indptr())?;
                    let seat = read_seat(l, &committed)?;
                    let (yv, next) = ssm::causal_conv1d_committed(
                        l.f(),
                        xv,
                        indptr,
                        seat,
                        w,
                        bank,
                        *conv_width,
                        *dilation,
                        ext_y.landing(),
                    )?;
                    l.land(ext_y, yv)?;
                    l.land(pool.new_conv_state, next)
                })?;
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
                self.gdn_prep(ext_ba, *dt_bias, *a_log, ext_gates)?;
                self.rs_land(OP, &seat, ext_gates, *gates)
            }
            Ssm::GatedDelta {
                qkv,
                z: _,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            }
            | Ssm::GatedDeltaChunked {
                qkv,
                z: _,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } if self.rs_seat().is_some() => {
                const OP: &str = "ssm.gated_delta_committed";
                let seat = self.rs_seat().expect("guarded");

                let ext_qkv = self.rs_ext_of(OP, &seat, *qkv)?;
                let ext_gates = self.rs_ext_of(OP, &seat, *gates)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                let pool = self.recurrent(*state);
                let committed = self.rs_committed(&seat);
                lower(self.ctx(), |l| {
                    let qkv = l.read(ext_qkv)?;
                    let gates = l.read(ext_gates)?;
                    let slab = l.read(pool.state)?;
                    let indptr = l.read(self.qo_indptr())?;
                    let seat = read_seat(l, &committed)?;
                    let tuning = l.tuning();
                    let (yv, next) = ssm::gated_delta_committed(
                        l.f(),
                        qkv,
                        indptr,
                        seat,
                        gates,
                        slab,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        ext_y.landing(),
                        tuning,
                    )?;
                    l.land(ext_y, yv)?;
                    l.land(pool.state, next)
                })?;
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
                let pool = self.recurrent(*state);
                let committed = self.rs_committed(&seat);
                lower(self.ctx(), |l| {
                    let mixed = l.read(ext_mixed)?;
                    let forget = l.read(ext_f)?;
                    let beta = l.read(ext_b)?;
                    let dt_bias = l.read(self.tensor(*dt_bias))?;
                    let a_log = l.read(self.tensor(*a_log))?;
                    let slab = l.read(pool.state)?;
                    let indptr = l.read(self.qo_indptr())?;
                    let seat = read_seat(l, &committed)?;
                    let tuning = l.tuning();
                    let (yv, next) = ssm::kda_committed(
                        l.f(),
                        mixed,
                        indptr,
                        seat,
                        forget,
                        beta,
                        dt_bias,
                        a_log,
                        slab,
                        *heads,
                        *head_dim,
                        *norm_eps,
                        *gate_floor,
                        ext_y.landing(),
                        tuning,
                    )?;
                    l.land(ext_y, yv)?;
                    l.land(pool.state, next)
                })?;
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
            } => {
                let pool = self.recurrent(*state);
                conv_bank("ssm.causal_conv1d", &pool)?;
                let y = self.tensor(*y);
                lower(self.ctx(), |l| {
                    let xv = l.read(self.tensor(*x))?;
                    let w = l.read(self.tensor(*weight))?;
                    let bank = l.read(pool.conv_state)?;
                    let slots = l.read(pool.slots)?;
                    let (yv, next) = ssm::causal_conv1d(
                        l.f(),
                        xv,
                        w,
                        bank,
                        slots,
                        *conv_width,
                        *dilation,
                        y.landing(),
                    )?;
                    l.land(y, yv)?;
                    l.land(pool.new_conv_state, next)
                })
            }
            Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } => {
                let pool = self.recurrent(*state);
                conv_bank("ssm.causal_conv1d_chunked", &pool)?;
                let (x, y) = (self.ragged(*x), self.tensor(*y));
                lower(self.ctx(), |l| {
                    let data = l.read(x.data)?;
                    let w = l.read(self.tensor(*weight))?;
                    let bank = l.read(pool.conv_state)?;
                    let indptr = l.read(x.indptr)?;
                    let slots = l.read(pool.slots)?;
                    let (yv, next) = ssm::causal_conv1d_chunked(
                        l.f(),
                        RaggedTensor { data, indptr },
                        w,
                        bank,
                        slots,
                        *conv_width,
                        *dilation,
                        y.landing(),
                    )?;
                    l.land(y, yv)?;
                    l.land(pool.new_conv_state, next)
                })
            }
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } => self.gdn_prep(self.tensor(*ba), *dt_bias, *a_log, self.tensor(*gates)),
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
            } => {
                let pool = self.recurrent(*state);
                let y = self.tensor(*y);
                lower(self.ctx(), |l| {
                    let qkv = l.read(self.tensor(*qkv))?;
                    let gates = l.read(self.tensor(*gates))?;
                    let slab = l.read(pool.state)?;
                    let slots = l.read(pool.slots)?;
                    let tuning = l.tuning();
                    let (yv, next) = ssm::gated_delta(
                        l.f(),
                        qkv,
                        gates,
                        slab,
                        slots,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        y.landing(),
                        tuning,
                    )?;
                    l.land(y, yv)?;
                    l.land(pool.state, next)
                })
            }
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
            } => {
                let pool = self.recurrent(*state);
                let (qkv, y) = (self.ragged(*qkv), self.tensor(*y));
                lower(self.ctx(), |l| {
                    let data = l.read(qkv.data)?;
                    let gates = l.read(self.tensor(*gates))?;
                    let slab = l.read(pool.state)?;
                    let indptr = l.read(qkv.indptr)?;
                    let slots = l.read(pool.slots)?;
                    let tuning = l.tuning();
                    let (yv, next) = ssm::gated_delta_chunked(
                        l.f(),
                        RaggedTensor { data, indptr },
                        gates,
                        slab,
                        slots,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        y.landing(),
                        tuning,
                    )?;
                    l.land(y, yv)?;
                    l.land(pool.state, next)
                })
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
            } => {
                let pool = self.recurrent(*state);
                let y = self.tensor(*y);
                lower(self.ctx(), |l| {
                    let mixed = l.read(self.tensor(*mixed))?;
                    let forget = l.read(self.tensor(*f))?;
                    let beta = l.read(self.tensor(*b))?;
                    let dt_bias = l.read(self.tensor(*dt_bias))?;
                    let a_log = l.read(self.tensor(*a_log))?;
                    let slab = l.read(pool.state)?;
                    let slots = l.read(pool.slots)?;
                    let tuning = l.tuning();
                    let (yv, next) = ssm::kda_step(
                        l.f(),
                        mixed,
                        forget,
                        beta,
                        dt_bias,
                        a_log,
                        slab,
                        slots,
                        *heads,
                        *head_dim,
                        *norm_eps,
                        *gate_floor,
                        y.landing(),
                        tuning,
                    )?;
                    l.land(y, yv)?;
                    l.land(pool.state, next)
                })
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
                let pool = self.recurrent(*state);
                let (mixed, y) = (self.ragged(*mixed), self.tensor(*y));
                lower(self.ctx(), |l| {
                    let data = l.read(mixed.data)?;
                    let forget = l.read(self.tensor(*f))?;
                    let beta = l.read(self.tensor(*b))?;
                    let dt_bias = l.read(self.tensor(*dt_bias))?;
                    let a_log = l.read(self.tensor(*a_log))?;
                    let slab = l.read(pool.state)?;
                    let indptr = l.read(mixed.indptr)?;
                    let slots = l.read(pool.slots)?;
                    let tuning = l.tuning();
                    let (yv, next) = ssm::kda_chunked(
                        l.f(),
                        RaggedTensor { data, indptr },
                        forget,
                        beta,
                        dt_bias,
                        a_log,
                        slab,
                        slots,
                        *heads,
                        *head_dim,
                        *norm_eps,
                        *gate_floor,
                        y.landing(),
                        tuning,
                    )?;
                    l.land(y, yv)?;
                    l.land(pool.state, next)
                })
            }
            Ssm::ShortConv {
                x,
                weight,
                state,
                conv_width,
                y,
            } => {
                let pool = self.recurrent(*state);
                conv_bank("ssm.short_conv", &pool)?;
                let y = self.tensor(*y);
                lower(self.ctx(), |l| {
                    let xv = l.read(self.tensor(*x))?;
                    let w = l.read(self.tensor(*weight))?;
                    let bank = l.read(pool.conv_state)?;
                    let slots = l.read(pool.slots)?;
                    let (yv, next) =
                        ssm::short_conv(l.f(), xv, w, bank, slots, *conv_width, y.landing())?;
                    l.land(y, yv)?;
                    l.land(pool.new_conv_state, next)
                })
            }
            Ssm::ShortConvChunked {
                x,
                weight,
                state,
                conv_width,
                y,
            } => {
                let pool = self.recurrent(*state);
                conv_bank("ssm.short_conv_chunked", &pool)?;
                let (x, y) = (self.ragged(*x), self.tensor(*y));
                lower(self.ctx(), |l| {
                    let data = l.read(x.data)?;
                    let w = l.read(self.tensor(*weight))?;
                    let bank = l.read(pool.conv_state)?;
                    let indptr = l.read(x.indptr)?;
                    let slots = l.read(pool.slots)?;
                    let (yv, next) = ssm::short_conv_chunked(
                        l.f(),
                        RaggedTensor { data, indptr },
                        w,
                        bank,
                        slots,
                        *conv_width,
                        y.landing(),
                    )?;
                    l.land(y, yv)?;
                    l.land(pool.new_conv_state, next)
                })
            }
            Ssm::DynConv {
                x,
                coeff,
                base,
                side,
                taps,
                group,
                y,
            } => lower(self.ctx(), |l| {
                let x_v = l.ragged(self.ragged(*x))?;
                let coeff_v = l.read(self.tensor(*coeff))?;
                let base_v = l.read(self.tensor(*base))?;
                let o_y = self.tensor(*y);
                let v = ssm::dyn_conv::block_dyn_conv(
                    l.f(),
                    x_v,
                    coeff_v,
                    base_v,
                    *side,
                    *taps,
                    *group,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
        }
    }

    /// `[b | a]` into `gates`.
    fn gdn_prep(
        &self,
        ba: Tensor,
        dt_bias: poem_ir::ValueId,
        a_log: poem_ir::ValueId,
        gates: Tensor,
    ) -> Result<(), kernels_xla::Error> {
        lower(self.ctx(), |l| {
            let ba = l.read(ba)?;
            let dt_bias = l.read(self.tensor(dt_bias))?;
            let a_log = l.read(self.tensor(a_log))?;
            let v = ssm::gdn_prep(l.f(), ba, dt_bias, a_log, gates.landing())?;
            l.land(gates, v)
        })
    }
}
