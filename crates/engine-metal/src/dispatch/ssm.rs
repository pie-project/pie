use crate::run::Run;
use kernels_metal::ssm;
use poem_exec::{DispatchSsm, KernelError};
use poem_ir::{Operands, Ssm};

impl DispatchSsm for Run<'_> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        self.ssm(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn ssm(&mut self, op: &Ssm) -> Result<(), kernels_metal::Error> {
        match op {
            // The committed walk's conv takes no bias.
            Ssm::CausalConv1d { bias: Some(_), .. }
            | Ssm::CausalConv1dChunked { bias: Some(_), .. }
                if self.rs_seat().is_some() =>
            {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            Ssm::Mamba2 { .. } | Ssm::Mamba2Chunked { .. } if self.rs_seat().is_some() => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            Ssm::Mamba2 {
                xbc,
                dt,
                z,
                a_log,
                dt_bias,
                d,
                norm,
                eps,
                state,
                heads,
                groups,
                state_dim,
                y,
            }
            | Ssm::Mamba2Chunked {
                xbc,
                dt,
                z,
                a_log,
                dt_bias,
                d,
                norm,
                eps,
                state,
                heads,
                groups,
                state_dim,
                y,
            } => {
                let m = ssm::Mamba2 {
                    a_log: self.tensor(*a_log),
                    dt_bias: self.tensor(*dt_bias),
                    d: self.tensor(*d),
                    norm: self.tensor(*norm),
                    eps: *eps,
                    heads: *heads,
                    groups: *groups,
                };
                let (dt, z, y) = (self.tensor(*dt), self.tensor(*z), self.tensor(*y));
                if matches!(op, Ssm::Mamba2 { .. }) {
                    return ssm::mamba2(
                        self.ctx(),
                        self.tensor(*xbc),
                        dt,
                        z,
                        m,
                        &self.recurrent(*state),
                        *state_dim,
                        y,
                    );
                }
                ssm::mamba2_chunked(
                    self.ctx(),
                    self.ragged(*xbc),
                    dt,
                    z,
                    m,
                    &self.recurrent(*state),
                    *state_dim,
                    y,
                    self.rs_snapshots(),
                )
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
                let seat = self.rs_seat().expect("guarded");
                let conv = crate::rs::Convolved {
                    x: *x,
                    weight: self.tensor(*weight),
                    state: *state,
                    conv_width: *conv_width,
                    dilation: *dilation,
                };
                self.rs_convolve(&seat, &conv, *y).map(|_| ())
            }
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } if self.rs_seat().is_some() => {
                let seat = self.rs_seat().expect("guarded");
                let prep = (*ba, self.tensor(*dt_bias), self.tensor(*a_log));
                // The scan that alone reads the gates derives them as it
                // stages its rows, and lands the window's own.
                if self.read_only_by_scan(*gates) {
                    seat.prepped.borrow_mut().insert(gates.0, prep);
                    return Ok(());
                }
                self.rs_prep(&seat, prep, *gates)
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
                let prepped = seat.prepped.borrow_mut().remove(&gates.0);
                if let Some((ba, dt_bias, a_log)) = prepped
                    && ssm::gated_delta_verify_fits(*k_dim, *v_dim)
                {
                    let paged = seat.paged(self.handles(), ba);
                    let ba = match paged {
                        Some(_) => self.tensor(ba),
                        None => self.rs_extend(OP, &seat, ba)?,
                    };
                    return ssm::gated_delta_verify(
                        self.ctx(),
                        ext_qkv,
                        self.qo_indptr(),
                        &self.rs_committed(&seat),
                        &ssm::VerifyGates {
                            ba,
                            dt_bias,
                            a_log,
                            gates: self.tensor(*gates),
                        },
                        paged,
                        &self.recurrent(*state),
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        self.tensor(*y),
                        self.rs_snapshots(),
                    );
                }
                // Heads the verify scan does not stage take the gate prep
                // the verify scan would have folded in.
                if let Some(prep) = prepped {
                    self.rs_prep(&seat, prep, *gates)?;
                }
                let ext_gates = self.rs_ext_of(OP, &seat, *gates)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                ssm::gated_delta_committed_landing(
                    self.ctx(),
                    self.tuning(),
                    ext_qkv,
                    self.qo_indptr(),
                    &self.rs_committed(&seat),
                    ext_gates,
                    &self.recurrent(*state),
                    seat.work,
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    ext_y,
                    self.rs_snapshots(),
                )?;
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                bias,
            } => ssm::causal_conv1d(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                bias.map(|b| self.tensor(b)),
                &self.recurrent(*state),
                *conv_width,
                *dilation,
                self.tensor(*y),
            ),
            Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                bias,
            } => ssm::causal_conv1d_chunked_landing(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*weight),
                bias.map(|b| self.tensor(b)),
                &self.recurrent(*state),
                *conv_width,
                *dilation,
                self.tensor(*y),
                self.rs_snapshots(),
            ),
            Ssm::ShortConv { .. } | Ssm::ShortConvChunked { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            Ssm::DynConv {
                x,
                coeff,
                base,
                side,
                taps,
                group,
                y,
            } => {
                // The convolved rows' projections on the register tile read
                // their input table as the convolution lays it.
                let (x, coeff, base, out) = (
                    self.ragged(*x),
                    self.tensor(*coeff),
                    self.tensor(*base),
                    self.tensor(*y),
                );
                match self.producer_table(*y) {
                    Some((table, counters)) => ssm::dyn_conv::block_dyn_conv_table(
                        self.ctx(),
                        x,
                        coeff,
                        base,
                        *side,
                        *taps,
                        *group,
                        out,
                        table,
                        counters,
                    ),
                    None => ssm::dyn_conv::block_dyn_conv(
                        self.ctx(),
                        x,
                        coeff,
                        base,
                        *side,
                        *taps,
                        *group,
                        out,
                    ),
                }
            }
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } => ssm::gdn_prep(
                self.ctx(),
                self.tensor(*ba),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                self.tensor(*gates),
            ),
            Ssm::GatedDelta {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => ssm::gated_delta(
                self.ctx(),
                self.tuning(),
                self.tensor(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                &self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y),
            ),
            Ssm::GatedDeltaChunked {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => ssm::gated_delta_chunked_landing(
                self.ctx(),
                self.tuning(),
                self.ragged(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                &self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y),
                self.rs_snapshots(),
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
                ssm::kda_committed(
                    self.ctx(),
                    ext_mixed,
                    self.qo_indptr(),
                    &self.rs_committed(&seat),
                    ext_f,
                    ext_b,
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    &self.recurrent(*state),
                    seat.work,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                    ext_y,
                )?;
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
            } => ssm::kda_step(
                self.ctx(),
                self.tensor(*mixed),
                self.tensor(*f),
                self.tensor(*b),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                &self.recurrent(*state),
                *heads,
                *head_dim,
                *norm_eps,
                *gate_floor,
                self.tensor(*y),
            ),
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
            } => ssm::kda_chunked(
                self.ctx(),
                self.ragged(*mixed),
                self.tensor(*f),
                self.tensor(*b),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                &self.recurrent(*state),
                *heads,
                *head_dim,
                *norm_eps,
                *gate_floor,
                self.tensor(*y),
            ),
        }
    }
}
