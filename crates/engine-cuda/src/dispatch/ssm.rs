use crate::run::Run;
use kernels_cuda::RaggedTensor;
use poem_exec::{DispatchSsm, KernelError};
use poem_ir::Ssm;
use poem_ir::ValueId;

impl DispatchSsm for Run<'_> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        self.ssm(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn ssm(&mut self, op: &Ssm) -> Result<(), kernels_cuda::Error> {
        match op {
            Ssm::CausalConv1d {
                x,
                weight,
                bias,
                state,
                conv_width,
                dilation,
                y,
            } => {
                if self.rs_extended() {
                    return self.conv1d_extended(
                        "ssm.causal_conv1d",
                        *x,
                        *weight,
                        *bias,
                        *state,
                        *conv_width,
                        *dilation,
                        *y,
                    );
                }
                self.rs_move("ssm.causal_conv1d", *x, self.tensor(*x))?;
                kernels_cuda::ssm::causal_conv1d(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*weight),
                    bias.map(|b| self.tensor(b)),
                    &self.recurrent_absolute(*state),
                    *conv_width,
                    *dilation,
                    &mut self.tensor(*y),
                )
            }
            Ssm::ShortConv {
                x,
                weight,
                state,
                conv_width,
                y,
            } => kernels_cuda::ssm::short_conv(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                &self.recurrent_absolute(*state),
                *conv_width,
                &mut self.tensor(*y),
            ),
            Ssm::ShortConvChunked {
                x,
                weight,
                state,
                conv_width,
                y,
            } => {
                self.rs_move("ssm.short_conv_chunked", *x, self.tensor(*x))?;
                let tail = self.recurrent_tail_absolute(*state);
                kernels_cuda::ssm::short_conv_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    &self.recurrent_absolute(*state),
                    *conv_width,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                kernels_cuda::ssm::short_conv_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    &tail,
                    *conv_width,
                    &mut self.tensor(*y),
                )
            }
            Ssm::CausalConv1dChunked {
                x,
                weight,
                bias,
                state,
                conv_width,
                dilation,
                y,
            } => {
                if self.rs_extended() {
                    return self.conv1d_extended(
                        "ssm.causal_conv1d_chunked",
                        *x,
                        *weight,
                        *bias,
                        *state,
                        *conv_width,
                        *dilation,
                        *y,
                    );
                }
                self.rs_move("ssm.causal_conv1d_chunked", *x, self.tensor(*x))?;
                let tail = self.recurrent_tail_absolute(*state);
                kernels_cuda::ssm::causal_conv1d_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    bias.map(|b| self.tensor(b)),
                    &self.recurrent_absolute(*state),
                    *conv_width,
                    *dilation,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                kernels_cuda::ssm::causal_conv1d_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    bias.map(|b| self.tensor(b)),
                    &tail,
                    *conv_width,
                    *dilation,
                    &mut self.tensor(*y),
                )
            }
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } => {
                if self.rs_extended() {
                    const OP: &str = "ssm.gdn_gates";
                    let ba_ext = self.rs_extend(OP, *ba, true)?;
                    let mut gates_ext = self.rs_out(OP, *gates)?;
                    kernels_cuda::ssm::gdn_gates(
                        self.ctx(),
                        ba_ext,
                        self.tensor(*dt_bias),
                        self.tensor(*a_log),
                        &mut gates_ext,
                    )?;
                    return self.rs_land(OP, gates_ext, *gates);
                }
                self.rs_move("ssm.gdn_gates", *ba, self.tensor(*ba))?;
                kernels_cuda::ssm::gdn_gates(
                    self.ctx(),
                    self.tensor(*ba),
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    &mut self.tensor(*gates),
                )
            }
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
            } => {
                if self.rs_extended() {
                    return self.gated_delta_extended(
                        "ssm.gated_delta",
                        *qkv,
                        *z,
                        *gates,
                        *state,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        *y,
                    );
                }
                kernels_cuda::ssm::gated_delta(
                    self.ctx(),
                    self.tensor(*qkv),
                    self.tensor(*z),
                    self.tensor(*gates),
                    &self.recurrent_absolute(*state),
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    &mut self.tensor(*y),
                )
            }
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
            } => {
                if self.rs_extended() {
                    return self.gated_delta_extended(
                        "ssm.gated_delta_chunked",
                        *qkv,
                        *z,
                        *gates,
                        *state,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        *y,
                    );
                }
                let tail = self.recurrent_tail_absolute(*state);
                kernels_cuda::ssm::gated_delta_chunked(
                    self.ctx(),
                    self.ragged_lanes(*qkv),
                    self.tensor(*z),
                    self.tensor(*gates),
                    &self.recurrent_absolute(*state),
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                kernels_cuda::ssm::gated_delta_chunked(
                    self.ctx(),
                    self.ragged_lanes(*qkv),
                    self.tensor(*z),
                    self.tensor(*gates),
                    &tail,
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    &mut self.tensor(*y),
                )
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
                if self.rs_extended() {
                    return self.kda_extended(
                        "ssm.kda",
                        *mixed,
                        *f,
                        *b,
                        *dt_bias,
                        *a_log,
                        *state,
                        *heads,
                        *head_dim,
                        *norm_eps,
                        *gate_floor,
                        *y,
                    );
                }
                self.rs_move("ssm.kda", *f, self.tensor(*f))?;
                self.rs_move("ssm.kda", *b, self.tensor(*b))?;
                kernels_cuda::ssm::kda(
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
                    &mut self.tensor(*y),
                )
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
                let chunked = matches!(op, Ssm::Mamba2Chunked { .. });
                const OP: &str = "ssm.mamba2";
                if self.rs_extended() {
                    return Err(kernels_cuda::Error::Unsupported { op: OP });
                }
                let m = kernels_cuda::ssm::Mamba2 {
                    a_log: self.tensor(*a_log),
                    dt_bias: self.tensor(*dt_bias),
                    d: self.tensor(*d),
                    norm: self.tensor(*norm),
                    eps: *eps,
                    heads: *heads,
                    groups: *groups,
                };
                let (dt, z) = (self.tensor(*dt), self.tensor(*z));
                if !chunked {
                    return kernels_cuda::ssm::mamba2(
                        self.ctx(),
                        self.tensor(*xbc),
                        dt,
                        z,
                        m,
                        &self.recurrent(*state),
                        *state_dim,
                        &mut self.tensor(*y),
                    );
                }
                let tail = self.recurrent_tail_absolute(*state);
                kernels_cuda::ssm::mamba2_chunked(
                    self.ctx(),
                    self.ragged_lanes(*xbc),
                    dt,
                    z,
                    m,
                    &self.recurrent_absolute(*state),
                    *state_dim,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                kernels_cuda::ssm::mamba2_chunked(
                    self.ctx(),
                    self.ragged_lanes(*xbc),
                    dt,
                    z,
                    m,
                    &tail,
                    *state_dim,
                    &mut self.tensor(*y),
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
                if self.rs_extended() {
                    return self.kda_extended(
                        "ssm.kda_chunked",
                        *mixed,
                        *f,
                        *b,
                        *dt_bias,
                        *a_log,
                        *state,
                        *heads,
                        *head_dim,
                        *norm_eps,
                        *gate_floor,
                        *y,
                    );
                }
                self.rs_move("ssm.kda_chunked", *f, self.tensor(*f))?;
                self.rs_move("ssm.kda_chunked", *b, self.tensor(*b))?;
                let tail = self.recurrent_tail_absolute(*state);
                kernels_cuda::ssm::kda_chunked(
                    self.ctx(),
                    self.ragged_lanes(*mixed),
                    self.tensor(*f),
                    self.tensor(*b),
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    &self.recurrent_absolute(*state),
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                kernels_cuda::ssm::kda_chunked(
                    self.ctx(),
                    self.ragged_lanes(*mixed),
                    self.tensor(*f),
                    self.tensor(*b),
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    &tail,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                    &mut self.tensor(*y),
                )
            }
            Ssm::DynConv {
                x,
                coeff,
                base,
                side,
                taps,
                group,
                y,
            } => kernels_cuda::ssm::dyn_conv(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*coeff),
                self.tensor(*base),
                *side,
                *taps,
                *group,
                &mut self.tensor(*y),
            ),
        }
    }
}

impl Run<'_> {
    #[allow(clippy::too_many_arguments)]
    fn conv1d_extended(
        &self,
        op: &'static str,
        x: ValueId,
        weight: ValueId,
        bias: Option<ValueId>,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let bias = bias.map(|b| self.tensor(b));
        let x_ext = self.rs_extend(op, x, true)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        kernels_cuda::ssm::causal_conv1d_chunked(
            self.ctx(),
            RaggedTensor {
                data: x_ext,
                indptr: csr,
            },
            self.tensor(weight),
            bias,
            &self.recurrent_absolute(state),
            conv_width,
            dilation,
            &mut y_ext,
        )?;
        if let Some(tail) = tail {
            kernels_cuda::ssm::causal_conv1d_chunked(
                self.ctx(),
                RaggedTensor {
                    data: x_ext,
                    indptr: csr,
                },
                self.tensor(weight),
                bias,
                &tail,
                conv_width,
                dilation,
                &mut y_ext,
            )?;
        }
        self.rs_land(op, y_ext, y)
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_delta_extended(
        &self,
        op: &'static str,
        qkv: ValueId,
        z: ValueId,
        gates: ValueId,
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let qkv_ext = self.rs_ext_of(op, qkv)?;
        let gates_ext = self.rs_ext_of(op, gates)?;
        let z_ext = self.rs_extend(op, z, false)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        kernels_cuda::ssm::gated_delta_chunked(
            self.ctx(),
            RaggedTensor {
                data: qkv_ext,
                indptr: csr,
            },
            z_ext,
            gates_ext,
            &self.recurrent_absolute(state),
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            &mut y_ext,
        )?;
        if let Some(tail) = tail {
            kernels_cuda::ssm::gated_delta_chunked(
                self.ctx(),
                RaggedTensor {
                    data: qkv_ext,
                    indptr: csr,
                },
                z_ext,
                gates_ext,
                &tail,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                &mut y_ext,
            )?;
        }
        self.rs_land(op, y_ext, y)?;
        self.rs_layer_done();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn kda_extended(
        &self,
        op: &'static str,
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        gate_floor: f32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let mixed_ext = self.rs_ext_of(op, mixed)?;
        let f_ext = self.rs_extend(op, f, true)?;
        let b_ext = self.rs_extend(op, b, true)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        kernels_cuda::ssm::kda_chunked(
            self.ctx(),
            RaggedTensor {
                data: mixed_ext,
                indptr: csr,
            },
            f_ext,
            b_ext,
            self.tensor(dt_bias),
            self.tensor(a_log),
            &self.recurrent_absolute(state),
            heads,
            head_dim,
            norm_eps,
            gate_floor,
            &mut y_ext,
        )?;
        if let Some(tail) = tail {
            kernels_cuda::ssm::kda_chunked(
                self.ctx(),
                RaggedTensor {
                    data: mixed_ext,
                    indptr: csr,
                },
                f_ext,
                b_ext,
                self.tensor(dt_bias),
                self.tensor(a_log),
                &tail,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                &mut y_ext,
            )?;
        }
        self.rs_land(op, y_ext, y)?;
        self.rs_layer_done();
        Ok(())
    }
}
