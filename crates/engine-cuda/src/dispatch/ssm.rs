use crate::run::{Run, one_token};
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
                #[cfg(not(feature = "rocm"))]
                if self.rs_buffered(*x) {
                    return self.conv1d_carried(
                        *x,
                        *weight,
                        *bias,
                        *state,
                        *conv_width,
                        *dilation,
                        *y,
                    );
                }
                // ROCm replays a decode step's carried rows extended.
                #[cfg(feature = "rocm")]
                {
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
                }
                kernels_cuda::ssm::causal_conv1d(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*weight),
                    bias.map(|b| self.tensor(b)),
                    &one_token(self.recurrent_absolute(*state)),
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
                &one_token(self.recurrent_absolute(*state)),
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
                // A step's `[b | a]` lands in the buffer with its recurrence
                // (`gdn_step_carried`).
                let step = self.rs_step_gates(*gates);
                if self.rs_extended() && !step {
                    const OP: &str = "ssm.gdn_gates";
                    self.rs_begin();
                    let ba_ext = self.rs_extend(OP, *ba, true)?;
                    let mut gates_ext = self.rs_out(OP, *gates)?;
                    kernels_cuda::ssm::gdn_gates(
                        self.rs_ext_ctx(OP)?,
                        ba_ext,
                        self.tensor(*dt_bias),
                        self.tensor(*a_log),
                        &mut gates_ext,
                    )?;
                    return self.rs_land(OP, gates_ext, *gates);
                }
                if !step {
                    self.rs_move("ssm.gdn_gates", *ba, self.tensor(*ba))?;
                }
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
                #[cfg(not(feature = "rocm"))]
                if self.rs_step_gates(*gates)
                    && let Some(gated) = self.gdn_gated(*gates)
                {
                    return self.gated_delta_carried(
                        *qkv, *gates, gated, *state, *k_heads, *v_heads, *k_dim, *v_dim, *y,
                    );
                }
                #[cfg(feature = "rocm")]
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
                    &one_token(self.recurrent_absolute(*state)),
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
                if let Some(tail) = tail {
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
                    )?;
                }
                #[cfg(not(feature = "rocm"))]
                self.gated_delta_tail(
                    self.tensor(*qkv),
                    self.tensor(*gates),
                    *gates,
                    false,
                    *state,
                    (*k_heads, *v_heads, *k_dim, *v_dim),
                    *y,
                )?;
                Ok(())
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
                #[cfg(not(feature = "rocm"))]
                if self.rs_buffered(*f) {
                    return self.kda_carried(
                        *mixed,
                        [*f, *b],
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
                #[cfg(feature = "rocm")]
                {
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
                }
                kernels_cuda::ssm::kda(
                    self.ctx(),
                    self.tensor(*mixed),
                    self.tensor(*f),
                    self.tensor(*b),
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    &one_token(self.recurrent(*state)),
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
                let m = kernels_cuda::ssm::Mamba2 {
                    a_log: self.tensor(*a_log),
                    dt_bias: self.tensor(*dt_bias),
                    d: self.tensor(*d),
                    norm: self.tensor(*norm),
                    eps: *eps,
                    heads: *heads,
                    groups: *groups,
                };
                #[cfg(not(feature = "rocm"))]
                if !chunked && self.rs_buffered(*dt) {
                    return self.mamba2_carried(*xbc, *dt, *z, m, *state, *state_dim, *y);
                }
                if (chunked || cfg!(feature = "rocm")) && self.rs_extended() {
                    return self.mamba2_extended(
                        "ssm.mamba2",
                        *xbc,
                        *dt,
                        *z,
                        m,
                        *state,
                        *state_dim,
                        *y,
                    );
                }
                self.rs_move("ssm.mamba2", *dt, self.tensor(*dt))?;
                let (dt, z) = (self.tensor(*dt), self.tensor(*z));
                if !chunked {
                    return kernels_cuda::ssm::mamba2(
                        self.ctx(),
                        self.tensor(*xbc),
                        dt,
                        z,
                        m,
                        &one_token(self.recurrent(*state)),
                        *state_dim,
                        &mut self.tensor(*y),
                    );
                }
                kernels_cuda::ssm::mamba2_chunked(
                    self.ctx(),
                    self.ragged_lanes(*xbc),
                    dt,
                    z,
                    m,
                    &self.recurrent_absolute(*state),
                    self.recurrent_tail_absolute(*state).as_ref(),
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
    /// The gate prep `gates` comes from: its `[b | a]`, `dt_bias` and
    /// `A_log`.
    #[cfg(not(feature = "rocm"))]
    fn gdn_gated(&self, gates: ValueId) -> Option<(ValueId, ValueId, ValueId)> {
        let poem_ir::Def::Op(node) = self.values().get(gates.0 as usize)?.def else {
            return None;
        };
        match &self.nodes().get(node as usize)?.op {
            poem_ir::Operation::Ssm(Ssm::GdnGates {
                ba, dt_bias, a_log, ..
            }) => Some((*ba, *dt_bias, *a_log)),
            _ => None,
        }
    }

    /// A gated-delta layer's buffered planes, as its kernels read them: the
    /// rows' `[b | a]`, keys and corrected values.
    #[cfg(not(feature = "rocm"))]
    fn gdn_planes(
        &self,
        op: &'static str,
        ba: ValueId,
        y: ValueId,
    ) -> Result<Vec<crate::store::rs::Plane>, kernels_cuda::Error> {
        let corrected = self
            .rs_planes()
            .and_then(|planes| planes.corrected(y))
            .ok_or_else(|| kernels_cuda::Error::Backend {
                op,
                detail: format!(
                    "the recurrence landing value {} keeps no corrected rows in its buffer, so \
                     a buffered row's step cannot be read back",
                    y.0
                ),
            })?;
        Ok(vec![self.rs_plane(op, ba)?, corrected.k, corrected.u])
    }

    /// The rows a chunked gated-delta step of a buffering fire lands past its
    /// lanes' commits, made ready for a later decode: their corrected values
    /// and keys into their buffer, from `qkv` and `gates` (the fire's own
    /// rows, or with `ext` its extended run's).
    #[cfg(not(feature = "rocm"))]
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_tail(
        &self,
        qkv: kernels_cuda::Tensor,
        gates_rows: kernels_cuda::Tensor,
        gates: ValueId,
        ext: bool,
        state: ValueId,
        (k_heads, v_heads, k_dim, v_dim): (u32, u32, u32, u32),
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.gated_delta_chunked";
        if self.rs_planes().is_none() {
            return Ok(());
        }
        let Some((ba, dt_bias, a_log)) = self.gdn_gated(gates) else {
            return Err(kernels_cuda::Error::Backend {
                op: OP,
                detail: "a buffering fire's chunked recurrence whose gates no gate prep forms"
                    .to_string(),
            });
        };
        let pool = self.recurrent_absolute(state);
        let carry = self.rs_carry(OP, &self.gdn_planes(OP, ba, y)?, 0, &pool)?;
        kernels_cuda::ssm::gated_delta_tail(
            self.ctx(),
            qkv,
            gates_rows,
            self.tensor(dt_bias),
            self.tensor(a_log),
            &pool,
            self.rs_qo_indptr(),
            ext,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            &carry,
        )
    }

    /// A decode conv of a buffering fire: each lane's own row reads its
    /// window from the committed cells and its last carried rows
    /// ([`kernels_cuda::ssm::Carry`]). A gated-delta layer's conv lands the
    /// row's keys in its buffer for the step's linear attention; a KDA or
    /// Mamba-2 layer's lands its carried rows' outputs in the carried
    /// scratch for the step that re-steps them; a lone conv's outputs
    /// nothing reads.
    #[cfg(not(feature = "rocm"))]
    #[allow(clippy::too_many_arguments)]
    fn conv1d_carried(
        &self,
        x: ValueId,
        weight: ValueId,
        bias: Option<ValueId>,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.causal_conv1d";
        let pool = self.recurrent_absolute(state);
        let x_plane = self.rs_plane(OP, x)?;
        let keys = self.rs_planes().and_then(|planes| planes.keys_of(y));
        let (scratch, planes, keys) = match keys {
            Some((k_plane, k_off, _)) => (
                0,
                vec![x_plane, k_plane],
                Some((k_off, u32::try_from(k_plane.width).unwrap_or(u32::MAX))),
            ),
            None if self.rs_lone(x) => (0, vec![x_plane], None),
            None => (self.rs_carried_out(OP, y)?, vec![x_plane], None),
        };
        let carry = self.rs_carry(OP, &planes, scratch, &pool)?;
        kernels_cuda::ssm::causal_conv1d_carried(
            self.ctx(),
            self.tensor(x),
            self.tensor(weight),
            bias.map(|b| self.tensor(b)),
            &one_token(pool),
            conv_width,
            dilation,
            &mut self.tensor(y),
            &carry,
            keys,
        )
    }

    /// A gated-delta decode step of a buffering fire, over each lane's
    /// buffered rows then its own: the rows' gates, keys and corrected values
    /// from their buffer, read as a linear attention.
    #[cfg(not(feature = "rocm"))]
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_carried(
        &self,
        qkv: ValueId,
        gates: ValueId,
        (ba, dt_bias, a_log): (ValueId, ValueId, ValueId),
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.gated_delta";
        let pool = self.recurrent_absolute(state);
        let _ = qkv;
        let carry = self.rs_carry(OP, &self.gdn_planes(OP, ba, y)?, 0, &pool)?;
        kernels_cuda::ssm::gated_delta_carried(
            self.ctx(),
            self.tensor(qkv),
            self.tensor(gates),
            &one_token(pool),
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            &mut self.tensor(y),
            kernels_cuda::ssm::GdnGated {
                ba: self.tensor(ba),
                dt_bias: self.tensor(dt_bias),
                a_log: self.tensor(a_log),
            },
            &carry,
        )
    }

    /// A KDA decode step of a buffering fire, over each lane's carried rows
    /// then its own: their `[q | k | v]` from the carried scratch, their
    /// forget and beta rows from their buffer.
    #[cfg(not(feature = "rocm"))]
    #[allow(clippy::too_many_arguments)]
    fn kda_carried(
        &self,
        mixed: ValueId,
        [f, b]: [ValueId; 2],
        dt_bias: ValueId,
        a_log: ValueId,
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        gate_floor: f32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.kda";
        let pool = self.recurrent_absolute(state);
        let scratch = self.rs_carried_of(OP, mixed)?;
        let carry = self.rs_carry(OP, &[self.rs_plane(OP, f)?, self.rs_plane(OP, b)?], scratch, &pool)?;
        kernels_cuda::ssm::kda_carried(
            self.ctx(),
            self.tensor(mixed),
            self.tensor(f),
            self.tensor(b),
            self.tensor(dt_bias),
            self.tensor(a_log),
            &one_token(pool),
            heads,
            head_dim,
            norm_eps,
            gate_floor,
            &mut self.tensor(y),
            &carry,
        )
    }

    /// A Mamba-2 decode step of a buffering fire, over each lane's carried
    /// rows then its own: their convolved `xbc` from the carried scratch,
    /// their step sizes from their buffer.
    #[cfg(not(feature = "rocm"))]
    #[allow(clippy::too_many_arguments)]
    fn mamba2_carried(
        &self,
        xbc: ValueId,
        dt: ValueId,
        z: ValueId,
        m: kernels_cuda::ssm::Mamba2,
        state: ValueId,
        state_dim: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.mamba2";
        let pool = self.recurrent_absolute(state);
        let scratch = self.rs_carried_of(OP, xbc)?;
        let carry = self.rs_carry(OP, &[self.rs_plane(OP, dt)?], scratch, &pool)?;
        kernels_cuda::ssm::mamba2_carried(
            self.ctx(),
            self.tensor(xbc),
            self.tensor(dt),
            self.tensor(z),
            m,
            &one_token(pool),
            state_dim,
            &mut self.tensor(y),
            &carry,
        )
    }

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
        self.rs_begin();
        // A conv no recurrence reads is a layer of its own.
        let lone = self.rs_lone(x);
        if lone {
            self.rs_begin_lone();
        }
        let x_ext = self.rs_extend(op, x, true)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        kernels_cuda::ssm::causal_conv1d_chunked(
            self.rs_ext_ctx(op)?,
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
                self.rs_ext_ctx(op)?,
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
        self.rs_land(op, y_ext, y)?;
        if lone {
            self.rs_layer_done();
        }
        Ok(())
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
        self.rs_begin();
        let qkv_ext = self.rs_ext_of(op, qkv)?;
        let gates_ext = self.rs_ext_of(op, gates)?;
        let z_ext = self.rs_extend(op, z, false)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        kernels_cuda::ssm::gated_delta_chunked(
            self.rs_ext_ctx(op)?,
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
                self.rs_ext_ctx(op)?,
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
        #[cfg(not(feature = "rocm"))]
        self.gated_delta_tail(
            qkv_ext,
            gates_ext,
            gates,
            true,
            state,
            (k_heads, v_heads, k_dim, v_dim),
            y,
        )?;
        self.rs_land(op, y_ext, y)?;
        self.rs_layer_done();
        Ok(())
    }

    /// A Mamba-2 layer over extended rows: the conv landed `xbc`'s
    /// ([`Run::rs_ext_of`]), the step sizes replay from the buffer, the gate
    /// reads only the rows the fire lands.
    #[allow(clippy::too_many_arguments)]
    fn mamba2_extended(
        &self,
        op: &'static str,
        xbc: ValueId,
        dt: ValueId,
        z: ValueId,
        m: kernels_cuda::ssm::Mamba2,
        state: ValueId,
        state_dim: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        self.rs_begin();
        let xbc_ext = self.rs_ext_of(op, xbc)?;
        let dt_ext = self.rs_extend(op, dt, true)?;
        let z_ext = self.rs_extend(op, z, false)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        kernels_cuda::ssm::mamba2_chunked(
            self.rs_ext_ctx(op)?,
            RaggedTensor {
                data: xbc_ext,
                indptr: csr,
            },
            dt_ext,
            z_ext,
            m,
            &self.recurrent_absolute(state),
            self.recurrent_tail_absolute(state).as_ref(),
            state_dim,
            &mut y_ext,
        )?;
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
        self.rs_begin();
        let mixed_ext = self.rs_ext_of(op, mixed)?;
        let f_ext = self.rs_extend(op, f, true)?;
        let b_ext = self.rs_extend(op, b, true)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        kernels_cuda::ssm::kda_chunked(
            self.rs_ext_ctx(op)?,
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
                self.rs_ext_ctx(op)?,
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
