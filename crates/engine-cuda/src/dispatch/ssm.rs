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
                // A buffering fire's decode step reads its lanes' carried
                // rows, through the gate prep whose `[b | a]` it lands: one
                // whose gates no buffered gate prep forms is refused by name,
                // never run as the plain step over the state alone.
                if self.rs_buffering() {
                    let gated = self
                        .rs_step_gates(*gates)
                        .then(|| self.gdn_gated(*gates))
                        .flatten()
                        .ok_or_else(|| kernels_cuda::Error::Backend {
                            op: "ssm.gated_delta",
                            detail: format!(
                                "a buffering fire's decode step whose gates (value {}) no \
                                 buffered gate prep forms: its carried rows would go unread",
                                gates.0
                            ),
                        })?;
                    return self.gated_delta_carried(
                        *qkv, *gates, gated, *state, *k_heads, *v_heads, *k_dim, *v_dim, *y,
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
                if !chunked && self.rs_buffered(*dt) {
                    return self.mamba2_carried(*xbc, *dt, *z, m, *state, *state_dim, *y);
                }
                if chunked && self.rs_extended() {
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
                let dt_id = *dt;
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
                )?;
                self.mamba2_tail(self.tensor(*xbc), false, [*xbc, dt_id], m, *state, *state_dim)?;
                Ok(())
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
                if let Some(tail) = tail {
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
                    )?;
                }
                self.kda_tail(
                    [self.tensor(*mixed), self.tensor(*f), self.tensor(*b)],
                    false,
                    [*mixed, *f, *b, *y],
                    *dt_bias,
                    *a_log,
                    *state,
                    (*heads, *head_dim, *norm_eps, *gate_floor),
                )?;
                Ok(())
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
    fn gdn_gated(&self, gates: ValueId) -> Option<(ValueId, ValueId, ValueId)> {
        let poem_ir::Def::Op(node) = self.values().get(gates.0 as usize)?.def else {
            return None;
        };
        match &self.nodes().get(node as usize)?.op {
            poem_ir::Operation::Ssm(Ssm::GdnGates {
                ba, dt_bias, a_log, ..
            })
            // The fused decode step's gate prep, run as its primitives.
            | poem_ir::Operation::Fused(poem_ir::Fused::GdnStep {
                ba, dt_bias, a_log, ..
            }) => Some((*ba, *dt_bias, *a_log)),
            _ => None,
        }
    }

    /// A gated-delta layer's buffered planes, as its kernels read them: the
    /// rows' `[b | a]`, keys and corrected values.
    pub(crate) fn gdn_planes(
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
        let carry = self.rs_carry(OP, &self.gdn_planes(OP, ba, y)?, &pool)?;
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

    /// The fire's beam group folds (`BEAM-PAGE-LISTS.md` §11.3), every
    /// recurrent layer's, launched on the fire's stream ahead of its body: a
    /// fold reads only the rows earlier fires kept in their buffer, never
    /// this fire's, so each layer's state is folded before any of the
    /// fire's steps reads it, by stream order alone. Each layer's state once,
    /// whichever of its decode ops names it.
    pub(crate) fn fold_groups_ahead(&self) -> Result<(), kernels_cuda::Error> {
        use poem_ir::{Fused, Operation};
        let mut folded: Vec<ValueId> = Vec::new();
        let mut first = |state: ValueId| {
            let new = !folded.contains(&state);
            if new {
                folded.push(state);
            }
            new
        };
        for node in self.nodes() {
            match &node.op {
                Operation::Ssm(Ssm::CausalConv1d {
                    x,
                    state,
                    conv_width,
                    dilation,
                    ..
                }) if first(*state) => {
                    self.conv1d_group_fold(*x, *state, *conv_width, *dilation)?;
                }
                Operation::Ssm(Ssm::GatedDelta {
                    gates,
                    state,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                }) if first(*state) => {
                    let gated = self.gdn_gated(*gates).ok_or_else(|| kernels_cuda::Error::Backend {
                        op: "ssm.gated_delta",
                        detail: format!(
                            "a beam group's fold over a step whose gates (value {}) no gate \
                             prep forms, so its rows' gates cannot be read back",
                            gates.0
                        ),
                    })?;
                    self.gated_delta_group_fold(gated, *state, (*k_heads, *v_heads, *k_dim, *v_dim), *y)?;
                }
                Operation::Fused(Fused::GdnStep {
                    qkv,
                    conv_state,
                    conv_width,
                    dilation,
                    ba,
                    dt_bias,
                    a_log,
                    state,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                }) => {
                    if first(*conv_state) {
                        self.conv1d_group_fold(*qkv, *conv_state, *conv_width, *dilation)?;
                    }
                    if first(*state) {
                        self.gated_delta_group_fold(
                            (*ba, *dt_bias, *a_log),
                            *state,
                            (*k_heads, *v_heads, *k_dim, *v_dim),
                            *y,
                        )?;
                    }
                }
                Operation::Ssm(Ssm::Kda {
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
                }) if first(*state) => {
                    const OP: &str = "ssm.kda";
                    let planes = self.rs_step_planes(OP, &[*f, *b], *mixed, Some(*y))?;
                    if let Some((carry, folds, groups)) = self.rs_group_folds(OP, &planes)? {
                        kernels_cuda::ssm::kda_fold(
                            self.ctx(),
                            &one_token(self.recurrent_absolute(*state)),
                            *heads,
                            *head_dim,
                            *norm_eps,
                            *gate_floor,
                            self.tensor(*dt_bias),
                            self.tensor(*a_log),
                            &carry,
                            folds,
                            groups,
                        )?;
                    }
                }
                Operation::Ssm(Ssm::Mamba2 {
                    xbc,
                    dt,
                    a_log,
                    dt_bias,
                    d,
                    norm,
                    eps,
                    state,
                    heads,
                    groups,
                    ..
                }) if first(*state) => {
                    const OP: &str = "ssm.mamba2";
                    let m = kernels_cuda::ssm::Mamba2 {
                        a_log: self.tensor(*a_log),
                        dt_bias: self.tensor(*dt_bias),
                        d: self.tensor(*d),
                        norm: self.tensor(*norm),
                        eps: *eps,
                        heads: *heads,
                        groups: *groups,
                    };
                    let planes = self.rs_step_planes(OP, &[*dt], *xbc, None)?;
                    if let Some((carry, folds, count)) = self.rs_group_folds(OP, &planes)? {
                        kernels_cuda::ssm::mamba2_fold(
                            self.ctx(),
                            &one_token(self.recurrent_absolute(*state)),
                            m,
                            &carry,
                            folds,
                            count,
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The beam groups' conv window folds before the layer's step (§11.3).
    pub(crate) fn conv1d_group_fold(
        &self,
        x: ValueId,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.causal_conv1d";
        let Some((carry, folds, groups)) = self.rs_group_folds(OP, &[self.rs_plane(OP, x)?])?
        else {
            return Ok(());
        };
        kernels_cuda::ssm::conv1d_fold(
            self.ctx(),
            &one_token(self.recurrent_absolute(state)),
            self.tensor(x).width,
            conv_width,
            dilation,
            &carry,
            folds,
            groups,
        )
    }

    /// The beam groups' gated-delta folds before the layer's step (§11.3).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gated_delta_group_fold(
        &self,
        (ba, dt_bias, a_log): (ValueId, ValueId, ValueId),
        state: ValueId,
        (k_heads, v_heads, k_dim, v_dim): (u32, u32, u32, u32),
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.gated_delta";
        let Some((carry, folds, groups)) =
            self.rs_group_folds(OP, &self.gdn_planes(OP, ba, y)?)?
        else {
            return Ok(());
        };
        kernels_cuda::ssm::gated_delta_fold(
            self.ctx(),
            &one_token(self.recurrent_absolute(state)),
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            self.tensor(dt_bias),
            self.tensor(a_log),
            &carry,
            folds,
            groups,
        )
    }

    /// A decode conv of a buffering fire: each lane's own row reads its
    /// window from the committed cells and its last carried rows
    /// ([`kernels_cuda::ssm::Carry`]), and lands in its buffer the channels
    /// its recurrence reads a buffered row by (a gated-delta or KDA layer's
    /// keys, a Mamba-2 layer's `x | B`); a lone conv's outputs nothing
    /// reads.
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
        let (planes, keys) = match self.rs_planes().and_then(|planes| planes.keys_of(y)) {
            Some((k_plane, k_off, k_len)) => (vec![x_plane, k_plane], Some((k_off, k_len))),
            None if self.rs_lone(x) => (vec![x_plane], None),
            None => {
                return Err(kernels_cuda::Error::Backend {
                    op: OP,
                    detail: format!(
                        "the decode conv landing value {} feeds a recurrence that keeps none \
                         of its channels in its buffer, so a later step cannot read the row",
                        y.0
                    ),
                });
            }
        };
        let carry = self.rs_carry(OP, &planes, &pool)?;
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
        let _ = (dt_bias, a_log);
        let pool = self.recurrent_absolute(state);
        let _ = qkv;
        let carry = self.rs_carry(OP, &self.gdn_planes(OP, ba, y)?, &pool)?;
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

    /// The buffered planes a KDA or Mamba-2 layer's kernels read: the
    /// planes of `reads` (its read values), then the conv output `conv_y`'s
    /// kept channels and, where the recurrence landing `y` keeps them, its
    /// corrected values.
    fn rs_step_planes(
        &self,
        op: &'static str,
        reads: &[ValueId],
        conv_y: ValueId,
        y: Option<ValueId>,
    ) -> Result<Vec<crate::store::rs::Plane>, kernels_cuda::Error> {
        let missing = |what: &str, id: ValueId| kernels_cuda::Error::Backend {
            op,
            detail: format!(
                "value {} keeps no {what} in its buffer, so a buffered row's step cannot be \
                 read back",
                id.0
            ),
        };
        let mut planes = reads
            .iter()
            .map(|&id| self.rs_plane(op, id))
            .collect::<Result<Vec<_>, _>>()?;
        let kept = self
            .rs_planes()
            .and_then(|planes| planes.keys_of(conv_y))
            .ok_or_else(|| missing("convolved channels", conv_y))?;
        planes.push(kept.0);
        if let Some(y) = y {
            let corrected = self
                .rs_planes()
                .and_then(|planes| planes.corrected(y))
                .ok_or_else(|| missing("corrected values", y))?;
            planes.push(corrected.u);
        }
        Ok(planes)
    }

    /// A KDA decode step of a buffering fire, over each lane's carried rows
    /// then its own: their forget and beta rows, keys and corrected values
    /// from their buffer, read as a linear attention.
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
        let planes = self.rs_step_planes(OP, &[f, b], mixed, Some(y))?;
        let carry = self.rs_carry(OP, &planes, &pool)?;
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

    /// The rows a chunked KDA step of a buffering fire lands past its lanes'
    /// commits, made ready for a later decode: their keys and corrected
    /// values into their buffer, from `mixed`, `f` and `b` (the fire's own
    /// rows, or with `ext` its extended run's).
    #[allow(clippy::too_many_arguments)]
    fn kda_tail(
        &self,
        rows: [kernels_cuda::Tensor; 3],
        ext: bool,
        [mixed, f, b, y]: [ValueId; 4],
        dt_bias: ValueId,
        a_log: ValueId,
        state: ValueId,
        (heads, head_dim, norm_eps, gate_floor): (u32, u32, f32, f32),
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.kda_chunked";
        if self.rs_planes().is_none() {
            return Ok(());
        }
        let pool = self.recurrent_absolute(state);
        let carry = self.rs_carry(OP, &self.rs_step_planes(OP, &[f, b], mixed, Some(y))?, &pool)?;
        let [mixed_rows, f_rows, b_rows] = rows;
        kernels_cuda::ssm::kda_tail(
            self.ctx(),
            mixed_rows,
            f_rows,
            b_rows,
            self.tensor(dt_bias),
            self.tensor(a_log),
            &pool,
            self.rs_qo_indptr(),
            ext,
            heads,
            head_dim,
            norm_eps,
            gate_floor,
            &carry,
        )
    }

    /// A Mamba-2 decode step of a buffering fire, over each lane's carried
    /// rows then its own: their step sizes and convolved `x | B` from their
    /// buffer, read as a linear attention.
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
        let planes = self.rs_step_planes(OP, &[dt], xbc, None)?;
        let carry = self.rs_carry(OP, &planes, &pool)?;
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

    /// The rows a chunked Mamba-2 step of a buffering fire lands past its
    /// lanes' commits, made ready for a later decode: their convolved
    /// `x | B` into their buffer, from `xbc_rows` (the fire's own rows, or
    /// with `ext` its extended run's).
    fn mamba2_tail(
        &self,
        xbc_rows: kernels_cuda::Tensor,
        ext: bool,
        [xbc, dt]: [ValueId; 2],
        m: kernels_cuda::ssm::Mamba2,
        state: ValueId,
        state_dim: u32,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "ssm.mamba2_chunked";
        if self.rs_planes().is_none() {
            return Ok(());
        }
        let pool = self.recurrent_absolute(state);
        let carry = self.rs_carry(OP, &self.rs_step_planes(OP, &[dt], xbc, None)?, &pool)?;
        kernels_cuda::ssm::mamba2_tail(
            self.ctx(),
            xbc_rows,
            m,
            &pool,
            self.rs_qo_indptr(),
            ext,
            state_dim,
            &carry,
        )
    }

    /// ROCm's KDA decode step reads no buffered rows as a linear attention
    /// yet (its kernels re-step them from a carried scratch the buffer no
    /// longer lays): a buffering fire over a KDA layer is refused by name.

    /// ROCm's Mamba-2 decode step, as [`Run::kda_carried`]'s: unported.

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
            self.mamba2_tail(xbc_ext, true, [xbc, dt], m, state, state_dim)?;
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
            self.kda_tail(
            [mixed_ext, f_ext, b_ext],
            true,
            [mixed, f, b, y],
            dt_bias,
            a_log,
            state,
            (heads, head_dim, norm_eps, gate_floor),
        )?;
        self.rs_land(op, y_ext, y)?;
        self.rs_layer_done();
        Ok(())
    }
}


/// The group folds a fire over `trace`'s recurrent steps can launch ahead
/// of its body ([`Run::fold_groups_ahead`]), each kind once: a load makes
/// their kernels before ready, since no arming fire carries a beam group's
/// fold.
pub(crate) fn group_folds(trace: &poem_ir::Trace) -> Vec<kernels_cuda::ssm::FoldKernel> {
    use kernels_cuda::ssm::FoldKernel;
    use poem_ir::{Fused, Operation};
    let mut folds = Vec::new();
    let mut add = |fold: FoldKernel| {
        if !folds.contains(&fold) {
            folds.push(fold);
        }
    };
    for node in &trace.nodes {
        match &node.op {
            Operation::Ssm(Ssm::CausalConv1d { .. }) => add(FoldKernel::Conv1d),
            Operation::Ssm(Ssm::GatedDelta { .. }) => add(FoldKernel::GatedDelta),
            Operation::Fused(Fused::GdnStep { .. }) => {
                add(FoldKernel::Conv1d);
                add(FoldKernel::GatedDelta);
            }
            Operation::Ssm(Ssm::Kda { head_dim, .. }) => add(FoldKernel::Kda {
                head_dim: *head_dim,
            }),
            Operation::Ssm(Ssm::Mamba2 { .. }) => add(FoldKernel::Mamba2),
            _ => {}
        }
    }
    folds
}
