use crate::encode::{from_slang, slang_ragged, slang_tensor};
use crate::run::Run;
use kernels_slang::Tier;
use kernels_slang::ssm::{StatePool, conv1d, gdn_prep, gdn_recurrent, kda};
use poem_exec::{DispatchSsm, KernelError};
use poem_ir::{Operands, Ssm, ValueId};

/// The recurrences fold their norms through workgroup memory, as WGSL does
/// everywhere; a chunked or committed recurrence holds its state in
/// registers from 64 rows, the threshold this engine has always run at.
const RECURRENT: gdn_recurrent::Tactic = gdn_recurrent::Tactic {
    tier: Tier::Baseline,
    register_min_rows: 64,
};

impl DispatchSsm for Run<'_> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        self.ssm(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn state_pool(&self, id: ValueId) -> StatePool {
        let pool = self.recurrent(id);
        StatePool {
            bank: slang_tensor(pool.state),
            slots: slang_tensor(pool.slots),
        }
    }

    fn slang_of(&self, id: ValueId) -> kernels_slang::Tensor {
        slang_tensor(self.tensor(id))
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

    fn ssm(&mut self, op: &Ssm) -> Result<(), kernels_wgpu::Error> {
        let ctx = self.slang();
        match op {
            Ssm::Mamba2 { .. }
            | Ssm::Mamba2Chunked { .. }
            | Ssm::DynConv { .. }
            | Ssm::ShortConv { .. }
            | Ssm::ShortConvChunked { .. } => {
                Err(kernels_wgpu::Error::Unsupported { op: op.name() })
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
                conv1d::conv1d_committed(
                    &ctx,
                    slang_tensor(ext_x),
                    slang_tensor(self.qo_indptr()),
                    &self.rs_committed(&seat),
                    self.slang_of(*weight),
                    &self.state_pool(*state),
                    *conv_width,
                    *dilation,
                    slang_tensor(ext_y),
                )
                .map_err(from_slang)?;
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
                    &ctx,
                    slang_tensor(ext_ba),
                    self.slang_of(*dt_bias),
                    self.slang_of(*a_log),
                    slang_tensor(ext_gates),
                )
                .map_err(from_slang)?;
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
                gdn_recurrent::gated_delta_committed(
                    &ctx,
                    RECURRENT,
                    slang_tensor(ext_qkv),
                    slang_tensor(self.qo_indptr()),
                    &self.rs_committed(&seat),
                    slang_tensor(ext_gates),
                    &self.state_pool(*state),
                    slang_tensor(seat.work),
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    slang_tensor(ext_y),
                )
                .map_err(from_slang)?;
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
                    slang_tensor(ext_f),
                    slang_tensor(ext_b),
                    *dt_bias,
                    *a_log,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                );
                kda::kda_committed(
                    &ctx,
                    RECURRENT.tier,
                    slang_tensor(ext_mixed),
                    slang_tensor(self.qo_indptr()),
                    &self.rs_committed(&seat),
                    &gates,
                    &self.state_pool(*state),
                    slang_tensor(seat.work),
                    slang_tensor(ext_y),
                )
                .map_err(from_slang)?;
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
                &ctx,
                self.slang_of(*x),
                self.slang_of(*weight),
                &self.state_pool(*state),
                *conv_width,
                *dilation,
                self.slang_of(*y),
            )
            .map_err(from_slang),
            Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } => conv1d::conv1d_chunked(
                &ctx,
                slang_ragged(self.ragged(*x)),
                self.slang_of(*weight),
                &self.state_pool(*state),
                *conv_width,
                *dilation,
                self.slang_of(*y),
            )
            .map_err(from_slang),
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } => gdn_prep::gdn_prep(
                &ctx,
                self.slang_of(*ba),
                self.slang_of(*dt_bias),
                self.slang_of(*a_log),
                self.slang_of(*gates),
            )
            .map_err(from_slang),
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
                &ctx,
                RECURRENT,
                self.slang_of(*qkv),
                self.slang_of(*gates),
                &self.state_pool(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.slang_of(*y),
            )
            .map_err(from_slang),
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
                &ctx,
                RECURRENT,
                slang_ragged(self.ragged(*qkv)),
                self.slang_of(*gates),
                &self.state_pool(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.slang_of(*y),
            )
            .map_err(from_slang),
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
                    &ctx,
                    RECURRENT.tier,
                    self.slang_of(*mixed),
                    &gates,
                    &self.state_pool(*state),
                    self.slang_of(*y),
                )
                .map_err(from_slang)
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
                    &ctx,
                    RECURRENT.tier,
                    slang_ragged(self.ragged(*mixed)),
                    &gates,
                    &self.state_pool(*state),
                    self.slang_of(*y),
                )
                .map_err(from_slang)
            }
        }
    }
}
