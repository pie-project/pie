use crate::run::Run;
use kernels_cerebras::attn;
use poem_exec::{DispatchSsm, KernelError};
use poem_ir::{Operands, Ssm};

impl DispatchSsm for Run<'_> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.ssm(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn ssm(&mut self, op: &Ssm) -> Result<(), kernels_cerebras::Error> {
        match op {
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
                ..
            } => attn::ssm::causal_conv1d(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
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
                ..
            } => attn::ssm::causal_conv1d_chunked(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*weight),
                &self.recurrent(*state),
                *conv_width,
                *dilation,
                self.tensor(*y),
            ),
            Ssm::GdnGates {
                ba,
                dt_bias,
                a_log,
                gates,
            } => attn::ssm::gdn_prep(
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
            } => attn::ssm::gated_delta(
                self.ctx(),
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
            } => attn::ssm::gated_delta_chunked(
                self.ctx(),
                self.ragged(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                &self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y),
            ),
            Ssm::DynConv {
                x,
                coeff,
                base,
                side,
                taps,
                group,
                y,
            } => attn::dynconv::block_dyn_conv(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*coeff),
                self.tensor(*base),
                *side,
                *taps,
                *group,
                self.tensor(*y),
            ),
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
