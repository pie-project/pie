//! The `elementwise` family: `impl DispatchElementwise for Run<'_>`, holding
//! the norm arms plus the absorbed `rope`, `gate`, and `hc` groups.

use kernels_metal::{Tensor, elemwise};
use model_exec::{DispatchElementwise, KernelError};
use model_ir::{Elementwise, MropeForm, Operands};

use crate::run::Run;

impl DispatchElementwise for Run<'_> {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        self.elementwise(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-metal`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn elementwise(&mut self, op: &Elementwise) -> Result<(), kernels_metal::Error> {
        match op {
            Elementwise::Rmsnorm { x, weight, eps, y } => elemwise::norm::rmsnorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormPerHead {
                x,
                weight,
                head_dim,
                eps,
                y,
            } => elemwise::norm::rmsnorm_per_head(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormPlusOne { x, weight, eps, y } => elemwise::norm::rmsnorm_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormPerHeadPlusOne {
                x,
                weight,
                head_dim,
                eps,
                y,
            } => elemwise::norm::rmsnorm_per_head_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormNoScale {
                x,
                head_dim,
                eps,
                y,
            } => elemwise::norm::rmsnorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            // **THE CENTRED NORMS AND THE CLAMP, REFUSED BY NAME**
            // (multimodal §6.1, §6.5, next.md B5). This plane's
            // `elemwise::norm` ships no centred entry — neither the scale-less
            // one nor the fused whole-`LayerNorm` beside it — and its
            // `elemwise` no free-standing clamp, so the arms that exist here
            // are the names — which is the family's rule
            // the other way round from `attn::dense`'s: a written shader gets
            // a forwarding arm, an unwritten one gets its own refusal rather
            // than a shader that norms the wrong thing. One `kernels-metal`
            // entry each retires these, and nothing else moves.
            Elementwise::LayernormNoScale { .. }
            | Elementwise::Layernorm { .. }
            | Elementwise::Clamp { .. }
            | Elementwise::ClampLearned { .. }
            // qwen4's gated-residual family: not on this plane yet.
            | Elementwise::RmsnormGroupedPlusOne { .. }
            | Elementwise::SiluScaled { .. }
            | Elementwise::HcMix { .. }
            | Elementwise::HcInject { .. }
            | Elementwise::PleGate { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            Elementwise::RmsnormGated {
                act: model_ir::GateActivation::Sigmoid,
                ..
            } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Elementwise::RmsnormGated {
                x,
                gate,
                weight,
                head_dim,
                eps,
                act: _,
                y,
            } => elemwise::norm::rmsnorm_gated(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormGatedBy {
                x,
                gate,
                weight,
                heads,
                eps,
                y,
            } => elemwise::norm::rmsnorm_gated_by(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *heads,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::ResidualAdd { x, y, y_out: _ } => {
                elemwise::norm::residual_add(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            Elementwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),
            Elementwise::MulScalar { s, x, x_out: _ } => {
                elemwise::norm::mul_scalar(self.ctx(), *s, self.tensor(*x))
            }
            Elementwise::Scale { s, x, x_out: _ } => {
                elemwise::norm::scale(self.ctx(), self.tensor(*s), self.tensor(*x))
            }
            Elementwise::ResBlend {
                prefix,
                blocks,
                weight,
                eps,
                proj,
                y,
            } => {
                let blocks: Vec<Tensor> = blocks.iter().map(|b| self.tensor(*b)).collect();
                elemwise::norm::res_blend(
                    self.ctx(),
                    self.tensor(*prefix),
                    &blocks,
                    self.tensor(*weight),
                    *eps,
                    self.tensor(*proj),
                    self.tensor(*y),
                )
            }

            // The absorbed `rope` family (`elementwise.rope_*`), calling into
            // `kernels_metal::elemwise::rope`.
            Elementwise::RopeFull {
                q,
                k,
                positions,
                head_dim,
                theta,
                interleaved,
                q_out: _,
                k_out: _,
            } => elemwise::rope::full(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *head_dim,
                *theta,
                *interleaved,
            ),
            Elementwise::RopePartial {
                q,
                k,
                positions,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => elemwise::rope::partial(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            // The interleaved section layout forwards, as it always did; the
            // tower's BLOCKED one refuses by name, because this plane's
            // `elemwise::rope_mrope` ships one entry and a rotation that
            // handed the sections out the other way would answer plausible
            // numbers for the wrong checkpoint. One entry beside
            // `interleaved` retires the refusal.
            Elementwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Interleaved,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => elemwise::rope_mrope::interleaved(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            Elementwise::RopeMrope {
                form: MropeForm::Blocked,
                ..
            } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Elementwise::RopePartialQ {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
            } => elemwise::rope::partial_q(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            Elementwise::RopePartialLast {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                interleaved,
                q_out: _,
            } => elemwise::rope::partial_last(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *interleaved,
            ),
            Elementwise::RopeYarn {
                q,
                k,
                positions,
                head_dim,
                theta,
                factor,
                beta_fast,
                beta_slow,
                attention_factor,
                original_max_position,
                interleaved,
                q_out: _,
                k_out: _,
            } => elemwise::rope::yarn(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *head_dim,
                *theta,
                *factor,
                *beta_fast,
                *beta_slow,
                *attention_factor,
                *original_max_position,
                *interleaved,
            ),

            // The absorbed `gate` family (`elementwise.gate_*`), calling into
            // `kernels_metal::elemwise::gate`.
            Elementwise::GateSigmoidMul { x, gate, x_out: _ } => {
                elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), self.tensor(*x))
            }

            // The absorbed `hc` family (`elementwise.hc_*`), calling into
            // `kernels_metal::elemwise::hc`.
            Elementwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, self.tensor(*y))
            }
            Elementwise::HcRmsnormF32 { streams, eps, y } => {
                elemwise::hc::rmsnorm_f32(self.ctx(), self.tensor(*streams), *eps, self.tensor(*y))
            }
            Elementwise::HcGates {
                normed,
                streams,
                scale,
                base,
                stream_count,
                gate_eps,
                alpha,
                sinkhorn,
                x,
                post_mix,
                comb_mix,
            } => elemwise::hc::gates(
                self.ctx(),
                self.tensor(*normed),
                self.tensor(*streams),
                self.tensor(*scale),
                self.tensor(*base),
                *stream_count,
                *gate_eps,
                *alpha,
                *sinkhorn,
                self.tensor(*x),
                self.tensor(*post_mix),
                self.tensor(*comb_mix),
            ),
            Elementwise::HcFold {
                x,
                streams,
                post_mix,
                comb_mix,
                y,
            } => elemwise::hc::fold(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*streams),
                self.tensor(*post_mix),
                self.tensor(*comb_mix),
                self.tensor(*y),
            ),
        }
    }
}
