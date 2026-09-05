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
    /// The arms themselves, in `kernels-metal`'s error vocabulary, lifted by
    /// [`kernel`](crate::error::kernel) above.
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
            // Fused: the centred row is never rounded between ops.
            Elementwise::Layernorm {
                x,
                weight,
                bias,
                eps,
                y,
            } => elemwise::norm::layernorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                self.tensor(*y),
            ),
            // No shipped shader for these; refuse by name.
            Elementwise::LayernormNoScale { .. }
            | Elementwise::Clamp { .. }
            | Elementwise::ClampLearned { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            // qwen4's gated-residual family.
            Elementwise::RmsnormGroupedPlusOne {
                x,
                weight,
                group,
                eps,
                y,
            } => elemwise::norm::rmsnorm_grouped_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *group,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::SiluScaled { s, x, x_out: _ } => {
                elemwise::norm::silu_scaled(self.ctx(), *s, self.tensor(*x))
            }
            Elementwise::HcMix {
                gates,
                normed,
                streams,
                y,
            } => elemwise::hc::mix(
                self.ctx(),
                self.tensor(*gates),
                self.tensor(*normed),
                *streams,
                self.tensor(*y),
            ),
            Elementwise::HcInject {
                o,
                gates,
                streams,
                hyper,
                hyper_out: _,
            } => elemwise::hc::inject(
                self.ctx(),
                self.tensor(*o),
                self.tensor(*gates),
                *streams,
                self.tensor(*hyper),
            ),
            Elementwise::PleGate {
                key,
                query,
                value,
                streams,
                y,
            } => elemwise::hc::ple_gate(
                self.ctx(),
                self.tensor(*key),
                self.tensor(*query),
                self.tensor(*value),
                *streams,
                self.tensor(*y),
            ),
            // Both gate curves: qwen4's GatedDeltaNet uses the sigmoid one.
            Elementwise::RmsnormGated {
                x,
                gate,
                weight,
                head_dim,
                eps,
                act,
                y,
            } => elemwise::norm::rmsnorm_gated(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *head_dim,
                *eps,
                matches!(act, model_ir::GateActivation::Sigmoid),
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
            Elementwise::ResidualAddRmsnorm {
                x,
                y,
                y_out: _,
                weight,
                plus_one,
                eps,
                out,
            } => elemwise::norm::residual_add_rmsnorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*y),
                self.tensor(*weight),
                *plus_one,
                *eps,
                self.tensor(*out),
            ),
            Elementwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),
            // Per-column affine (x - bias) * scale, in place; its own entry
            // since scale reads one device-held scalar.
            Elementwise::Standardize {
                x,
                bias,
                scale,
                x_out: _,
            } => elemwise::norm::standardize(
                self.ctx(),
                self.tensor(*bias),
                self.tensor(*scale),
                self.tensor(*x),
            ),
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
            // Gemma's tower: contiguous channel blocks, `rotate_half` inside each.
            Elementwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Split,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => elemwise::rope_mrope::split(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            // Contiguous sections, each restarting the frequency ladder at a
            // denominator that is sum(sections), not the head.
            Elementwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Blocked,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => elemwise::rope_mrope::blocked(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),
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
                inverse,
                yarn,
                q_out: _,
            } => elemwise::rope::partial_last(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *interleaved,
                *inverse,
                yarn.map(|y| elemwise::rope::Yarn {
                    factor: y.factor,
                    beta_fast: y.beta_fast,
                    beta_slow: y.beta_slow,
                    original_max_position: y.original_max_position,
                }),
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

            Elementwise::GateSigmoidMul { x, gate, x_out: _ } => {
                elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), self.tensor(*x))
            }

            Elementwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, self.tensor(*y))
            }
            Elementwise::HcRmsnormF32 { streams, eps, y } => {
                elemwise::hc::rmsnorm_f32(self.ctx(), self.tensor(*streams), *eps, self.tensor(*y))
            }
            Elementwise::HcProject {
                normed,
                weight,
                stream_count,
                mixes,
            } => elemwise::hc::project(
                self.ctx(),
                self.tensor(*normed),
                self.tensor(*weight),
                *stream_count,
                self.tensor(*mixes),
            ),
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
            Elementwise::HcCollapse {
                mixes,
                streams,
                scale,
                base,
                stream_count,
                hc_eps,
                y,
            } => elemwise::hc::collapse(
                self.ctx(),
                self.tensor(*mixes),
                self.tensor(*streams),
                self.tensor(*scale),
                self.tensor(*base),
                *stream_count,
                *hc_eps,
                self.tensor(*y),
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
