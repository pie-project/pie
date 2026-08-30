//! `Elementwise`: the norm anchor, rope, gate, and hc arms.

use kernels_cuda::{Tensor, elemwise};
use model_exec::{DispatchElementwise, KernelError};
use model_ir::{Elementwise, MropeForm};

use crate::run::Run;

impl DispatchElementwise for Run<'_> {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        self.elementwise(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-cuda`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn elementwise(&mut self, op: &Elementwise) -> Result<(), kernels_cuda::Error> {
        match op {
            // ---- norm (anchor) ----
            Elementwise::Rmsnorm { x, weight, eps, y } => elemwise::norm::rmsnorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
            Elementwise::RmsnormPlusOne { x, weight, eps, y } => elemwise::norm::rmsnorm_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
            // **THE CENTRED NORM** (multimodal §6.1). The one part of the
            // towers' `nn.LayerNorm` that does not fold into the GEMM behind
            // it; the scale and the bias do, at import.
            Elementwise::LayernormNoScale { x, eps, y } => elemwise::layernorm::layernorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *eps,
                &mut self.tensor(*y),
            ),
            // **THE CLIPPED LINEAR'S CLAMP** (multimodal §6.5), in place on
            // `x` — the IR aliases `x_out` onto it.
            Elementwise::Clamp { x, lo, hi, x_out: _ } => {
                elemwise::clip::clamp(self.ctx(), *lo, *hi, &mut self.tensor(*x))
            }
            // **AND THE FORM WHOSE BOUNDS THE CHECKPOINT SHIPS** (multimodal
            // §12.2): two `[1]` planes resolved like any other weight, which
            // is `Elementwise::Scale`'s arm one row up read for a bound
            // instead of a gain.
            Elementwise::ClampLearned {
                x,
                lo,
                hi,
                x_out: _,
            } => elemwise::clip::clamp_learned(
                self.ctx(),
                self.tensor(*lo),
                self.tensor(*hi),
                &mut self.tensor(*x),
            ),
            Elementwise::RmsnormGated {
                x,
                gate,
                weight,
                head_dim,
                eps,
                y,
            } => elemwise::norm::rmsnorm_gated(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *head_dim,
                *eps,
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
            Elementwise::ResidualAdd { x, y, y_out: _ } => {
                elemwise::norm::residual_add(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Elementwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), &mut self.tensor(*out)),
            Elementwise::MulScalar { s, x, x_out: _ } => {
                elemwise::norm::mul_scalar(self.ctx(), *s, &mut self.tensor(*x))
            }
            Elementwise::Scale { s, x, x_out: _ } => {
                elemwise::norm::scale(self.ctx(), self.tensor(*s), &mut self.tensor(*x))
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
                    &mut self.tensor(*y),
                )
            }
            // ---- rope ----
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
                &mut self.tensor(*q),
                &mut self.tensor(*k),
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
                &mut self.tensor(*q),
                &mut self.tensor(*k),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            // **THE SECTION LAYOUT PICKS THE ENTRY** (multimodal §6.3). The
            // trunk states `mrope_interleaved: true` and the tower's
            // `apply_rotary_pos_emb_vision` hands its sections out in
            // contiguous blocks; the two entries share every refusal and
            // differ in one symbol, so this arm is a choice of function and
            // not a second arm.
            Elementwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => (match form {
                MropeForm::Interleaved => elemwise::rope_mrope::interleaved,
                MropeForm::Blocked => elemwise::rope_mrope::blocked,
            })(
                self.ctx(),
                &mut self.tensor(*q),
                &mut self.tensor(*k),
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
                &mut self.tensor(*q),
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
                &mut self.tensor(*q),
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
                &mut self.tensor(*q),
                &mut self.tensor(*k),
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
            // ---- gate ----
            Elementwise::GateSigmoidMul { x, gate, x_out: _ } => {
                elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), &mut self.tensor(*x))
            }
            // ---- hc ----
            Elementwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, &mut self.tensor(*y))
            }
            Elementwise::HcRmsnormF32 { streams, eps, y } => elemwise::hc::rmsnorm_f32(
                self.ctx(),
                self.tensor(*streams),
                *eps,
                &mut self.tensor(*y),
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
                &mut self.tensor(*x),
                &mut self.tensor(*post_mix),
                &mut self.tensor(*comb_mix),
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
                &mut self.tensor(*y),
            ),
        }
    }
}
