use kernels_xla::{Tensor, elemwise};
use model_exec::{DispatchElementwise, KernelError};
use model_ir::{Elementwise, ModulateForm, MropeForm, NormKind, RopeForm, ValueId};

use model_ir::Operands;

use crate::run::Run;

impl DispatchElementwise for Run<'_> {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.elementwise(op).map_err(crate::error::kernel)
    }
}

fn modulate_form(form: ModulateForm) -> elemwise::act::Form {
    match form {
        ModulateForm::ScaleShift => elemwise::act::Form::ScaleShift,
        ModulateForm::Scale => elemwise::act::Form::Scale,
        ModulateForm::TanhGate => elemwise::act::Form::TanhGate,
    }
}

fn norm_kind(norm: NormKind) -> elemwise::act::NormKind {
    match norm {
        NormKind::Layernorm { eps } => elemwise::act::NormKind::Layernorm { eps },
        NormKind::Rmsnorm { head_dim, eps } => elemwise::act::NormKind::Rmsnorm { head_dim, eps },
    }
}

impl Run<'_> {
    /// A modulation plane: with a row→lane map its rows are indexed by the
    /// map's (fire-wide) lane ids, so it is the uncut plane; without one it
    /// rides the rows and is cut to the window like them.
    fn per_lane(&self, m: ValueId, lane_of_row: Option<ValueId>) -> Tensor {
        match lane_of_row {
            Some(_) => self.uncut(m),
            None => self.tensor(m),
        }
    }

    fn elementwise(&mut self, op: &Elementwise) -> Result<(), kernels_xla::Error> {
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

            Elementwise::Clamp {
                x,
                lo,
                hi,
                x_out: _,
            } => elemwise::clip::clamp(self.ctx(), *lo, *hi, self.tensor(*x)),

            Elementwise::ClampLearned {
                x,
                lo,
                hi,
                x_out: _,
            } => elemwise::clip::clamp_learned(
                self.ctx(),
                self.tensor(*lo),
                self.tensor(*hi),
                self.tensor(*x),
            ),

            Elementwise::LayernormNoScale { x, eps, y } => elemwise::norm::layernorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormResidualAdd {
                x,
                weight,
                eps,
                t,
                y,
                y_out: _,
                scale,
                post,
            } => elemwise::norm::rmsnorm_residual_add(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*t),
                self.tensor(*y),
                scale.map(|(s, scaled)| (self.tensor(s), self.tensor(scaled))),
                post.as_ref().map(|post| elemwise::norm::PostNorm {
                    weight: self.tensor(post.weight),
                    plus_one: post.plus_one,
                    eps: post.eps,
                    out: self.tensor(post.out),
                }),
            ),
            Elementwise::EmbedScaleAdd {
                ids,
                table,
                vocab,
                e,
                embed_scale,
                e_scaled,
                y,
                y_out: _,
                out_scale,
                y_scaled,
            } => elemwise::act::embed_scale_add(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                self.tensor(*e),
                *embed_scale,
                self.tensor(*e_scaled),
                self.tensor(*y),
                *out_scale,
                self.tensor(*y_scaled),
            ),
            Elementwise::EmbedScaleAddSelect {
                ids,
                table,
                vocab,
                e,
                embed_scale,
                e_scaled,
                stacked,
                layer,
                width,
                y_out,
                out_scale,
                y_scaled,
            } => elemwise::act::embed_scale_add_select(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                self.tensor(*e),
                *embed_scale,
                self.tensor(*e_scaled),
                self.tensor(*stacked),
                *layer,
                *width,
                self.tensor(*y_out),
                *out_scale,
                self.tensor(*y_scaled),
            ),
            Elementwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                y,
            } => elemwise::act::modulate(
                self.ctx(),
                modulate_form(*form),
                self.tensor(*x),
                self.per_lane(*m, *lane_of_row),
                lane_of_row.map(|lanes| self.tensor(lanes)),
                self.tensor(*y),
            ),
            Elementwise::GatedResidualAdd {
                r,
                g,
                y,
                lane_of_row,
                r_out: _,
            } => {
                let r = self.tensor(*r);
                elemwise::act::gated_residual_add(
                    self.ctx(),
                    r,
                    self.per_lane(*g, *lane_of_row),
                    self.tensor(*y),
                    lane_of_row.map(|lanes| self.tensor(lanes)),
                    r,
                )
            }
            Elementwise::NormModulate {
                x,
                norm,
                normed,
                m,
                lane_of_row,
                form,
                y,
            } => elemwise::act::norm_modulate(
                self.ctx(),
                self.tensor(*x),
                norm_kind(*norm),
                self.tensor(*normed),
                self.per_lane(*m, *lane_of_row),
                lane_of_row.map(|lanes| self.tensor(lanes)),
                modulate_form(*form),
                self.tensor(*y),
            ),
            Elementwise::GatedResidualNormModulate {
                r,
                g,
                y,
                lane_of_row,
                r_out: _,
                norm,
                normed,
                m,
                form,
                out,
            } => {
                let r = self.tensor(*r);
                elemwise::act::gated_residual_norm_modulate(
                    self.ctx(),
                    r,
                    self.per_lane(*g, *lane_of_row),
                    self.tensor(*y),
                    lane_of_row.map(|lanes| self.tensor(lanes)),
                    r,
                    norm_kind(*norm),
                    self.tensor(*normed),
                    self.per_lane(*m, *lane_of_row),
                    modulate_form(*form),
                    self.tensor(*out),
                )
            }
            Elementwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => elemwise::act::sinusoid(
                self.ctx(),
                self.tensor(*t),
                *dim,
                *max_period,
                *flip_sin_cos,
                *scale,
                self.tensor(*y),
            ),
            Elementwise::RelativeBucketBias {
                embedding,
                max_len,
                num_buckets,
                max_distance,
                bidirectional,
                y,
            } => elemwise::act::relative_bucket_bias(
                self.ctx(),
                self.tensor(*embedding),
                *max_len,
                *num_buckets,
                *max_distance,
                *bidirectional,
                self.tensor(*y),
            ),
            Elementwise::Silu { x, x_out: _ } => {
                let x = self.tensor(*x);
                elemwise::act::silu(self.ctx(), x, x)
            }
            Elementwise::Gelu { x, tanh, x_out: _ } => {
                let x = self.tensor(*x);
                if *tanh {
                    elemwise::act::gelu_tanh(self.ctx(), x, x)
                } else {
                    elemwise::act::gelu_erf(self.ctx(), x, x)
                }
            }
            Elementwise::Tanh { x, x_out: _ } => {
                let x = self.tensor(*x);
                elemwise::act::tanh(self.ctx(), x, x)
            }
            Elementwise::Mul { x, y, z } => elemwise::act::mul(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*y),
                self.tensor(*z),
            ),
            Elementwise::Add { x, y, z } => elemwise::act::add(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*y),
                self.tensor(*z),
            ),
            Elementwise::RopeAxes {
                x,
                positions,
                dims,
                thetas,
                form,
                rotary_dim,
                head_dim,
                x_out: _,
            } => {
                let x = self.tensor(*x);
                elemwise::act::rope_axes(
                    self.ctx(),
                    x,
                    self.tensor(*positions),
                    *dims,
                    *thetas,
                    match form {
                        RopeForm::Interleaved => elemwise::act::RopeForm::Interleaved,
                        RopeForm::Neox => elemwise::act::RopeForm::Neox,
                        RopeForm::Split => elemwise::act::RopeForm::Split,
                        RopeForm::SplitLadder => elemwise::act::RopeForm::SplitLadder,
                    },
                    *rotary_dim,
                    *head_dim,
                    x,
                )
            }
            Elementwise::GateSigmoidMulHeads {
                x,
                gate,
                head_dim,
                scale,
                x_out: _,
            } => elemwise::gate::sigmoid_mul_heads(
                self.ctx(),
                self.tensor(*gate),
                *head_dim,
                *scale,
                self.tensor(*x),
            ),
            Elementwise::RmsnormRopePartialQ {
                x,
                weight,
                head_dim,
                eps,
                positions,
                rotary_dim,
                theta,
                y,
                q_out: _,
            } => elemwise::rope::rmsnorm_rope_partial_q(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*positions),
                *rotary_dim,
                *theta,
                self.tensor(*y),
            ),

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
                // A readout-rowed `x` into a token-rowed `y` (an MTP head
                // over the trunk's hidden rows): the GPU kernel adds row
                // for row from the first, and only the readout rows mean
                // anything past it.
                let (x, y) = (self.tensor(*x), self.tensor(*y));
                let y = if x.rows < y.rows {
                    self.handles().cut(y, 0, x.rows)
                } else {
                    y
                };
                elemwise::norm::residual_add(self.ctx(), x, y)
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
                let y = self.tensor(*y);
                let blocks = self.stacked_blocks(blocks, y)?;
                elemwise::norm::res_blend(
                    self.ctx(),
                    self.tensor(*prefix),
                    &blocks,
                    self.tensor(*weight),
                    *eps,
                    self.tensor(*proj),
                    y,
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

impl Run<'_> {
    /// The candidate blocks, each its own handle: the XLA entry reads them
    /// one by one, so they need not be stacked planes.
    fn stacked_blocks(
        &self,
        blocks: &[model_ir::ValueId],
        y: Tensor,
    ) -> Result<Vec<Tensor>, kernels_xla::Error> {
        let _ = y;
        Ok(blocks.iter().map(|b| self.tensor(*b)).collect())
    }
}
