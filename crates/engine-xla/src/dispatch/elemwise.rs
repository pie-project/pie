use kernels_xla::{Tensor, elemwise};
use poem_exec::{DispatchElemwise, KernelError};
use poem_ir::{Elemwise, ModulateForm, MropeForm, RopeForm, ValueId};
use poem_ir::{NormAxis, NormWeight};

use poem_ir::Operands;

use crate::run::Run;

impl DispatchElemwise for Run<'_> {
    fn dispatch(&mut self, op: &Elemwise) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.elemwise(op).map_err(crate::error::kernel)
    }
}

fn modulate_form(form: ModulateForm) -> elemwise::act::Form {
    match form {
        ModulateForm::ScaleShift => elemwise::act::Form::ScaleShift,
        ModulateForm::Scale => elemwise::act::Form::Scale,
        ModulateForm::TanhGate => elemwise::act::Form::TanhGate,
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

    fn elemwise(&mut self, op: &Elemwise) -> Result<(), kernels_xla::Error> {
        match op {
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::Scale(weight),
                eps,
                y,
                axis: NormAxis::Row,
            } => elemwise::norm::rmsnorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*y),
            ),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::Scale(weight),
                axis: NormAxis::Head(head_dim),
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
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                eps,
                y,
                axis: NormAxis::Row,
            } => elemwise::norm::rmsnorm_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*y),
            ),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Head(head_dim),
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
            // Muse Spark's post-norm gate is served on CUDA.
            Elemwise::NormGate { .. } => Err(kernels_xla::Error::Unsupported {
                op: "elemwise.norm_gate",
            }),
            // A rotation at its own frequencies (a checkpoint's rope scaling) is
            // served on CUDA.
            Elemwise::RopeFull { freqs: Some(_), .. }
            | Elemwise::RopePartial { freqs: Some(_), .. }
            | Elemwise::RopePartialQ { freqs: Some(_), .. }
            | Elemwise::RopeMrope { freqs: Some(_), .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
            Elemwise::Rmsnorm {
                x,
                axis: NormAxis::Head(head_dim),
                eps,
                y,
                weight: NormWeight::None,
            } => elemwise::norm::rmsnorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),

            Elemwise::Layernorm {
                x,
                weight: Some(weight),
                bias: Some(bias),
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

            Elemwise::Clamp {
                x,
                lo,
                hi,
                x_out: _,
            } => elemwise::clip::clamp(self.ctx(), *lo, *hi, self.tensor(*x)),

            Elemwise::ClampLearned {
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

            Elemwise::Layernorm {
                x,
                eps,
                y,
                weight: None,
                bias: None,
            } => elemwise::norm::layernorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *eps,
                self.tensor(*y),
            ),
            Elemwise::Modulate {
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
            Elemwise::GatedResidualAdd {
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
            Elemwise::Sinusoid {
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
            Elemwise::RelativeBucketBias {
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
            Elemwise::Silu { x, x_out: _ } => {
                let x = self.tensor(*x);
                elemwise::act::silu(self.ctx(), x, x)
            }
            Elemwise::Gelu { x, tanh, x_out: _ } => {
                let x = self.tensor(*x);
                if *tanh {
                    elemwise::act::gelu_tanh(self.ctx(), x, x)
                } else {
                    elemwise::act::gelu_erf(self.ctx(), x, x)
                }
            }
            Elemwise::Tanh { x, x_out: _ } => {
                let x = self.tensor(*x);
                elemwise::act::tanh(self.ctx(), x, x)
            }
            Elemwise::Mul { x, y, z } => elemwise::act::mul(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*y),
                self.tensor(*z),
            ),
            Elemwise::Add { x, y, z } => elemwise::act::add(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*y),
                self.tensor(*z),
            ),
            Elemwise::RopeAxes {
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
            Elemwise::GateSigmoidMulHeads {
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

            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Group(group),
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
            Elemwise::SiluScaled { s, x, x_out: _ } => {
                elemwise::norm::silu_scaled(self.ctx(), *s, self.tensor(*x))
            }
            Elemwise::HcMix {
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
            Elemwise::HcInject {
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
            Elemwise::PleGate {
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

            // A scaleless gated norm runs on CUDA alone (Muse Spark's).
            Elemwise::RmsnormGated { weight: None, .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
            Elemwise::RmsnormGated {
                x,
                gate,
                weight: Some(weight),
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
                matches!(act, poem_ir::GateActivation::Sigmoid),
                self.tensor(*y),
            ),
            Elemwise::RmsnormGatedBy {
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
            Elemwise::ResidualAdd { x, y, y_out: _ } => {
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
            Elemwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),

            Elemwise::Standardize {
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
            Elemwise::MulScalar { s, x, x_out: _ } => {
                elemwise::norm::mul_scalar(self.ctx(), *s, self.tensor(*x))
            }
            Elemwise::Scale { s, x, x_out: _ } => {
                elemwise::norm::scale(self.ctx(), self.tensor(*s), self.tensor(*x))
            }
            Elemwise::ResBlend {
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

            Elemwise::RopeFull {
                q,
                k,
                positions,
                head_dim,
                theta,
                interleaved,
                freqs: None,
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
            Elemwise::RopePartial {
                q,
                k,
                positions,
                rotary_dim,
                head_dim,
                theta,
                span,
                freqs: None,
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
                *span,
            ),
            Elemwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Interleaved,
                rotary_dim,
                head_dim,
                theta,
                freqs: None,
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

            Elemwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Split,
                rotary_dim,
                head_dim,
                theta,
                freqs: None,
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

            Elemwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Blocked,
                rotary_dim,
                head_dim,
                theta,
                freqs: None,
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
            Elemwise::RopePartialQ {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                span,
                freqs: None,
                q_out: _,
            } => elemwise::rope::partial_q(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *span,
            ),
            Elemwise::RopePartialLast {
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
            Elemwise::RopeYarn {
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

            Elemwise::GateSigmoidMul { x, gate, x_out: _ } => {
                elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), self.tensor(*x))
            }

            Elemwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, self.tensor(*y))
            }
            Elemwise::HcRmsnormF32 { streams, eps, y } => {
                elemwise::hc::rmsnorm_f32(self.ctx(), self.tensor(*streams), *eps, self.tensor(*y))
            }
            Elemwise::HcProject {
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
            // The independent form (no combine) is landed on CUDA alone.
            Elemwise::HcGates { comb: false, .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
            Elemwise::HcGates {
                normed,
                streams,
                scale,
                base,
                stream_count,
                gate_eps,
                alpha,
                sinkhorn,
                comb: true,
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
            Elemwise::HcCollapse {
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
            Elemwise::HcFold {
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
            // The XLA backend has no Hadamard/FWHT kernel; report it as
            // unsupported, the same way the wgpu and vulkan backends do.
            Elemwise::Hadamard { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            // A weight and axis no builder pairs.
            Elemwise::Rmsnorm { .. } | Elemwise::Layernorm { .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
        }
    }
}

impl Run<'_> {
    /// The candidate blocks, each its own handle: the XLA entry reads them
    /// one by one, so they need not be stacked planes.
    fn stacked_blocks(
        &self,
        blocks: &[poem_ir::ValueId],
        y: Tensor,
    ) -> Result<Vec<Tensor>, kernels_xla::Error> {
        let _ = y;
        Ok(blocks.iter().map(|b| self.tensor(*b)).collect())
    }
}
