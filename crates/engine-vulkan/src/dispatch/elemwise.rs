use kernels_slang::{Error, Tensor};
use poem_exec::{DispatchElemwise, KernelError};
use poem_ir::{Elemwise, MropeForm, Operands};
use poem_ir::{NormAxis, NormWeight};

use crate::run::Run;

impl DispatchElemwise for Run<'_> {
    fn dispatch(&mut self, op: &Elemwise) -> Result<(), KernelError> {
        self.elemwise(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn elemwise(&mut self, op: &Elemwise) -> Result<(), kernels_slang::Error> {
        match op {
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::Scale(weight),
                eps,
                y,
                axis: NormAxis::Row,
            } => kernels_slang::elemwise::norm::rmsnorm(
                self.ctx,
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
            } => kernels_slang::elemwise::norm::rmsnorm_per_head(
                self.ctx,
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
            } => kernels_slang::elemwise::norm::rmsnorm_plus_one(
                self.ctx,
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
            } => kernels_slang::elemwise::norm::rmsnorm_per_head_plus_one(
                self.ctx,
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            // Muse Spark's post-norm gate is served on CUDA.
            Elemwise::NormGate { .. } => Err(kernels_slang::Error::Unsupported {
                op: "elemwise.norm_gate",
            }),
            // A rotation at its own frequencies (a checkpoint's rope scaling) is
            // served on CUDA.
            Elemwise::Rope { freqs: Some(_), .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }
            Elemwise::Rmsnorm {
                x,
                axis: NormAxis::Head(head_dim),
                eps,
                y,
                weight: NormWeight::None,
            } => kernels_slang::elemwise::norm::rmsnorm_no_scale(
                self.ctx,
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
            } => kernels_slang::elemwise::norm::layernorm(
                self.ctx,
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
            } => kernels_slang::elemwise::pointwise::clamp(self.ctx, *lo, *hi, self.tensor(*x)),

            Elemwise::ClampLearned {
                x,
                lo,
                hi,
                x_out: _,
            } => kernels_slang::elemwise::pointwise::clamp_learned(
                self.ctx,
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
            } => kernels_slang::elemwise::norm::layernorm_no_scale(
                self.ctx,
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
            } => kernels_slang::elemwise::modulate::modulate(
                self.ctx,
                match form {
                    poem_ir::ModulateForm::ScaleShift => {
                        kernels_slang::elemwise::modulate::Form::ScaleShift
                    }
                    poem_ir::ModulateForm::Scale => kernels_slang::elemwise::modulate::Form::Scale,
                    poem_ir::ModulateForm::TanhGate => {
                        kernels_slang::elemwise::modulate::Form::TanhGate
                    }
                },
                self.tensor(*x),
                match lane_of_row {
                    Some(_) => self.uncut(*m),
                    None => self.tensor(*m),
                },
                lane_of_row.map(|lanes| self.tensor(lanes)),
                self.tensor(*y),
            ),
            Elemwise::GatedResidualAdd {
                r,
                g,
                y,
                lane_of_row,
                r_out: _,
            } => kernels_slang::elemwise::modulate::gated_residual_add(
                self.ctx,
                self.tensor(*r),
                match lane_of_row {
                    Some(_) => self.uncut(*g),
                    None => self.tensor(*g),
                },
                self.tensor(*y),
                lane_of_row.map(|lanes| self.tensor(lanes)),
            ),
            Elemwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => kernels_slang::elemwise::posenc::sinusoid(
                self.ctx,
                self.tensor(*t),
                *dim,
                *max_period,
                *flip_sin_cos,
                *scale,
                self.tensor(*y),
            ),
            Elemwise::Silu { x, x_out: _ } => {
                kernels_slang::elemwise::pointwise::silu(self.ctx, self.tensor(*x))
            }
            Elemwise::Tanh { x, x_out: _ } => {
                kernels_slang::elemwise::pointwise::tanh(self.ctx, self.tensor(*x))
            }
            Elemwise::Add { x, y, z } => kernels_slang::elemwise::pointwise::add(
                self.ctx,
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
            } => kernels_slang::elemwise::rope::axes(
                self.ctx,
                self.tensor(*x),
                self.tensor(*positions),
                *dims,
                *thetas,
                match form {
                    poem_ir::RopeForm::Interleaved => {
                        kernels_slang::elemwise::rope::RopeForm::Interleaved
                    }
                    poem_ir::RopeForm::Neox => kernels_slang::elemwise::rope::RopeForm::Neox,
                    poem_ir::RopeForm::Split => kernels_slang::elemwise::rope::RopeForm::Split,
                    poem_ir::RopeForm::SplitLadder => {
                        kernels_slang::elemwise::rope::RopeForm::SplitLadder
                    }
                },
                *rotary_dim,
                *head_dim,
            ),
            Elemwise::RelativeBucketBias { .. }
            | Elemwise::Gelu { .. }
            | Elemwise::Hadamard { .. }
            | Elemwise::Mul { .. }
            | Elemwise::GateSigmoidMulHeads { .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }

            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Group(group),
                eps,
                y,
            } => kernels_slang::elemwise::norm::rmsnorm_grouped_plus_one(
                self.ctx,
                self.tensor(*x),
                self.tensor(*weight),
                *group,
                *eps,
                self.tensor(*y),
            ),
            Elemwise::SiluScaled { s, x, x_out: _ } => {
                kernels_slang::elemwise::pointwise::silu_scaled(self.ctx, *s, self.tensor(*x))
            }
            Elemwise::HcMix {
                gates,
                normed,
                streams,
                y,
            } => kernels_slang::elemwise::hc::mix(
                self.ctx,
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
            } => kernels_slang::elemwise::hc::inject(
                self.ctx,
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
            } => kernels_slang::elemwise::hc::ple_gate(
                self.ctx,
                self.tensor(*key),
                self.tensor(*query),
                self.tensor(*value),
                *streams,
                self.tensor(*y),
            ),

            // A scaleless gated norm runs on CUDA alone (Muse Spark's).
            Elemwise::RmsnormGated { weight: None, .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }
            Elemwise::RmsnormGated {
                x,
                gate,
                weight: Some(weight),
                head_dim,
                eps,
                act,
                y,
            } => kernels_slang::elemwise::norm::rmsnorm_gated(
                self.ctx,
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
            } => kernels_slang::elemwise::norm::rmsnorm_gated_by(
                self.ctx,
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *heads,
                *eps,
                self.tensor(*y),
            ),
            Elemwise::ResidualAdd { x, y, y_out: _ } => {
                kernels_slang::elemwise::pointwise::residual_add(
                    self.ctx,
                    self.tensor(*x),
                    self.tensor(*y),
                )
            }
            Elemwise::AddBias {
                bias,
                out,
                out_out: _,
            } => kernels_slang::elemwise::pointwise::add_bias(
                self.ctx,
                self.tensor(*bias),
                self.tensor(*out),
            ),

            Elemwise::Standardize {
                x,
                bias,
                scale,
                x_out: _,
            } => kernels_slang::elemwise::norm::standardize(
                self.ctx,
                self.tensor(*bias),
                self.tensor(*scale),
                self.tensor(*x),
            ),
            Elemwise::MulScalar { s, x, x_out: _ } => {
                kernels_slang::elemwise::pointwise::mul_scalar(self.ctx, *s, self.tensor(*x))
            }
            Elemwise::Scale { s, x, x_out: _ } => kernels_slang::elemwise::pointwise::scale(
                self.ctx,
                self.tensor(*s),
                self.tensor(*x),
            ),
            Elemwise::ResBlend {
                prefix,
                blocks,
                weight,
                eps,
                proj,
                y,
            } => {
                let y = self.tensor(*y);
                let blocks: Vec<_> = self.stacked_blocks(blocks, y)?;
                kernels_slang::elemwise::norm::res_blend(
                    self.ctx,
                    self.tensor(*prefix),
                    &blocks,
                    self.tensor(*weight),
                    *eps,
                    self.tensor(*proj),
                    y,
                )
            }

            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                theta,
                freqs: None,
                q_out: _,
                k_out: _,
                turn: poem_ir::Turn::Full { interleaved, .. },
                ..
            } => kernels_slang::elemwise::rope::full(
                self.ctx,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *head_dim,
                *theta,
                *interleaved,
            ),
            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                theta,
                freqs: None,
                q_out: _,
                k_out: _,
                turn:
                    poem_ir::Turn::Partial {
                        rotary_dim, span, ..
                    },
                ..
            } => kernels_slang::elemwise::rope::partial(
                self.ctx,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *span,
            ),
            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                theta,
                freqs: None,
                q_out: _,
                k_out: _,
                turn:
                    poem_ir::Turn::Mrope {
                        sections,
                        form: MropeForm::Interleaved,
                        rotary_dim,
                        ..
                    },
                ..
            } => kernels_slang::elemwise::rope::mrope_interleaved(
                self.ctx,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),

            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                theta,
                freqs: None,
                q_out: _,
                k_out: _,
                turn:
                    poem_ir::Turn::Mrope {
                        sections,
                        form: MropeForm::Split,
                        rotary_dim,
                        ..
                    },
                ..
            } => kernels_slang::elemwise::rope::mrope_split(
                self.ctx,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),

            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                theta,
                freqs: None,
                q_out: _,
                k_out: _,
                turn:
                    poem_ir::Turn::Mrope {
                        sections,
                        form: MropeForm::Blocked,
                        rotary_dim,
                        ..
                    },
                ..
            } => kernels_slang::elemwise::rope::mrope_blocked(
                self.ctx,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            Elemwise::Rope {
                q,
                positions,
                head_dim,
                theta,
                freqs: None,
                q_out: _,
                k: None,
                turn:
                    poem_ir::Turn::Partial {
                        rotary_dim, span, ..
                    },
                ..
            } => kernels_slang::elemwise::rope::partial_q(
                self.ctx,
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *span,
            ),
            Elemwise::Rope {
                q,
                positions,
                head_dim,
                theta,
                q_out: _,
                k: None,
                turn:
                    poem_ir::Turn::Last {
                        rotary_dim,
                        interleaved,
                        inverse,
                        yarn,
                        ..
                    },
                ..
            } => kernels_slang::elemwise::rope::partial_last(
                self.ctx,
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *interleaved,
                *inverse,
                yarn.map(|y| kernels_slang::elemwise::rope::Yarn {
                    factor: y.factor,
                    beta_fast: y.beta_fast,
                    beta_slow: y.beta_slow,
                    original_max_position: y.original_max_position,
                }),
            ),
            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
                turn:
                    poem_ir::Turn::Yarn {
                        yarn:
                            poem_ir::Yarn {
                                factor,
                                beta_fast,
                                beta_slow,
                                original_max_position,
                                ..
                            },
                        attention_factor,
                        interleaved,
                        ..
                    },
                ..
            } => kernels_slang::elemwise::rope::yarn(
                self.ctx,
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

            // The pairs no builder makes.
            Elemwise::Rope { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Elemwise::GateSigmoidMul { x, gate, x_out: _ } => {
                kernels_slang::elemwise::pointwise::sigmoid_mul(
                    self.ctx,
                    self.tensor(*gate),
                    self.tensor(*x),
                )
            }

            Elemwise::HcExpand { x, streams, y } => kernels_slang::elemwise::hc::expand(
                self.ctx,
                self.tensor(*x),
                *streams,
                self.tensor(*y),
            ),
            Elemwise::HcRmsnormF32 { streams, eps, y } => kernels_slang::elemwise::hc::rmsnorm_f32(
                self.ctx,
                self.tensor(*streams),
                *eps,
                self.tensor(*y),
            ),
            Elemwise::HcProject {
                normed,
                weight,
                stream_count,
                mixes,
            } => kernels_slang::elemwise::hc_project::project(
                self.ctx,
                self.tensor(*normed),
                self.tensor(*weight),
                *stream_count,
                self.tensor(*mixes),
            ),
            // The independent form (no combine) is landed on CUDA alone.
            Elemwise::HcGates { comb: false, .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
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
            } => kernels_slang::elemwise::hc::gates(
                self.ctx,
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
            } => kernels_slang::elemwise::hc::collapse(
                self.ctx,
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
            } => kernels_slang::elemwise::hc::fold(
                self.ctx,
                self.tensor(*x),
                self.tensor(*streams),
                self.tensor(*post_mix),
                self.tensor(*comb_mix),
                self.tensor(*y),
            ),
            // A weight and axis no builder pairs.
            Elemwise::Rmsnorm { .. } | Elemwise::Layernorm { .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }
        }
    }
}

impl Run<'_> {
    fn stacked_blocks(
        &self,
        blocks: &[poem_ir::ValueId],
        y: Tensor,
    ) -> Result<Vec<Tensor>, kernels_slang::Error> {
        const OP: &str = "elemwise.res_blend";
        let mut out: Vec<Tensor> = blocks.iter().map(|b| self.tensor(*b)).collect();
        let Some(first) = out.first().copied() else {
            return Ok(out);
        };
        let plane = u64::from(y.rows) * u64::from(y.width) * y.dtype.bytes_ceil();
        let mut want = self.address(first.buf).ok_or(Error::Backend {
            op: OP,
            detail: "the first candidate block has no minted handle".to_string(),
        })?;
        for block in &out {
            let at = self.address(block.buf).ok_or(Error::Backend {
                op: OP,
                detail: "a candidate block has no minted handle".to_string(),
            })?;
            if at != want {
                return Err(Error::Backend {
                    op: OP,
                    detail: "the candidate blocks do not land as stacked planes; the shader \
                             walks `blocks[(j * rows + t) * hidden]` and cannot gather \
                             scattered slots"
                        .to_string(),
                });
            }
            want.1 += plane;
        }
        let span = plane * out.len() as u64;
        let stack = self
            .handles()
            .cut(first.buf, 0, span)
            .map_err(|fault| Error::Backend {
                op: OP,
                detail: fault.to_string(),
            })?;
        out[0] = Tensor::new(stack, y.rows, y.width, y.dtype);
        Ok(out)
    }
}
