use kernels_vulkan::{Error, Tensor};
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
    fn elemwise(&mut self, op: &Elemwise) -> Result<(), kernels_vulkan::Error> {
        match op {
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::Scale(weight),
                eps,
                y,
                axis: NormAxis::Row,
            } => kernels_slang::elemwise::norm::rmsnorm(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::Scale(weight),
                axis: NormAxis::Head(head_dim),
                eps,
                y,
            } => kernels_slang::elemwise::norm::rmsnorm_per_head(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *head_dim,
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                eps,
                y,
                axis: NormAxis::Row,
            } => kernels_slang::elemwise::norm::rmsnorm_plus_one(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Head(head_dim),
                eps,
                y,
            } => kernels_slang::elemwise::norm::rmsnorm_per_head_plus_one(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *head_dim,
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            // Muse Spark's post-norm gate is served on CUDA.
            Elemwise::NormGate { .. } => Err(kernels_vulkan::Error::Unsupported {
                op: "elemwise.norm_gate",
            }),
            // A rotation at its own frequencies (a checkpoint's rope scaling) is
            // served on CUDA.
            Elemwise::RopeFull { freqs: Some(_), .. }
            | Elemwise::RopePartial { freqs: Some(_), .. }
            | Elemwise::RopePartialQ { freqs: Some(_), .. }
            | Elemwise::RopeMrope { freqs: Some(_), .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Elemwise::Rmsnorm {
                x,
                axis: NormAxis::Head(head_dim),
                eps,
                y,
                weight: NormWeight::None,
            } => kernels_slang::elemwise::norm::rmsnorm_no_scale(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                *head_dim,
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            Elemwise::Layernorm {
                x,
                weight: Some(weight),
                bias: Some(bias),
                eps,
                y,
            } => kernels_slang::elemwise::norm::layernorm(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                crate::encode::slang_tensor(self.tensor(*bias)),
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            Elemwise::Clamp {
                x,
                lo,
                hi,
                x_out: _,
            } => kernels_slang::elemwise::pointwise::clamp(
                &self.slang(),
                *lo,
                *hi,
                crate::encode::slang_tensor(self.tensor(*x)),
            )
            .map_err(crate::encode::from_slang),

            Elemwise::ClampLearned {
                x,
                lo,
                hi,
                x_out: _,
            } => kernels_slang::elemwise::pointwise::clamp_learned(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*lo)),
                crate::encode::slang_tensor(self.tensor(*hi)),
                crate::encode::slang_tensor(self.tensor(*x)),
            )
            .map_err(crate::encode::from_slang),

            Elemwise::Layernorm {
                x,
                eps,
                y,
                weight: None,
                bias: None,
            } => kernels_slang::elemwise::norm::layernorm_no_scale(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                y,
            } => kernels_slang::elemwise::modulate::modulate(
                &self.slang(),
                match form {
                    poem_ir::ModulateForm::ScaleShift => {
                        kernels_slang::elemwise::modulate::Form::ScaleShift
                    }
                    poem_ir::ModulateForm::Scale => kernels_slang::elemwise::modulate::Form::Scale,
                    poem_ir::ModulateForm::TanhGate => {
                        kernels_slang::elemwise::modulate::Form::TanhGate
                    }
                },
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(match lane_of_row {
                    Some(_) => self.uncut(*m),
                    None => self.tensor(*m),
                }),
                lane_of_row.map(|lanes| crate::encode::slang_tensor(self.tensor(lanes))),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::GatedResidualAdd {
                r,
                g,
                y,
                lane_of_row,
                r_out: _,
            } => kernels_slang::elemwise::modulate::gated_residual_add(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*r)),
                crate::encode::slang_tensor(match lane_of_row {
                    Some(_) => self.uncut(*g),
                    None => self.tensor(*g),
                }),
                crate::encode::slang_tensor(self.tensor(*y)),
                lane_of_row.map(|lanes| crate::encode::slang_tensor(self.tensor(lanes))),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => kernels_slang::elemwise::posenc::sinusoid(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*t)),
                *dim,
                *max_period,
                *flip_sin_cos,
                *scale,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Silu { x, x_out: _ } => kernels_slang::elemwise::pointwise::silu(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Tanh { x, x_out: _ } => kernels_slang::elemwise::pointwise::tanh(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::Add { x, y, z } => kernels_slang::elemwise::pointwise::add(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*y)),
                crate::encode::slang_tensor(self.tensor(*z)),
            )
            .map_err(crate::encode::from_slang),
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
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*positions)),
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
            )
            .map_err(crate::encode::from_slang),
            Elemwise::RelativeBucketBias { .. }
            | Elemwise::Gelu { .. }
            | Elemwise::Hadamard { .. }
            | Elemwise::Mul { .. }
            | Elemwise::GateSigmoidMulHeads { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }

            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Group(group),
                eps,
                y,
            } => kernels_slang::elemwise::norm::rmsnorm_grouped_plus_one(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *group,
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::SiluScaled { s, x, x_out: _ } => {
                kernels_slang::elemwise::pointwise::silu_scaled(
                    &self.slang(),
                    *s,
                    crate::encode::slang_tensor(self.tensor(*x)),
                )
                .map_err(crate::encode::from_slang)
            }
            Elemwise::HcMix {
                gates,
                normed,
                streams,
                y,
            } => kernels_slang::elemwise::hc::mix(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*gates)),
                crate::encode::slang_tensor(self.tensor(*normed)),
                *streams,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::HcInject {
                o,
                gates,
                streams,
                hyper,
                hyper_out: _,
            } => kernels_slang::elemwise::hc::inject(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*o)),
                crate::encode::slang_tensor(self.tensor(*gates)),
                *streams,
                crate::encode::slang_tensor(self.tensor(*hyper)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::PleGate {
                key,
                query,
                value,
                streams,
                y,
            } => kernels_slang::elemwise::hc::ple_gate(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*key)),
                crate::encode::slang_tensor(self.tensor(*query)),
                crate::encode::slang_tensor(self.tensor(*value)),
                *streams,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            // A scaleless gated norm runs on CUDA alone (Muse Spark's).
            Elemwise::RmsnormGated { weight: None, .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
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
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*gate)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *head_dim,
                *eps,
                matches!(act, poem_ir::GateActivation::Sigmoid),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::RmsnormGatedBy {
                x,
                gate,
                weight,
                heads,
                eps,
                y,
            } => kernels_slang::elemwise::norm::rmsnorm_gated_by(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*gate)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *heads,
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::ResidualAdd { x, y, y_out: _ } => {
                kernels_slang::elemwise::pointwise::residual_add(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*x)),
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
            }
            Elemwise::AddBias {
                bias,
                out,
                out_out: _,
            } => kernels_slang::elemwise::pointwise::add_bias(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*bias)),
                crate::encode::slang_tensor(self.tensor(*out)),
            )
            .map_err(crate::encode::from_slang),

            Elemwise::Standardize {
                x,
                bias,
                scale,
                x_out: _,
            } => kernels_slang::elemwise::norm::standardize(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*bias)),
                crate::encode::slang_tensor(self.tensor(*scale)),
                crate::encode::slang_tensor(self.tensor(*x)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::MulScalar { s, x, x_out: _ } => {
                kernels_slang::elemwise::pointwise::mul_scalar(
                    &self.slang(),
                    *s,
                    crate::encode::slang_tensor(self.tensor(*x)),
                )
                .map_err(crate::encode::from_slang)
            }
            Elemwise::Scale { s, x, x_out: _ } => kernels_slang::elemwise::pointwise::scale(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*s)),
                crate::encode::slang_tensor(self.tensor(*x)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::ResBlend {
                prefix,
                blocks,
                weight,
                eps,
                proj,
                y,
            } => {
                let y = self.tensor(*y);
                let blocks: Vec<_> = self
                    .stacked_blocks(blocks, y)?
                    .into_iter()
                    .map(crate::encode::slang_tensor)
                    .collect();
                kernels_slang::elemwise::norm::res_blend(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*prefix)),
                    &blocks,
                    crate::encode::slang_tensor(self.tensor(*weight)),
                    *eps,
                    crate::encode::slang_tensor(self.tensor(*proj)),
                    crate::encode::slang_tensor(y),
                )
                .map_err(crate::encode::from_slang)
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
            } => kernels_slang::elemwise::rope::full(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *head_dim,
                *theta,
                *interleaved,
            )
            .map_err(crate::encode::from_slang),
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
            } => kernels_slang::elemwise::rope::partial(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *rotary_dim,
                *head_dim,
                *theta,
                *span,
            )
            .map_err(crate::encode::from_slang),
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
            } => kernels_slang::elemwise::rope::mrope_interleaved(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            )
            .map_err(crate::encode::from_slang),

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
            } => kernels_slang::elemwise::rope::mrope_split(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            )
            .map_err(crate::encode::from_slang),

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
            } => kernels_slang::elemwise::rope::mrope_blocked(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            )
            .map_err(crate::encode::from_slang),
            Elemwise::RopePartialQ {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                span,
                freqs: None,
                q_out: _,
            } => kernels_slang::elemwise::rope::partial_q(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *rotary_dim,
                *head_dim,
                *theta,
                *span,
            )
            .map_err(crate::encode::from_slang),
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
            } => kernels_slang::elemwise::rope::partial_last(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*positions)),
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
            )
            .map_err(crate::encode::from_slang),
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
            } => kernels_slang::elemwise::rope::yarn(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *head_dim,
                *theta,
                *factor,
                *beta_fast,
                *beta_slow,
                *attention_factor,
                *original_max_position,
                *interleaved,
            )
            .map_err(crate::encode::from_slang),

            Elemwise::GateSigmoidMul { x, gate, x_out: _ } => {
                kernels_slang::elemwise::pointwise::sigmoid_mul(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*gate)),
                    crate::encode::slang_tensor(self.tensor(*x)),
                )
                .map_err(crate::encode::from_slang)
            }

            Elemwise::HcExpand { x, streams, y } => kernels_slang::elemwise::hc::expand(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                *streams,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::HcRmsnormF32 { streams, eps, y } => kernels_slang::elemwise::hc::rmsnorm_f32(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*streams)),
                *eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::HcProject {
                normed,
                weight,
                stream_count,
                mixes,
            } => kernels_slang::elemwise::hc_project::project(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*normed)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *stream_count,
                crate::encode::slang_tensor(self.tensor(*mixes)),
            )
            .map_err(crate::encode::from_slang),
            // The independent form (no combine) is landed on CUDA alone.
            Elemwise::HcGates { comb: false, .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
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
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*normed)),
                crate::encode::slang_tensor(self.tensor(*streams)),
                crate::encode::slang_tensor(self.tensor(*scale)),
                crate::encode::slang_tensor(self.tensor(*base)),
                *stream_count,
                *gate_eps,
                *alpha,
                *sinkhorn,
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*post_mix)),
                crate::encode::slang_tensor(self.tensor(*comb_mix)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::HcCollapse {
                mixes,
                streams,
                scale,
                base,
                stream_count,
                hc_eps,
                y,
            } => kernels_slang::elemwise::hc::collapse(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*mixes)),
                crate::encode::slang_tensor(self.tensor(*streams)),
                crate::encode::slang_tensor(self.tensor(*scale)),
                crate::encode::slang_tensor(self.tensor(*base)),
                *stream_count,
                *hc_eps,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Elemwise::HcFold {
                x,
                streams,
                post_mix,
                comb_mix,
                y,
            } => kernels_slang::elemwise::hc::fold(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*streams)),
                crate::encode::slang_tensor(self.tensor(*post_mix)),
                crate::encode::slang_tensor(self.tensor(*comb_mix)),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            // A weight and axis no builder pairs.
            Elemwise::Rmsnorm { .. } | Elemwise::Layernorm { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
        }
    }
}

impl Run<'_> {
    fn stacked_blocks(
        &self,
        blocks: &[poem_ir::ValueId],
        y: Tensor,
    ) -> Result<Vec<Tensor>, kernels_vulkan::Error> {
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
