use kernels_cuda::{Tensor, elemwise};
use model_exec::{DispatchElementwise, KernelError};
use model_ir::{Elementwise, ModulateForm, MropeForm, NormKind, Operands, RopeForm};

use crate::run::Run;

impl DispatchElementwise for Run<'_> {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        let mut outputs = Vec::new();
        op.outputs(&mut outputs);
        let lanes = outputs.first().is_some_and(|out| self.lane_shaped(*out));
        if lanes {
            self.unseated(|| self.elementwise(op))
        } else {
            self.elementwise(op)
        }
        .map_err(crate::error::kernel)
    }
}

fn fused_norm(norm: NormKind, width: u32) -> Option<elemwise::modulate::NormKind> {
    match norm {
        NormKind::Layernorm { eps } => {
            Some(elemwise::modulate::NormKind::LayerNormNoAffine { eps })
        }
        NormKind::Rmsnorm { head_dim, eps } if head_dim == width => {
            Some(elemwise::modulate::NormKind::RmsNormNoScale { eps })
        }
        NormKind::Rmsnorm { .. } => None,
    }
}

impl Run<'_> {
    fn scale_free_norm(
        &self,
        norm: NormKind,
        x: Tensor,
        normed: &mut Tensor,
    ) -> Result<(), kernels_cuda::Error> {
        match norm {
            NormKind::Layernorm { eps } => {
                elemwise::layernorm::layernorm_no_scale(self.ctx(), x, eps, normed)
            }
            NormKind::Rmsnorm { head_dim, eps } => {
                elemwise::norm::rmsnorm_no_scale(self.ctx(), x, head_dim, eps, normed)
            }
        }
    }

    fn modulate(
        &self,
        form: ModulateForm,
        x: Tensor,
        m: Tensor,
        lane_of_row: Option<Tensor>,
        y: &mut Tensor,
    ) -> Result<(), kernels_cuda::Error> {
        match form {
            ModulateForm::ScaleShift => {
                elemwise::modulate::scale_shift(self.ctx(), x, m, lane_of_row, y)
            }
            ModulateForm::Scale => elemwise::modulate::scale(self.ctx(), x, m, lane_of_row, y),
            ModulateForm::TanhGate => {
                elemwise::modulate::tanh_gate(self.ctx(), x, m, lane_of_row, y)
            }
        }
    }

    fn elementwise(&self, op: &Elementwise) -> Result<(), kernels_cuda::Error> {
        match op {
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
            Elementwise::LayernormNoScale { x, eps, y } => elemwise::layernorm::layernorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *eps,
                &mut self.tensor(*y),
            ),
            Elementwise::Layernorm {
                x,
                weight,
                bias,
                eps,
                y,
            } => elemwise::layernorm::layernorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                &mut self.tensor(*y),
            ),
            Elementwise::Clamp {
                x,
                lo,
                hi,
                x_out: _,
            } => elemwise::clip::clamp(self.ctx(), *lo, *hi, &mut self.tensor(*x)),
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
                &mut self.tensor(*y),
                self.tensor(*weight),
                *plus_one,
                *eps,
                &mut self.tensor(*out),
            ),
            Elementwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), &mut self.tensor(*out)),
            Elementwise::Standardize {
                x,
                bias,
                scale,
                x_out: _,
            } => elemwise::norm::standardize(
                self.ctx(),
                self.tensor(*bias),
                self.tensor(*scale),
                &mut self.tensor(*x),
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
            } => {
                let mut scaled = scale.map(|(_, scaled)| self.tensor(scaled));
                let mut out = post.as_ref().map(|post| self.tensor(post.out));
                elemwise::norm::rmsnorm_residual_add(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*weight),
                    *eps,
                    &mut self.tensor(*t),
                    &mut self.tensor(*y),
                    match (scale, scaled.as_mut()) {
                        (Some((s, _)), Some(scaled)) => Some((self.tensor(*s), scaled)),
                        _ => None,
                    },
                    match (post, out.as_mut()) {
                        (Some(post), Some(out)) => Some(elemwise::norm::PostNorm {
                            weight: self.tensor(post.weight),
                            plus_one: post.plus_one,
                            eps: post.eps,
                            out,
                        }),
                        _ => None,
                    },
                )
            }
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
            } => kernels_cuda::layout::embed_scale_add(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                &mut self.tensor(*e),
                *embed_scale,
                &mut self.tensor(*e_scaled),
                &mut self.tensor(*y),
                *out_scale,
                &mut self.tensor(*y_scaled),
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
            } => kernels_cuda::layout::embed_scale_add_select(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                &mut self.tensor(*e),
                *embed_scale,
                &mut self.tensor(*e_scaled),
                self.tensor(*stacked),
                *layer,
                *width,
                &mut self.tensor(*y_out),
                *out_scale,
                &mut self.tensor(*y_scaled),
            ),
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
                MropeForm::Split => {
                    return Err(kernels_cuda::Error::Unsupported {
                        op: "elementwise.rope_mrope",
                    });
                }
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*q),
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
            Elementwise::GateSigmoidMul { x, gate, x_out: _ } => {
                let fan = self.plane_fan(self.tensor(*x).rows);
                elemwise::gate::sigmoid_mul(
                    self.ctx(),
                    self.tensor(*gate),
                    fan,
                    &mut self.tensor(*x),
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
                &mut self.tensor(*x),
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
                &mut self.tensor(*y),
            ),
            Elementwise::SiluScaled { s, x, x_out: _ } => {
                elemwise::norm::silu_scaled(self.ctx(), *s, &mut self.tensor(*x))
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*hyper),
            ),
            Elementwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                y,
            } => self.modulate(
                *form,
                self.tensor(*x),
                self.tensor(*m),
                lane_of_row.map(|lanes| self.tensor(lanes)),
                &mut self.tensor(*y),
            ),
            Elementwise::GatedResidualAdd {
                r,
                g,
                y,
                lane_of_row,
                r_out: _,
            } => elemwise::modulate::gated_residual_add(
                self.ctx(),
                self.tensor(*r),
                self.tensor(*g),
                self.tensor(*y),
                lane_of_row.map(|lanes| self.tensor(lanes)),
                &mut self.tensor(*r),
            ),
            Elementwise::NormModulate {
                x,
                norm,
                normed,
                m,
                lane_of_row,
                form,
                y,
            } => {
                let x_t = self.tensor(*x);
                let lanes = lane_of_row.map(|lanes| self.tensor(lanes));
                match (fused_norm(*norm, x_t.width), form) {
                    (Some(kernel_norm), ModulateForm::ScaleShift) => {
                        elemwise::modulate::norm_modulate(
                            self.ctx(),
                            x_t,
                            self.tensor(*m),
                            lanes,
                            kernel_norm,
                            &mut self.tensor(*y),
                        )?;
                        if self.read_elsewhere(*normed) {
                            self.scale_free_norm(*norm, x_t, &mut self.tensor(*normed))?;
                        }
                        Ok(())
                    }
                    _ => {
                        self.scale_free_norm(*norm, x_t, &mut self.tensor(*normed))?;
                        self.modulate(
                            *form,
                            self.tensor(*normed),
                            self.tensor(*m),
                            lanes,
                            &mut self.tensor(*y),
                        )
                    }
                }
            }
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
                let r_t = self.tensor(*r);
                let lanes = lane_of_row.map(|lanes| self.tensor(lanes));
                match (fused_norm(*norm, r_t.width), form) {
                    (Some(kernel_norm), ModulateForm::ScaleShift) => {
                        elemwise::modulate::gated_residual_norm_modulate(
                            self.ctx(),
                            r_t,
                            self.tensor(*g),
                            self.tensor(*y),
                            self.tensor(*m),
                            lanes,
                            kernel_norm,
                            &mut self.tensor(*r),
                            &mut self.tensor(*out),
                        )?;
                        if self.read_elsewhere(*normed) {
                            self.scale_free_norm(
                                *norm,
                                self.tensor(*r),
                                &mut self.tensor(*normed),
                            )?;
                        }
                        Ok(())
                    }
                    _ => {
                        elemwise::modulate::gated_residual_add(
                            self.ctx(),
                            r_t,
                            self.tensor(*g),
                            self.tensor(*y),
                            lanes,
                            &mut self.tensor(*r),
                        )?;
                        self.scale_free_norm(*norm, self.tensor(*r), &mut self.tensor(*normed))?;
                        self.modulate(
                            *form,
                            self.tensor(*normed),
                            self.tensor(*m),
                            lanes,
                            &mut self.tensor(*out),
                        )
                    }
                }
            }
            Elementwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => elemwise::sinusoid(
                self.ctx(),
                self.tensor(*t),
                *dim,
                *max_period,
                *flip_sin_cos,
                *scale,
                &mut self.tensor(*y),
            ),
            Elementwise::RelativeBucketBias {
                embedding,
                max_len,
                num_buckets,
                max_distance,
                bidirectional,
                y,
            } => elemwise::relative_bucket_bias(
                self.ctx(),
                self.tensor(*embedding),
                *max_len,
                *num_buckets,
                *max_distance,
                *bidirectional,
                &mut self.tensor(*y),
            ),
            Elementwise::Silu { x, x_out: _ } => {
                elemwise::activation::silu(self.ctx(), self.tensor(*x), &mut self.tensor(*x))
            }
            Elementwise::Gelu { x, tanh, x_out: _ } => {
                if !*tanh {
                    return Err(kernels_cuda::Error::Backend {
                        op: "elementwise.gelu",
                        detail: "the erf gelu has no CUDA entry; the tanh approximation \
                                 (`gelu(x, tanh = true)`) is what every DiT under study runs"
                            .to_string(),
                    });
                }
                elemwise::activation::gelu_tanh(self.ctx(), self.tensor(*x), &mut self.tensor(*x))
            }
            Elementwise::Tanh { x, x_out: _ } => {
                elemwise::activation::tanh(self.ctx(), self.tensor(*x), &mut self.tensor(*x))
            }
            Elementwise::Mul { x, y, z } => elemwise::binary::mul(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*y),
                &mut self.tensor(*z),
            ),
            Elementwise::Add { x, y, z } => elemwise::binary::add(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*y),
                &mut self.tensor(*z),
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
            } => elemwise::rope_axes::rope_axes(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*positions),
                *dims,
                *thetas,
                match form {
                    RopeForm::Interleaved => elemwise::rope_axes::RopeForm::Interleaved,
                    RopeForm::Neox => elemwise::rope_axes::RopeForm::Neox,
                    RopeForm::Split => elemwise::rope_axes::RopeForm::Split,
                    RopeForm::SplitLadder => elemwise::rope_axes::RopeForm::SplitLadder,
                },
                *rotary_dim,
                *head_dim,
                &mut self.tensor(*x),
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
                &mut self.tensor(*y),
            ),
            Elementwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, &mut self.tensor(*y))
            }
            Elementwise::HcRmsnormF32 { streams, eps, y } => elemwise::hc::rmsnorm_f32(
                self.ctx(),
                self.tensor(*streams),
                *eps,
                &mut self.tensor(*y),
            ),
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
                &mut self.tensor(*mixes),
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
                &mut self.tensor(*y),
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
