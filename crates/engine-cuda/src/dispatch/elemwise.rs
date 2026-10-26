use kernels_cuda::{Tensor, elemwise};
use poem_exec::{DispatchElemwise, KernelError};
#[cfg(feature = "rocm")]
use poem_ir::NormKind;
use poem_ir::{Elemwise, Fused, ModulateForm, MropeForm, Operands, RopeForm, Turn};
use poem_ir::{GateHeads, NormAxis, NormWeight, ResidualGate};

use crate::run::Run;

impl DispatchElemwise for Run<'_> {
    fn dispatch(&mut self, op: &Elemwise) -> Result<(), KernelError> {
        let mut outputs = Vec::new();
        op.outputs(&mut outputs);
        let lanes = outputs.first().is_some_and(|out| self.lane_shaped(*out));
        if lanes {
            self.unseated(|| self.elemwise(op))
        } else {
            self.elemwise(op)
        }
        .map_err(crate::error::kernel)
    }
}

#[cfg(feature = "rocm")]
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
    #[cfg(feature = "rocm")]
    fn scale_free_norm(
        &self,
        norm: NormKind,
        x: Tensor,
        normed: &mut Tensor,
    ) -> Result<(), kernels_cuda::Error> {
        match norm {
            NormKind::Layernorm { eps } => {
                elemwise::norm::layernorm_no_scale(self.ctx(), x, eps, normed)
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

    pub(super) fn fused_elemwise(&self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        match op {
            // CUDA runs the residual norms as generated regions; ROCm forms
            // their fused ops.
            #[cfg(feature = "rocm")]
            Fused::ResidualAddRmsnorm {
                x,
                y,
                y_out: _,
                weight,
                plus_one,
                eps,
                out,
            } => elemwise::norm_residual::residual_add_rmsnorm(
                self.ctx(),
                self.tensor(*x),
                &mut self.tensor(*y),
                self.tensor(*weight),
                *plus_one,
                *eps,
                &mut self.tensor(*out),
            ),
            #[cfg(feature = "rocm")]
            Fused::RmsnormResidualAdd {
                x,
                weight,
                eps,
                t,
                y,
                y_out,
                scale,
                post,
            } => {
                let mut scaled = scale.map(|scale| self.tensor(scale.out));
                let mut out = post.as_ref().map(|post| self.tensor(post.out));
                let _ = y_out;
                let mut t = self.tensor(*t);
                let mut y = self.tensor(*y);
                let scale = match (scale, scaled.as_mut()) {
                    (Some(scale), Some(scaled)) => Some((self.tensor(scale.by), scaled)),
                    _ => None,
                };
                let post = match (post, out.as_mut()) {
                    (Some(post), Some(out)) => Some(elemwise::norm_residual::PostNorm {
                        weight: self.tensor(post.weight),
                        plus_one: post.plus_one,
                        eps: post.eps,
                        out,
                    }),
                    _ => None,
                };
                let (x, weight) = (self.tensor(*x), self.tensor(*weight));
                elemwise::norm_residual::rmsnorm_residual_add(
                    self.ctx(),
                    x,
                    weight,
                    *eps,
                    &mut t,
                    &mut y,
                    scale,
                    post,
                )
            }
            // CUDA runs a scaled embedding's chain and a DiT block's norm and
            // modulation as generated regions; ROCm forms their fused ops.
            #[cfg(feature = "rocm")]
            Fused::EmbedScaleAdd {
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
            } => kernels_cuda::gather::embed_scale_add(
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
            #[cfg(feature = "rocm")]
            Fused::EmbedScaleAddSelect {
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
            } => kernels_cuda::gather::embed_scale_add_select(
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
            Fused::NormRope {
                x,
                weight: Some(weight),
                head_dim,
                eps,
                positions,
                theta,
                turn: Turn::Partial { rotary_dim, span },
                y,
                q_out: _,
            } => elemwise::norm_rope::rmsnorm_rope_partial_q(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*positions),
                *rotary_dim,
                *theta,
                *span,
                &mut self.tensor(*y),
            ),
            #[cfg(feature = "rocm")]
            Fused::HcNormProject {
                streams,
                eps,
                weight,
                stream_count,
                mixes,
            } => elemwise::hc_project::norm_project(
                self.ctx(),
                self.tensor(*streams),
                self.tensor(*weight),
                *eps,
                *stream_count,
                &mut self.tensor(*mixes),
            ),
            Fused::NormRope {
                x,
                weight,
                head_dim,
                eps,
                positions,
                theta,
                turn:
                    Turn::Last {
                        rotary_dim,
                        interleaved,
                        inverse,
                        yarn,
                    },
                y,
                q_out: _,
            } => elemwise::norm_rope::norm_rope_last(
                self.ctx(),
                self.tensor(*x),
                weight.map(|w| self.tensor(w)),
                *head_dim,
                *eps,
                self.tensor(*positions),
                *rotary_dim,
                *theta,
                *interleaved,
                *inverse,
                yarn.map(|y| elemwise::rope::Yarn {
                    factor: y.factor,
                    beta_fast: y.beta_fast,
                    beta_slow: y.beta_slow,
                    original_max_position: y.original_max_position,
                }),
                &mut self.tensor(*y),
            ),
            #[cfg(feature = "rocm")]
            Fused::NormModulate {
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
            #[cfg(feature = "rocm")]
            Fused::GatedResidualNormModulate {
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
            _ => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
        }
    }

    /// A rotation at its own frequencies (a checkpoint's rope scaling): `k`
    /// absent turns q alone; `pairs` is the head width, the pairs' span,
    /// whether they interleave and an interleaved M-RoPE's sections.
    #[cfg(not(feature = "rocm"))]
    fn rope_table(
        &self,
        op: &'static str,
        q: poem_ir::ValueId,
        k: Option<poem_ir::ValueId>,
        positions: poem_ir::ValueId,
        pairs: (u32, u32, bool, Option<[u32; 3]>),
        freqs: &poem_ir::RopeFreqs,
    ) -> Result<(), kernels_cuda::Error> {
        let (head_dim, span, interleaved, sections) = pairs;
        let mut q = self.tensor(q);
        let mut k = k.map_or_else(
            || Tensor::new(q.ptr, q.rows, 0, q.dtype),
            |k| self.tensor(k),
        );
        elemwise::rope::rope(
            self.ctx(),
            op,
            &mut q,
            &mut k,
            self.tensor(positions),
            elemwise::rope::Pairs {
                head_dim,
                span,
                interleaved,
                sections,
            },
            elemwise::rope::Freqs {
                inv_freq: &freqs.inv_freq,
                mscale: freqs.mscale,
            },
        )
    }

    /// ROCm turns no stated table yet.
    #[cfg(feature = "rocm")]
    fn rope_table(
        &self,
        op: &'static str,
        _q: poem_ir::ValueId,
        _k: Option<poem_ir::ValueId>,
        _positions: poem_ir::ValueId,
        _pairs: (u32, u32, bool, Option<[u32; 3]>),
        _freqs: &poem_ir::RopeFreqs,
    ) -> Result<(), kernels_cuda::Error> {
        Err(kernels_cuda::Error::Unsupported { op })
    }

    pub(super) fn elemwise(&self, op: &Elemwise) -> Result<(), kernels_cuda::Error> {
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
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
                &mut self.tensor(*y),
            ),
            Elemwise::SumSquares { x, ss } => {
                elemwise::norm::sum_squares(self.ctx(), self.tensor(*x), &mut self.tensor(*ss))
            }
            Elemwise::RmsnormBy {
                x,
                ss,
                weight,
                width,
                eps,
                y,
            } => {
                let (weight, plus_one) = match weight {
                    NormWeight::None => (None, false),
                    NormWeight::Scale(w) => (Some(self.tensor(*w)), false),
                    NormWeight::PlusOne(w) => (Some(self.tensor(*w)), true),
                };
                elemwise::norm::rmsnorm_by(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*ss),
                    weight,
                    plus_one,
                    *width,
                    *eps,
                    &mut self.tensor(*y),
                )
            }
            Elemwise::NormGate {
                x,
                scale,
                res,
                gate,
                eps,
                temperature,
                next,
                y,
            } => elemwise::modulate::norm_gate(
                self.ctx(),
                self.tensor(*x),
                scale.map(|s| self.tensor(s)),
                res.map(|r| self.tensor(r)),
                gate.map(|g| self.tensor(g)),
                *eps,
                *temperature,
                self.plane_fan(*y),
                next.map(|n| elemwise::norm::Next {
                    weight: self.tensor(n.weight),
                    eps: n.eps,
                    plus_one: n.plus_one,
                    y: self.tensor(n.y),
                }),
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
            Elemwise::Clamp {
                x,
                x_out: _,
                lo: None,
                hi: None,
                bounds: poem_ir::Bounds::Fixed { lo, hi, .. },
                ..
            } => elemwise::pointwise::clamp(self.ctx(), *lo, *hi, &mut self.tensor(*x)),
            Elemwise::Clamp {
                x,
                lo: Some(lo),
                hi: Some(hi),
                x_out: _,
                bounds: poem_ir::Bounds::Learned,
                ..
            } => elemwise::pointwise::clamp_learned(
                self.ctx(),
                self.tensor(*lo),
                self.tensor(*hi),
                &mut self.tensor(*x),
            ),
            // The forms no builder makes.
            Elemwise::Clamp { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Elemwise::RmsnormGated {
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
                weight.map(|w| self.tensor(w)),
                *head_dim,
                *eps,
                matches!(act, poem_ir::GateActivation::Sigmoid),
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
            Elemwise::ResidualAdd {
                x,
                gate: None,
                y,
                y_out: _,
            } => {
                elemwise::pointwise::residual_add(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Elemwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::pointwise::add_bias(
                self.ctx(),
                self.tensor(*bias),
                &mut self.tensor(*out),
            ),
            Elemwise::Standardize {
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
            Elemwise::Act {
                scale: Some(s),
                x,
                x_out: _,
                s: None,
                act: poem_ir::ActFn::Identity,
                ..
            } => elemwise::pointwise::mul_scalar(self.ctx(), *s, &mut self.tensor(*x)),
            Elemwise::Act {
                s: Some(s),
                x,
                x_out: _,
                scale: None,
                act: poem_ir::ActFn::Identity,
                ..
            } => elemwise::pointwise::scale(self.ctx(), self.tensor(*s), &mut self.tensor(*x)),
            Elemwise::ResBlend {
                prefix,
                blocks,
                weight,
                eps,
                proj,
                y,
            } => {
                let blocks: Vec<Tensor> = blocks.iter().map(|b| self.tensor(*b)).collect();
                elemwise::modulate::res_blend(
                    self.ctx(),
                    self.tensor(*prefix),
                    &blocks,
                    self.tensor(*weight),
                    *eps,
                    self.tensor(*proj),
                    &mut self.tensor(*y),
                )
            }
            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                freqs: Some(freqs),
                turn: poem_ir::Turn::Full { interleaved, .. },
                ..
            } => self.rope_table(
                "elemwise.rope",
                *q,
                Some(*k),
                *positions,
                (*head_dim, *head_dim / 2, *interleaved, None),
                freqs,
            ),
            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                freqs: Some(freqs),
                turn: poem_ir::Turn::Partial { span, .. },
                ..
            } => self.rope_table(
                "elemwise.rope",
                *q,
                Some(*k),
                *positions,
                (*head_dim, *span, false, None),
                freqs,
            ),
            Elemwise::Rope {
                q,
                positions,
                head_dim,
                freqs: Some(freqs),
                k: None,
                turn: poem_ir::Turn::Partial { span, .. },
                ..
            } => self.rope_table(
                "elemwise.rope",
                *q,
                None,
                *positions,
                (*head_dim, *span, false, None),
                freqs,
            ),
            Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                freqs: Some(freqs),
                turn:
                    poem_ir::Turn::Mrope {
                        sections,
                        form: MropeForm::Interleaved,
                        rotary_dim,
                        ..
                    },
                ..
            } => self.rope_table(
                "elemwise.rope",
                *q,
                Some(*k),
                *positions,
                (*head_dim, *rotary_dim / 2, false, Some(*sections)),
                freqs,
            ),
            Elemwise::Rope {
                freqs: Some(_),
                k: Some(_),
                turn: poem_ir::Turn::Mrope { .. },
                ..
            } => Err(kernels_cuda::Error::Unsupported {
                op: "elemwise.rope",
            }),
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
            } => elemwise::rope::full(
                self.ctx(),
                &mut self.tensor(*q),
                &mut self.tensor(*k),
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
            } => elemwise::rope::partial(
                self.ctx(),
                &mut self.tensor(*q),
                &mut self.tensor(*k),
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
                        form,
                        rotary_dim,
                        ..
                    },
                ..
            } => (match form {
                MropeForm::Interleaved => elemwise::rope::interleaved,
                MropeForm::Blocked => elemwise::rope::blocked,
                #[cfg(feature = "rocm")]
                MropeForm::Split => elemwise::rope::split,
                #[cfg(not(feature = "rocm"))]
                MropeForm::Split => {
                    return Err(kernels_cuda::Error::Unsupported {
                        op: "elemwise.rope",
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
            } => elemwise::rope::partial_q(
                self.ctx(),
                &mut self.tensor(*q),
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
            } => {
                use elemwise::rope;
                let r = rope::YarnRope {
                    head_dim: *head_dim,
                    theta: *theta,
                    yarn: rope::Yarn {
                        factor: *factor,
                        beta_fast: *beta_fast,
                        beta_slow: *beta_slow,
                        original_max_position: *original_max_position,
                    },
                    attention_factor: *attention_factor,
                    interleaved: *interleaved,
                };
                let positions = self.tensor(*positions);
                rope::yarn(
                    self.ctx(),
                    &mut self.tensor(*q),
                    &mut self.tensor(*k),
                    positions,
                    r,
                )
            }
            // The pairs no builder makes.
            Elemwise::Rope { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Elemwise::GateSigmoidMul {
                x,
                gate,
                heads: None,
                x_out: _,
            } => {
                let fan = self.plane_fan(*x);
                elemwise::gate::sigmoid_mul(
                    self.ctx(),
                    self.tensor(*gate),
                    fan,
                    &mut self.tensor(*x),
                )
            }
            Elemwise::GateSigmoidMul {
                x,
                gate,
                heads: Some(GateHeads { head_dim, scale }),
                x_out: _,
            } => elemwise::gate::sigmoid_mul_heads(
                self.ctx(),
                self.tensor(*gate),
                *head_dim,
                *scale,
                &mut self.tensor(*x),
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
                &mut self.tensor(*y),
            ),
            Elemwise::Act {
                scale: Some(s),
                x,
                x_out: _,
                s: None,
                act: poem_ir::ActFn::Silu,
                ..
            } => elemwise::pointwise::silu_scaled(self.ctx(), *s, &mut self.tensor(*x)),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*hyper),
            ),
            Elemwise::Modulate {
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
            Elemwise::ResidualAdd {
                x: y,
                gate: Some(ResidualGate { g, lane_of_row }),
                y: r,
                y_out: _,
            } => elemwise::modulate::gated_residual_add(
                self.ctx(),
                self.tensor(*r),
                self.tensor(*g),
                self.tensor(*y),
                lane_of_row.map(|lanes| self.tensor(lanes)),
                &mut self.tensor(*r),
            ),
            Elemwise::Softcap { x, cap, x_out: _ } => {
                kernels_cuda::attn::merge::logit_softcap(self.ctx(), &mut self.tensor(*x), *cap)
            }
            Elemwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => elemwise::posenc::sinusoid(
                self.ctx(),
                self.tensor(*t),
                *dim,
                *max_period,
                *flip_sin_cos,
                *scale,
                &mut self.tensor(*y),
            ),
            Elemwise::RelativeBucketBias {
                embedding,
                max_len,
                num_buckets,
                max_distance,
                bidirectional,
                y,
            } => elemwise::posenc::relative_bucket_bias(
                self.ctx(),
                self.tensor(*embedding),
                *max_len,
                *num_buckets,
                *max_distance,
                *bidirectional,
                &mut self.tensor(*y),
            ),
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Silu,
                ..
            } => elemwise::pointwise::silu(self.ctx(), self.tensor(*x), &mut self.tensor(*x)),
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Gelu { tanh, .. },
                ..
            } => {
                if !*tanh {
                    return Err(kernels_cuda::Error::Backend {
                        op: "elemwise.act",
                        detail: "the erf gelu has no CUDA entry; the tanh approximation \
                                 (`gelu(x, tanh = true)`) is what every DiT under study runs"
                            .to_string(),
                    });
                }
                elemwise::pointwise::gelu_tanh(self.ctx(), self.tensor(*x), &mut self.tensor(*x))
            }
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Tanh,
                ..
            } => elemwise::pointwise::tanh(self.ctx(), self.tensor(*x), &mut self.tensor(*x)),
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Sqrt,
                ..
            } => elemwise::pointwise::sqrt(self.ctx(), self.tensor(*x), &mut self.tensor(*x)),
            // The forms no builder makes.
            Elemwise::Act { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Elemwise::Hadamard { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Elemwise::Binary {
                x,
                y,
                z,
                op: binary,
            } => {
                let f = match binary {
                    poem_ir::BinaryOp::Add => elemwise::pointwise::add,
                    poem_ir::BinaryOp::Mul => elemwise::pointwise::mul,
                    poem_ir::BinaryOp::Div => elemwise::pointwise::div,
                    poem_ir::BinaryOp::Mod => elemwise::pointwise::modulo,
                };
                f(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*y),
                    &mut self.tensor(*z),
                )
            }
            Elemwise::RopeAxes {
                x,
                positions,
                dims,
                thetas,
                form,
                rotary_dim,
                head_dim,
                x_out: _,
            } => elemwise::rope::rope_axes(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*positions),
                *dims,
                *thetas,
                match form {
                    RopeForm::Interleaved => elemwise::rope::RopeForm::Interleaved,
                    RopeForm::Neox => elemwise::rope::RopeForm::Neox,
                    RopeForm::Split => elemwise::rope::RopeForm::Split,
                    RopeForm::SplitLadder => elemwise::rope::RopeForm::SplitLadder,
                },
                *rotary_dim,
                *head_dim,
                &mut self.tensor(*x),
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
                &mut self.tensor(*y),
            ),
            Elemwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, &mut self.tensor(*y))
            }
            Elemwise::HcRmsnormF32 { streams, eps, y } => elemwise::hc::rmsnorm_f32(
                self.ctx(),
                self.tensor(*streams),
                *eps,
                &mut self.tensor(*y),
            ),
            Elemwise::HcProject {
                normed,
                weight,
                stream_count,
                mixes,
            } => elemwise::hc_project::project(
                self.ctx(),
                self.tensor(*normed),
                self.tensor(*weight),
                *stream_count,
                &mut self.tensor(*mixes),
            ),
            Elemwise::HcGates {
                normed,
                streams,
                scale,
                base,
                stream_count,
                gate_eps,
                alpha,
                sinkhorn,
                comb,
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
                *comb,
                &mut self.tensor(*x),
                &mut self.tensor(*post_mix),
                &mut self.tensor(*comb_mix),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
            // A weight and axis no builder pairs.
            Elemwise::Rmsnorm { .. } | Elemwise::Layernorm { .. } => {
                Err(kernels_cuda::Error::Unsupported { op: op.name() })
            }
        }
    }
}
