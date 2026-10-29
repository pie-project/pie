use kernels_metal::{Tensor, elemwise};
use poem_exec::{DispatchElemwise, KernelError};
use poem_ir::{Elemwise, Fused, ModulateForm, MropeForm, Operands, RopeForm};
use poem_ir::{GateHeads, NormAxis, NormWeight, ResidualGate};

use crate::run::Run;

impl DispatchElemwise for Run<'_> {
    fn dispatch(&mut self, op: &Elemwise) -> Result<(), KernelError> {
        self.elemwise(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    pub(super) fn fused_elemwise(&mut self, op: &Fused) -> Result<(), kernels_metal::Error> {
        match op {
            Fused::ResidualAddRmsnorm {
                x,
                y,
                y_out: _,
                weight,
                plus_one,
                eps,
                out,
            } => {
                // The normed rows' projections on the register tile read
                // their input table as this norm lays it.
                if let Some((table, counters)) = self.producer_table(*out) {
                    return elemwise::norm::residual_add_rmsnorm_table(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*y),
                        self.tensor(*weight),
                        *plus_one,
                        *eps,
                        self.tensor(*out),
                        table,
                        counters,
                    );
                }
                elemwise::norm::residual_add_rmsnorm(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*y),
                    self.tensor(*weight),
                    *plus_one,
                    *eps,
                    self.tensor(*out),
                )
            }
            _ => Err(kernels_metal::Error::Unsupported { op: op.name() }),
        }
    }

    fn elemwise(&mut self, op: &Elemwise) -> Result<(), kernels_metal::Error> {
        match op {
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::Scale(weight),
                eps,
                y,
                axis: NormAxis::Row,
            } => {
                // The normed rows' projections on the register tile read
                // their input table as this norm lays it.
                match (self.tensor(*x).width == self.tensor(*y).width)
                    .then(|| self.producer_table(*y))
                    .flatten()
                {
                    Some((table, counters)) => elemwise::norm::rmsnorm_table(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*weight),
                        false,
                        *eps,
                        self.tensor(*y),
                        table,
                        counters,
                    ),
                    None => elemwise::norm::rmsnorm(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*weight),
                        *eps,
                        self.tensor(*y),
                    ),
                }
            }
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
            } => {
                // As `Rmsnorm`: the projections read the table this lays.
                match (self.tensor(*x).width == self.tensor(*y).width)
                    .then(|| self.producer_table(*y))
                    .flatten()
                {
                    Some((table, counters)) => elemwise::norm::rmsnorm_table(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*weight),
                        true,
                        *eps,
                        self.tensor(*y),
                        table,
                        counters,
                    ),
                    None => elemwise::norm::rmsnorm_plus_one(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*weight),
                        *eps,
                        self.tensor(*y),
                    ),
                }
            }
            // The pair launch of the norm before it landed these rows.
            Elemwise::Rmsnorm {
                y,
                weight: NormWeight::PlusOne(_),
                axis: NormAxis::Head(_),
                ..
            } if self.landed_ahead(*y) => Ok(()),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Head(head_dim),
                eps,
                y,
            } => match self.norm_pair(*x, *y, *head_dim, *eps) {
                Some((x2, weight2, y2)) => {
                    elemwise::norm::rmsnorm_per_head_plus_one_pair(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*weight),
                        self.tensor(*y),
                        self.tensor(x2),
                        self.tensor(weight2),
                        self.tensor(y2),
                        *head_dim,
                        *eps,
                    )?;
                    self.land_ahead([Some(y2), None, None]);
                    Ok(())
                }
                None => elemwise::norm::rmsnorm_per_head_plus_one(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*weight),
                    *head_dim,
                    *eps,
                    self.tensor(*y),
                ),
            },
            // A row norm the tensor-parallel ranks split is served on CUDA.
            Elemwise::SumSquares { .. } | Elemwise::RmsnormBy { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            // Muse Spark's post-norm gate.
            Elemwise::NormGate {
                x,
                scale,
                res,
                gate,
                eps,
                temperature,
                next,
                y,
            } => elemwise::norm::norm_gate(
                self.ctx(),
                self.tensor(*x),
                scale.map(|s| self.tensor(s)),
                res.map(|r| self.tensor(r)),
                gate.map(|g| self.tensor(g)),
                *eps,
                *temperature,
                next.map(|n| elemwise::norm::Next {
                    weight: self.tensor(n.weight),
                    eps: n.eps,
                    plus_one: n.plus_one,
                    y: self.tensor(n.y),
                }),
                self.tensor(*y),
            ),
            // A rotation at its own frequencies (a checkpoint's rope scaling) is
            // served on CUDA.
            Elemwise::Rope { freqs: Some(_), .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
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
            Elemwise::Clamp {
                x,
                x_out: _,
                lo: None,
                hi: None,
                bounds: poem_ir::Bounds::Fixed { lo, hi, .. },
                ..
            } => elemwise::pointwise::clamp(self.ctx(), *lo, *hi, self.tensor(*x)),
            Elemwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                y,
            } => elemwise::modulate::modulate(
                self.ctx(),
                match form {
                    ModulateForm::ScaleShift => elemwise::modulate::Form::ScaleShift,
                    ModulateForm::Scale => elemwise::modulate::Form::Scale,
                    ModulateForm::TanhGate => elemwise::modulate::Form::TanhGate,
                },
                self.tensor(*x),
                match lane_of_row {
                    Some(_) => self.uncut(*m),
                    None => self.tensor(*m),
                },
                lane_of_row.map(|lanes| self.tensor(lanes)),
                self.tensor(*y),
            ),
            Elemwise::ResidualAdd {
                x: y,
                gate: Some(ResidualGate { g, lane_of_row }),
                y: r,
                y_out: _,
            } => elemwise::modulate::gated_residual_add(
                self.ctx(),
                self.tensor(*r),
                match lane_of_row {
                    Some(_) => self.uncut(*g),
                    None => self.tensor(*g),
                },
                self.tensor(*y),
                lane_of_row.map(|lanes| self.tensor(lanes)),
                self.tensor(*r),
            ),
            Elemwise::Softcap { x, cap, x_out: _ } => {
                kernels_metal::attn::logit_softcap(self.ctx(), self.tensor(*x), *cap)
            }
            Elemwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => elemwise::sinusoid::sinusoid(
                self.ctx(),
                self.tensor(*t),
                *dim,
                *max_period,
                *flip_sin_cos,
                *scale,
                self.tensor(*y),
            ),
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Silu,
                ..
            } => elemwise::pointwise::silu(self.ctx(), self.tensor(*x), self.tensor(*x)),
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Gelu { tanh, .. },
                ..
            } if *tanh => {
                elemwise::pointwise::gelu_tanh(self.ctx(), self.tensor(*x), self.tensor(*x))
            }
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Gelu { .. },
                ..
            } => elemwise::pointwise::gelu(self.ctx(), self.tensor(*x), self.tensor(*x)),
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Tanh,
                ..
            } => elemwise::pointwise::tanh(self.ctx(), self.tensor(*x), self.tensor(*x)),
            Elemwise::Act {
                x,
                x_out: _,
                s: None,
                scale: None,
                act: poem_ir::ActFn::Sqrt,
                ..
            } => elemwise::pointwise::sqrt(self.ctx(), self.tensor(*x), self.tensor(*x)),
            Elemwise::Snake {
                x,
                alpha,
                beta,
                x_out: _,
            } => elemwise::pointwise::snake(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*alpha),
                self.tensor(*beta),
            ),
            Elemwise::Hadamard {
                x,
                x_out: _,
                block,
                signs,
            } => elemwise::pointwise::hadamard(
                self.ctx(),
                self.tensor(*x),
                *block,
                signs.as_ref().map(|s| self.tensor(*s)),
            ),
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
                    self.tensor(*z),
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
                self.tensor(*x),
            ),
            Elemwise::RelativeBucketBias {
                embedding,
                max_len,
                num_buckets,
                max_distance,
                bidirectional,
                y,
            } => elemwise::pointwise::relative_bucket_bias(
                self.ctx(),
                self.tensor(*embedding),
                *max_len,
                *num_buckets,
                *max_distance,
                *bidirectional,
                self.tensor(*y),
            ),
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
                self.tensor(*x),
            ),
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
                self.tensor(*x),
            ),
            // The forms no builder makes.
            Elemwise::Clamp { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
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
            Elemwise::Act {
                scale: Some(s),
                x,
                x_out: _,
                s: None,
                act: poem_ir::ActFn::Silu,
                ..
            } => elemwise::pointwise::silu_scaled(self.ctx(), *s, self.tensor(*x)),
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
            // A scaleless gated norm (Muse Spark's).
            Elemwise::RmsnormGated {
                x,
                gate,
                weight: None,
                head_dim,
                eps,
                act,
                y,
            } => elemwise::norm::rmsnorm_gated(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                None,
                *head_dim,
                *eps,
                matches!(act, poem_ir::GateActivation::Sigmoid),
                self.tensor(*y),
            ),
            Elemwise::RmsnormGated {
                x,
                gate,
                weight: Some(weight),
                head_dim,
                eps,
                act,
                y,
            } => {
                let sigmoid = matches!(act, poem_ir::GateActivation::Sigmoid);
                // The out projection on the register tile reads its input
                // table as the norm lays it: a GDN layer's projection reads
                // the merge of its scan's arms, which this arm's rows land in.
                let laid = (!sigmoid
                    && self.tensor(*x).dtype == poem_ir::Dtype::F32
                    && head_dim.is_multiple_of(32))
                .then(|| self.producer_table(self.merged_into(*y).unwrap_or(*y)))
                .flatten();
                match laid {
                    Some((table, counters)) => elemwise::norm::rmsnorm_gated_table(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*gate),
                        self.tensor(*weight),
                        *head_dim,
                        *eps,
                        self.tensor(*y),
                        table,
                        counters,
                    ),
                    None => elemwise::norm::rmsnorm_gated(
                        self.ctx(),
                        self.tensor(*x),
                        self.tensor(*gate),
                        Some(self.tensor(*weight)),
                        *head_dim,
                        *eps,
                        sigmoid,
                        self.tensor(*y),
                    ),
                }
            }
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
            Elemwise::ResidualAdd {
                x,
                gate: None,
                y,
                y_out: _,
            } => elemwise::pointwise::residual_add(self.ctx(), self.tensor(*x), self.tensor(*y)),
            Elemwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::pointwise::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),
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
            Elemwise::Act {
                scale: Some(s),
                x,
                x_out: _,
                s: None,
                act: poem_ir::ActFn::Identity,
                ..
            } => elemwise::pointwise::mul_scalar(self.ctx(), *s, self.tensor(*x)),
            Elemwise::Act {
                s: Some(s),
                x,
                x_out: _,
                scale: None,
                act: poem_ir::ActFn::Identity,
                ..
            } => elemwise::pointwise::scale(self.ctx(), self.tensor(*s), self.tensor(*x)),
            // The forms no builder makes.
            Elemwise::Act { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Elemwise::ResBlend {
                prefix,
                blocks,
                weight,
                eps,
                proj,
                y,
            } => {
                let planes: Vec<Tensor> = blocks.iter().map(|b| self.tensor(*b)).collect();
                let Some(first) = planes.first().copied() else {
                    return Err(kernels_metal::Error::Backend {
                        op: "elemwise.res_blend",
                        detail: "a blend of no candidate blocks".to_string(),
                    });
                };
                let elem = u64::from(first.width)
                    * poem_compiler::arena::elem_bytes(first.dtype).unwrap_or(2);
                let plane_bytes = u64::from(first.rows) * elem;
                let mut want = self.handles().get(first.buf).map(|row| row.offset());
                for plane in &planes {
                    let at = self.handles().get(plane.buf).map(|row| row.offset());
                    if at != want || plane.rows != first.rows || plane.width != first.width {
                        // Blocks that lie apart are each bound where they lie.
                        return elemwise::norm::res_blend_scattered(
                            self.ctx(),
                            self.tensor(*prefix),
                            &planes,
                            self.tensor(*weight),
                            *eps,
                            self.tensor(*proj),
                            self.tensor(*y),
                        );
                    }
                    want = want.map(|at| at + plane_bytes);
                }
                let stacked = Tensor::new(
                    first.buf,
                    first
                        .rows
                        .saturating_mul(u32::try_from(planes.len()).unwrap_or(u32::MAX)),
                    first.width,
                    first.dtype,
                );
                elemwise::norm::res_blend(
                    self.ctx(),
                    self.tensor(*prefix),
                    stacked,
                    u32::try_from(planes.len()).unwrap_or(u32::MAX),
                    self.tensor(*weight),
                    *eps,
                    self.tensor(*proj),
                    self.tensor(*y),
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
            } => elemwise::rope::full(
                self.ctx(),
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

            // The pairs no builder makes.
            Elemwise::Rope { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Elemwise::GateSigmoidMul {
                x,
                gate,
                heads: None,
                x_out,
            } => {
                // The gated rows' projection on the register tile reads its
                // input table as the gate lays it.
                match self.producer_table(*x_out) {
                    Some((table, counters)) => elemwise::gate::sigmoid_mul_table(
                        self.ctx(),
                        self.tensor(*gate),
                        self.tensor(*x),
                        table,
                        counters,
                    ),
                    None => {
                        elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), self.tensor(*x))
                    }
                }
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
            // A weight and axis no builder pairs.
            Elemwise::Rmsnorm { .. } | Elemwise::Layernorm { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
        }
    }
}
