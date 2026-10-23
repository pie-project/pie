use crate::handle::Tensor;
use kernels_xla::elemwise;
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
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::rmsnorm(l.f(), x_v, weight_v, *eps)?;
                l.land(o_y, v)
            }),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::Scale(weight),
                axis: NormAxis::Head(head_dim),
                eps,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::rmsnorm_per_head(l.f(), x_v, weight_v, *head_dim, *eps)?;
                l.land(o_y, v)
            }),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                eps,
                y,
                axis: NormAxis::Row,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::rmsnorm_plus_one(l.f(), x_v, weight_v, *eps)?;
                l.land(o_y, v)
            }),
            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Head(head_dim),
                eps,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::rmsnorm_per_head_plus_one(
                    l.f(),
                    x_v,
                    weight_v,
                    *head_dim,
                    *eps,
                )?;
                l.land(o_y, v)
            }),
            // Muse Spark's post-norm gate is served on CUDA.
            Elemwise::NormGate { .. } => Err(kernels_xla::Error::Unsupported {
                op: "elemwise.norm_gate",
            }),
            // A rotation at its own frequencies (a checkpoint's rope scaling) is
            // served on CUDA.
            Elemwise::Rope { freqs: Some(_), .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
            Elemwise::Rmsnorm {
                x,
                axis: NormAxis::Head(head_dim),
                eps,
                y,
                weight: NormWeight::None,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::rmsnorm_no_scale(l.f(), x_v, *head_dim, *eps)?;
                l.land(o_y, v)
            }),

            Elemwise::Layernorm {
                x,
                weight: Some(weight),
                bias: Some(bias),
                eps,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let bias_v = l.read(self.tensor(*bias))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::layernorm(l.f(), x_v, weight_v, bias_v, *eps)?;
                l.land(o_y, v)
            }),

            Elemwise::Clamp {
                x,
                lo,
                hi,
                x_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_x = self.tensor(*x);
                let v = elemwise::clip::clamp(l.f(), *lo, *hi, x_v)?;
                l.land(o_x, v)
            }),

            Elemwise::ClampLearned {
                x,
                lo,
                hi,
                x_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let lo_v = l.read(self.tensor(*lo))?;
                let hi_v = l.read(self.tensor(*hi))?;
                let o_x = self.tensor(*x);
                let v = elemwise::clip::clamp_learned(l.f(), lo_v, hi_v, x_v)?;
                l.land(o_x, v)
            }),

            Elemwise::Layernorm {
                x,
                eps,
                y,
                weight: None,
                bias: None,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::layernorm_no_scale(l.f(), x_v, *eps, o_y.landing())?;
                l.land(o_y, v)
            }),
            Elemwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let m_v = l.read(self.per_lane(*m, *lane_of_row))?;
                let lane_of_row_v = l.read_opt(lane_of_row.map(|lanes| self.tensor(lanes)))?;
                let o_y = self.tensor(*y);
                let v = elemwise::act::modulate(
                    l.f(),
                    modulate_form(*form),
                    x_v,
                    m_v,
                    lane_of_row_v,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Elemwise::GatedResidualAdd {
                r,
                g,
                y,
                lane_of_row,
                r_out: _,
            } => {
                let r = self.tensor(*r);
                crate::lower::lower(self.ctx(), |l| {
                    let r_v = l.read(r)?;
                    let g_v = l.read(self.per_lane(*g, *lane_of_row))?;
                    let y_v = l.read(self.tensor(*y))?;
                    let lane_of_row_v = l.read_opt(lane_of_row.map(|lanes| self.tensor(lanes)))?;
                    let o_r_out = r;
                    let v = elemwise::act::gated_residual_add(
                        l.f(),
                        r_v,
                        g_v,
                        y_v,
                        lane_of_row_v,
                        o_r_out.landing(),
                    )?;
                    l.land(o_r_out, v)
                })
            }
            Elemwise::Sinusoid {
                t,
                dim,
                max_period,
                flip_sin_cos,
                scale,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let t_v = l.read(self.tensor(*t))?;
                let o_y = self.tensor(*y);
                let v = elemwise::act::sinusoid(
                    l.f(),
                    t_v,
                    *dim,
                    *max_period,
                    *flip_sin_cos,
                    *scale,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Elemwise::RelativeBucketBias {
                embedding,
                max_len,
                num_buckets,
                max_distance,
                bidirectional,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let embedding_v = l.read(self.tensor(*embedding))?;
                let o_y = self.tensor(*y);
                let v = elemwise::act::relative_bucket_bias(
                    l.f(),
                    embedding_v,
                    *max_len,
                    *num_buckets,
                    *max_distance,
                    *bidirectional,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Elemwise::Silu { x, x_out: _ } => {
                let x = self.tensor(*x);
                crate::lower::lower(self.ctx(), |l| {
                    let x_v = l.read(x)?;
                    let o_o = x;
                    let v = elemwise::act::silu(l.f(), x_v, o_o.landing())?;
                    l.land(o_o, v)
                })
            }
            Elemwise::Gelu { x, tanh, x_out: _ } => {
                let x = self.tensor(*x);
                if *tanh {
                    crate::lower::lower(self.ctx(), |l| {
                        let x_v = l.read(x)?;
                        let o_o = x;
                        let v = elemwise::act::gelu_tanh(l.f(), x_v, o_o.landing())?;
                        l.land(o_o, v)
                    })
                } else {
                    crate::lower::lower(self.ctx(), |l| {
                        let x_v = l.read(x)?;
                        let o_o = x;
                        let v = elemwise::act::gelu_erf(l.f(), x_v, o_o.landing())?;
                        l.land(o_o, v)
                    })
                }
            }
            Elemwise::Tanh { x, x_out: _ } => {
                let x = self.tensor(*x);
                crate::lower::lower(self.ctx(), |l| {
                    let x_v = l.read(x)?;
                    let o_o = x;
                    let v = elemwise::act::tanh(l.f(), x_v, o_o.landing())?;
                    l.land(o_o, v)
                })
            }
            Elemwise::Mul { x, y, z } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let y_v = l.read(self.tensor(*y))?;
                let o_z = self.tensor(*z);
                let v = elemwise::act::mul(l.f(), x_v, y_v, o_z.landing())?;
                l.land(o_z, v)
            }),
            Elemwise::Add { x, y, z } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let y_v = l.read(self.tensor(*y))?;
                let o_z = self.tensor(*z);
                let v = elemwise::act::add(l.f(), x_v, y_v, o_z.landing())?;
                l.land(o_z, v)
            }),
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
                crate::lower::lower(self.ctx(), |l| {
                    let x_v = l.read(x)?;
                    let positions_v = l.read(self.tensor(*positions))?;
                    let o_o = x;
                    let v = elemwise::act::rope_axes(
                        l.f(),
                        x_v,
                        positions_v,
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
                        o_o.landing(),
                    )?;
                    l.land(o_o, v)
                })
            }
            Elemwise::GateSigmoidMulHeads {
                x,
                gate,
                head_dim,
                scale,
                x_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let gate_v = l.read(self.tensor(*gate))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_x = self.tensor(*x);
                let v = elemwise::gate::sigmoid_mul_heads(l.f(), gate_v, *head_dim, *scale, x_v)?;
                l.land(o_x, v)
            }),

            Elemwise::Rmsnorm {
                x,
                weight: NormWeight::PlusOne(weight),
                axis: NormAxis::Group(group),
                eps,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let o_y = self.tensor(*y);
                let v =
                    elemwise::norm::rmsnorm_grouped_plus_one(l.f(), x_v, weight_v, *group, *eps)?;
                l.land(o_y, v)
            }),
            Elemwise::SiluScaled { s, x, x_out: _ } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_x = self.tensor(*x);
                let v = elemwise::norm::silu_scaled(l.f(), *s, x_v)?;
                l.land(o_x, v)
            }),
            Elemwise::HcMix {
                gates,
                normed,
                streams,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let gates_v = l.read(self.tensor(*gates))?;
                let normed_v = l.read(self.tensor(*normed))?;
                let o_y = self.tensor(*y);
                let v = elemwise::hc::mix(l.f(), gates_v, normed_v, *streams, o_y.landing())?;
                l.land(o_y, v)
            }),
            Elemwise::HcInject {
                o,
                gates,
                streams,
                hyper,
                hyper_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let gates_v = l.read(self.tensor(*gates))?;
                let o_v = l.read(self.tensor(*o))?;
                let hyper_v = l.read(self.tensor(*hyper))?;
                let o_hyper = self.tensor(*hyper);
                let v = elemwise::hc::inject(l.f(), o_v, gates_v, *streams, hyper_v)?;
                l.land(o_hyper, v)
            }),
            Elemwise::PleGate {
                key,
                query,
                value,
                streams,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let value_v = l.read(self.tensor(*value))?;
                let key_v = l.read(self.tensor(*key))?;
                let query_v = l.read(self.tensor(*query))?;
                let o_y = self.tensor(*y);
                let v = elemwise::hc::ple_gate(
                    l.f(),
                    key_v,
                    query_v,
                    value_v,
                    *streams,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),

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
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let gate_v = l.read(self.tensor(*gate))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::rmsnorm_gated(
                    l.f(),
                    x_v,
                    gate_v,
                    weight_v,
                    *head_dim,
                    *eps,
                    matches!(act, poem_ir::GateActivation::Sigmoid),
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Elemwise::RmsnormGatedBy {
                x,
                gate,
                weight,
                heads,
                eps,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let gate_v = l.read(self.tensor(*gate))?;
                let weight_v = l.read(self.tensor(*weight))?;
                let o_y = self.tensor(*y);
                let v = elemwise::norm::rmsnorm_gated_by(
                    l.f(),
                    x_v,
                    gate_v,
                    weight_v,
                    *heads,
                    *eps,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
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
                crate::lower::lower(self.ctx(), |l| {
                    let x_v = l.read(x)?;
                    let y_v = l.read(y)?;
                    let o_y = y;
                    let v = elemwise::norm::residual_add(l.f(), x_v, y_v)?;
                    l.land(o_y, v)
                })
            }
            Elemwise::AddBias {
                bias,
                out,
                out_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let bias_v = l.read(self.tensor(*bias))?;
                let out_v = l.read(self.tensor(*out))?;
                let o_out = self.tensor(*out);
                let v = elemwise::norm::add_bias(l.f(), bias_v, out_v)?;
                l.land(o_out, v)
            }),

            Elemwise::Standardize {
                x,
                bias,
                scale,
                x_out: _,
            } => crate::lower::lower(self.ctx(), |l| {
                let bias_v = l.read(self.tensor(*bias))?;
                let scale_v = l.read(self.tensor(*scale))?;
                let out_v = l.read(self.tensor(*x))?;
                let o_out = self.tensor(*x);
                let v = elemwise::norm::standardize(l.f(), bias_v, scale_v, out_v)?;
                l.land(o_out, v)
            }),
            Elemwise::MulScalar { s, x, x_out: _ } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_x = self.tensor(*x);
                let v = elemwise::norm::mul_scalar(l.f(), *s, x_v)?;
                l.land(o_x, v)
            }),
            Elemwise::Scale { s, x, x_out: _ } => crate::lower::lower(self.ctx(), |l| {
                let s_v = l.read(self.tensor(*s))?;
                let x_v = l.read(self.tensor(*x))?;
                let o_x = self.tensor(*x);
                let v = elemwise::norm::scale(l.f(), s_v, x_v)?;
                l.land(o_x, v)
            }),
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
                crate::lower::lower(self.ctx(), |l| {
                    let blocks_v = blocks
                        .iter()
                        .map(|t| l.read(*t))
                        .collect::<Result<Vec<_>, _>>()?;
                    let prefix_v = l.read(self.tensor(*prefix))?;
                    let weight_v = l.read(self.tensor(*weight))?;
                    let proj_v = l.read(self.tensor(*proj))?;
                    let o_y = y;
                    let v = elemwise::norm::res_blend(
                        l.f(),
                        prefix_v,
                        &blocks_v,
                        weight_v,
                        *eps,
                        proj_v,
                        o_y.landing(),
                    )?;
                    l.land(o_y, v)
                })
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
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let k_v = l.read(self.tensor(*k))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                let o_k = self.tensor(*k);
                let vs = elemwise::rope::full(
                    l.f(),
                    q_v,
                    k_v,
                    positions_v,
                    *head_dim,
                    *theta,
                    *interleaved,
                )?;
                if let Some(v) = vs[0] {
                    l.land(o_q, v)?;
                }
                if let Some(v) = vs[1] {
                    l.land(o_k, v)?;
                }
                Ok(())
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
                turn:
                    poem_ir::Turn::Partial {
                        rotary_dim, span, ..
                    },
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let k_v = l.read(self.tensor(*k))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                let o_k = self.tensor(*k);
                let vs = elemwise::rope::partial(
                    l.f(),
                    q_v,
                    k_v,
                    positions_v,
                    *rotary_dim,
                    *head_dim,
                    *theta,
                    *span,
                )?;
                if let Some(v) = vs[0] {
                    l.land(o_q, v)?;
                }
                if let Some(v) = vs[1] {
                    l.land(o_k, v)?;
                }
                Ok(())
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
                turn:
                    poem_ir::Turn::Mrope {
                        sections,
                        form: MropeForm::Interleaved,
                        rotary_dim,
                        ..
                    },
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let k_v = l.read(self.tensor(*k))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                let o_k = self.tensor(*k);
                let vs = elemwise::rope_mrope::interleaved(
                    l.f(),
                    q_v,
                    k_v,
                    positions_v,
                    *sections,
                    *rotary_dim,
                    *head_dim,
                    *theta,
                )?;
                if let Some(v) = vs[0] {
                    l.land(o_q, v)?;
                }
                if let Some(v) = vs[1] {
                    l.land(o_k, v)?;
                }
                Ok(())
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
                turn:
                    poem_ir::Turn::Mrope {
                        sections,
                        form: MropeForm::Split,
                        rotary_dim,
                        ..
                    },
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let k_v = l.read(self.tensor(*k))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                let o_k = self.tensor(*k);
                let vs = elemwise::rope_mrope::split(
                    l.f(),
                    q_v,
                    k_v,
                    positions_v,
                    *sections,
                    *rotary_dim,
                    *head_dim,
                    *theta,
                )?;
                if let Some(v) = vs[0] {
                    l.land(o_q, v)?;
                }
                if let Some(v) = vs[1] {
                    l.land(o_k, v)?;
                }
                Ok(())
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
                turn:
                    poem_ir::Turn::Mrope {
                        sections,
                        form: MropeForm::Blocked,
                        rotary_dim,
                        ..
                    },
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let k_v = l.read(self.tensor(*k))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                let o_k = self.tensor(*k);
                let vs = elemwise::rope_mrope::blocked(
                    l.f(),
                    q_v,
                    k_v,
                    positions_v,
                    *sections,
                    *rotary_dim,
                    *head_dim,
                    *theta,
                )?;
                if let Some(v) = vs[0] {
                    l.land(o_q, v)?;
                }
                if let Some(v) = vs[1] {
                    l.land(o_k, v)?;
                }
                Ok(())
            }),
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
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                if let Some(v) = elemwise::rope::partial_q(
                    l.f(),
                    q_v,
                    positions_v,
                    *rotary_dim,
                    *head_dim,
                    *theta,
                    *span,
                )? {
                    l.land(o_q, v)?;
                }
                Ok(())
            }),
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
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                if let Some(v) = elemwise::rope::partial_last(
                    l.f(),
                    q_v,
                    positions_v,
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
                )? {
                    l.land(o_q, v)?;
                }
                Ok(())
            }),
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
            } => crate::lower::lower(self.ctx(), |l| {
                let q_v = l.read(self.tensor(*q))?;
                let k_v = l.read(self.tensor(*k))?;
                let positions_v = l.read(self.tensor(*positions))?;
                let o_q = self.tensor(*q);
                let o_k = self.tensor(*k);
                let vs = elemwise::rope::yarn(
                    l.f(),
                    q_v,
                    k_v,
                    positions_v,
                    *head_dim,
                    *theta,
                    *factor,
                    *beta_fast,
                    *beta_slow,
                    *attention_factor,
                    *original_max_position,
                    *interleaved,
                )?;
                if let Some(v) = vs[0] {
                    l.land(o_q, v)?;
                }
                if let Some(v) = vs[1] {
                    l.land(o_k, v)?;
                }
                Ok(())
            }),

            // The pairs no builder makes.
            Elemwise::Rope { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Elemwise::GateSigmoidMul { x, gate, x_out: _ } => {
                crate::lower::lower(self.ctx(), |l| {
                    let gate_v = l.read(self.tensor(*gate))?;
                    let x_v = l.read(self.tensor(*x))?;
                    let o_x = self.tensor(*x);
                    let v = elemwise::gate::sigmoid_mul(l.f(), gate_v, x_v)?;
                    l.land(o_x, v)
                })
            }

            Elemwise::HcExpand { x, streams, y } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = elemwise::hc::expand(l.f(), x_v, *streams, o_y.landing())?;
                l.land(o_y, v)
            }),
            Elemwise::HcRmsnormF32 { streams, eps, y } => crate::lower::lower(self.ctx(), |l| {
                let streams_v = l.read(self.tensor(*streams))?;
                let o_y = self.tensor(*y);
                let v = elemwise::hc::rmsnorm_f32(l.f(), streams_v, *eps, o_y.landing())?;
                l.land(o_y, v)
            }),
            Elemwise::HcProject {
                normed,
                weight,
                stream_count,
                mixes,
            } => crate::lower::lower(self.ctx(), |l| {
                let normed_v = l.read(self.tensor(*normed))?;
                let hc_fn_v = l.read(self.tensor(*weight))?;
                let o_mixes = self.tensor(*mixes);
                let v = elemwise::hc::project(
                    l.f(),
                    normed_v,
                    hc_fn_v,
                    *stream_count,
                    o_mixes.landing(),
                )?;
                l.land(o_mixes, v)
            }),
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
            } => crate::lower::lower(self.ctx(), |l| {
                let normed_v = l.read(self.tensor(*normed))?;
                let scale_v = l.read(self.tensor(*scale))?;
                let base_v = l.read(self.tensor(*base))?;
                let streams_v = l.read(self.tensor(*streams))?;
                let o_x = self.tensor(*x);
                let o_post_mix = self.tensor(*post_mix);
                let o_comb_mix = self.tensor(*comb_mix);
                let (v0, v1, v2) = elemwise::hc::gates(
                    l.f(),
                    normed_v,
                    streams_v,
                    scale_v,
                    base_v,
                    *stream_count,
                    *gate_eps,
                    *alpha,
                    *sinkhorn,
                    o_x.landing(),
                    o_post_mix.landing(),
                    o_comb_mix.landing(),
                )?;
                l.land(o_post_mix, v0)?;
                l.land(o_comb_mix, v1)?;
                l.land(o_x, v2)?;
                Ok(())
            }),
            Elemwise::HcCollapse {
                mixes,
                streams,
                scale,
                base,
                stream_count,
                hc_eps,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let mixes_v = l.read(self.tensor(*mixes))?;
                let streams_v = l.read(self.tensor(*streams))?;
                let scale_v = l.read(self.tensor(*scale))?;
                let base_v = l.read(self.tensor(*base))?;
                let o_y = self.tensor(*y);
                let v = elemwise::hc::collapse(
                    l.f(),
                    mixes_v,
                    streams_v,
                    scale_v,
                    base_v,
                    *stream_count,
                    *hc_eps,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Elemwise::HcFold {
                x,
                streams,
                post_mix,
                comb_mix,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let post_mix_v = l.read(self.tensor(*post_mix))?;
                let comb_mix_v = l.read(self.tensor(*comb_mix))?;
                let streams_v = l.read(self.tensor(*streams))?;
                let o_y = self.tensor(*y);
                let v = elemwise::hc::fold(
                    l.f(),
                    x_v,
                    streams_v,
                    post_mix_v,
                    comb_mix_v,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
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
