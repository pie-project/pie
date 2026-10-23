use kernels_cerebras::elemwise;
use poem_exec::{DispatchElemwise, KernelError};
use poem_ir::{Elemwise, GateActivation, MropeForm, Operands};
use poem_ir::{NormAxis, NormWeight};

use crate::run::Run;

impl DispatchElemwise for Run<'_> {
    fn dispatch(&mut self, op: &Elemwise) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.elemwise(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn elemwise(&mut self, op: &Elemwise) -> Result<(), kernels_cerebras::Error> {
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
                matches!(act, GateActivation::Sigmoid),
                self.tensor(*y),
            ),
            Elemwise::ResidualAdd { x, y, y_out: _ } => {
                // A readout-rowed `x` into a token-rowed `y`: the kernel adds
                // row for row from the first.
                let (x, y) = (self.tensor(*x), self.tensor(*y));
                let y = if x.rows < y.rows {
                    self.handles().cut(y, 0, x.rows)
                } else {
                    y
                };
                elemwise::norm::residual_add(self.ctx(), x, y)
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
                turn:
                    poem_ir::Turn::Partial {
                        rotary_dim, span, ..
                    },
                ..
            } if *span == rotary_dim / 2 => elemwise::rope::partial(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
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
            } if *span == rotary_dim / 2 => elemwise::rope::partial_q(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
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
                        form,
                        rotary_dim,
                        ..
                    },
                ..
            } => {
                let rotate = match form {
                    MropeForm::Interleaved => elemwise::rope_mrope::interleaved,
                    MropeForm::Blocked => elemwise::rope_mrope::blocked,
                    MropeForm::Split => elemwise::rope_mrope::split,
                };
                rotate(
                    self.ctx(),
                    self.tensor(*q),
                    self.tensor(*k),
                    self.tensor(*positions),
                    *sections,
                    *rotary_dim,
                    *head_dim,
                    *theta,
                )
            }
            Elemwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),
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
            Elemwise::GateSigmoidMul { x, gate, x_out: _ } => {
                elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), self.tensor(*x))
            }
            Elemwise::SiluScaled { s, x, x_out: _ } => {
                elemwise::gate::silu_scaled(self.ctx(), *s, self.tensor(*x))
            }
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
            Elemwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, self.tensor(*y))
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
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
