use kernels_cerebras::elemwise;
use model_exec::{DispatchElementwise, KernelError};
use model_ir::{Elementwise, GateActivation, MropeForm, Operands};

use crate::run::Run;

impl DispatchElementwise for Run<'_> {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.elementwise(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn elementwise(&mut self, op: &Elementwise) -> Result<(), kernels_cerebras::Error> {
        match op {
            Elementwise::Rmsnorm { x, weight, eps, y } => elemwise::norm::rmsnorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
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
                matches!(act, GateActivation::Sigmoid),
                self.tensor(*y),
            ),
            Elementwise::ResidualAdd { x, y, y_out: _ } => {
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
            Elementwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),
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
            Elementwise::GateSigmoidMul { x, gate, x_out: _ } => {
                elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), self.tensor(*x))
            }
            Elementwise::SiluScaled { s, x, x_out: _ } => {
                elemwise::gate::silu_scaled(self.ctx(), *s, self.tensor(*x))
            }
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
            Elementwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, self.tensor(*y))
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
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
