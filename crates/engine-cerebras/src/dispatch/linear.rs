use kernels_cerebras::linear;
use model_exec::{DispatchLinear, KernelError};
use model_ir::{Linear, Operands};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_cerebras::Error> {
        match op {
            Linear::Matmul { act, w, y } => linear::gemm::matmul(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                self.tensor(*y),
            ),
            Linear::LmHead { act, w, y } => linear::gemm::lm_head(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                self.tensor(*y),
            ),
            Linear::MlpSwiglu {
                packed,
                intermediate,
                y,
            } => linear::mlp::swiglu(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                self.tensor(*y),
            ),
            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            } => linear::lora::correct(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank_a),
                self.tensor(*bank_b),
                self.tensor(*routes),
                self.tensor(*y_out),
            ),
            Linear::MlpGeluTanh { x, y } => {
                linear::mlp::gelu_tanh(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            Linear::MoeTopkSoftmax {
                logits,
                experts,
                top_k,
                routes,
                weights,
            } => linear::moe::topk_softmax(
                self.ctx(),
                self.tensor(*logits),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::GroupRoutes { groups, routes } => {
                linear::moe::group_routes(self.ctx(), *groups, self.tensor(*routes))
            }
            Linear::MatmulGrouped {
                x,
                w,
                routes,
                groups,
                y,
            } => linear::moe::matmul_grouped(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*w),
                self.tensor(*routes),
                *groups,
                self.tensor(*y),
            ),
            Linear::MoeMatmulSelect { x, bank, routes, y }
            | Linear::MoeMatmulSelectQuant { x, bank, routes, y } => linear::moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                self.tensor(*y),
            ),
            Linear::MoeWeightedSum { routed, weights, y } => linear::moe::weighted_sum(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*weights),
                self.tensor(*y),
            ),
            Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y,
            } => linear::moe::sigmoid_gate_add(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*shared),
                self.tensor(*gate),
                self.tensor(*y),
            ),
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
