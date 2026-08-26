//! `Linear`: the gemm anchor, the mlp activations, and the moe router,
//! bank, and combine arms.

use kernels::{DispatchLinear, KernelError};
use kernels_cuda::linear;
use model_ir::Linear;

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        match op {
            // ---- gemm (anchor) ----
            Linear::Matmul { act, w, y } => linear::gemm::matmul(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                &mut self.tensor(*y),
            ),
            Linear::LmHead { act, w, y } => linear::gemm::lm_head(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                &mut self.tensor(*y),
            ),
            Linear::AttentionLanding { act, w, layer, y } => linear::gemm::attention_landing(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                *layer,
                &mut self.tensor(*y),
            ),
            // ---- mlp ----
            Linear::MlpSwiglu {
                packed,
                intermediate,
                y,
            } => linear::mlp::swiglu(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                &mut self.tensor(*y),
            ),
            Linear::MlpSwigluClamp {
                packed,
                intermediate,
                limit,
                y,
            } => linear::mlp::swiglu_clamp(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *limit,
                &mut self.tensor(*y),
            ),
            Linear::MlpSwigluClampAlpha {
                packed,
                intermediate,
                limit,
                alpha,
                y,
            } => linear::mlp::swiglu_clamp_alpha(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *limit,
                *alpha,
                &mut self.tensor(*y),
            ),
            Linear::MlpGegluTanh { gate, up, y } => linear::mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                &mut self.tensor(*y),
            ),
            Linear::MlpGegluTanhPacked {
                packed,
                intermediate,
                y,
            } => linear::mlp::geglu_tanh_packed(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                &mut self.tensor(*y),
            ),
            Linear::MlpSitu {
                packed,
                intermediate,
                beta,
                up_cap,
                y,
            } => linear::mlp::situ(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *beta,
                *up_cap,
                &mut self.tensor(*y),
            ),
            // ---- moe ----
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
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Linear::MoeTopkSigmoid {
                logits,
                experts,
                top_k,
                renormalize,
                scaling,
                routes,
                weights,
            } => linear::moe::topk_sigmoid(
                self.ctx(),
                self.tensor(*logits),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Linear::MoeTopkSqrtSoftplus {
                logits,
                bias,
                experts,
                top_k,
                renormalize,
                scaling,
                routes,
                weights,
            } => linear::moe::topk_sqrt_softplus(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*bias),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Linear::MoeMatmulSelect { x, bank, routes, y } => linear::moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                &mut self.tensor(*y),
            ),
            // MENLO-SEAM: the IR's one `bank` id is two device planes — the
            // (codes, scales) pair the entry reads. The metal shell's
            // one-handle weight rows refused this form; here the weight
            // table seats it (`WeightRow::Planes`) and the id resolves
            // through `Run::planes`.
            Linear::MoeMatmulSelectBias {
                x,
                bank,
                bias,
                routes,
                y,
            } => {
                let (codes, scales) = self.planes(*bank);
                linear::moe::matmul_select_bias(
                    self.ctx(),
                    self.tensor(*x),
                    codes,
                    scales,
                    self.tensor(*bias),
                    self.tensor(*routes),
                    &mut self.tensor(*y),
                )
            }
            Linear::MoeWeightedSum { routed, weights, y } => linear::moe::weighted_sum(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*weights),
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
        }
    }
}
