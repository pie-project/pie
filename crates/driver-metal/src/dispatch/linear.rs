//! The `linear` family: `impl DispatchLinear for Run<'_>`, holding the gemm
//! arms plus the absorbed `mlp` and `moe` groups.

use kernels::{DispatchLinear, KernelError};
use kernels_metal::linear;
use model_ir::Linear;

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
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

            // The absorbed `mlp` family (`linear.mlp_*`), calling into
            // `kernels_metal::linear::mlp`.
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
                self.tensor(*y),
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
                self.tensor(*y),
            ),
            Linear::MlpGegluTanh { gate, up, y } => linear::mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                self.tensor(*y),
            ),
            Linear::MlpGegluTanhPacked {
                packed,
                intermediate,
                y,
            } => linear::mlp::geglu_tanh_packed(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                self.tensor(*y),
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
                self.tensor(*y),
            ),

            // The absorbed `moe` family (`linear.moe_*`), calling into
            // `kernels_metal::linear::moe`.
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
                self.tensor(*routes),
                self.tensor(*weights),
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
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::MoeMatmulSelect { x, bank, routes, y } => linear::moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                self.tensor(*y),
            ),
            // MENLO-SEAM: the IR's one `bank` id is two device planes — the
            // (codes, scales) pair the entry reads. The weight table seats
            // that form (`WeightRow::Planes`) and the id resolves through
            // `Run::planes`, never as one dense handle.
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
                    self.tensor(*y),
                )
            }
            // The same split-plane bank as `MoeMatmulSelectBias`, with
            // nothing added inside the fold — the routed bias of a rows-cut
            // expert lands afterwards, through `MoeBiasSum`.
            Linear::MoeMatmulSelectQuant { x, bank, routes, y } => {
                let (codes, scales) = self.planes(*bank);
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    codes,
                    scales,
                    self.tensor(*routes),
                    self.tensor(*y),
                )
            }
            // **THE CORRECTION CLASS IS NOT ON THIS PLANE** (palo C2). An
            // honest per-op refusal and not a fake: `kernels-metal` stamps no
            // routed low-rank pair, and a silently-skipped correction is a
            // lane that asked for an adapter and got the base model — which
            // is exactly the wrong output design §8 makes the capacity a
            // budget to avoid. The arm exists because the match is total by
            // construction; what would fill it is a metal `linear::lora`.
            Linear::LoraCorrect { .. } => Err(KernelError::Unsupported {
                op: "linear.lora_correct",
            }),
            Linear::MoeWeightedSum { routed, weights, y } => linear::moe::weighted_sum(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*weights),
                self.tensor(*y),
            ),
            Linear::MoeBiasSum {
                x,
                bias,
                routes,
                weights,
                y,
            } => linear::moe::bias_sum(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bias),
                self.tensor(*routes),
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
        }
    }
}
