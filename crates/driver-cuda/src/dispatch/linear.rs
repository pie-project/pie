//! `Linear`: the gemm anchor, the mlp activations, the moe router, bank and
//! combine arms, and the LoRA correction over a routed adapter bank.

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
            // The same two-plane bank as the biased twin above, with nothing
            // added inside the fold: the down leg's routed bias lands after
            // the reduce, through `MoeBiasSum`.
            Linear::MoeMatmulSelectQuant { x, bank, routes, y } => {
                let (codes, scales) = self.planes(*bank);
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    codes,
                    scales,
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
                &mut self.tensor(*y),
            ),
            // ---- the correction class (design §8) ----
            //
            // `y` and `y_out` are ONE arena column — the compiler folded the
            // in-place pair — so the arm resolves `y_out` and writes through
            // it, which is the same address `y` names. Resolving `y` here
            // instead would be right today and wrong the first time a pass
            // stops aliasing; the output seat is the one the walk owns.
            //
            // Both banks resolve through `Run::tensor` like any other weight,
            // because that is exactly what they are: `Def::Weight` rows whose
            // bytes came from `register_adapter` instead of the checkpoint
            // (`ParamSource::Registered`), and the runtime index rides in
            // `routes` inside the op — the MoE precedent, followed.
            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            //
            // AND THE SEGMENTS, WHICH ARE THIS ARM'S ALONE ON THIS PLANE.
            // `Run::segments` is `None` for a window P4 seated — every row of
            // the rectangle is a row of the correction — and `Some` for one it
            // answered `Fallback::Grouped` for, where the rectangle is the
            // union of the correction's intervals and the list says which of
            // its rows are its own. This is the only op `driver_cuda::GROUPED`
            // names, and it is the only arm here that may take a grouped
            // window: the compiler wrote that row BECAUSE this shell named
            // this op, so the two statements are one.
            } => linear::lora::correct(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank_a),
                self.tensor(*bank_b),
                self.tensor(*routes),
                &mut self.tensor(*y_out),
                self.segments(),
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
