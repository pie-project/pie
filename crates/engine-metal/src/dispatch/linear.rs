use kernels_metal::linear;
use model_exec::{DispatchLinear, KernelError};
use model_ir::{Linear, Operands};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_metal::Error> {
        match op {
            Linear::Matmul { act, w, y }
                if self.tensor(*act).dtype == model_ir::Dtype::F32 && self.banked(*w).is_none() =>
            {
                linear::lane_gemm::act_x_wt(
                    self.ctx(),
                    "linear.matmul",
                    self.tensor(*act),
                    self.tensor(*w),
                    self.tensor(*y),
                )
            }
            Linear::Matmul { act, w, y } => match self.banked(*w) {
                Some(bank) => linear::quant::matmul(
                    self.ctx(),
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                    linear::quant::Scratch {
                        precast: &|rows, contraction| self.precast(rows, contraction),
                        partials: &|rows, width| self.partials(rows, width),
                    },
                    self.capacity(*act).min(self.capacity(*y)),
                ),
                None => linear::gemm::matmul(
                    self.ctx(),
                    self.tensor(*act),
                    self.tensor(*w),
                    self.tensor(*y),
                ),
            },
            Linear::LmHead { act, w, y } => match self.banked(*w) {
                Some(bank) => linear::quant::lm_head(
                    self.ctx(),
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                    linear::quant::Scratch {
                        precast: &|rows, contraction| self.precast(rows, contraction),
                        partials: &|rows, width| self.partials(rows, width),
                    },
                    self.capacity(*act).min(self.capacity(*y)),
                ),
                None => linear::gemm::lm_head(
                    self.ctx(),
                    self.tensor(*act),
                    self.tensor(*w),
                    self.tensor(*y),
                ),
            },

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
            Linear::MlpSwigluClampSplit { gate, up, limit, y } => linear::mlp::swiglu_clamp_split(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                *limit,
                self.tensor(*y),
            ),
            Linear::MlpGegluTanh { gate, up, y } => linear::mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                self.tensor(*y),
            ),
            Linear::MlpGeluTanh { x, y } => {
                linear::mlp::gelu_tanh(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            Linear::MatmulGeglu { .. }
            | Linear::LmHeadSoftcap { .. }
            | Linear::RelBias { .. }
            | Linear::MoeTopkSigmoidSink { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
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
            Linear::MoeTopkSoftmaxScaled {
                logits,
                scale,
                experts,
                top_k,
                routes,
                weights,
            } => linear::moe::topk_softmax_scaled(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*scale),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::MoeTopkSigmoid {
                logits,
                bias,
                experts,
                top_k,
                renormalize,
                scaling,
                routes,
                weights,
                hint: _,
            } => match bias {
                Some(bias) => linear::moe::topk_sigmoid_biased(
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
                None => linear::moe::topk_sigmoid(
                    self.ctx(),
                    self.tensor(*logits),
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    self.tensor(*routes),
                    self.tensor(*weights),
                ),
            },
            Linear::MoePredictRoute {
                logits,
                bias,
                experts,
                top_k,
                routes,
                weights,
            } => linear::moe::predict_route(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*bias),
                *experts,
                *top_k,
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
                hint: _,
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
            Linear::MoeHashRoute {
                ids,
                tid2eid,
                logits,
                vocab,
                experts: _,
                top_k,
                renormalize,
                scaling,
                routes,
                weights,
            } => linear::moe::hash_route(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*tid2eid),
                self.tensor(*logits),
                *vocab,
                *top_k,
                *renormalize,
                *scaling,
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
                match self.banked(*w) {
                    Some(bank) => linear::moe::GroupedPlane::Bank(bank),
                    None => linear::moe::GroupedPlane::Dense(self.tensor(*w)),
                },
                self.tensor(*routes),
                *groups,
                self.tensor(*y),
            ),
            Linear::MoeMatmulSelect { x, bank, routes, y } => linear::moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                self.tensor(*y),
            ),
            Linear::MoeMatmulSelectBias {
                x,
                bank,
                bias,
                routes,
                y,
            } => {
                const OP: &str = "linear.moe_matmul_select_bias";
                if let Some(scratch) = self.routed_scratch()
                    && linear::moe::matmul_select_batched(
                        self.ctx(),
                        OP,
                        self.tensor(*x),
                        self.planes(*bank),
                        Some(self.tensor(*bias)),
                        self.tensor(*routes),
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                        &kernels_metal::tuning::current(),
                    )?
                {
                    return Ok(());
                }
                linear::moe::matmul_select_bias(
                    self.ctx(),
                    self.tensor(*x),
                    self.planes(*bank),
                    self.tensor(*bias),
                    self.tensor(*routes),
                    self.tensor(*y),
                )
            }
            Linear::MoeMatmulSelectQuant { x, bank, routes, y } => {
                const OP: &str = "linear.moe_matmul_select_quant";
                if let Some(scratch) = self.routed_scratch()
                    && linear::moe::matmul_select_batched(
                        self.ctx(),
                        OP,
                        self.tensor(*x),
                        self.planes(*bank),
                        None,
                        self.tensor(*routes),
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                        &kernels_metal::tuning::current(),
                    )?
                {
                    return Ok(());
                }
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    self.planes(*bank),
                    self.tensor(*routes),
                    self.tensor(*y),
                )
            }
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
