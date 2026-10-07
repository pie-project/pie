use kernels_xla::linear;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Linear, Operands};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// Whether weight `w` is one dense plane (not a bank, not a K-quant
    /// block format).
    fn dense_weight(&self, w: poem_ir::ValueId) -> bool {
        self.banked(w).is_none() && self.maybe_stored(w).is_none()
    }

    fn linear(&mut self, op: &Linear) -> Result<(), kernels_xla::Error> {
        match op {
            Linear::Matmul { act, w, y } => match self.banked(*w) {
                Some(bank) => {
                    linear::quant::matmul(self.ctx(), self.tensor(*act), bank, self.tensor(*y))
                }

                None => match self.maybe_stored(*w) {
                    Some(block) => linear::kquant::matmul(
                        self.ctx(),
                        self.tensor(*act),
                        block,
                        self.tensor(*y),
                    ),
                    None => linear::gemm::matmul(
                        self.ctx(),
                        self.tensor(*act),
                        self.tensor(*w),
                        self.tensor(*y),
                    ),
                },
            },
            Linear::LmHead { act, w, y } => match self.banked(*w) {
                Some(bank) => {
                    linear::quant::lm_head(self.ctx(), self.tensor(*act), bank, self.tensor(*y))
                }

                None => match self.maybe_stored(*w) {
                    Some(block) => linear::kquant::lm_head(
                        self.ctx(),
                        self.tensor(*act),
                        block,
                        self.tensor(*y),
                    ),
                    None => linear::gemm::lm_head(
                        self.ctx(),
                        self.tensor(*act),
                        self.tensor(*w),
                        self.tensor(*y),
                    ),
                },
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
            // The fused forms take one dense weight; a quantized weight is
            // the plain matmul followed by the epilogue, as engine-cuda does.
            Linear::MatmulGeglu {
                act,
                w,
                intermediate,
                packed,
                y,
            } => {
                if self.dense_weight(*w) {
                    return linear::gemm::matmul_geglu(
                        self.ctx(),
                        self.tensor(*act),
                        self.tensor(*w),
                        *intermediate,
                        self.tensor(*packed),
                        self.tensor(*y),
                    );
                }
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *packed,
                })?;
                self.linear(&Linear::MlpGegluTanhPacked {
                    packed: *packed,
                    intermediate: *intermediate,
                    y: *y,
                })
            }
            Linear::MatmulBias {
                act,
                w,
                bias,
                y,
                y_out: _,
            } => {
                if self.dense_weight(*w) {
                    return linear::gemm::matmul_bias(
                        self.ctx(),
                        self.tensor(*act),
                        self.tensor(*w),
                        self.tensor(*bias),
                        self.tensor(*y),
                    );
                }
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *y,
                })?;
                kernels_xla::elemwise::norm::add_bias(
                    self.ctx(),
                    self.tensor(*bias),
                    self.tensor(*y),
                )
            }
            Linear::LmHeadSoftcap {
                act,
                w,
                cap,
                y,
                y_out: _,
            } => {
                if self.dense_weight(*w) {
                    return linear::gemm::lm_head_softcap(
                        self.ctx(),
                        self.tensor(*act),
                        self.tensor(*w),
                        *cap,
                        self.tensor(*y),
                    );
                }
                self.linear(&Linear::LmHead {
                    act: *act,
                    w: *w,
                    y: *y,
                })?;
                kernels_xla::attn::logit_softcap(self.ctx(), self.tensor(*y), *cap)
            }
            Linear::RelBias {
                x,
                w,
                heads,
                d_rel,
                extent,
                y,
            } => linear::gemm::rel_bias(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*w),
                *heads,
                *d_rel,
                *extent,
                self.tensor(*y),
            ),
            Linear::MoeTopkSigmoidSink {
                logits,
                bias,
                scale,
                experts,
                top_k,
                sink,
                scaling,
                routes,
                weights,
            } => linear::moe::topk_sigmoid_sink(
                self.ctx(),
                self.tensor(*logits),
                bias.map(|b| self.tensor(b)),
                scale.map(|s| self.tensor(s)),
                *experts,
                *top_k,
                *sink,
                *scaling,
                self.tensor(*routes),
                self.tensor(*weights),
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
                // A readout-rowed `x` (an MTP head) against token-rowed
                // routes and result: the GPU kernel counts rows off `x`.
                self.first_rows(self.tensor(*routes), *x),
                *groups,
                self.first_rows(self.tensor(*y), *x),
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
            } => linear::moe::matmul_select_bias(
                self.ctx(),
                self.tensor(*x),
                self.planes(*bank),
                self.tensor(*bias),
                self.tensor(*routes),
                self.tensor(*y),
            ),
            Linear::MoeMatmulSelectQuant { x, bank, routes, y } => {
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    self.planes(*bank),
                    self.tensor(*routes),
                    self.tensor(*y),
                )
            }
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
