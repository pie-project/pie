use kernels_wgpu::linear;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Linear, Operands};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_wgpu::Error> {
        match op {
            Linear::Relu2 { .. } => Err(kernels_wgpu::Error::Unsupported { op: op.name() }),
            Linear::Matmul { act, w, y } => match self.banked(*w) {
                Some(bank) => linear::quant::matmul(
                    self.ctx(),
                    self.tuning(),
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                    self.capacity(*act).min(self.capacity(*y)),
                ),

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
                Some(bank) => linear::quant::lm_head(
                    self.ctx(),
                    self.tuning(),
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                    self.capacity(*act).min(self.capacity(*y)),
                ),

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

            Linear::Swiglu {
                packed,
                intermediate,
                y,
            } => kernels_slang::linear::glu::swiglu(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::SwigluClamp {
                packed,
                intermediate,
                limit,
                y,
            } => kernels_slang::linear::glu::swiglu_clamp(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                *limit,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::SwigluClampAlpha {
                packed,
                intermediate,
                limit,
                alpha,
                y,
            } => kernels_slang::linear::glu::swiglu_clamp_alpha(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                *limit,
                *alpha,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            Linear::SwigluClampSplit { gate, up, limit, y } => {
                kernels_slang::linear::glu::swiglu_clamp_split(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*gate)),
                    crate::encode::slang_tensor(self.tensor(*up)),
                    *limit,
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
            }
            Linear::GegluTanh { gate, up, y } => kernels_slang::linear::glu::geglu_tanh(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*gate)),
                crate::encode::slang_tensor(self.tensor(*up)),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::GeluTanh { x, y } => kernels_slang::linear::glu::gelu_tanh(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            // A product that quantizes its activations runs only on CUDA's
            // tcgen05 GEMMs.
            Linear::RelBias { .. }
            | Linear::MoeTopkSigmoidSink { .. }
            | Linear::MatmulActQuant { .. }
            | Linear::MoeMatmulSelectActQuant { .. }
            | Linear::SwigluClampSplitRouted { .. }
            | Linear::SwigluClampRouted { .. }
            | Linear::MoeSumAdd { .. } => Err(kernels_wgpu::Error::Unsupported { op: op.name() }),
            Linear::GegluTanhPacked {
                packed,
                intermediate,
                y,
            } => kernels_slang::linear::glu::geglu_tanh_packed(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::Situ {
                packed,
                intermediate,
                beta,
                up_cap,
                y,
            } => kernels_slang::linear::glu::situ(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                *beta,
                *up_cap,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            Linear::MoeTopkSoftmax {
                logits,
                experts,
                top_k,
                routes,
                weights,
            } => kernels_slang::linear::moe::route::topk_softmax(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*logits)),
                *experts,
                *top_k,
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.tensor(*weights)),
            )
            .map_err(crate::encode::from_slang),
            Linear::MoeTopkSoftmaxScaled {
                logits,
                scale,
                experts,
                top_k,
                routes,
                weights,
            } => kernels_slang::linear::moe::route::topk_softmax_scaled(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*logits)),
                crate::encode::slang_tensor(self.tensor(*scale)),
                *experts,
                *top_k,
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.tensor(*weights)),
            )
            .map_err(crate::encode::from_slang),

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
                Some(bias) => kernels_slang::linear::moe::route::topk_sigmoid_biased(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*logits)),
                    crate::encode::slang_tensor(self.tensor(*bias)),
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    crate::encode::slang_tensor(self.tensor(*routes)),
                    crate::encode::slang_tensor(self.tensor(*weights)),
                )
                .map_err(crate::encode::from_slang),
                None => kernels_slang::linear::moe::route::topk_sigmoid(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*logits)),
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    crate::encode::slang_tensor(self.tensor(*routes)),
                    crate::encode::slang_tensor(self.tensor(*weights)),
                )
                .map_err(crate::encode::from_slang),
            },
            Linear::MoePredictRoute {
                logits,
                bias,
                experts,
                top_k,
                routes,
                weights,
            } => kernels_slang::linear::moe::route::predict_route(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*logits)),
                crate::encode::slang_tensor(self.tensor(*bias)),
                *experts,
                *top_k,
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.tensor(*weights)),
            )
            .map_err(crate::encode::from_slang),

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
            } => kernels_slang::linear::moe::route::topk_sqrt_softplus(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*logits)),
                crate::encode::slang_tensor(self.tensor(*bias)),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.tensor(*weights)),
            )
            .map_err(crate::encode::from_slang),

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
            } => kernels_slang::linear::moe::route::hash_route(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*ids)),
                crate::encode::slang_tensor(self.tensor(*tid2eid)),
                crate::encode::slang_tensor(self.tensor(*logits)),
                *vocab,
                *top_k,
                *renormalize,
                *scaling,
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.tensor(*weights)),
            )
            .map_err(crate::encode::from_slang),
            Linear::GroupRoutes { groups, routes } => {
                kernels_slang::linear::moe::route::group_routes(
                    &self.slang(),
                    *groups,
                    crate::encode::slang_tensor(self.tensor(*routes)),
                )
                .map_err(crate::encode::from_slang)
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

                let (planes, bias_plane, routed, staged) = match self.staged(
                    Some(self.planes(*bank)),
                    Some(self.tensor(*bias)),
                    self.tensor(*routes),
                )? {
                    Some((planes, bias_plane, routed)) => (
                        planes.expect("a staged bank answers a bank"),
                        bias_plane,
                        routed,
                        true,
                    ),
                    None => (self.planes(*bank), None, self.tensor(*routes), false),
                };
                let bias_plane = bias_plane.unwrap_or_else(|| self.tensor(*bias));
                if !staged
                    && let Some(scratch) = self.routed_scratch()
                    && linear::moe::matmul_select_batched(
                        self.ctx(),
                        OP,
                        self.tensor(*x),
                        planes,
                        Some(bias_plane),
                        routed,
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                        self.tuning(),
                    )?
                {
                    return Ok(());
                }
                linear::moe::matmul_select_bias(
                    self.ctx(),
                    self.tensor(*x),
                    planes,
                    bias_plane,
                    routed,
                    self.tensor(*y),
                )
            }

            Linear::MoeMatmulSelectQuant { x, bank, routes, y } => {
                const OP: &str = "linear.moe_matmul_select_quant";
                let (planes, routed, staged) =
                    match self.staged(Some(self.planes(*bank)), None, self.tensor(*routes))? {
                        Some((planes, _, routed)) => {
                            (planes.expect("a staged bank answers a bank"), routed, true)
                        }
                        None => (self.planes(*bank), self.tensor(*routes), false),
                    };
                if !staged
                    && let Some(scratch) = self.routed_scratch()
                    && linear::moe::matmul_select_batched(
                        self.ctx(),
                        OP,
                        self.tensor(*x),
                        planes,
                        None,
                        routed,
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                        self.tuning(),
                    )?
                {
                    return Ok(());
                }
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    planes,
                    routed,
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
            Linear::MoeWeightedSum { routed, weights, y } => {
                kernels_slang::linear::moe::combine::weighted_sum(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*routed)),
                    crate::encode::slang_tensor(self.tensor(*weights)),
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
            }

            Linear::MoeBiasSum {
                x,
                bias,
                routes,
                weights,
                y,
            } => {
                let (bias_plane, routed) =
                    match self.staged(None, Some(self.tensor(*bias)), self.tensor(*routes))? {
                        Some((_, bias_plane, routed)) => {
                            (bias_plane.expect("a staged bias answers a plane"), routed)
                        }
                        None => (self.tensor(*bias), self.tensor(*routes)),
                    };
                kernels_slang::linear::moe::combine::bias_sum(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*x)),
                    crate::encode::slang_tensor(bias_plane),
                    crate::encode::slang_tensor(routed),
                    crate::encode::slang_tensor(self.tensor(*weights)),
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
            }
            Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y,
            } => kernels_slang::linear::moe::combine::sigmoid_gate_add(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*routed)),
                crate::encode::slang_tensor(self.tensor(*shared)),
                crate::encode::slang_tensor(self.tensor(*gate)),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
        }
    }
}
