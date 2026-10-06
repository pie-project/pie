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
            Linear::MlpAne {
                act,
                gate_up,
                down,
                intermediate,
                layer,
                packed,
                h,
                y,
            } => match crate::ane::plan(*layer, self.tensor(*act).rows) {
                Some(plan) => self.mlp_split(&plan, *act, *packed, *h, *y),
                None => {
                    self.linear(&Linear::Matmul {
                        act: *act,
                        w: *gate_up,
                        y: *packed,
                    })?;
                    self.linear(&Linear::MlpSwiglu {
                        packed: *packed,
                        intermediate: *intermediate,
                        y: *h,
                    })?;
                    self.linear(&Linear::Matmul {
                        act: *h,
                        w: *down,
                        y: *y,
                    })
                }
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
            | Linear::MatmulBias { .. }
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

impl Run<'_> {
    /// The GPU's columns of a split MLP, bracketed by the stage that hands
    /// the Neural Engine its input and the join that adds its output back.
    fn mlp_split(
        &self,
        plan: &crate::ane::Plan,
        act: model_ir::ValueId,
        packed: model_ir::ValueId,
        h: model_ir::ValueId,
        y: model_ir::ValueId,
    ) -> Result<(), kernels_metal::Error> {
        let keep = plan.split.keep;
        let x = self.tensor(act);
        let out = self.tensor(y);
        // The gate rows land in `packed`'s scratch and the up rows beside it;
        // `h` takes the GPU's columns of the activation.
        let gate = kernels_metal::Tensor {
            width: keep,
            ..self.tensor(packed)
        };
        let h_gpu = kernels_metal::Tensor {
            width: keep,
            ..self.tensor(h)
        };
        linear::ane::stage(self.ctx(), x, plan.staged, plan.stage)?;
        plan.submit();
        let precast = |rows, contraction| self.precast(rows, contraction);
        let partials = |rows, width| self.partials(rows, width);
        let scratch = || linear::quant::Scratch {
            precast: &precast,
            partials: &partials,
        };
        let rows = self.capacity(act);
        linear::quant::matmul(
            self.ctx(),
            x,
            plan.split.gate,
            gate,
            scratch(),
            rows.min(self.capacity(packed)),
        )?;
        linear::quant::matmul(self.ctx(), x, plan.split.up, plan.up_rows, scratch(), rows)?;
        linear::mlp::swiglu_clamp_split(self.ctx(), gate, plan.up_rows, f32::MAX, h_gpu)?;
        linear::quant::matmul(
            self.ctx(),
            h_gpu,
            plan.split.down,
            out,
            scratch(),
            self.capacity(h).min(self.capacity(y)),
        )?;
        linear::ane::join(self.ctx(), out, plan.other, plan.done)
    }
}
