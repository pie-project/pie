use kernels_vulkan::linear;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Linear, Operands};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_vulkan::Error> {
        match op {
            Linear::Relu2 { .. } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } if self.tensor(*act).dtype == poem_ir::Dtype::F32 && self.banked(*w).is_none() => {
                kernels_slang::linear::gemm::lane(
                    &self.slang(),
                    "linear.matmul",
                    crate::encode::slang_tensor(self.tensor(*act)),
                    crate::encode::slang_tensor(self.tensor(*w)),
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
            }
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } => match self.banked(*w) {
                Some(bank) => {
                    let capacity = self.capacity(*act).min(self.capacity(*y));
                    let (act, y) = (self.tensor(*act), self.tensor(*y));
                    let ladder = crate::encode::qmm_ladder(
                        self.tuning(),
                        act.rows,
                        y.width,
                        act.width,
                        bank,
                    );
                    kernels_slang::linear::qmm::matmul(
                        &self.slang(),
                        &ladder,
                        crate::encode::slang_tensor(act),
                        crate::encode::slang_bank(bank),
                        crate::encode::slang_tensor(y),
                        capacity,
                    )
                    .map_err(crate::encode::from_slang)
                }

                None => match self.maybe_stored(*w) {
                    Some(block) => kernels_slang::linear::kquant::matmul(
                        &self.slang(),
                        self.kquant_tactic(),
                        crate::encode::slang_tensor(self.tensor(*act)),
                        crate::encode::slang_tensor(block),
                        crate::encode::slang_tensor(self.tensor(*y)),
                    )
                    .map_err(crate::encode::from_slang),
                    None => kernels_slang::linear::gemm::matmul(
                        &self.slang(),
                        self.gemm_tactic(),
                        crate::encode::slang_tensor(self.tensor(*act)),
                        crate::encode::slang_tensor(self.tensor(*w)),
                        crate::encode::slang_tensor(self.tensor(*y)),
                    )
                    .map_err(crate::encode::from_slang),
                },
            },
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: true,
                ..
            } => match self.banked(*w) {
                Some(bank) => {
                    let capacity = self.capacity(*act).min(self.capacity(*y));
                    let (act, y) = (self.tensor(*act), self.tensor(*y));
                    let ladder = crate::encode::qmm_ladder(
                        self.tuning(),
                        act.rows,
                        y.width,
                        act.width,
                        bank,
                    );
                    kernels_slang::linear::qmm::lm_head(
                        &self.slang(),
                        &ladder,
                        crate::encode::slang_tensor(act),
                        crate::encode::slang_bank(bank),
                        crate::encode::slang_tensor(y),
                        capacity,
                    )
                    .map_err(crate::encode::from_slang)
                }

                None => match self.maybe_stored(*w) {
                    Some(block) => kernels_slang::linear::kquant::lm_head(
                        &self.slang(),
                        self.kquant_tactic(),
                        crate::encode::slang_tensor(self.tensor(*act)),
                        crate::encode::slang_tensor(block),
                        crate::encode::slang_tensor(self.tensor(*y)),
                    )
                    .map_err(crate::encode::from_slang),
                    None => kernels_slang::linear::gemm::lm_head(
                        &self.slang(),
                        self.gemm_tactic(),
                        crate::encode::slang_tensor(self.tensor(*act)),
                        crate::encode::slang_tensor(self.tensor(*w)),
                        crate::encode::slang_tensor(self.tensor(*y)),
                    )
                    .map_err(crate::encode::from_slang),
                },
            },

            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => kernels_slang::linear::glu::swiglu(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => kernels_slang::linear::glu::swiglu_clamp(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                *limit,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
            } => kernels_slang::linear::glu::swiglu_clamp_alpha(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                *limit,
                *alpha,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                limit: Some(limit),
                y,
                packed: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => kernels_slang::linear::glu::swiglu_clamp_split(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*gate)),
                crate::encode::slang_tensor(self.tensor(*up)),
                *limit,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                y,
                packed: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => kernels_slang::linear::glu::geglu_tanh(
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
            | Linear::MoeRoute {
                score: poem_ir::RouteScore::SigmoidSink { .. },
                ..
            }
            | Linear::Matmul {
                quant: Some(_),
                logits: false,
                ..
            }
            | Linear::MoeMatmulSelectActQuant { .. }
            | Linear::Glu {
                packed: None,
                gate: Some(_),
                weights: Some(_),
                limit: Some(_),
                act: poem_ir::GluAct::Swiglu,
                ..
            }
            | Linear::Glu {
                packed: Some(_),
                gate: None,
                weights: Some(_),
                limit: Some(_),
                act: poem_ir::GluAct::Swiglu,
                ..
            }
            | Linear::MoeSumAdd { .. } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            // The forms no builder makes.
            Linear::Matmul { .. } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => kernels_slang::linear::glu::geglu_tanh_packed(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Situ { beta, up_cap, .. },
                ..
            } => kernels_slang::linear::glu::situ(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *intermediate,
                *beta,
                *up_cap,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            // The forms no builder makes.
            Linear::Glu { .. } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            Linear::MoeRoute {
                logits,
                experts,
                top_k,
                routes,
                weights,
                scale: None,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => kernels_slang::linear::moe::route::topk_softmax(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*logits)),
                *experts,
                *top_k,
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.tensor(*weights)),
            )
            .map_err(crate::encode::from_slang),
            Linear::MoeRoute {
                logits,
                scale: Some(scale),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Softmax,
                ..
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

            Linear::MoeRoute {
                logits,
                bias,
                experts,
                top_k,
                routes,
                weights,
                hint: _,
                score:
                    poem_ir::RouteScore::Sigmoid {
                        renormalize,
                        scaling,
                        ..
                    },
                ..
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
            Linear::MoeRoute {
                logits,
                bias: Some(bias),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Predict,
                ..
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

            Linear::MoeRoute {
                logits,
                bias: Some(bias),
                experts,
                top_k,
                hint: _,
                routes,
                weights,
                score:
                    poem_ir::RouteScore::SqrtSoftplus {
                        renormalize,
                        scaling,
                        ..
                    },
                ..
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

            Linear::MoeRoute {
                ids: Some(ids),
                table: Some(tid2eid),
                logits,
                experts: _,
                top_k,
                routes,
                weights,
                score:
                    poem_ir::RouteScore::Hash {
                        vocab,
                        renormalize,
                        scaling,
                        ..
                    },
                ..
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
            // The forms no builder makes.
            Linear::MoeRoute { .. } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
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
            } => kernels_slang::linear::moe::select::matmul_grouped(
                &self.slang(),
                kernels_slang::Tier::Baseline,
                crate::encode::slang_tensor(self.tensor(*x)),
                match self.banked(*w) {
                    Some(bank) => kernels_slang::linear::moe::select::GroupedPlane::Bank(
                        crate::encode::slang_bank(bank),
                    ),
                    None => kernels_slang::linear::moe::select::GroupedPlane::Dense(
                        crate::encode::slang_tensor(self.tensor(*w)),
                    ),
                },
                crate::encode::slang_tensor(self.tensor(*routes)),
                *groups,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Linear::MoeMatmulSelect { x, bank, routes, y } => {
                kernels_slang::linear::moe::select::matmul_select(
                    &self.slang(),
                    kernels_slang::Tier::Baseline,
                    crate::encode::slang_tensor(self.tensor(*x)),
                    crate::encode::slang_tensor(self.tensor(*bank)),
                    crate::encode::slang_tensor(self.tensor(*routes)),
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
            }

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
                    && self.batched(
                        OP,
                        self.tensor(*x),
                        planes,
                        Some(bias_plane),
                        routed,
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                    )?
                {
                    return Ok(());
                }
                kernels_slang::linear::moe::select::matmul_select_bias(
                    &self.slang(),
                    kernels_slang::Tier::Baseline,
                    crate::encode::slang_tensor(self.tensor(*x)),
                    crate::encode::slang_bank(planes),
                    crate::encode::slang_tensor(bias_plane),
                    crate::encode::slang_tensor(routed),
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
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
                    && self.batched(
                        OP,
                        self.tensor(*x),
                        planes,
                        None,
                        routed,
                        self.experts(*routes),
                        scratch,
                        self.tensor(*y),
                    )?
                {
                    return Ok(());
                }
                kernels_slang::linear::moe::select::matmul_select_quant(
                    &self.slang(),
                    kernels_slang::Tier::Baseline,
                    crate::encode::slang_tensor(self.tensor(*x)),
                    crate::encode::slang_bank(planes),
                    crate::encode::slang_tensor(routed),
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang)
            }

            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            } => kernels_slang::linear::lora::correct(
                &self.slang(),
                kernels_slang::Tier::Subgroup,
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*bank_a)),
                crate::encode::slang_tensor(self.tensor(*bank_b)),
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.tensor(*y_out)),
            )
            .map_err(crate::encode::from_slang),
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

impl Run<'_> {
    /// The subgroup tier where every subgroup holds a 32-lane strip; four
    /// strips a workgroup, the 4090's best at every scheme and row count
    /// measured (kernels-slang.md §15).
    fn kquant_tactic(&self) -> kernels_slang::linear::kquant::Tactic {
        kernels_slang::linear::kquant::Tactic {
            tier: if self.tuning().limits().lanes >= 32 {
                kernels_slang::Tier::Subgroup
            } else {
                kernels_slang::Tier::Baseline
            },
            slices: 4,
        }
    }
}

impl Run<'_> {
    /// Cooperative-matrix tiles where the profile takes them.
    fn gemm_tactic(&self) -> kernels_slang::linear::gemm::Tactic {
        kernels_slang::linear::gemm::Tactic {
            tier: if self.tuning().gemm_coopmat() {
                kernels_slang::Tier::Coopmat
            } else {
                kernels_slang::Tier::Baseline
            },
        }
    }
}

impl Run<'_> {
    /// The routed matmul over sorted expert tiles, at the device's tile
    /// rows, sort width and tier; `Ok(false)` where it declines.
    #[allow(clippy::too_many_arguments)]
    fn batched(
        &self,
        op: &'static str,
        x: kernels_vulkan::Tensor,
        bank: kernels_vulkan::Bank,
        bias: Option<kernels_vulkan::Tensor>,
        routes: kernels_vulkan::Tensor,
        experts: u32,
        scratch: linear::moe::RoutedScratch,
        y: kernels_vulkan::Tensor,
    ) -> Result<bool, kernels_vulkan::Error> {
        use crate::encode::slang_tensor as st;
        let pairs = y.rows;
        let batch = kernels_slang::linear::moe::select::Batch {
            tile: linear::moe::tile_rows(pairs, experts, self.tuning()),
            padded: linear::moe::sorted_rows(pairs, experts, self.tuning()),
            tier: if self.tuning().moe_coopmat() {
                kernels_slang::Tier::Coopmat
            } else {
                kernels_slang::Tier::Baseline
            },
            sort_lanes: self.tuning().sort_lanes(),
        };
        let scratch = kernels_slang::linear::moe::select::RoutedScratch {
            perm: st(scratch.perm),
            row_expert: st(scratch.row_expert),
            tile_expert: st(scratch.tile_expert),
            inv: st(scratch.inv),
            x: st(scratch.x),
            y: st(scratch.y),
        };
        kernels_slang::linear::moe::select::matmul_select_batched(
            &self.slang(),
            op,
            st(x),
            crate::encode::slang_bank(bank),
            bias.map(st),
            st(routes),
            experts,
            scratch,
            st(y),
            batch,
        )
        .map_err(crate::encode::from_slang)
    }
}
