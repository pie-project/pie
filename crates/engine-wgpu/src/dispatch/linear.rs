use kernels_wgpu::linear;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Linear, Operands};

use crate::run::Run;

/// WGSL has no subgroup tier; four strips a workgroup (kernels-slang.md §15).
const KQUANT: kernels_slang::linear::kquant::Tactic = kernels_slang::linear::kquant::Tactic {
    tier: kernels_slang::Tier::Baseline,
    slices: 4,
};

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
                Some(bank) => {
                    let capacity = self.capacity(*act).min(self.capacity(*y));
                    let (act, y) = (self.tensor(*act), self.tensor(*y));
                    let ladder = qmm_ladder(self.tuning(), act.rows, y.width, act.width, bank);
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
                        KQUANT,
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
            Linear::LmHead { act, w, y } => match self.banked(*w) {
                Some(bank) => {
                    let capacity = self.capacity(*act).min(self.capacity(*y));
                    let (act, y) = (self.tensor(*act), self.tensor(*y));
                    let ladder = qmm_ladder(self.tuning(), act.rows, y.width, act.width, bank);
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
                        KQUANT,
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
            } => kernels_slang::linear::moe::select::matmul_grouped(
                &self.slang(),
                self.tier(),
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
                    self.tier(),
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
                    self.tier(),
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
                    self.tier(),
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
                kernels_slang::Tier::Baseline,
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

/// The device's `kernel.qmm` ladder at this shape, as kernels-slang's rungs:
/// the vector rung reads two activation rows a workgroup, which on the 4090
/// beat kernels-wgpu's one-row kernel at every row count from 1 to 11 where
/// four rows lost at two (kernels-slang.md §17).
fn qmm_ladder(
    tuning: &kernels_wgpu::Tuning,
    rows: u32,
    columns: u32,
    contraction: u32,
    bank: kernels_wgpu::Bank,
) -> Vec<kernels_slang::linear::qmm::Qmm> {
    use kernels_slang::linear::qmm::Qmm as Rung;
    use kernels_wgpu::tuning::Qmm;
    let at = tuning.qmm_at(rows, columns, contraction, bank.bits, bank.group);
    tuning
        .qmm(&at)
        .iter()
        .map(|q| match *q {
            Qmm::Tiled { crossover_tg } => Rung::Tiled { crossover_tg },
            Qmm::Qmv => Rung::Qmv { vecs: 2 },
        })
        .collect()
}

impl Run<'_> {
    /// The vector kernel folds by shuffles where the adapter's subgroups
    /// hold a 32-lane strip.
    fn gemm_tactic(&self) -> kernels_slang::linear::gemm::Tactic {
        kernels_slang::linear::gemm::Tactic { tier: self.tier() }
    }
}

impl Run<'_> {
    /// The routed matmul over sorted expert tiles, at the device's tile
    /// rows, sort width and tier; `Ok(false)` where it declines.
    #[allow(clippy::too_many_arguments)]
    fn batched(
        &self,
        op: &'static str,
        x: kernels_wgpu::Tensor,
        bank: kernels_wgpu::Bank,
        bias: Option<kernels_wgpu::Tensor>,
        routes: kernels_wgpu::Tensor,
        experts: u32,
        scratch: linear::moe::RoutedScratch,
        y: kernels_wgpu::Tensor,
    ) -> Result<bool, kernels_wgpu::Error> {
        use crate::encode::slang_tensor as st;
        let pairs = y.rows;
        let batch = kernels_slang::linear::moe::select::Batch {
            tile: linear::moe::tile_rows(pairs, experts, self.tuning()),
            padded: linear::moe::sorted_rows(pairs, experts, self.tuning()),
            tier: kernels_slang::Tier::Baseline,
            sort_lanes: 256,
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
