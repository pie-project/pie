use kernels_cuda::Tensor;
use kernels_cuda::linear;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::OffsetKind;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Dtype, Fused, Linear, Operands, ValueId};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    pub(super) fn fused_linear(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        match op {
            Fused::MatmulGeglu {
                act,
                w,
                intermediate,
                packed,
                y,
            } => {
                // The skinny leg reads the weight as one dense handle; a split-plane
                // bank resolves only through the matmul below.
                if self.dense_weight(*w) {
                    let a = self.tensor(*act);
                    let weight = self.tensor(*w);
                    let out = self.tensor(*y);
                    let (m, k) = (a.rows as i32, a.width as i32);
                    let i = i32::try_from(*intermediate).unwrap_or(0);
                    if weight.rows == 2 * *intermediate
                        && linear::skinny::covers(
                            m,
                            i,
                            k,
                            linear::skinny::Epilogue::Geglu,
                            self.ctx().tuning().gemm_skinny_rows,
                        )
                    {
                        return linear::skinny::skinny_bf16(
                            self.ctx(),
                            weight.ptr,
                            a.ptr,
                            out.ptr,
                            m,
                            i,
                            k,
                            linear::skinny::Epilogue::Geglu,
                        );
                    }
                }
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *packed,
                })?;
                self.linear(&Linear::GegluTanhPacked {
                    packed: *packed,
                    intermediate: *intermediate,
                    y: *y,
                })
            }
            // A decode's few rows take the projection and the swiglu as one
            // skinny fire; any other shape takes the three primitives.
            Fused::MlpSwiglu {
                act,
                gate_up,
                down,
                intermediate,
                packed,
                h,
                y,
            } => {
                let held = crate::run::HeldShared {
                    y: *y,
                    gate_up: *gate_up,
                    down: *down,
                };
                if self.holds_shared(*act, held) {
                    self.shared_mlp = Some(held);
                    return Ok(());
                }
                if self.tensor(*act).dtype == Dtype::Bf16
                    && self.dense_weight(*gate_up)
                    && linear::gemm::matmul_swiglu(
                        self.ctx(),
                        self.tensor(*act),
                        self.tensor(*gate_up),
                        &mut self.tensor(*h),
                    )?
                {
                    return self.linear(&Linear::Matmul {
                        act: *h,
                        w: *down,
                        y: *y,
                    });
                }
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *gate_up,
                    y: *packed,
                })?;
                self.linear(&Linear::Swiglu {
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
            Fused::MatmulBias {
                act,
                w,
                bias,
                y,
                y_out: _,
            } => {
                if self.tensor(*act).dtype == Dtype::Bf16 && self.dense_weight(*w) {
                    return linear::gemm::matmul_bias(
                        self.ctx(),
                        self.tensor(*act),
                        self.tensor(*w),
                        self.tensor(*bias),
                        &mut self.tensor(*y),
                    );
                }
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *y,
                })?;
                kernels_cuda::elemwise::pointwise::add_bias(
                    self.ctx(),
                    self.tensor(*bias),
                    &mut self.tensor(*y),
                )
            }
            Fused::LmHeadSoftcap {
                act,
                w,
                cap,
                y,
                y_out: _,
            } => {
                if self.dense_weight(*w) {
                    let a = self.tensor(*act);
                    let weight = self.tensor(*w);
                    let out = self.tensor(*y);
                    let (m, n, k) = (a.rows as i32, weight.rows as i32, a.width as i32);
                    let epilogue = linear::skinny::Epilogue::Softcap(*cap);
                    if linear::skinny::covers(
                        m,
                        n,
                        k,
                        epilogue,
                        self.ctx().tuning().gemm_skinny_rows,
                    ) {
                        return linear::skinny::skinny_bf16(
                            self.ctx(),
                            weight.ptr,
                            a.ptr,
                            out.ptr,
                            m,
                            n,
                            k,
                            epilogue,
                        );
                    }
                }
                self.linear(&Linear::LmHead {
                    act: *act,
                    w: *w,
                    y: *y,
                })?;
                kernels_cuda::attn::logit_softcap(self.ctx(), &mut self.tensor(*y), *cap)
            }
            _ => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
        }
    }

    fn linear(&mut self, op: &Linear) -> Result<(), kernels_cuda::Error> {
        if let Linear::Matmul { act, w, y } = op {
            let held = crate::run::HeldGate { gate: *y, w: *w };
            if self.tensor(*y).width == 1 && self.holds_gate(*act, held) {
                self.shared_gate_w = Some(held);
                return Ok(());
            }
        }
        match op {
            Linear::Matmul { act, w, y } if self.tensor(*act).dtype == Dtype::F32 => {
                let weight = self.dense_or_decoded(
                    "linear.matmul",
                    *w,
                    self.tensor(*y).width,
                    self.tensor(*act).width,
                )?;
                linear::lane_gemm::act_x_wt(
                    self.ctx(),
                    "linear.matmul",
                    self.tensor(*act),
                    weight,
                    &mut self.tensor(*y),
                )
            }
            Linear::Matmul { act, w, y } if self.tensor(*y).dtype == Dtype::F32 => {
                linear::gemm::matmul_f32(
                    self.ctx(),
                    self.tensor(*act),
                    self.tensor(*w),
                    &mut self.tensor(*y),
                )
            }
            Linear::Matmul { act, w, y } => match self.maybe_tiled_planes(*w) {
                Some((codes, scales, biases, seat)) => {
                    let act = self.tensor(*act);
                    // The GEMV lands one 16-row tile; past it the GEMM, whatever
                    // the dense arm's threshold.
                    let entry = if act.rows >= self.ctx().tuning().qmm_gemm_from
                        || act.rows > linear::affine::GEMV_ROWS
                    {
                        linear::affine::matmul
                    } else {
                        linear::affine::matmul_gemv
                    };
                    entry(
                        self.ctx(),
                        act,
                        codes,
                        scales,
                        biases,
                        &mut self.tensor(*y),
                        seat,
                    )
                }
                None => self.row_major_matmul(act, w, y),
            },
            Linear::LmHead { act, w, y } => match self.maybe_tiled_planes(*w) {
                Some((codes, scales, biases, seat)) => {
                    let act = self.tensor(*act);
                    // The GEMV lands one 16-row tile; past it the GEMM, whatever
                    // the dense arm's threshold.
                    let entry = if act.rows >= self.ctx().tuning().qmm_gemm_from
                        || act.rows > linear::affine::GEMV_ROWS
                    {
                        linear::affine::lm_head
                    } else {
                        linear::affine::lm_head_gemv
                    };
                    entry(
                        self.ctx(),
                        act,
                        codes,
                        scales,
                        biases,
                        &mut self.tensor(*y),
                        seat,
                    )
                }
                None => self.row_major_lm_head(act, w, y),
            },
            Linear::Swiglu {
                packed,
                intermediate,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::swiglu(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    &mut self.tensor(*y),
                )
            }
            Linear::SwigluClamp {
                packed,
                intermediate,
                limit,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::swiglu_clamp(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::SwigluClampAlpha {
                packed,
                intermediate,
                limit,
                alpha,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::swiglu_clamp_alpha(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    *limit,
                    *alpha,
                    &mut self.tensor(*y),
                )
            }
            Linear::SwigluClampSplitRouted {
                gate,
                up,
                weights,
                limit,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::swiglu_clamp_split_routed(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
                    self.tensor(*weights),
                    fan,
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::SwigluClampRouted {
                packed,
                intermediate,
                weights,
                limit,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::swiglu_clamp_routed(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    self.tensor(*weights),
                    fan,
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::MoeSumAdd { routed, shared, y } => linear::moe::sum_add(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*shared),
                &mut self.tensor(*y),
            ),
            Linear::MatmulActQuant {
                act,
                w,
                quant,
                input_scales: _,
                scales,
                y,
            } => self.matmul_act_quant(*act, *w, *quant, *scales, *y),
            Linear::MoeMatmulSelectActQuant {
                x,
                bank,
                routes,
                quant,
                input_scales: _,
                y,
            } => {
                const OP: &str = "linear.moe_matmul_select_act_quant";
                // The routes' rows read x's rows, so x's rows are what quantize.
                // An fp8 bank (f32 factors) reads e4m3 rows under f32 scales
                // a 1x128 block; any other bank, ue8m0 ones.
                let (codes, scales, biases, _) = self.planes(*bank);
                let fp8_bank = biases.is_none()
                    && linear::moe::fp8_tiles(self.tensor(*x), codes, scales, &self.tensor(*y));
                match *quant {
                    poem_ir::ActQuant::E4m3 { gran: 128 } if fp8_bank => {
                        self.select_fp8(*x, *bank, *routes, *y)
                    }
                    poem_ir::ActQuant::E4m3 { gran } if fp8_bank => {
                        Err(kernels_cuda::Error::Backend {
                            op: OP,
                            detail: format!(
                                "an fp8 bank folds f32 activation scales over its 128-wide blocks, not {gran}-wide ones"
                            ),
                        })
                    }
                    poem_ir::ActQuant::E4m3 { gran } => {
                        let x = self.tensor(*x);
                        let fan = self.plane_fan(x.rows);
                        let x = linear::act_quant::quantized(self.ctx(), OP, x, gran, fan)?;
                        self.select_quant(x, *bank, *routes, *y)
                    }
                    poem_ir::ActQuant::Nvfp4 => Err(nvfp4_activations(OP)),
                }
            }
            Linear::SwigluClampSplit { gate, up, limit, y } => linear::glu::swiglu_clamp_split(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                *limit,
                &mut self.tensor(*y),
            ),
            Linear::GegluTanh { gate, up, y } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::geglu_tanh(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
                    fan,
                    &mut self.tensor(*y),
                )
            }
            Linear::GeluTanh { x, y } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::gelu_tanh(self.ctx(), self.tensor(*x), fan, &mut self.tensor(*y))
            }
            Linear::Relu2 { x, y } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::relu2(self.ctx(), self.tensor(*x), fan, &mut self.tensor(*y))
            }
            Linear::GegluTanhPacked {
                packed,
                intermediate,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::geglu_tanh_packed(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    &mut self.tensor(*y),
                )
            }
            Linear::Situ {
                packed,
                intermediate,
                beta,
                up_cap,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::situ(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    *beta,
                    *up_cap,
                    &mut self.tensor(*y),
                )
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
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
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
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
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
            } => linear::moe::topk_sigmoid(
                self.ctx(),
                self.tensor(*logits),
                bias.map(|bias| self.tensor(bias)),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
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
                bias.map(|bias| self.tensor(bias)),
                scale.map(|scale| self.tensor(scale)),
                *experts,
                *top_k,
                *sink,
                *scaling,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Linear::RelBias {
                x,
                w,
                heads,
                d_rel,
                extent,
                y,
            } => kernels_cuda::elemwise::posenc::rel_bias(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*w),
                *heads,
                *d_rel,
                *extent,
                &mut self.tensor(*y),
            ),
            Linear::MoePredictRoute {
                logits,
                bias,
                experts,
                top_k,
                routes,
                weights,
            } => linear::moe::topk_sqrt_softplus(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*bias),
                *experts,
                *top_k,
                false,
                1.0,
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
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
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
            } => linear::moe_route::hash_route(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*tid2eid),
                self.tensor(*logits),
                *vocab,
                *top_k,
                *renormalize,
                *scaling,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Linear::GroupRoutes { groups, routes } => {
                linear::moe_route::group_routes(self.ctx(), *groups, &mut self.tensor(*routes))
            }
            Linear::MatmulGrouped {
                x,
                w,
                routes,
                groups,
                y,
            } => {
                const OP: &str = "linear.matmul_grouped";
                if self.maybe_tiled_planes(*w).is_some() {
                    return Err(kernels_cuda::Error::Backend {
                        op: OP,
                        detail: "a quantized o-projection plane is not yet read grouped on the \
                                 CUDA arm"
                            .to_string(),
                    });
                }
                let x = self.tensor(*x);
                let y = self.tensor(*y);
                let groups_nz = *groups;
                if groups_nz == 0
                    || !x.width.is_multiple_of(groups_nz)
                    || !y.width.is_multiple_of(groups_nz)
                {
                    return Err(kernels_cuda::Error::Backend {
                        op: OP,
                        detail: format!(
                            "{groups_nz} groups do not divide a {}-wide row into a {}-wide one",
                            x.width, y.width
                        ),
                    });
                }
                let rows = x.rows * groups_nz;
                let x =
                    kernels_cuda::tensor::Tensor::new(x.ptr, rows, x.width / groups_nz, x.dtype);
                let mut y =
                    kernels_cuda::tensor::Tensor::new(y.ptr, rows, y.width / groups_nz, y.dtype);
                let (bank, experts) = self.expert_bank(*w);
                // The weight is the groups' [n, k] blocks stacked as one
                // [groups·n, k] matrix; the select reads it as `groups`
                // experts of n·k, the shape its expert-ordered legs take.
                let (n, k) = (u64::from(y.width), u64::from(x.width));
                let bank = if !experts.streams()
                    && u64::from(bank.rows) * u64::from(bank.width) == u64::from(groups_nz) * n * k
                    && let Ok(stride) = u32::try_from(n * k)
                {
                    kernels_cuda::tensor::Tensor::new(bank.ptr, groups_nz, stride, bank.dtype)
                } else {
                    bank
                };
                let identity = self.group_routes_of(*routes, groups_nz);
                linear::moe::matmul_grouped(
                    self.ctx(),
                    x,
                    bank,
                    self.tensor(*routes),
                    &mut y,
                    experts,
                    identity,
                )
            }
            Linear::MoeMatmulSelect { x, bank, routes, y } => {
                let (bank, experts) = self.expert_bank(*bank);
                linear::moe::matmul_select(
                    self.ctx(),
                    self.tensor(*x),
                    bank,
                    self.tensor(*routes),
                    &mut self.tensor(*y),
                    experts,
                )
            }
            Linear::MoeMatmulSelectBias {
                x,
                bank,
                bias,
                routes,
                y,
            } => {
                let (codes, scales, affine, seat) = self.planes(*bank);
                debug_assert!(
                    affine.is_none(),
                    "the biased select is the mxfp4 gate/up leg's; an affine bank's \
                     zero points ride the quant twin"
                );
                linear::moe::matmul_select_bias(
                    self.ctx(),
                    self.tensor(*x),
                    codes,
                    scales,
                    self.tensor(*bias),
                    self.tensor(*routes),
                    &mut self.tensor(*y),
                    seat,
                )
            }
            Linear::MoeMatmulSelectQuant { x, bank, routes, y } => {
                let x = self.tensor(*x);
                self.select_quant(x, *bank, *routes, *y)
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

    pub(super) fn dense_weight(&mut self, w: ValueId) -> bool {
        self.maybe_tiled_planes(w).is_none() && self.maybe_planes(w).is_none()
    }

    /// A split-plane bank's routed select of `x`'s rows.
    fn select_quant(
        &mut self,
        x: kernels_cuda::Tensor,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let (codes, scales, biases, seat) = self.planes(bank);
        let routes = self.tensor(routes);
        if self.planes_in_lanes(bank) {
            return linear::moe::matmul_select_mxfp4_lanes(
                self.ctx(),
                x,
                codes,
                scales,
                routes,
                &mut self.tensor(y),
            );
        }
        let (codes, scales, biases, routes, seat) =
            match self.staged_experts(codes, scales, biases, routes) {
                Some((codes, scales, biases, routes)) => {
                    (codes, scales, biases, routes, GroupSeat::RESIDENT)
                }
                None => (codes, scales, biases, routes, seat),
            };
        linear::moe::matmul_select_quant(
            self.ctx(),
            x,
            codes,
            scales,
            biases,
            routes,
            &mut self.tensor(y),
            seat,
        )
    }

    /// `act`'s rows quantized for the FP8 products: as an earlier product
    /// of this walk quantized them (same rows, window and stream), or now,
    /// and held for the later ones when the batch is a decode batch's.
    fn fp8_quantized(
        &mut self,
        op: &'static str,
        act: ValueId,
        pow2: bool,
    ) -> Result<(kernels_cuda::Tensor, kernels_cuda::Tensor), kernels_cuda::Error> {
        let stage = match self.ctx().stage() {
            kernels_cuda::jit::ArgValue::Ptr(p) => p,
            _ => 0,
        };
        let ctx = core::ptr::from_ref(self.ctx()) as usize;
        if let Some(held) = self.fp8_rows.iter().find(|held| {
            held.value == act.0 && held.pow2 == pow2 && held.stage == stage && held.ctx == ctx
        }) {
            return Ok((held.codes, held.scales));
        }
        let x = self.tensor(act);
        let region = self.at_region();
        let hold = match self.fp8_holds.iter_mut().find(|(r, _)| *r == region) {
            Some((_, next)) => {
                *next += 1;
                *next - 1
            }
            None => {
                self.fp8_holds.push((region, 1));
                0
            }
        } % linear::act_quant::FP8_HOLDS;
        let fan = self.plane_fan(x.rows);
        let (q, s) = linear::act_quant::quantized_codes_held(self.ctx(), op, x, hold, pow2, fan)?;
        // The hold's earlier rows are gone; past a decode batch nothing is
        // held (the stream's slab).
        self.fp8_rows
            .retain(|held| !(held.region == region && held.hold == hold));
        if x.rows <= 64 {
            self.fp8_rows.push(crate::run::Fp8Rows {
                value: act.0,
                pow2,
                stage,
                ctx,
                region,
                hold,
                codes: q,
                scales: s,
            });
        }
        Ok((q, s))
    }

    /// An fp8 bank's select against the rows quantized as e4m3 under f32
    /// scales a 1x128 block.
    fn select_fp8(
        &mut self,
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let (q, act_scales) = self.fp8_quantized("linear.moe_matmul_select_act_quant", x, false)?;
        let x = self.tensor(x);
        let (codes, scales, biases, seat) = self.planes(bank);
        let routes = self.tensor(routes);
        let (codes, scales, routes, seat) = match self.staged_experts(codes, scales, biases, routes)
        {
            Some((codes, scales, _, routes)) => (codes, scales, routes, GroupSeat::RESIDENT),
            None => (codes, scales, routes, seat),
        };
        linear::moe::matmul_select_fp8_quantized(
            self.ctx(),
            x,
            q,
            act_scales,
            codes,
            scales,
            routes,
            &mut self.tensor(y),
            seat,
        )
    }

    /// A projection whose activations quantize first: an fp8b128 one
    /// (DeepSeek's W8A8) against e4m3 in its `gran`s.
    fn matmul_act_quant(
        &mut self,
        act: ValueId,
        w: ValueId,
        quant: poem_ir::ActQuant,
        scales: Option<ValueId>,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "linear.matmul_act_quant";
        let gran = match quant {
            poem_ir::ActQuant::E4m3 { gran } => gran,
            poem_ir::ActQuant::Nvfp4 => return Err(nvfp4_activations(OP)),
        };
        let Some(scales) = scales else {
            return Err(kernels_cuda::Error::Backend {
                op: OP,
                detail: "an MXFP4 projection against e4m3 activations has no dense product on \
                         rocm (its MXFP4 products are the routed selects); no catalog \
                         deployment declares one"
                    .to_string(),
            });
        };
        let codes = self.tensor(w);
        let scales = self.tensor(scales);
        let ordered = self.ordered_bank(w).is_some();
        // f32 factors: an e4m3tile projection (transformers' FP8 release),
        // against e4m3 activations under f32 scales on the FP8 matrix core.
        if scales.dtype == Dtype::F32 {
            if gran != linear::act_quant::FP8_BLOCK {
                return Err(kernels_cuda::Error::Backend {
                    op: OP,
                    detail: format!(
                        "an e4m3tile128 projection folds f32 activation scales over its 128-wide blocks, not \
                         {gran}-wide ones"
                    ),
                });
            }
            // An ordered plane's factors are an fp8b128 one's exponents
            // (`weights::hip::order_fp8b_planes`): its rows quantize under
            // DeepSeek's power-of-two scales.
            let (q, act_scales) = self.fp8_quantized(OP, act, ordered)?;
            return linear::act_quant::matmul_fp8_quantized(
                self.ctx(),
                OP,
                q,
                act_scales,
                codes,
                scales,
                &mut self.tensor(y),
            );
        }
        // An fp8b32 plane in the FP8 matrix core's order.
        if ordered {
            return linear::act_quant::matmul_fp8b32_lanes(
                self.ctx(),
                OP,
                self.tensor(act),
                codes,
                scales,
                &mut self.tensor(y),
            );
        }
        let (n, k) = (codes.rows, codes.width);
        let holds = self.decoded_tile_holds(n, k);
        linear::act_quant::matmul_fp8b128(
            self.ctx(),
            OP,
            self.tensor(act),
            codes,
            scales,
            gran,
            holds,
            &mut self.tensor(y),
        )
    }

    fn row_major_matmul(
        &mut self,
        act: &ValueId,
        w: &ValueId,
        y: &ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        match self.maybe_planes(*w) {
            Some((codes, scales, biases, seat)) => {
                let act = self.tensor(*act);
                let held = self.decoded_tile_holds(self.tensor(*y).width, act.width);
                let entry =
                    if act.rows >= self.ctx().tuning().qmm_gemm_from && !seat.streams() && held {
                        linear::quant::matmul_via_dense
                    } else {
                        linear::quant::matmul
                    };
                entry(
                    self.ctx(),
                    act,
                    codes,
                    scales,
                    OffsetKind::Post,
                    biases,
                    Dtype::Bf16,
                    &mut self.tensor(*y),
                    seat,
                )
            }
            None => linear::gemm::matmul(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                &mut self.tensor(*y),
            ),
        }
    }

    fn row_major_lm_head(
        &mut self,
        act: &ValueId,
        w: &ValueId,
        y: &ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        match self.maybe_planes(*w) {
            Some((codes, scales, biases, seat)) => {
                let act = self.tensor(*act);
                let held = self.decoded_tile_holds(self.tensor(*y).width, act.width);
                let entry =
                    if act.rows >= self.ctx().tuning().qmm_gemm_from && !seat.streams() && held {
                        linear::quant::lm_head_via_dense
                    } else {
                        linear::quant::lm_head
                    };
                entry(
                    self.ctx(),
                    act,
                    codes,
                    scales,
                    OffsetKind::Post,
                    biases,
                    Dtype::Bf16,
                    &mut self.tensor(*y),
                    seat,
                )
            }
            None => {
                // Over a body's readouts, only the live ones (a staged word).
                let act = self.tensor(*act);
                let readouts = self.bindings().readout_rows;
                let live = if act.rows == readouts.rows {
                    self.bindings().readout_live
                } else {
                    0
                };
                linear::gemm::lm_head_live(
                    self.ctx(),
                    act,
                    self.tensor(*w),
                    live,
                    &mut self.tensor(*y),
                )
            }
        }
    }
}

const STAGED_EXPERT_BYTES: u64 = 1536 * 1024 * 1024;

impl Run<'_> {
    fn staged_experts(
        &self,
        codes: Tensor,
        scales: Tensor,
        biases: Option<Tensor>,
        routes: Tensor,
    ) -> Option<(Tensor, Tensor, Option<Tensor>, Tensor)> {
        if routes.rows == 0 || !crate::device::alloc::is_host_pointer(codes.ptr) {
            return None;
        }
        let stream = self.ctx().stream();
        if crate::device::alloc::is_capturing(stream) {
            return None;
        }
        let count = routes.rows as usize * routes.width as usize;
        let mut picked = vec![0i32; count];
        if crate::device::copy_any(stream, picked.as_mut_ptr() as u64, routes.ptr, count * 4)
            .is_err()
        {
            return None;
        }
        let mut unique: Vec<i32> = picked.iter().copied().filter(|e| *e >= 0).collect();
        unique.sort_unstable();
        unique.dedup();
        if unique.is_empty() {
            return None;
        }
        let per_expert = |plane: Tensor| u64::from(plane.width);
        let bytes_each = per_expert(codes) + per_expert(scales) + biases.map_or(0, per_expert);
        if unique.len() as u64 * bytes_each > STAGED_EXPERT_BYTES {
            return None;
        }
        let slot_of = |expert: i32| unique.binary_search(&expert).map_or(-1, |at| at as i32);
        let remapped: Vec<i32> = picked
            .iter()
            .map(|e| if *e < 0 { -1 } else { slot_of(*e) })
            .collect();
        let stage = |name: &'static str, plane: Tensor| -> Option<Tensor> {
            let width = plane.width as usize;
            let ptr = staging(name, unique.len() * width)?;
            for (slot, expert) in unique.iter().enumerate() {
                crate::device::copy_any(
                    stream,
                    ptr + (slot * width) as u64,
                    plane.ptr + u64::from(*expert as u32) * width as u64,
                    width,
                )
                .ok()?;
            }
            Some(Tensor::new(
                ptr,
                unique.len() as u32,
                plane.width,
                plane.dtype,
            ))
        };
        let staged_codes = stage("moe.staged.codes", codes)?;
        let staged_scales = stage("moe.staged.scales", scales)?;
        let staged_biases = match biases {
            Some(biases) => Some(stage("moe.staged.biases", biases)?),
            None => None,
        };
        let routes_ptr = staging("moe.staged.routes", count * 4)?;
        crate::device::copy_any(stream, routes_ptr, remapped.as_ptr() as u64, count * 4).ok()?;
        Some((
            staged_codes,
            staged_scales,
            staged_biases,
            Tensor::new(routes_ptr, routes.rows, routes.width, routes.dtype),
        ))
    }
}

/// A staging buffer per `name`, grown by doubling. A buffer it outgrows is
/// kept, never freed: copies queued on it may still be in flight, and a
/// free here would either race them or synchronise the device (illegal
/// mid-capture). Doubling keeps what is retired under the live buffer's
/// size.
fn staging(name: &'static str, bytes: usize) -> Option<u64> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    struct Held {
        live: HashMap<&'static str, (u64, usize)>,
        retired: Vec<u64>,
    }
    static HELD: Mutex<Option<Held>> = Mutex::new(None);
    if bytes == 0 {
        return None;
    }
    let mut held = HELD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let held = held.get_or_insert_with(|| Held {
        live: HashMap::new(),
        retired: Vec::new(),
    });
    let cap = held.live.get(name).map_or(0, |(_, cap)| *cap);
    if let Some((ptr, _)) = held.live.get(name).filter(|_| cap >= bytes) {
        return Some(*ptr);
    }
    let size = bytes.max(cap * 2);
    let fresh = crate::device::alloc::raw_alloc(size)?;
    if let Some((old, _)) = held.live.insert(name, (fresh, size)) {
        held.retired.push(old);
    }
    Some(fresh)
}

/// NVFP4 activations (`nvfp4a4`) are a tcgen05 product, CUDA's alone.
fn nvfp4_activations(op: &'static str) -> kernels_cuda::Error {
    kernels_cuda::Error::Backend {
        op,
        detail: "NVFP4 activations (`nvfp4a4`) are multiplied on CUDA's tcgen05 GEMMs, which \
                 rocm has none of; serve the deployment that reads bf16 activations instead"
            .to_string(),
    }
}
