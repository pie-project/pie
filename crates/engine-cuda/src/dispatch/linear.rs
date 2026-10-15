use kernels_cuda::Tensor;
use kernels_cuda::linear;
#[cfg(feature = "cuda")]
use kernels_cuda::linear::act_quant;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::OffsetKind;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{ActQuant, Dtype, Fused, Linear, Operands, ValueId};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    pub(super) fn fused_linear(&mut self, op: &Fused) -> Result<(), kernels_cuda::Error> {
        match op {
            Fused::MoeSelectGlu { select, glu } => {
                #[cfg(feature = "cuda")]
                if self.select_glu(select, glu)? {
                    return Ok(());
                }
                self.linear(select)?;
                self.linear(glu)
            }
            Fused::MoeExperts {
                select,
                glu,
                down,
                sum,
                gate_add,
            } => {
                #[cfg(feature = "cuda")]
                if self.moe_experts(select, glu, down, sum, gate_add.as_deref())? {
                    return Ok(());
                }
                // Apart, the gate-up select and its GLU as their own fusion.
                self.fused_linear(&Fused::MoeSelectGlu {
                    select: select.clone(),
                    glu: glu.clone(),
                })?;
                self.linear(down)?;
                self.linear(sum)?;
                match gate_add {
                    Some(gate_add) => self.linear(gate_add),
                    None => Ok(()),
                }
            }
            Fused::MatmulSwiglu {
                act,
                w,
                intermediate,
                packed,
                y,
            } => {
                #[cfg(feature = "cuda")]
                if self.dense_glu(*act, *w, *y, linear::tc5::Glu::Swiglu)? {
                    return Ok(());
                }
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *packed,
                })?;
                self.linear(&Linear::MlpSwiglu {
                    packed: *packed,
                    intermediate: *intermediate,
                    y: *y,
                })
            }
            Fused::MatmulGeglu {
                act,
                w,
                intermediate,
                packed,
                y,
            } => {
                #[cfg(feature = "cuda")]
                if self.dense_glu(*act, *w, *y, linear::tc5::Glu::GegluTanh)? {
                    return Ok(());
                }
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
                self.linear(&Linear::MlpGegluTanhPacked {
                    packed: *packed,
                    intermediate: *intermediate,
                    y: *y,
                })
            }
            // Not in `FUSED`, so never formed here; whole, it is its three
            // primitives.
            Fused::MlpSwiglu {
                act,
                gate_up,
                down,
                intermediate,
                packed,
                h,
                y,
            } => {
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
                kernels_cuda::elemwise::norm::add_bias(
                    self.ctx(),
                    self.tensor(*bias),
                    &mut self.tensor(*y),
                )
            }
            Fused::MatmulSplitRows {
                act,
                w,
                width,
                y,
                left,
                right,
            } => {
                // Each half from its own slice of the weight rows, so the
                // packed rows are never written and cut; a short fire, or a
                // chip that keeps the old forms, runs the pair.
                let a = self.tensor(*act);
                if self.ctx().tuning().rows_vec
                    && a.dtype == Dtype::Bf16
                    && a.rows >= self.ctx().tuning().gemm_split_rows_from
                    && self.dense_weight(*w)
                {
                    let weight = self.tensor(*w);
                    let (l, r) = (self.tensor(*left), self.tensor(*right));
                    if weight.dtype == Dtype::Bf16 && weight.rows == l.width + r.width {
                        let cut = u64::from(*width) * u64::from(weight.width) * 2;
                        let top = Tensor::new(weight.ptr, *width, weight.width, weight.dtype);
                        let bottom = Tensor::new(
                            weight.ptr + cut,
                            weight.rows - *width,
                            weight.width,
                            weight.dtype,
                        );
                        linear::gemm::matmul(self.ctx(), a, top, &mut self.tensor(*left))?;
                        return linear::gemm::matmul(
                            self.ctx(),
                            a,
                            bottom,
                            &mut self.tensor(*right),
                        );
                    }
                }
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *y,
                })?;
                kernels_cuda::layout::split_rows(
                    self.ctx(),
                    self.tensor(*y),
                    *width,
                    &mut self.tensor(*left),
                    &mut self.tensor(*right),
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
            Linear::MatmulActQuant {
                act,
                w,
                quant,
                scales,
                y,
                ..
            } => self.matmul_act_quant(*act, *w, *quant, *scales, *y),
            Linear::Matmul { act, w, y } => match self.maybe_tiled_planes(*w) {
                Some((codes, scales, biases, seat)) => {
                    let act = self.tensor(*act);
                    let entry = if act.rows >= self.ctx().tuning().qmm_gemm_from {
                        linear::tiled::matmul
                    } else {
                        linear::tiled::matmul_gemv
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
                    let entry = if act.rows >= self.ctx().tuning().qmm_gemm_from {
                        linear::tiled::lm_head
                    } else {
                        linear::tiled::lm_head_gemv
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
            Linear::MlpSwiglu {
                packed,
                intermediate,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::swiglu(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpSwigluClamp {
                packed,
                intermediate,
                limit,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::swiglu_clamp(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpSwigluClampAlpha {
                packed,
                intermediate,
                limit,
                alpha,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::swiglu_clamp_alpha(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    *limit,
                    *alpha,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpSwigluClampSplitRouted {
                gate,
                up,
                weights,
                limit,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::swiglu_clamp_split_routed(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
                    self.tensor(*weights),
                    fan,
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpSwigluClampRouted {
                packed,
                intermediate,
                weights,
                limit,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::swiglu_clamp_routed(
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
            Linear::MlpSwigluClampSplit { gate, up, limit, y } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::swiglu_clamp_split(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
                    fan,
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpGegluTanh { gate, up, y } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::geglu_tanh(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
                    fan,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpGeluTanh { x, y } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::gelu_tanh(self.ctx(), self.tensor(*x), fan, &mut self.tensor(*y))
            }
            Linear::MlpGegluTanhPacked {
                packed,
                intermediate,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::geglu_tanh_packed(
                    self.ctx(),
                    self.tensor(*packed),
                    *intermediate,
                    fan,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpSitu {
                packed,
                intermediate,
                beta,
                up_cap,
                y,
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::situ(
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
            } => linear::rel_bias::rel_bias(
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
                // One "expert" a group: the plane viewed `[groups, n * k]`, so
                // a select that lays routes out by expert (the grouped and
                // tcgen05 legs) counts groups, not the plane's rows.
                let span = u64::from(bank.rows) * u64::from(bank.width);
                let bank = match u32::try_from(span / u64::from(groups_nz)) {
                    Ok(per) if span % u64::from(groups_nz) == 0 => {
                        kernels_cuda::tensor::Tensor::new(bank.ptr, groups_nz, per, bank.dtype)
                    }
                    _ => bank,
                };
                linear::moe::matmul_select(
                    self.ctx(),
                    x,
                    bank,
                    self.tensor(*routes),
                    &mut y,
                    experts,
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
                if codes.dtype == Dtype::Nvfp4 {
                    return Err(kernels_cuda::Error::Backend {
                        op: "linear.moe_matmul_select_bias",
                        detail: "an nvfp4 bank's biased select has no cuda point; its \
                                 select reads no expert bias"
                            .to_string(),
                    });
                }
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
            Linear::MoeMatmulSelectActQuant {
                x,
                bank,
                routes,
                quant,
                input_scales,
                y,
            } => self.select_act_quant(*x, *bank, *routes, *quant, *input_scales, *y),
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

    /// A dense bf16 gate-up projection and its GLU as one tcgen05 GEMM, the
    /// GLU in its epilogue, from `kernel.gemm.tc5_glu_from` rows on; `false`
    /// leaves the projection and the GLU to run apart.
    #[cfg(feature = "cuda")]
    fn dense_glu(
        &mut self,
        act: ValueId,
        w: ValueId,
        y: ValueId,
        glu: linear::tc5::Glu,
    ) -> Result<bool, kernels_cuda::Error> {
        use linear::tc5;
        let a = self.tensor(act);
        let out = self.tensor(y);
        if a.dtype != Dtype::Bf16
            || !self.dense_weight(w)
            || a.rows < self.ctx().tuning().gemm_tc5_glu_from
            || !tc5::supported()
        {
            return Ok(false);
        }
        let weight = self.tensor(w);
        let (k, inter) = (a.width, out.width);
        if weight.dtype != Dtype::Bf16
            || weight.rows != 2 * inter
            || weight.width != k
            || !k.is_multiple_of(64)
            || !inter.is_multiple_of(8)
        {
            return Ok(false);
        }
        // Every row the padded extent holds, as the GEMM it stands for.
        let m = u32::try_from(
            self.ctx()
                .opaque_rows(i32::try_from(a.rows).unwrap_or(i32::MAX)),
        )
        .unwrap_or(a.rows);
        // A GLU tile is 128 gate rows and their up rows over every K: a
        // product of too few tiles to reach a quarter of the SMs (a narrow
        // shared expert at decode widths) streams its weight through those
        // few, slower than the GEMM and the GLU apart.
        let sms = self.ctx().multiprocessors().unwrap_or(148);
        if inter.div_ceil(128) * m.div_ceil(128) < sms / 4 {
            return Ok(false);
        }
        // A tcgen05 launch that fails leaves the projection and the GLU to
        // run apart.
        Ok(tc5::gemm_epi(
            self.ctx(),
            tc5::Operand::plain(a.ptr, tc5::Elem::Bf16),
            tc5::Operand::plain(weight.ptr, tc5::Elem::Bf16),
            out.ptr,
            [m, inter, k],
            1.0,
            tc5::Epilogue {
                glu: Some(glu),
                ..tc5::Epilogue::STORE
            },
        )
        .is_ok())
    }

    /// A gate-up select and its GLU as one grouped tcgen05 GEMM, the GLU in
    /// its epilogue (a bf16 bank, or an mxfp4 one, biased or not), when the
    /// fire takes it; `false` leaves the two to run apart.
    #[cfg(feature = "cuda")]
    fn select_glu(&mut self, select: &Linear, glu: &Linear) -> Result<bool, kernels_cuda::Error> {
        use linear::tc5::Glu;
        // The device and the profile first: nothing resolved off tcgen05.
        if !self.ctx().tuning().moe_tc5 || !linear::tc5::supported() {
            return Ok(false);
        }
        let (packed, y, glu) = match *glu {
            Linear::MlpSwiglu { packed, y, .. } => (packed, y, Glu::Swiglu),
            Linear::MlpSwigluClamp {
                packed, y, limit, ..
            } => (packed, y, Glu::SwigluClamp { limit }),
            Linear::MlpSwigluClampAlpha {
                packed,
                y,
                limit,
                alpha,
                ..
            } => (packed, y, Glu::SwigluClampAlpha { limit, alpha }),
            Linear::MlpGegluTanhPacked { packed, y, .. } => (packed, y, Glu::GegluTanh),
            _ => return Ok(false),
        };
        let Some((x, routes, sy, bank, experts)) = self.routed_bank(select) else {
            return Ok(false);
        };
        if sy != packed {
            return Ok(false);
        }
        let x = self.tensor(x);
        let routes = self.tensor(routes);
        let mut out = self.tensor(y);
        linear::moe::select_glu(self.ctx(), x, bank, routes, &mut out, glu, experts)
    }

    /// A routed select's operands for the grouped tcgen05 GEMM: its input,
    /// routes and output, and the bank as the GEMM reads it. `None` for a
    /// bank it does not read (streamed, scales in atoms, host-seated, a
    /// biased NVFP4 or affine one).
    #[cfg(feature = "cuda")]
    fn routed_bank(
        &mut self,
        select: &Linear,
    ) -> Option<(
        ValueId,
        ValueId,
        ValueId,
        linear::moe::GluBank,
        linear::moe::ExpertTable,
    )> {
        use linear::moe::{ExpertTable, GluBank};
        Some(match *select {
            Linear::MoeMatmulSelect {
                x,
                bank,
                routes,
                y: sy,
            } => {
                let (bank, experts) = self.expert_bank(bank);
                (x, routes, sy, GluBank::Bf16(bank), experts)
            }
            Linear::MoeMatmulSelectBias {
                x,
                bank,
                bias,
                routes,
                y: sy,
            } => {
                let (codes, scales, affine, seat) = self.planes(bank);
                if codes.dtype == Dtype::Nvfp4
                    || affine.is_some()
                    || seat.streams()
                    || self.scales_in_atoms(bank)
                {
                    return None;
                }
                let bias = Some(self.tensor(bias));
                (
                    x,
                    routes,
                    sy,
                    GluBank::Mxfp4 {
                        codes,
                        scales,
                        bias,
                    },
                    ExpertTable::RESIDENT,
                )
            }
            Linear::MoeMatmulSelectQuant {
                x,
                bank,
                routes,
                y: sy,
            } => {
                let (codes, scales, biases, seat) = self.planes(bank);
                if seat.streams()
                    || self.scales_in_atoms(bank)
                    || crate::device::alloc::is_host_pointer(codes.ptr)
                {
                    return None;
                }
                // NVFP4 carries its tensor scales as `biases`, an affine
                // bank its zeros, mxfp4 none.
                let bank = match (codes.dtype == Dtype::Nvfp4, biases) {
                    (true, Some(tensor_scales)) => GluBank::Nvfp4 {
                        codes,
                        scales,
                        tensor_scales,
                    },
                    (true, None) => return None,
                    (false, Some(zeros)) => GluBank::Affine {
                        codes,
                        scales,
                        zeros,
                    },
                    (false, None) => GluBank::Mxfp4 {
                        codes,
                        scales,
                        bias: None,
                    },
                };
                (x, routes, sy, bank, ExpertTable::RESIDENT)
            }
            // An NVFP4 bank against NVFP4 activations (W4A4), its scales laid
            // in atoms and its input scales stated.
            Linear::MoeMatmulSelectActQuant {
                x,
                bank,
                routes,
                quant: ActQuant::Nvfp4,
                input_scales: Some(input_scales),
                y: sy,
            } => {
                let (codes, atoms, biases, seat) = self.planes(bank);
                if codes.dtype != Dtype::Nvfp4
                    || seat.streams()
                    || !self.scales_in_atoms(bank)
                    || crate::device::alloc::is_host_pointer(codes.ptr)
                {
                    return None;
                }
                let tensor_scales = biases?;
                (
                    x,
                    routes,
                    sy,
                    GluBank::Nvfp4A4 {
                        codes,
                        atoms,
                        tensor_scales,
                        input_scales: self.tensor(input_scales),
                    },
                    ExpertTable::RESIDENT,
                )
            }
            // An MXFP4 bank against e4m3 activations in 1x128 blocks (W4A8),
            // its scales laid in atoms.
            Linear::MoeMatmulSelectActQuant {
                x,
                bank,
                routes,
                quant: ActQuant::E4m3 { gran: 128 },
                input_scales: None,
                y: sy,
            } => {
                let (codes, atoms, biases, seat) = self.planes(bank);
                if !matches!(codes.dtype, Dtype::Mxfp4 | Dtype::U8)
                    || biases.is_some()
                    || seat.streams()
                    || !self.scales_in_atoms(bank)
                    || crate::device::alloc::is_host_pointer(codes.ptr)
                {
                    return None;
                }
                (
                    x,
                    routes,
                    sy,
                    GluBank::Mxfp4A8 { codes, atoms },
                    ExpertTable::RESIDENT,
                )
            }
            _ => return None,
        })
    }

    /// A routed MLP whose rows stay laid out by expert from the gate-up select
    /// to the weighted sum (`linear::moe::experts`); false when it does not
    /// take this one.
    #[cfg(feature = "cuda")]
    fn moe_experts(
        &mut self,
        select: &Linear,
        glu: &Linear,
        down: &Linear,
        sum: &Linear,
        gate_add: Option<&Linear>,
    ) -> Result<bool, kernels_cuda::Error> {
        use linear::tc5::Glu;
        if !self.ctx().tuning().moe_tc5 || !linear::tc5::supported() {
            return Ok(false);
        }
        // DeepSeek's form: the route weights scale the GLU's rows before the
        // down leg, and the routes sum plainly with the shared row added.
        if let (
            Linear::MlpSwigluClampRouted {
                packed,
                intermediate,
                weights,
                limit,
                y: hidden,
            },
            Linear::MoeSumAdd {
                routed: summed,
                shared,
                y,
            },
        ) = (glu, sum)
        {
            let (packed, intermediate, weights, limit, hidden) =
                (*packed, *intermediate, *weights, *limit, *hidden);
            let (summed, shared, y) = (*summed, *shared, *y);
            if gate_add.is_some() {
                return Ok(false);
            }
            let (Some((x, routes, up_y, up, _)), Some((down_x, down_routes, down_y, down, _))) =
                (self.routed_bank(select), self.routed_bank(down))
            else {
                return Ok(false);
            };
            if up_y != packed || down_x != hidden || down_routes != routes || down_y != summed {
                return Ok(false);
            }
            let x = self.tensor(x);
            let routes = self.tensor(routes);
            let weights = self.tensor(weights);
            let shared = self.tensor(shared);
            let mut out = self.tensor(y);
            return linear::moe::experts_weighted_first(
                self.ctx(),
                x,
                up,
                Glu::SwigluClamp { limit },
                down,
                routes,
                weights,
                intermediate,
                shared,
                &mut out,
            );
        }
        let (packed, hidden, intermediate, glu) = match *glu {
            Linear::MlpSwiglu {
                packed,
                y,
                intermediate,
            } => (packed, y, intermediate, Glu::Swiglu),
            Linear::MlpSwigluClamp {
                packed,
                y,
                intermediate,
                limit,
            } => (packed, y, intermediate, Glu::SwigluClamp { limit }),
            Linear::MlpSwigluClampAlpha {
                packed,
                y,
                intermediate,
                limit,
                alpha,
            } => (
                packed,
                y,
                intermediate,
                Glu::SwigluClampAlpha { limit, alpha },
            ),
            Linear::MlpGegluTanhPacked {
                packed,
                y,
                intermediate,
            } => (packed, y, intermediate, Glu::GegluTanh),
            _ => return Ok(false),
        };
        let Linear::MoeWeightedSum {
            routed: summed,
            weights,
            y,
        } = *sum
        else {
            return Ok(false);
        };
        let (
            Some((x, routes, up_y, up, up_experts)),
            Some((down_x, down_routes, down_y, down, down_experts)),
        ) = (self.routed_bank(select), self.routed_bank(down))
        else {
            return Ok(false);
        };
        if up_y != packed || down_x != hidden || down_routes != routes || down_y != summed {
            return Ok(false);
        }
        // Under a gate, the gate-add's row lands, the shared row added.
        let (y, gate) = match gate_add {
            None => (y, None),
            Some(&Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y: gated,
            }) if routed == y => (gated, Some((self.tensor(shared), self.tensor(gate)))),
            Some(_) => return Ok(false),
        };
        let x = self.tensor(x);
        let routes = self.tensor(routes);
        let weights = self.tensor(weights);
        let mut out = self.tensor(y);
        linear::moe::experts(
            self.ctx(),
            x,
            up,
            glu,
            down,
            routes,
            weights,
            intermediate,
            &mut out,
            up_experts,
            down_experts,
            gate,
        )
    }

    /// A quantized bank's select of `x`.
    fn select_quant(
        &mut self,
        x: Tensor,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let (codes, scales, biases, seat) = self.planes(bank);
        let routes = self.tensor(routes);
        per_tensor_e4m3_is_a_table("linear.moe_matmul_select_quant", codes)?;
        if codes.dtype == Dtype::Nvfp4 {
            let tensor_scales = tensor_scales("linear.moe_matmul_select_quant", biases)?;
            let (codes, scales, tensor_scales, routes, seat) =
                match self.staged_experts(codes, scales, Some(tensor_scales), routes) {
                    Some((codes, scales, Some(tensor_scales), routes)) => {
                        (codes, scales, tensor_scales, routes, GroupSeat::RESIDENT)
                    }
                    _ => (codes, scales, tensor_scales, routes, seat),
                };
            return linear::nvfp4::matmul_select(
                self.ctx(),
                x,
                codes,
                scales,
                tensor_scales,
                routes,
                &mut self.tensor(y),
                seat,
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

    #[cfg(not(feature = "cuda"))]
    fn select_act_quant(
        &mut self,
        _: ValueId,
        _: ValueId,
        _: ValueId,
        _: ActQuant,
        _: Option<ValueId>,
        _: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        lowp("linear.moe_matmul_select_act_quant")
    }

    #[cfg(not(feature = "cuda"))]
    fn matmul_act_quant(
        &mut self,
        _: ValueId,
        _: ValueId,
        _: ActQuant,
        _: Option<ValueId>,
        _: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        lowp("linear.matmul_act_quant")
    }

    /// A select whose activations are quantized first, on the grouped
    /// tcgen05 GEMM: NVFP4 against an NVFP4 bank, each route under its
    /// expert's static global, or e4m3 in `gran`s against an MXFP4 bank.
    #[cfg(feature = "cuda")]
    fn select_act_quant(
        &mut self,
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        quant: ActQuant,
        input_scales: Option<ValueId>,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "linear.moe_matmul_select_act_quant";
        lowp(OP)?;
        let (codes, scales, biases, seat) = self.planes(bank);
        per_tensor_e4m3_is_a_table(OP, codes)?;
        if seat.streams() || crate::device::alloc::is_host_pointer(codes.ptr) {
            return Err(kernels_cuda::Error::Backend {
                op: OP,
                detail: "a low-precision select reads a resident bank, and this one is \
                         seated by a streaming tier"
                    .to_string(),
            });
        }
        let x = self.tensor(x);
        let route_t = self.tensor(routes);
        let mut out = self.tensor(y);
        let (experts, n, k) = (codes.rows, out.width, x.width);
        let (fmt, elem, gran, alpha_e, alpha_from) = match (codes.dtype, quant) {
            (Dtype::Nvfp4, ActQuant::Nvfp4) => {
                let globals = self.tensor(input_scales.ok_or_else(|| {
                    kernels_cuda::Error::Backend {
                        op: OP,
                        detail: "an nvfp4 bank's W4A4 select is traced without its input \
                                 scales"
                            .to_string(),
                    }
                })?);
                let tensor_scales = tensor_scales(OP, biases)?;
                let alpha = self
                    .ctx()
                    .scratch_stream(OP, EXPERT_ALPHA, experts as usize * 4)?
                    as u64;
                // The select lays it with the fire's routes.
                let from = act_quant::AlphaFrom {
                    input_scales: globals.ptr,
                    tensor_scales: tensor_scales.ptr,
                    rows: n,
                };
                (
                    act_quant::Format::Nvfp4,
                    linear::tc5::Elem::E2m1,
                    16,
                    alpha,
                    Some(from),
                )
            }
            // An MXFP4 bank's codes are bound as bytes; only an MXFP4 bank pairs
            // with e4m3 activations.
            (Dtype::Mxfp4 | Dtype::U8, ActQuant::E4m3 { gran }) => (
                act_quant::Format::E4m3 { gran },
                linear::tc5::Elem::E2m1,
                32,
                0,
                None,
            ),
            (stored, quant) => {
                return Err(kernels_cuda::Error::Backend {
                    op: OP,
                    detail: format!(
                        "a {stored:?} bank has no select against {quant:?} activations; \
                         NVFP4 banks take NVFP4 activations and MXFP4 banks e4m3"
                    ),
                });
            }
        };
        if !self.scales_in_atoms(bank) {
            return Err(unlaid(OP));
        }
        let atoms = scales.ptr;
        // At decode widths an MXFP4 bank's e4m3 select reads its activations
        // quantized and read back to bf16 (exact, the same products) on the
        // e2m1-convert register-dequant kernel, which streams the bank far
        // faster than the grouped GEMM does a handful of rows an expert.
        if let act_quant::Format::E4m3 { .. } = fmt
            && kernels_cuda::linear::moe::select_mxfp4_atoms_takes(self.ctx(), route_t, x)
        {
            let back =
                self.ctx()
                    .scratch_stream(OP, READ_BACK, x.rows as usize * x.width as usize * 2)?
                    as u64;
            let back =
                act_quant::read_back(self.ctx(), OP, fmt, act_quant::Source::rows(x, None), back)?;
            let bank_codes =
                kernels_cuda::tensor::Tensor::new(codes.ptr, experts, n * k / 2, Dtype::U8);
            if kernels_cuda::linear::moe::select_mxfp4_atoms(
                self.ctx(),
                back,
                bank_codes,
                atoms,
                route_t,
                &mut out,
            )? {
                return Ok(());
            }
        }
        act_quant::select(
            self.ctx(),
            OP,
            fmt,
            x,
            route_t,
            input_scales.map(|v| self.tensor(v)),
            act_quant::Bank {
                codes: codes.ptr,
                atoms,
                experts,
                n,
                k,
                elem,
                gran: gran as u32,
                alpha_e,
                alpha_from,
            },
            &mut out,
        )
    }

    /// A projection whose activations are quantized first, on the tcgen05
    /// GEMM: e4m3 in `gran`s against MXFP4 codes.
    #[cfg(feature = "cuda")]
    fn matmul_act_quant(
        &mut self,
        act: ValueId,
        w: ValueId,
        quant: ActQuant,
        block_scales: Option<ValueId>,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "linear.matmul_act_quant";
        lowp(OP)?;
        if let Some(block_scales) = block_scales {
            return self.fp8_block_matmul(act, w, quant, block_scales, y);
        }
        let (codes, scales, _, seat) = self.planes(w);
        if seat.streams() {
            return Err(kernels_cuda::Error::Backend {
                op: OP,
                detail: "a low-precision projection reads resident planes, and these are \
                         seated by a streaming tier"
                    .to_string(),
            });
        }
        if !self.scales_in_atoms(w) {
            return Err(unlaid(OP));
        }
        let act = self.tensor(act);
        let mut out = self.tensor(y);
        let (n, k) = (out.width, act.width);
        match (codes.dtype, quant) {
            (Dtype::Mxfp4 | Dtype::U8, ActQuant::E4m3 { gran }) => act_quant::matmul(
                self.ctx(),
                OP,
                act_quant::Format::E4m3 { gran },
                act_quant::Source::rows(act, None),
                act_quant::Weight {
                    codes: codes.ptr,
                    scales: scales.ptr,
                    n,
                    k,
                    elem: linear::tc5::Elem::E2m1,
                    gran: 32,
                    laid: true,
                },
                &mut out,
                1.0,
            ),
            // The tcgen05 GEMM takes its alpha from the host, and an NVFP4
            // projection's (`input_scale · tensor scale`) lives on the device.
            (stored, quant) => Err(kernels_cuda::Error::Backend {
                op: OP,
                detail: format!(
                    "a {stored:?} projection has no dense product against {quant:?} \
                     activations yet; MXFP4 projections take e4m3"
                ),
            }),
        }
    }

    /// An fp8b128 projection against e4m3 activations in 1x128 blocks on
    /// one tcgen05 GEMM (DeepSeek's W8A8): raw e4m3 codes beside the block
    /// exponents the load spread into atoms.
    #[cfg(feature = "cuda")]
    fn fp8_block_matmul(
        &mut self,
        act: ValueId,
        w: ValueId,
        quant: ActQuant,
        block_scales: ValueId,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "linear.matmul_act_quant";
        if quant != (ActQuant::E4m3 { gran: 128 }) {
            return Err(kernels_cuda::Error::Backend {
                op: OP,
                detail: format!(
                    "an fp8b128 projection reads e4m3 activations in 1x128 blocks, not \
                     {quant:?}"
                ),
            });
        }
        let codes = self.tensor(w);
        let atoms = self.tensor(block_scales);
        if atoms.dtype != Dtype::U8 {
            return Err(unlaid(OP));
        }
        let act = self.tensor(act);
        let mut out = self.tensor(y);
        let (n, k) = (out.width, act.width);
        act_quant::matmul(
            self.ctx(),
            OP,
            act_quant::Format::E4m3 { gran: 128 },
            act_quant::Source::rows(act, None),
            act_quant::Weight {
                codes: codes.ptr,
                scales: atoms.ptr,
                n,
                k,
                elem: linear::tc5::Elem::E4m3,
                gran: 128,
                laid: true,
            },
            &mut out,
            1.0,
        )
    }

    fn dense_weight(&mut self, w: ValueId) -> bool {
        self.maybe_tiled_planes(w).is_none()
            && self.maybe_planes(w).is_none()
            && self.maybe_stored(w).is_none()
    }

    fn row_major_matmul(
        &mut self,
        act: &ValueId,
        w: &ValueId,
        y: &ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        if let Some((codes, ..)) = self.maybe_planes(*w) {
            per_tensor_e4m3_is_a_table("linear.matmul", codes)?;
        }
        match self.maybe_planes(*w) {
            Some((codes, scales, biases, seat)) if codes.dtype == Dtype::Nvfp4 => {
                linear::nvfp4::matmul(
                    self.ctx(),
                    "linear.matmul",
                    self.tensor(*act),
                    codes,
                    scales,
                    tensor_scales("linear.matmul", biases)?,
                    &mut self.tensor(*y),
                    seat,
                )
            }
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
            None => match self.maybe_stored(*w) {
                Some(block) => linear::kquant::matmul(
                    self.ctx(),
                    self.tensor(*act),
                    block,
                    &mut self.tensor(*y),
                ),
                None => linear::gemm::matmul(
                    self.ctx(),
                    self.tensor(*act),
                    self.tensor(*w),
                    &mut self.tensor(*y),
                ),
            },
        }
    }

    fn row_major_lm_head(
        &mut self,
        act: &ValueId,
        w: &ValueId,
        y: &ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        if let Some((codes, ..)) = self.maybe_planes(*w) {
            per_tensor_e4m3_is_a_table("linear.lm_head", codes)?;
        }
        match self.maybe_planes(*w) {
            Some((codes, scales, biases, seat)) if codes.dtype == Dtype::Nvfp4 => {
                linear::nvfp4::matmul(
                    self.ctx(),
                    "linear.lm_head",
                    self.tensor(*act),
                    codes,
                    scales,
                    tensor_scales("linear.lm_head", biases)?,
                    &mut self.tensor(*y),
                    seat,
                )
            }
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
            None => match self.maybe_stored(*w) {
                Some(block) => linear::kquant::lm_head(
                    self.ctx(),
                    self.tensor(*act),
                    block,
                    &mut self.tensor(*y),
                ),
                None => linear::gemm::lm_head(
                    self.ctx(),
                    self.tensor(*act),
                    self.tensor(*w),
                    &mut self.tensor(*y),
                ),
            },
        }
    }
}

/// A per-tensor e4m3 plane is a gathered table's, which no product reads.
fn per_tensor_e4m3_is_a_table(op: &'static str, codes: Tensor) -> Result<(), kernels_cuda::Error> {
    if codes.dtype == Dtype::E4m3tensor {
        return Err(kernels_cuda::Error::Backend {
            op,
            detail: "a per-tensor e4m3 plane is read by the n-gram gather alone; no cuda \
                     product multiplies one"
                .to_string(),
        });
    }
    Ok(())
}

/// An nvfp4 bank's tensor scales, which its planes carry where an affine
/// bank's zero points ride.
pub(crate) fn tensor_scales(
    op: &'static str,
    planes: Option<Tensor>,
) -> Result<Tensor, kernels_cuda::Error> {
    planes.ok_or_else(|| kernels_cuda::Error::Backend {
        op,
        detail: "an nvfp4 bank is bound without its tensor scales".to_string(),
    })
}

/// A low-precision product runs on the tcgen05 GEMMs or not at all: an
/// A4/A8 deployment does not fall back to bf16 activations.
fn lowp(op: &'static str) -> Result<(), kernels_cuda::Error> {
    #[cfg(feature = "cuda")]
    if linear::tc5::supported() {
        return Ok(());
    }
    Err(kernels_cuda::Error::Backend {
        op,
        detail: "a product that quantizes its activations runs on the tcgen05 GEMMs \
                 (sm_100a, sm_101a, sm_103a or sm_110a, NVRTC 12.9 or newer), which this \
                 device or NVRTC lacks; serve the deployment that reads bf16 activations \
                 instead"
            .to_string(),
    })
}

/// A low-precision bank whose scales the load could not lay into atoms.
#[cfg(feature = "cuda")]
fn unlaid(op: &'static str) -> kernels_cuda::Error {
    kernels_cuda::Error::Backend {
        op,
        detail: "a low-precision bank's scales are laid into tcgen05 atoms at load, which \
                 takes a resident bank of whole 128-row blocks whose rows hold whole fours \
                 of scales; this one is not"
            .to_string(),
    }
}

/// An NVFP4 bank's per-expert product scales, one a stream.
#[cfg(feature = "cuda")]
const EXPERT_ALPHA: &str = "linear.act_quant.expert_alpha";
/// A W4A8 select's activations read back to bf16, for the register-dequant
/// kernel.
#[cfg(feature = "cuda")]
const READ_BACK: &str = "linear.act_quant.read_back";

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
