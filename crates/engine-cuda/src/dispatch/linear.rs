use kernels_cuda::Tensor;
use kernels_cuda::linear;
#[cfg(feature = "cuda")]
use kernels_cuda::linear::act_quant;
use kernels_cuda::linear::affine::OffsetKind;
#[cfg(feature = "cuda")]
use kernels_cuda::linear::gemv_fp8;
use kernels_cuda::linear::moe::GroupSeat;
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
                norm,
                sum,
                gate_add,
            } => {
                #[cfg(feature = "cuda")]
                if self.moe_experts(select, glu, down, norm.as_deref(), sum, gate_add.as_deref())? {
                    return Ok(());
                }
                // Apart: the gate-up select and its GLU as their own fusion, keeping route
                // order for the down select; the GLU lands its rows e4m3-read-back where the
                // down select reads them so.
                #[cfg(feature = "cuda")]
                {
                    let pair = Fused::MoeSelectGlu {
                        select: select.clone(),
                        glu: glu.clone(),
                    };
                    let shares = matches!(
                        (routes_of(select), routes_of(down)),
                        (Some(a), Some(b)) if a == b
                    );
                    let read_back = match (&**glu, &**down) {
                        (
                            Linear::Glu {
                                y: rows,
                                packed: Some(_),
                                gate: None,
                                weights: Some(_),
                                limit: Some(_),
                                act: poem_ir::GluAct::Swiglu,
                                ..
                            },
                            Linear::MoeMatmulSelectActQuant {
                                x,
                                quant: ActQuant::E4m3 { gran: 128 },
                                ..
                            },
                        ) if x == rows => Some(*rows),
                        _ => None,
                    };
                    let apart = |me: &mut Self| -> Result<(), kernels_cuda::Error> {
                        if shares {
                            linear::moe::keeping_order(|| me.fused_linear(&pair))?;
                        } else {
                            me.fused_linear(&pair)?;
                        }
                        me.linear(down)
                    };
                    match read_back {
                        Some(rows) => self.reading_back(rows, apart)?,
                        None => apart(self)?,
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    self.fused_linear(&Fused::MoeSelectGlu {
                        select: select.clone(),
                        glu: glu.clone(),
                    })?;
                    self.linear(down)?;
                }
                if let Some(norm) = norm {
                    self.elemwise(norm)?;
                }
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
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *packed,
                    input_scales: None,
                    scales: None,
                    quant: None,
                    logits: false,
                })?;
                self.linear(&Linear::Glu {
                    packed: Some(*packed),
                    intermediate: *intermediate,
                    y: *y,
                    gate: None,
                    weights: None,
                    limit: None,
                    act: poem_ir::GluAct::Swiglu,
                    up: None,
                })
            }
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
                    input_scales: None,
                    scales: None,
                    quant: None,
                    logits: false,
                })?;
                self.linear(&Linear::Glu {
                    packed: Some(*packed),
                    intermediate: *intermediate,
                    y: *y,
                    gate: None,
                    weights: None,
                    limit: None,
                    act: poem_ir::GluAct::GegluTanh,
                    up: None,
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
                    input_scales: None,
                    scales: None,
                    quant: None,
                    logits: false,
                })?;
                self.linear(&Linear::Glu {
                    packed: Some(*packed),
                    intermediate: *intermediate,
                    y: *h,
                    gate: None,
                    weights: None,
                    limit: None,
                    act: poem_ir::GluAct::Swiglu,
                    up: None,
                })?;
                self.linear(&Linear::Matmul {
                    act: *h,
                    w: *down,
                    y: *y,
                    input_scales: None,
                    scales: None,
                    quant: None,
                    logits: false,
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
                    input_scales: None,
                    scales: None,
                    quant: None,
                    logits: false,
                })?;
                kernels_cuda::elemwise::pointwise::add_bias(
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
                    input_scales: None,
                    scales: None,
                    quant: None,
                    logits: false,
                })?;
                kernels_cuda::gather::split_rows(
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
                self.linear(&Linear::Matmul {
                    act: *act,
                    w: *w,
                    y: *y,
                    input_scales: None,
                    scales: None,
                    quant: None,
                    logits: true,
                })?;
                kernels_cuda::attn::merge::logit_softcap(self.ctx(), &mut self.tensor(*y), *cap)
            }
            _ => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
        }
    }

    fn linear(&mut self, op: &Linear) -> Result<(), kernels_cuda::Error> {
        match op {
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } if self.tensor(*act).dtype == Dtype::F32 => {
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
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } if self.tensor(*y).dtype == Dtype::F32 => linear::gemm::matmul_f32(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                &mut self.tensor(*y),
            ),
            Linear::Matmul {
                act,
                w,
                quant: Some(quant),
                scales,
                y,
                logits: false,
                ..
            } => self.matmul_act_quant(*act, *w, *quant, *scales, *y),
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } => match self.maybe_tiled_planes(*w) {
                Some((codes, scales, biases, seat)) => {
                    let act = self.tensor(*act);
                    let entry = if act.rows >= self.ctx().tuning().qmm_gemm_from {
                        linear::affine::tiled::matmul
                    } else {
                        linear::affine::tiled::matmul_gemv
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
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: true,
                ..
            } => match self.maybe_tiled_planes(*w) {
                Some((codes, scales, biases, seat)) => {
                    let act = self.tensor(*act);
                    let entry = if act.rows >= self.ctx().tuning().qmm_gemm_from {
                        linear::affine::tiled::lm_head
                    } else {
                        linear::affine::tiled::lm_head_gemv
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
            // The forms no builder makes.
            Linear::Matmul { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
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
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
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
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
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
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                weights: Some(weights),
                limit: Some(limit),
                y,
                packed: None,
                act: poem_ir::GluAct::Swiglu,
                ..
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
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                weights: Some(weights),
                limit: Some(limit),
                y,
                gate: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                // Read back through e4m3 here, for the W4A8 select that
                // would read it back.
                #[cfg(feature = "cuda")]
                if self.lands_read_back(*y)
                    && linear::glu::swiglu_clamp_routed_e4m3(
                        self.ctx(),
                        self.tensor(*packed),
                        *intermediate,
                        self.tensor(*weights),
                        fan,
                        *limit,
                        &mut self.tensor(*y),
                    )?
                {
                    return Ok(());
                }
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
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                limit: Some(limit),
                y,
                packed: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::glu::swiglu_clamp_split(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
                    fan,
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                y,
                packed: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => {
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
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
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
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Situ { beta, up_cap, .. },
                ..
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
            // The forms no builder makes.
            Linear::Glu { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Linear::MoeRoute {
                logits,
                experts,
                top_k,
                routes,
                weights,
                scale: None,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => linear::moe::topk_softmax(
                self.ctx(),
                self.tensor(*logits),
                *experts,
                *top_k,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Linear::MoeRoute {
                logits,
                scale: Some(scale),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => linear::moe::topk_softmax_scaled(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*scale),
                *experts,
                *top_k,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
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
            Linear::MoeRoute {
                logits,
                bias,
                scale,
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::SigmoidSink { sink, scaling, .. },
                ..
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
            Linear::MoeRoute {
                logits,
                bias: Some(bias),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Predict,
                ..
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
            } => linear::moe::hash_route(
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
            // The forms no builder makes.
            Linear::MoeRoute { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Linear::GroupRoutes { groups, routes } => {
                linear::moe::group_routes(self.ctx(), *groups, &mut self.tensor(*routes))
            }
            Linear::MatmulGrouped {
                x,
                w,
                routes,
                groups,
                y,
            } => {
                const OP: &str = "linear.matmul_grouped";
                // A plane requantized to fp8b128 at load (`kernel.gemm.fp8_o_down`): decode
                // rows run on it; a prefill's on its bf16 again, via a scratch for cuBLAS.
                #[cfg(feature = "cuda")]
                if let Some((codes, atoms)) = self.fp8_grouped_plane(*w) {
                    let (x, mut out) = (self.tensor(*x), self.tensor(*y));
                    if out.rows <= linear::moe::GROUPED_FP8_ROWS {
                        return linear::moe::matmul_grouped_fp8(
                            self.ctx(),
                            x,
                            codes.ptr,
                            atoms.ptr,
                            &mut out,
                            *groups,
                        );
                    }
                    // Past them (`kernel.gemm.fp8_grouped_tc5_until`): each group on the tcgen05
                    // GEMM against its rows of the fp8 plane, no bf16 widened.
                    if out.rows <= self.ctx().tuning().gemm_fp8_grouped_tc5_until
                        && x.width.is_multiple_of(*groups)
                        && out.width.is_multiple_of(*groups)
                    {
                        // A row's groups are its routes (`linear.group_routes`):
                        // the rows a group each, its slice quantized to e4m3
                        // 1x128, on the grouped tcgen05 GEMM.
                        let (k, n) = (x.width / groups, out.width / groups);
                        let xr = Tensor::new(x.ptr, x.rows * groups, k, Dtype::Bf16);
                        let mut yr = Tensor::new(out.ptr, out.rows * groups, n, Dtype::Bf16);
                        return linear::act_quant::select(
                            self.ctx(),
                            OP,
                            linear::act_quant::Format::E4m3 { gran: 128 },
                            xr,
                            self.tensor(*routes),
                            None,
                            linear::act_quant::Bank {
                                codes: codes.ptr,
                                atoms: atoms.ptr,
                                experts: *groups,
                                n,
                                k,
                                elem: linear::gemm_sm100::Elem::E4m3,
                                gran: 128,
                                alpha_e: 0,
                                alpha_from: None,
                            },
                            &mut yr,
                        );
                    }
                    let bytes = codes.rows as usize * codes.width as usize * 2;
                    let plane = self.ctx().scratch_stream(OP, FP8_PLANE_BF16, bytes)? as u64;
                    linear::moe::dequantize_fp8b128(
                        self.ctx(),
                        codes.ptr,
                        atoms.ptr,
                        plane,
                        codes.rows,
                        codes.width,
                    )?;
                    let bank = Tensor::new(plane, codes.rows, codes.width, Dtype::Bf16);
                    return linear::moe::matmul_grouped_strided(
                        self.ctx(),
                        x,
                        bank,
                        &mut out,
                        *groups,
                    );
                }
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
                // Routes that send each row's group g to plane g (`group_routes`
                // made them): one strided batched GEMM, no layout or reorder.
                #[cfg(feature = "cuda")]
                let trivial = matches!(
                    self.values()[routes.0 as usize].def,
                    poem_ir::Def::Op(at) if matches!(
                        &self.nodes()[at as usize].op,
                        poem_ir::Operation::Linear(Linear::GroupRoutes { groups: g, .. }) if *g == groups_nz
                    )
                );
                #[cfg(feature = "cuda")]
                if trivial {
                    let (bank, _) = self.expert_bank(*w);
                    if bank.dtype == Dtype::Bf16 && x.dtype == Dtype::Bf16 && y.dtype == Dtype::Bf16
                    {
                        let mut y = y;
                        return linear::moe::matmul_grouped_strided(
                            self.ctx(),
                            x,
                            bank,
                            &mut y,
                            groups_nz,
                        );
                    }
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

    /// A gate-up select and its GLU as one grouped tcgen05 GEMM, the GLU in
    /// its epilogue (a bf16 bank, or an mxfp4 one, biased or not), when the
    /// fire takes it; `false` leaves the two to run apart.
    #[cfg(feature = "cuda")]
    fn select_glu(&mut self, select: &Linear, glu: &Linear) -> Result<bool, kernels_cuda::Error> {
        use linear::gemm_sm100::Glu;
        // The device and the profile first: nothing resolved off tcgen05.
        if !self.ctx().tuning().moe_tc5 || !linear::gemm_sm100::supported() {
            return Ok(false);
        }
        let (packed, y, glu) = match *glu {
            Linear::Glu {
                packed: Some(packed),
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => (packed, y, Glu::Swiglu),
            Linear::Glu {
                packed: Some(packed),
                y,
                limit: Some(limit),
                gate: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => (packed, y, Glu::SwigluClamp { limit }),
            Linear::Glu {
                packed: Some(packed),
                y,
                limit: Some(limit),
                gate: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
            } => (packed, y, Glu::SwigluClampAlpha { limit, alpha }),
            Linear::Glu {
                packed: Some(packed),
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => (packed, y, Glu::GegluTanh),
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

    /// A routed select's operands for the grouped tcgen05 GEMM. `None` for a bank
    /// it does not read (streamed, scales in atoms, host-seated, a biased NVFP4 or
    /// affine one).
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
                if !matches!(codes.dtype, Dtype::Mxfp4 | Dtype::U8 | Dtype::E4m3)
                    || biases.is_some()
                    || seat.streams()
                    || !self.scales_in_atoms(bank)
                    || crate::device::alloc::is_host_pointer(codes.ptr)
                {
                    return None;
                }
                // A block-FP8 bank's e4m3 codes (W8A8), else MXFP4's (W4A8).
                let bank = if codes.dtype == Dtype::E4m3 {
                    GluBank::Fp8A8 { codes, atoms }
                } else {
                    GluBank::Mxfp4A8 { codes, atoms }
                };
                (x, routes, sy, bank, ExpertTable::RESIDENT)
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
        norm: Option<&poem_ir::Elemwise>,
        sum: &Linear,
        gate_add: Option<&Linear>,
    ) -> Result<bool, kernels_cuda::Error> {
        use linear::gemm_sm100::Glu;
        if !self.ctx().tuning().moe_tc5 || !linear::gemm_sm100::supported() {
            return Ok(false);
        }
        // The routes' norm: a scale, no residual, the down rows in, the sum's
        // rows out.
        let normed = match norm {
            None => None,
            Some(&poem_ir::Elemwise::NormGate {
                x,
                scale: Some(scale),
                res: None,
                gate: None,
                eps,
                temperature: 1.0,
                next: None,
                y,
            }) => Some((x, scale, eps, y)),
            Some(_) => return Ok(false),
        };
        // DeepSeek's form: the route weights scale the GLU's rows before the
        // down leg, and the routes sum plainly with the shared row added.
        if let (
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                weights: Some(weights),
                limit: Some(limit),
                y: hidden,
                gate: None,
                act: poem_ir::GluAct::Swiglu,
                ..
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
            if gate_add.is_some() || normed.is_some() {
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
            let glu = Glu::SwigluClamp { limit };
            #[rustfmt::skip]
            let m = linear::moe::RoutedMlp { x, up, glu, down, routes, weights, inter: intermediate };
            return linear::moe::experts_weighted_first(self.ctx(), m, shared, &mut out);
        }
        let (packed, hidden, intermediate, glu) = match *glu {
            Linear::Glu {
                packed: Some(packed),
                y,
                intermediate,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => (packed, y, intermediate, Glu::Swiglu),
            Linear::Glu {
                packed: Some(packed),
                y,
                intermediate,
                limit: Some(limit),
                gate: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => (packed, y, intermediate, Glu::SwigluClamp { limit }),
            Linear::Glu {
                packed: Some(packed),
                y,
                intermediate,
                limit: Some(limit),
                gate: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
            } => (
                packed,
                y,
                intermediate,
                Glu::SwigluClampAlpha { limit, alpha },
            ),
            Linear::Glu {
                packed: Some(packed),
                y,
                intermediate,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => (packed, y, intermediate, Glu::GegluTanh),
            Linear::Glu {
                packed: Some(packed),
                y,
                intermediate,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Situ { beta, up_cap, .. },
                ..
            } => (
                packed,
                y,
                intermediate,
                Glu::Situ {
                    beta,
                    cap: up_cap.unwrap_or(0.0),
                },
            ),
            Linear::Relu2 { x, y } => (x, y, self.tensor(y).width, Glu::Relu2),
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
        // The sum reads the down rows, or their norm's.
        let sum_reads = normed.map_or(down_y, |(_, _, _, y)| y);
        if up_y != packed
            || down_x != hidden
            || down_routes != routes
            || sum_reads != summed
            || normed.is_some_and(|(x, ..)| x != down_y)
        {
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
        let norm = normed.map(|(_, scale, eps, _)| (self.tensor(scale), eps));
        #[rustfmt::skip]
        let m = linear::moe::RoutedMlp { x, up, glu, down, routes, weights, inter: intermediate };
        linear::moe::experts(
            self.ctx(),
            m,
            &mut out,
            up_experts,
            down_experts,
            gate,
            norm,
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
            return linear::moe::select_nvfp4(
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
        lowp("linear.matmul")
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
        if !linear::gemm_sm100::supported() {
            return self.select_a16(OP, x, bank, routes, y);
        }
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
        let x_id = x;
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
                    linear::gemm_sm100::Elem::E2m1,
                    16,
                    alpha,
                    Some(from),
                )
            }
            // An MXFP4 bank's codes are bound as bytes; only an MXFP4 bank pairs
            // with e4m3 activations.
            (Dtype::Mxfp4 | Dtype::U8, ActQuant::E4m3 { gran }) => (
                act_quant::Format::E4m3 { gran },
                linear::gemm_sm100::Elem::E2m1,
                32,
                0,
                None,
            ),
            // A 128x128-block FP8 bank (a block-FP8 release's experts, its
            // factors requantized to ue8m0 at load and spread a row an atom
            // row) against e4m3 activations in 1x128 blocks (W8A8); an MXFP8
            // bank (its e8m0 exponents a 1x32 block of each row, laid as they
            // are) against e4m3 in 1x32 blocks. The laid atoms count the K a
            // scale covers.
            (Dtype::E4m3, ActQuant::E4m3 { gran }) if fp8_bank_gran(scales, k) == Some(gran) => (
                act_quant::Format::E4m3 { gran },
                linear::gemm_sm100::Elem::E4m3,
                gran,
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
        // At decode widths an MXFP4 bank's e4m3 select reads its activations read back
        // to bf16 (exact) on the register-dequant kernel, which streams the bank far
        // faster than the grouped GEMM at a handful of rows an expert.
        if let act_quant::Format::E4m3 { .. } = fmt
            && elem == linear::gemm_sm100::Elem::E2m1
            && kernels_cuda::linear::moe::select_mxfp4_atoms_takes(self.ctx(), route_t, x)
        {
            // A routed GLU's rows land read back already.
            let back = if self.lands_read_back(x_id)
                && linear::glu::routed_e4m3_spans(&x)
                && fmt == (act_quant::Format::E4m3 { gran: 128 })
            {
                x
            } else {
                let back = self.ctx().scratch_stream(
                    OP,
                    READ_BACK,
                    x.rows as usize * x.width as usize * 2,
                )? as u64;
                act_quant::read_back(self.ctx(), OP, fmt, act_quant::Source::rows(x, None), back)?
            };
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
                gran,
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
        const OP: &str = "linear.matmul";
        if !linear::gemm_sm100::supported() {
            return self.matmul_a16(OP, act, w, block_scales, y);
        }
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
        let act_id = act;
        let act = self.tensor(act);
        let mut out = self.tensor(y);
        let (n, k) = (out.width, act.width);
        // Its producer landed the 1x128 codes itself: no quantize launch.
        if let (Dtype::Mxfp4 | Dtype::U8, ActQuant::E4m3 { gran: 128 }) = (codes.dtype, quant)
            && let Some(planes) = self.held_e4m3(act_id, &act)
        {
            let weight = act_quant::Weight {
                codes: codes.ptr,
                scales: scales.ptr,
                n,
                k,
                elem: linear::gemm_sm100::Elem::E2m1,
                gran: 32,
                laid: true,
            };
            return act_quant::matmul_held(self.ctx(), OP, planes, act.rows, weight, &mut out, 1.0);
        }
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
                    elem: linear::gemm_sm100::Elem::E2m1,
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

    /// An activation-quantizing select where there are no tcgen05 GEMMs: the
    /// bank against the rows as they are, bf16 (W4A16 for an MXFP4 or NVFP4
    /// bank, W8A16 for a block-FP8 one).
    #[cfg(feature = "cuda")]
    fn select_a16(
        &mut self,
        op: &'static str,
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let (codes, scales, _, seat) = self.planes(bank);
        match codes.dtype {
            Dtype::Nvfp4 | Dtype::Mxfp4 | Dtype::U8 => {
                let x = self.tensor(x);
                self.select_quant(x, bank, routes, y)
            }
            // A block-FP8 release's experts, their factors laid as ue8m0
            // exponents a weight row a 128 at load (`lay_tile_banks`), or an
            // MXFP8 bank's exponents a weight row a 32.
            Dtype::E4m3 => {
                if !self.scales_in_atoms(bank) {
                    return Err(unlaid(op));
                }
                let Some(gran) = fp8_bank_gran(scales, self.tensor(x).width) else {
                    return Err(unlaid(op));
                };
                if seat.streams() || crate::device::alloc::is_host_pointer(codes.ptr) {
                    return Err(kernels_cuda::Error::Backend {
                        op,
                        detail: "a W8A16 select reads a resident bank, and this one is seated \
                                 by a streaming tier"
                            .to_string(),
                    });
                }
                let x = self.tensor(x);
                let routes = self.tensor(routes);
                // A batch's routes quantized as the deployment says, on the
                // FP8 MMA where there is one (W8A8).
                if linear::moe::fp8_w8a8_capable()
                    && routes.rows * routes.width >= linear::moe::FP8_W8A8_FROM
                    && linear::moe::fp8_w8a8_takes(x, self.tensor(y).width)
                {
                    return linear::moe::select_fp8_w8a8(
                        self.ctx(),
                        x,
                        codes.ptr,
                        scales.ptr,
                        codes.rows,
                        routes,
                        &mut self.tensor(y),
                    );
                }
                linear::moe::select_fp8_w8a16(
                    self.ctx(),
                    op,
                    x,
                    codes.ptr,
                    scales.ptr,
                    codes.rows,
                    gran,
                    routes,
                    &mut self.tensor(y),
                    None,
                )
            }
            stored => Err(kernels_cuda::Error::Backend {
                op,
                detail: format!(
                    "a {stored:?} bank's select against bf16 activations has no point where \
                     there are no tcgen05 GEMMs"
                ),
            }),
        }
    }

    /// An activation-quantizing projection where there are no tcgen05 GEMMs:
    /// a block-FP8 plane against the rows as they are, bf16 (W8A16), its
    /// codes widened in registers (`linear::w8a16`); a 4-bit one as its
    /// bf16-activation product reads it.
    #[cfg(feature = "cuda")]
    fn matmul_a16(
        &mut self,
        op: &'static str,
        act: ValueId,
        w: ValueId,
        block_scales: Option<ValueId>,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let Some(block_scales) = block_scales else {
            return self.row_major_matmul(&act, &w, &y);
        };
        let codes = self.tensor(w);
        let atoms = self.tensor(block_scales);
        if atoms.dtype != Dtype::U8 {
            return Err(unlaid(op));
        }
        let act = self.tensor(act);
        let mut out = self.tensor(y);
        // A batch's rows on the plane's bf16 copy (`kernel.gemm.fp8_shadow`).
        if act.rows > linear::w8a16::MMA_UNTIL
            && let Some(shadow) = self.fp8_shadow(w)
        {
            linear::gemm::act_x_wt(self.ctx(), op, act, shadow, None, &mut out)?;
            return Ok(());
        }
        let k = act.width;
        let block = if atoms.width == 0 {
            0
        } else {
            k.div_ceil(atoms.width)
        };
        // A decode's few rows over 128- or 32-wide blocks as the GEMV over bf16
        // rows (`kernel.gemm.fp8_gemv_until`), as the tcgen05 path takes them.
        if matches!(block, 32 | 128)
            && gemv_fp8::fp8_block_gemv(self.ctx(), act, codes.ptr, atoms.ptr, block, &mut out)?
        {
            return Ok(());
        }
        // A batch's rows (past the decode widths the W8A16 kernels take)
        // quantized as the deployment says, on cuBLASLt's block-scaled FP8
        // GEMM where there is an FP8 MMA (W8A8), else the mma.sync one.
        if block == 128
            && act.rows > linear::w8a16::MMA_UNTIL
            && linear::moe::fp8_w8a8_capable()
            && linear::gemm::cublas_lt_fp8::matmul(
                self.ctx(),
                op,
                act,
                codes.ptr,
                atoms.ptr,
                &mut out,
            )?
        {
            return Ok(());
        }
        if block == 128
            && linear::moe::FP8_W8A8_DENSE.contains(&act.rows)
            && linear::moe::fp8_w8a8_capable()
            && linear::moe::fp8_w8a8_takes(act, out.width)
        {
            return linear::moe::matmul_fp8_w8a8(self.ctx(), act, codes.ptr, atoms.ptr, &mut out);
        }
        if linear::w8a16::matmul(self.ctx(), op, act, codes.ptr, atoms.ptr, block, &mut out)? {
            return Ok(());
        }
        Err(kernels_cuda::Error::Backend {
            op,
            detail: format!(
                "a block-fp8 projection of {block}-wide blocks over K {k} has no W8A16 point"
            ),
        })
    }

    /// A block-FP8 projection against e4m3 activations on one tcgen05 GEMM (W8A8):
    /// 128x128 blocks against 1x128 (V4's fp8b128) or 32x32 against 1x32 (V4.1's MXFP8).
    #[cfg(feature = "cuda")]
    fn fp8_block_matmul(
        &mut self,
        act: ValueId,
        w: ValueId,
        quant: ActQuant,
        block_scales: ValueId,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "linear.matmul";
        let codes = self.tensor(w);
        let atoms = self.tensor(block_scales);
        if atoms.dtype != Dtype::U8 {
            return Err(unlaid(OP));
        }
        let act_id = act;
        let act = self.tensor(act);
        let mut out = self.tensor(y);
        let (n, k) = (out.width, act.width);
        // The atoms hold one exponent a block of K a row.
        let block = if atoms.width == 0 {
            0
        } else {
            k.div_ceil(atoms.width)
        };
        if !matches!(block, 32 | 128) || quant != (ActQuant::E4m3 { gran: block }) {
            return Err(kernels_cuda::Error::Backend {
                op: OP,
                detail: format!(
                    "a block-fp8 projection of {block}-wide blocks reads e4m3 activations in \
                     1x{block} blocks, not {quant:?}"
                ),
            });
        }
        let weight = act_quant::Weight {
            codes: codes.ptr,
            scales: atoms.ptr,
            n,
            k,
            elem: linear::gemm_sm100::Elem::E4m3,
            gran: block,
            laid: true,
        };
        if block == 128 {
            // A decode's few rows on the mma.sync fp8 kernel, its rows quantized
            // in the same launch, where it leads (`kernel.gemm.fp8_mma_gemv_until`,
            // up to `kernel.gemm.fp8_mma_k_until` of K, `moe::dense_fp8_shape`).
            let tuning = self.ctx().tuning();
            if act.rows <= tuning.gemm_fp8_mma_gemv_until
                && (tuning.gemm_fp8_mma_k_until == 0 || k <= tuning.gemm_fp8_mma_k_until)
                && let Some(shape) = linear::moe::dense_fp8_shape(act.rows, n, k)
            {
                return linear::moe::matmul_grouped_fp8_with(
                    self.ctx(),
                    act,
                    codes.ptr,
                    atoms.ptr,
                    &mut out,
                    1,
                    shape,
                );
            }
            // Else a GEMV over the activations as they are, where the profile says
            // (`kernel.gemm.fp8_gemv_until`): ahead of the tcgen05 GEMM even on held codes.
            if gemv_fp8::fp8_block_gemv(self.ctx(), act, codes.ptr, atoms.ptr, 128, &mut out)? {
                return Ok(());
            }
            // Its producer landed the codes itself: no quantize launch.
            if let Some(planes) = self.held_e4m3(act_id, &act) {
                return act_quant::matmul_held(
                    self.ctx(),
                    OP,
                    planes,
                    act.rows,
                    weight,
                    &mut out,
                    1.0,
                );
            }
        }
        act_quant::matmul(
            self.ctx(),
            OP,
            act_quant::Format::E4m3 { gran: block },
            act_quant::Source::rows(act, None),
            weight,
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
                linear::dequant::nvfp4_matmul(
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
                        linear::dequant::matmul_via_dense
                    } else {
                        linear::affine::matmul
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
            per_tensor_e4m3_is_a_table("linear.matmul", codes)?;
        }
        match self.maybe_planes(*w) {
            Some((codes, scales, biases, seat)) if codes.dtype == Dtype::Nvfp4 => {
                linear::dequant::nvfp4_matmul(
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
                        linear::dequant::lm_head_via_dense
                    } else {
                        linear::affine::lm_head
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

/// The K one scale of a block-FP8 bank covers, read off its laid atoms (one
/// a block of K a row): 128 or 32.
#[cfg(feature = "cuda")]
fn fp8_bank_gran(atoms: Tensor, k: u32) -> Option<u32> {
    (atoms.width > 0 && k.is_multiple_of(atoms.width))
        .then(|| k / atoms.width)
        .filter(|gran| matches!(gran, 32 | 128))
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

/// A low-precision product runs on the tcgen05 GEMMs; on CUDA without them
/// it reads bf16 activations (`select_a16`, `matmul_a16`), and a build with
/// no CUDA has neither.
#[cfg(not(feature = "cuda"))]
fn lowp(op: &'static str) -> Result<(), kernels_cuda::Error> {
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

/// The routes a routed select reads.
#[cfg(feature = "cuda")]
fn routes_of(op: &Linear) -> Option<ValueId> {
    match op {
        Linear::MoeMatmulSelect { routes, .. }
        | Linear::MoeMatmulSelectBias { routes, .. }
        | Linear::MoeMatmulSelectQuant { routes, .. }
        | Linear::MoeMatmulSelectActQuant { routes, .. } => Some(*routes),
        _ => None,
    }
}

/// The bf16 of a plane requantized to fp8b128 at load, for a fire too wide
/// for the fp8 kernel; one a stream.
#[cfg(feature = "cuda")]
const FP8_PLANE_BF16: &str = "linear.matmul_grouped.fp8_plane_bf16";
