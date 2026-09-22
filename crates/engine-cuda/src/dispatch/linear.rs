use kernels_cuda::Tensor;
use kernels_cuda::linear;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::OffsetKind;
use model_exec::{DispatchLinear, KernelError};
use model_ir::{Dtype, Linear, ValueId};

use crate::run::Run;

const PREFILL_ROWS: u32 = 16;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
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
            Linear::Matmul { act, w, y } => match self.maybe_tiled_planes(*w) {
                Some((codes, scales, biases, seat)) => {
                    let act = self.tensor(*act);
                    let entry = if act.rows >= PREFILL_ROWS {
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
                    let entry = if act.rows >= PREFILL_ROWS {
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
            Linear::MatmulGeglu {
                act,
                w,
                intermediate,
                packed,
                y,
            } => {
                let a = self.tensor(*act);
                let weight = self.tensor(*w);
                let out = self.tensor(*y);
                let (m, k) = (a.rows as i32, a.width as i32);
                let i = i32::try_from(*intermediate).unwrap_or(0);
                if self.dense_weight(*w)
                    && weight.rows == 2 * *intermediate
                    && linear::skinny::covers(m, i, k, linear::skinny::Epilogue::Geglu)
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
            Linear::LmHeadSoftcap {
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
                    if linear::skinny::covers(m, n, k, epilogue) {
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
            Linear::MlpSwigluClampSplit { gate, up, limit, y } => linear::mlp::swiglu_clamp_split(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                *limit,
                &mut self.tensor(*y),
            ),
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
                if groups_nz == 0 || x.width % groups_nz != 0 || y.width % groups_nz != 0 {
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
                let (codes, scales, biases, seat) = self.planes(*bank);
                let routes = self.tensor(*routes);
                let (codes, scales, biases, routes, seat) =
                    match self.staged_experts(codes, scales, biases, routes) {
                        Some((codes, scales, biases, routes)) => {
                            (codes, scales, biases, routes, GroupSeat::RESIDENT)
                        }
                        None => (codes, scales, biases, routes, seat),
                    };
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    codes,
                    scales,
                    biases,
                    routes,
                    &mut self.tensor(*y),
                    seat,
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
        match self.maybe_planes(*w) {
            Some((codes, scales, biases, seat)) => {
                let act = self.tensor(*act);
                let entry = if act.rows >= PREFILL_ROWS && !seat.streams() {
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
        match self.maybe_planes(*w) {
            Some((codes, scales, biases, seat)) => {
                let act = self.tensor(*act);
                let entry = if act.rows >= PREFILL_ROWS && !seat.streams() {
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

fn staging(name: &'static str, bytes: usize) -> Option<u64> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static HELD: Mutex<Option<HashMap<&'static str, (u64, usize)>>> = Mutex::new(None);
    if bytes == 0 {
        return None;
    }
    let mut held = HELD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let map = held.get_or_insert_with(HashMap::new);
    if let Some((ptr, cap)) = map.get(name)
        && *cap >= bytes
    {
        return Some(*ptr);
    }
    let fresh =
        crate::device::alloc::raw_alloc(bytes.max(map.get(name).map_or(0, |(_, cap)| cap * 2)))?;
    if let Some((old, _)) = map.insert(
        name,
        (
            fresh,
            bytes.max(map.get(name).map_or(0, |(_, cap)| cap * 2)),
        ),
    ) {
        crate::device::alloc::raw_free(old);
    }
    Some(fresh)
}
