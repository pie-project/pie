//! `Linear`: the gemm anchor, the mlp activations, the moe router, bank and
//! combine arms, and the LoRA correction over a routed adapter bank.

use kernels_cuda::linear;
use kernels_cuda::Tensor;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::OffsetKind;
use model_exec::{DispatchLinear, KernelError};
use model_ir::{Dtype, Linear, ValueId};

use crate::run::Run;

// Row count where a quantized projection switches from the fused (folding)
// kernel to its `_via_dense` twin. The fused kernel re-reads the whole
// weight per activation row: cheap at decode sizes, far slower at prefill
// sizes. Kept above 6 so a 6-token first-light prompt stays on the fused arm
// (the two arms don't agree bit-for-bit, only to bf16 rounding).
const PREFILL_ROWS: u32 = 16;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    // Arms are in `kernels-cuda`'s error vocabulary, not the contract's, so
    // each is a plain tail call with `?`; `kernel()` lifts the error family.
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_cuda::Error> {
        match op {
            // ---- gemm (anchor) ----
            //
            // Weight seating picks the arm: a repacked plane (m16n8k16
            // fragment order, written by `pie model import`) takes the tiled
            // road first, since the row-major arms below can't read it. An
            // MLX affine triplet (`WeightRow::Planes`) takes the quant
            // point; a stored super-block row (ggml, no companion plane)
            // takes the kquant path, which discriminates the k-quant scheme
            // by the row's byte width; anything else takes the dense gemm.
            // Nothing is chosen here beyond that: the trace declares which
            // rows seat which way.
            //
            // `PREFILL_ROWS` picks fused vs. `_via_dense`/gemv within a
            // seating: the fused kernel reads the weight once per row (cheap
            // at decode sizes, far slower at prefill), the alternate decodes
            // the weight once into scratch. A STREAMED seat always takes the
            // fused arm since its planes have no fixed rectangle.
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
            // ---- mlp ----
            Linear::MlpSwiglu {
                packed,
                intermediate,
                y,
            } => {
                // Fan-out the staged seat is scaled by (`Run::plane_fan`):
                // 1 for a dense MLP, or the routed fan-out for a leg whose
                // packed input is a select's output.
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
                // Fan-out the staged seat is scaled by (`Run::plane_fan`):
                // 1 for a dense MLP, or the routed fan-out for a leg whose
                // packed input is a select's output.
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
                // Fan-out the staged seat is scaled by (`Run::plane_fan`):
                // 1 for a dense MLP, or the routed fan-out for a leg whose
                // packed input is a select's output.
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
            // Unfused swiglu-clamp pair: the 2-bit MLX expert path's combine.
            // This plane serves no MLX affine bank; the arm exists because
            // the match is exhaustive, and the kernel itself refuses rather
            // than compute a shape it has no unit for.
            Linear::MlpSwigluClampSplit { gate, up, limit, y } => {
                linear::mlp::swiglu_clamp_split(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
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
            // Ungated GELU: the towers' MLP and merger, `fc2(act(fc1(x)))`
            // with nothing to multiply.
            Linear::MlpGeluTanh { x, y } => {
                let fan = self.plane_fan(self.tensor(*y).rows);
                linear::mlp::gelu_tanh(self.ctx(), self.tensor(*x), fan, &mut self.tensor(*y))
            }
            Linear::MlpGegluTanhPacked {
                packed,
                intermediate,
                y,
            } => {
                // Fan-out the staged seat is scaled by (`Run::plane_fan`):
                // 1 for a dense MLP, or the routed fan-out for a leg whose
                // packed input is a select's output.
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
                // Fan-out the staged seat is scaled by (`Run::plane_fan`):
                // 1 for a dense MLP, or the routed fan-out for a leg whose
                // packed input is a select's output.
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
            // ---- moe ----
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
            // The prediction ranks like the router and cuts nothing on this
            // arm either (its op is not one `exports::writer_classes` names).
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
            // Lookup router: no logits are read. `tid2eid` is the
            // `[vocab, top_k]` I64 table, `ids` the fire's own token stream;
            // the pair landed is the same pair the ranked routers above
            // land, so the selects behind it need no arm of their own.
            // `experts` is only used by host-side passes dividing a band by
            // it, not a kernel argument.
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
            // **THE BLOCK-DIAGONAL PROJECTION, AS THE ROUTED SELECT OVER A
            // RESTATED RECTANGLE.** `[tokens, G·K]` is `[tokens·G, K]` byte for
            // byte, `[G·N, K]` is a `G`-expert bank, `[tokens, G·N]` is
            // `[tokens·G, N]`; with `group_routes`' `g` in slot `g`, the
            // by-route dense select computes exactly the grouped product.
            //
            // **THE DENSE PLANE ONLY, ON THIS ARM.** A quantized o-projection
            // plane lands in this shell's TILED layout, which the split-plane
            // routed select does not read; it is refused by name here rather
            // than read wrong. (The Metal arm reads both; this arm was
            // mirrored without a device to run it on.)
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
                let x = kernels_cuda::tensor::Tensor::new(x.ptr, rows, x.width / groups_nz, x.dtype);
                let mut y = kernels_cuda::tensor::Tensor::new(y.ptr, rows, y.width / groups_nz, y.dtype);
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
            // `Run::expert_bank` answers the same rectangle `Run::tensor`
            // would, plus the two device addresses a STREAMED bank needs
            // (indirection table, routing counters). A resident bank
            // answers `ExpertTable::RESIDENT` (two nulls), unchanged from
            // before the tier existed.
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
            // The IR's one `bank` id is two device planes: the (codes,
            // scales) pair the entry reads, resolved through `Run::planes`
            // (`WeightRow::Planes`).
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
            // The same two-plane bank as the biased twin above, with nothing
            // added inside the fold: the down leg's routed bias lands after
            // the reduce, through `MoeBiasSum`.
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
            // ---- the correction class ----
            //
            // `y` and `y_out` are one arena column (compiler folded the
            // in-place pair), so the arm resolves `y_out` and writes
            // through it — the same address `y` names.
            //
            // Both banks resolve through `Run::tensor`: they're
            // `Def::Weight` rows whose bytes came from `register_adapter`
            // (`ParamSource::Registered`), with the runtime index riding in
            // `routes`.
            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            // `Run::segments` is `None` for a window the compiler seated
            // whole (every row is a row of the correction) and `Some` for a
            // `Fallback::Grouped`
            // window; this is the only arm that may take a grouped window.
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

    /// The row-major roads: a projection whose planes the checkpoint landed
    /// in declared order, a weight seated as one stored quantization block,
    /// and the dense bf16 rectangle beside them.
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

    /// [`Run::row_major_matmul`] under the head's own entries, and the same
    /// lift.
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
            // The head's own stored-block arm, and not a courtesy
            // pairing: a Q4_K_M mix stores `output.weight` at q6_k, so
            // the head is the busiest consumer the entry has.
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

/// The most expert bytes one select stages into device scratch; a wider
/// routing (a long prefill) stays on the bound planes.
const STAGED_EXPERT_BYTES: u64 = 1536 * 1024 * 1024;

impl Run<'_> {
    /// A routed bank held on the host (T1) read at device speed: the experts
    /// this fire routes to are copied into scratch and the routes renumbered
    /// onto those slots. `None` keeps the bound planes — a device-resident
    /// bank, a routing too wide to stage, or a copy the runtime refused.
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
        // Ordered behind the routing on the same stream; a pageable
        // destination makes the copy synchronous, so `picked` is whole after it.
        if crate::device::copy_any(stream, picked.as_mut_ptr() as u64, routes.ptr, count * 4).is_err() {
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
        let remapped: Vec<i32> = picked.iter().map(|e| if *e < 0 { -1 } else { slot_of(*e) }).collect();
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
            Some(Tensor::new(ptr, unique.len() as u32, plane.width, plane.dtype))
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

/// One device buffer per staging plane for the whole process, grown when a
/// wider routing needs it — not fire scratch, whose per-region slabs would
/// multiply a gigabyte of experts by the walk's regions.
fn staging(name: &'static str, bytes: usize) -> Option<u64> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static HELD: Mutex<Option<HashMap<&'static str, (u64, usize)>>> = Mutex::new(None);
    if bytes == 0 {
        return None;
    }
    let mut held = HELD.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let map = held.get_or_insert_with(HashMap::new);
    if let Some((ptr, cap)) = map.get(name)
        && *cap >= bytes
    {
        return Some(*ptr);
    }
    let fresh = crate::device::alloc::raw_alloc(bytes.max(map.get(name).map_or(0, |(_, cap)| cap * 2)))?;
    if let Some((old, _)) = map.insert(name, (fresh, bytes.max(map.get(name).map_or(0, |(_, cap)| cap * 2)))) {
        crate::device::alloc::raw_free(old);
    }
    Some(fresh)
}
