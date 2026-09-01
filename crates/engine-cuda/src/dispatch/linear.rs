//! `Linear`: the gemm anchor, the mlp activations, the moe router, bank and
//! combine arms, and the LoRA correction over a routed adapter bank.

use kernels_cuda::linear;
use kernels_cuda::linear::quant::OffsetKind;
use model_exec::{DispatchLinear, KernelError};
use model_ir::{Dtype, Linear, ValueId};

use crate::run::Run;

/// **THE ROW COUNT AT WHICH A QUANTIZED PROJECTION STOPS FOLDING AND STARTS
/// DECODING** — the shape gate between `linear::quant::matmul` and its
/// `_via_dense` twin, and the INTERIM answer to a measured cliff.
///
/// The fused point carves one block column per activation row and re-reads
/// the whole weight inside each of them. Measured against cuBLAS bf16 on the
/// same rectangle: parity at ONE row, and 98–189× slower over 128–2048 rows.
/// So the crossover is somewhere well below sixteen, and sixteen is a
/// deliberate over-estimate of it: on the wrong side of a badly placed
/// threshold the decode arm pays one `n·k` decode for a handful of rows,
/// which is the cheap mistake.
///
/// **AND SIXTEEN IS ALSO A PIN.** qwen4 first-light prompts six tokens, and
/// six is below this, so that path takes the FUSED arm exactly as it did
/// before this gate existed and its numbers do not move. Anyone tempted to
/// tune the threshold down: below seven it changes what first-light computes
/// (the two arms agree to bf16 rounding, not to the bit), and that pin is
/// the reason this number is not four.
const PREFILL_ROWS: u32 = 16;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-cuda`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_cuda::Error> {
        match op {
            // ---- gemm (anchor) ----
            //
            // **THE ANCHOR RESOLVES ITS WEIGHT'S SEATING FIRST** (qwen4
            // stored-form wave) — `engine_metal::dispatch::linear`'s own
            // ladder, mirrored: a projection whose weight seats as an MLX
            // affine triplet (`WeightRow::Planes`) takes the quant point,
            // and a one-handle row takes the dense gemm it always took.
            // Nothing is chosen here: which rows seat as planes is the
            // trace's declaration, made when the text stated what the file
            // holds.
            //
            // The affine point serves a family now (QNF P1), so the two
            // things a byte rectangle cannot say are said here instead:
            // `Post` and `Bf16`, which is the MLX triplet this shell binds
            // today — a bf16 (scale, bias) pair per group, the offset added
            // in the value domain. The other arms wait on a `WeightRow` that
            // carries the stored form; until it does, a guess here would be
            // the silent-wrong-number the entry's refusals exist to prevent.
            //
            // **AND A THIRD SEATING, WHICH IS NEITHER OF THOSE TWO** (QNF
            // §J5, the GGUF connective wave): a weight the store seats as ONE
            // stored quantization block — a ggml super-block row, scales
            // braided into the payload, no companion plane to pair in. It is
            // a `WeightRow::Dense` like the bf16 rectangle below it and is
            // told apart from one by its DECLARATION, which is a
            // stored super-block variant ([`Run::maybe_stored`]). Nothing is chosen
            // here either: the text said what the file holds.
            //
            // **AND WHICH K-QUANT IT IS, THE ENTRY READS OFF THE ROW.** There
            // is deliberately no second table here mapping a term to a point.
            // `linear::kquant` discriminates on the row's byte width, that
            // discrimination is TOTAL over the five schemes and provably
            // unambiguous (its module doc carries the argument), and it
            // refuses a width that is none of them by naming all five. A
            // table here would be a second answer to a question that already
            // has one — and the terms that do NOT braid (the MLX affine
            // triplet, mxfp4's code/scale pair) never arrive on this arm at
            // all: their container is leaf-per-plane and they seat as
            // `WeightRow::Planes` above.
            //
            // **AND THE SHAPE PICKS THE POINT** ([`PREFILL_ROWS`]). The
            // fused arm reads the weight once per activation row, which is
            // right at a decode step's handful and two orders of magnitude
            // wrong at a prefill's hundreds; the `_via_dense` twin decodes
            // the weight once into scratch and hands cuBLAS the rectangle.
            // Same planes, same declared arm, same stored form — only the
            // kernel differs, and only above the row count where it wins.
            // A STREAMED seat takes the fused arm at every shape: the decode
            // arm refuses one (its planes have no fixed rectangle), and
            // asking it here would be asking for a refusal.
            //
            // **AND A REPACKED PLANE TAKES THE TILED ROADS** (§J4b). A
            // projection whose three rectangles `pie model import` wrote in
            // m16n8k16 fragment order is asked for FIRST, because the two
            // resolutions are disjoint and the row-major arms below cannot
            // read it — same dtype, same rows, same width, and an answer
            // that would be nonsense rather than a refusal
            // ([`Run::maybe_tiled_planes`]). Non-repacked planes fall
            // through to exactly the roads they took before this arm
            // existed.
            //
            // The shape picks the point here too, at the same
            // [`PREFILL_ROWS`]: `linear::tiled::matmul` is the tensor-core
            // tile that reads each weight word for a whole row tile, and
            // `matmul_gemv` is the decode reading of the SAME layout —
            // measured on an L40S at 1.4-3.1x the fused GEMV it replaces,
            // over both projection directions at one to sixteen rows, which
            // is what made the flip safe to take.
            //
            // [`Run::maybe_tiled_planes`]: crate::run::Run::maybe_tiled_planes
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
            } => linear::mlp::swiglu(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
            // **THE UNFUSED SWIGLU-CLAMP PAIR, REFUSED BY NAME.** The op is
            // the 2-bit MLX expert path's combine; this plane serves no MLX
            // affine bank, so the arm exists (the match is exhaustive over the
            // IR) and its body says exactly that rather than computing a shape
            // it has no unit for. See `kernels_cuda::linear::mlp`.
            Linear::MlpSwigluClampSplit { gate, up, limit, y } => {
                linear::mlp::swiglu_clamp_split(
                    self.ctx(),
                    self.tensor(*gate),
                    self.tensor(*up),
                    *limit,
                    &mut self.tensor(*y),
                )
            }
            Linear::MlpGegluTanh { gate, up, y } => linear::mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                &mut self.tensor(*y),
            ),
            // **THE UNGATED GELU** (multimodal §6.2) — the towers' MLP and
            // merger, which are `fc2(act(fc1(x)))` with nothing to multiply.
            Linear::MlpGeluTanh { x, y } => {
                linear::mlp::gelu_tanh(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Linear::MlpGegluTanhPacked {
                packed,
                intermediate,
                y,
            } => linear::mlp::geglu_tanh_packed(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                &mut self.tensor(*y),
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
                &mut self.tensor(*y),
            ),
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
                experts,
                top_k,
                renormalize,
                scaling,
                routes,
                weights,
            } => linear::moe::topk_sigmoid(
                self.ctx(),
                self.tensor(*logits),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
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
            // **THE LOOKUP ROUTER, AND THE DAY THE REFUSAL NAMED HAS COME.**
            // This arm used to answer `Unsupported`: the metal shell served
            // `linear.moe_hash_route` off `linear/moe_route.metal`'s
            // `hash_route_gather` and `kernels-cuda` shipped no twin, so the
            // arm said which op was missing rather than routing the layer
            // through a softmax gate that would answer DIFFERENT experts.
            // `kernels/linear/moe_route.cuh` is that twin, and this is the
            // one line the refusal said would change.
            //
            // No logits are read at all: `tid2eid` is the `[vocab, top_k]`
            // I64 table, `ids` the fire's own token stream, and the pair this
            // lands is the pair the four ranked routers above land — which is
            // exactly why the selects behind it need no arm of their own.
            // `experts` is the router's field for the host-side passes that
            // divide a band by it, and no argument of the kernel.
            Linear::MoeHashRoute {
                ids,
                tid2eid,
                vocab,
                experts: _,
                top_k,
                routes,
                weights,
            } => linear::moe_route::hash_route(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*tid2eid),
                *vocab,
                *top_k,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            // **THE ROUTED SELECT RESOLVES ITS BANK THROUGH THE TIER** (alto
            // design §7, wave D2). `Run::expert_bank` answers the same
            // rectangle `Run::tensor` would, plus the two device addresses a
            // STREAMED bank needs: the indirection table the kernel reads each
            // expert's base out of, and the counters it notes the routing in.
            // A resident bank answers `ExpertTable::RESIDENT` — two nulls —
            // and the launch below is byte for byte the launch this arm made
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
            // MENLO-SEAM: the IR's one `bank` id is two device planes — the
            // (codes, scales) pair the entry reads. The metal shell's
            // one-handle weight rows refused this form; here the weight
            // table seats it (`WeightRow::Planes`) and the id resolves
            // through `Run::planes`.
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
                linear::moe::matmul_select_quant(
                    self.ctx(),
                    self.tensor(*x),
                    codes,
                    scales,
                    biases,
                    self.tensor(*routes),
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
            // ---- the correction class (design §8) ----
            //
            // `y` and `y_out` are ONE arena column — the compiler folded the
            // in-place pair — so the arm resolves `y_out` and writes through
            // it, which is the same address `y` names. Resolving `y` here
            // instead would be right today and wrong the first time a pass
            // stops aliasing; the output seat is the one the walk owns.
            //
            // Both banks resolve through `Run::tensor` like any other weight,
            // because that is exactly what they are: `Def::Weight` rows whose
            // bytes came from `register_adapter` instead of the checkpoint
            // (`ParamSource::Registered`), and the runtime index rides in
            // `routes` inside the op — the MoE precedent, followed.
            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            //
            // AND THE SEGMENTS, WHICH ARE THIS ARM'S ALONE ON THIS PLANE.
            // `Run::segments` is `None` for a window P4 seated — every row of
            // the rectangle is a row of the correction — and `Some` for one it
            // answered `Fallback::Grouped` for, where the rectangle is the
            // union of the correction's intervals and the list says which of
            // its rows are its own. This is the only op `engine_cuda::GROUPED`
            // names, and it is the only arm here that may take a grouped
            // window: the compiler wrote that row BECAUSE this shell named
            // this op, so the two statements are one.
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

    /// **THE ROW-MAJOR ROADS, EXACTLY AS THEY WERE** — a projection whose
    /// planes the checkpoint landed in the order it stated them, a weight
    /// seated as one stored quantization block, and the dense bf16
    /// rectangle beside them.
    ///
    /// Lifted out of the `Matmul` arm unchanged when the tiled roads joined
    /// it (§J4b), so that the flip is a question asked BEFORE this ladder
    /// and not a fourth rung inside it: a plane that was not repacked takes
    /// the same three arms, in the same order, on the same conditions.
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
