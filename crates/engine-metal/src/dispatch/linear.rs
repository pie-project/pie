//! The `linear` family: `impl DispatchLinear for Run<'_>`, holding the gemm
//! arms plus the absorbed `mlp` and `moe` groups.

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
    /// The arms themselves, in `kernels-metal`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_metal::Error> {
        match op {
            // **THE OP NAMES A WEIGHT, AND THE ROW NAMES THE FORMAT.** A
            // projection against a quantized bank is the same statement
            // against the same `Def::Weight` id; what differs is what the
            // loader seated there, so the selection is one question to the
            // weight table and never a second IR variant. `Run::banked`
            // answers `None` for a dense row, which is why this reads as a
            // choice rather than as a rescue.
            // The quantized arm's own selection — the GEMV/GEMM crossover,
            // the row rung and the column tile — is `linear::quant`'s and
            // reads `kernels_metal::tuning`. Nothing is chosen here.
            //
            // **THE PLANE IS SEATED NOW, AND STILL NOTHING IS CHOSEN HERE.**
            // `crate::scratch` reserves the FP16 staging rectangle and the
            // split-K partials at the budget's ceiling; what these two arms
            // hand over is the two MINTS ([`Run::precast`](crate::run::Run),
            // [`Run::partials`](crate::run::Run)) and one number
            // ([`Run::capacity`](crate::run::Run)) — never a rectangle,
            // because how many rows the staging covers and how deep the
            // partials go are `quant`'s own selection, several guards inside
            // that entry. **THE CLOSURES ARE WHAT CARRIES THE QUESTION RATHER
            // THAN AN ANSWER**, and a `None` out of one is this shell saying
            // the reservation does not hold that shape — which the ladder
            // answers by taking the rung that needs no plane.
            //
            // The capacity is the MINIMUM of the two slots a padded launch
            // touches: it reads `act` and writes `y` at the padded row count,
            // so a rung either rectangle cannot hold is a rung neither takes.
            Linear::Matmul { act, w, y } => match self.banked(*w) {
                Some(bank) => linear::quant::matmul(
                    self.ctx(),
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                    linear::quant::Scratch {
                        precast: &|rows, contraction| self.precast(rows, contraction),
                        partials: &|split, rows, width| self.partials(split, rows, width),
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
            Linear::LmHead { act, w, y } => match self.banked(*w) {
                Some(bank) => linear::quant::lm_head(
                    self.ctx(),
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                    linear::quant::Scratch {
                        precast: &|rows, contraction| self.precast(rows, contraction),
                        partials: &|split, rows, width| self.partials(split, rows, width),
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

            // The absorbed `mlp` family (`linear.mlp_*`), calling into
            // `kernels_metal::linear::mlp`.
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
            Linear::MlpGegluTanh { gate, up, y } => linear::mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                self.tensor(*y),
            ),
            // **THE UNGATED GELU, REFUSED BY NAME** (multimodal §6.2): this
            // plane's `linear::mlp` ships `geglu_tanh` and its packed twin,
            // both of which multiply by an `up` half, and no ungated entry.
            Linear::MlpGeluTanh { .. } => {
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

            // The absorbed `moe` family (`linear.moe_*`), calling into
            // `kernels_metal::linear::moe`.
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
            Linear::MoeMatmulSelect { x, bank, routes, y } => linear::moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                self.tensor(*y),
            ),
            // The IR's one `bank` id is a weight row of two or three device
            // planes, and the entry picks its point off the row's own
            // `(affine?, group, bits)` — so this arm resolves and hands over,
            // and nothing here knows which of the two four-bit formats
            // arrived.
            //
            // **THE SORTED ARM IS OFFERED FIRST AND DECLINES BY ANSWERING
            // `false`.** `matmul_select_batched` is the whole selection — it
            // reads `moe::tile_rows` off the tuning, checks the column tile
            // divides, checks `K % BK`, and asks whether a point is stamped
            // for this bank at this tile — and every one of those is a
            // question about the fire, so it is asked at the fire and not
            // here. What this arm supplies is the two things the entry cannot
            // reach for itself: the working plane
            // ([`Run::routed_scratch`](crate::run::Run)) and the expert count
            // the router named ([`Run::experts`](crate::run::Run)).
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
            // The same banked bank as `MoeMatmulSelectBias`, with nothing
            // added inside the fold — the routed bias of a rows-cut expert
            // lands afterwards, through `MoeBiasSum`.
            //
            // **THE MATVEC IS THE BELOW-THRESHOLD ARM AND IT IS NO LONGER
            // THE ONLY ONE.** `linear::moe::matmul_select_batched` is the
            // sorted arm — counting sort by expert onto tile boundaries,
            // gather, one routed GEMM over each expert's contiguous run,
            // scatter back through the inverse permutation — and it is worth
            // 134 -> 311 tok/s on gpt-oss-20b at 16 lanes. The two things
            // that stood between it and this call site were a working plane
            // and a number, and both arrive from the shell now: the
            // rectangles from `crate::scratch`'s load-time reservation, and
            // the expert count off the router that wrote this op's own
            // `routes`. Below the threshold the entry answers `false` without
            // encoding anything, and the matvec below is what runs.
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
            // **THE CORRECTION CLASS, AND IT IS IN PLACE** (palo design §8,
            // decision 17). `y_out` aliases `y`, so this arm resolves one
            // column and hands it over to be read, added to, and written
            // back; a class whose guard does not hold never runs the node and
            // reads the uncorrected value at the same address, which is the
            // identity without a merge and without an arm.
            //
            // AND IT IS TOLD NOTHING ABOUT SEGMENTS, where the CUDA arm is.
            // `crate::window` serves `Fallback::Split` and names no op in
            // `DeviceProfile::grouped`, so every row of the rectangle this arm
            // is handed is a row of the correction — the day this shell names
            // one is the day the entry grows the list.
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
