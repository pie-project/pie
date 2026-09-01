//! The `elementwise` family: `impl DispatchElementwise for Run<'_>`, holding
//! the norm arms plus the absorbed `rope`, `gate`, and `hc` groups.

use kernels_metal::{Tensor, elemwise};
use model_exec::{DispatchElementwise, KernelError};
use model_ir::{Elementwise, MropeForm, Operands};

use crate::run::Run;

impl DispatchElementwise for Run<'_> {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        self.elementwise(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-metal`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn elementwise(&mut self, op: &Elementwise) -> Result<(), kernels_metal::Error> {
        match op {
            Elementwise::Rmsnorm { x, weight, eps, y } => elemwise::norm::rmsnorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormPerHead {
                x,
                weight,
                head_dim,
                eps,
                y,
            } => elemwise::norm::rmsnorm_per_head(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormPlusOne { x, weight, eps, y } => elemwise::norm::rmsnorm_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormPerHeadPlusOne {
                x,
                weight,
                head_dim,
                eps,
                y,
            } => elemwise::norm::rmsnorm_per_head_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::RmsnormNoScale {
                x,
                head_dim,
                eps,
                y,
            } => elemwise::norm::rmsnorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *head_dim,
                *eps,
                self.tensor(*y),
            ),
            // **THE WHOLE `nn.LayerNorm`, IN ONE LAUNCH** (multimodal §9.1,
            // next.md B5): every qwen vision block's `norm1`/`norm2` and its
            // merger's, twenty-five of them per dev-tower fire. It is the
            // fused row and not the three-op spelling because the centred row
            // is never rounded to the activation's element on the way
            // through, which the composition cannot claim.
            Elementwise::Layernorm {
                x,
                weight,
                bias,
                eps,
                y,
            } => elemwise::norm::layernorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                self.tensor(*y),
            ),
            // **THE SCALE-LESS CENTRED NORM AND THE CLAMP, REFUSED BY NAME**
            // (multimodal §6.1, §6.5). This plane's `elemwise::norm` ships no
            // scale-less centred entry — what a text writes when the scale
            // really does bake into the GEMM behind it, which no shipping row
            // does — and its `elemwise` no free-standing clamp. That is the
            // family's rule the other way round from `attn::dense`'s: a
            // written shader gets a forwarding arm, an unwritten one gets its
            // own refusal rather than a shader that norms the wrong thing.
            //
            // `clamp_learned` stays owed by `gemma4-e4b-vision-bf16` alone —
            // the wide gemma tower publishes `use_clipped_linears: false` and
            // spells no clamp — and its checkpoint is not on this box.
            Elementwise::LayernormNoScale { .. }
            | Elementwise::Clamp { .. }
            | Elementwise::ClampLearned { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            // **THE GATED-RESIDUAL FAMILY (qwen4), WHICH THIS BLOCK USED TO
            // REFUSE WHOLE.** Five arms, ported off `elemwise/hc.cuh` and
            // `elemwise/norm.cuh`: the per-group norm whose gain bank spans
            // the wide row, the shared gate's scaled silu, the stream collapse
            // and its injection back, and the PLE's key·query gate. They were
            // the reason a `qwen38-flash-*` fire could not reach its second
            // layer even after the n-gram hasher landed — the census called
            // every one of those rows "clears the bake" while these five
            // refused at the first fire, which is a bake's blind spot and not
            // a lie it can be talked out of.
            Elementwise::RmsnormGroupedPlusOne {
                x,
                weight,
                group,
                eps,
                y,
            } => elemwise::norm::rmsnorm_grouped_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *group,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::SiluScaled { s, x, x_out: _ } => {
                elemwise::norm::silu_scaled(self.ctx(), *s, self.tensor(*x))
            }
            Elementwise::HcMix {
                gates,
                normed,
                streams,
                y,
            } => elemwise::hc::mix(
                self.ctx(),
                self.tensor(*gates),
                self.tensor(*normed),
                *streams,
                self.tensor(*y),
            ),
            Elementwise::HcInject {
                o,
                gates,
                streams,
                hyper,
                hyper_out: _,
            } => elemwise::hc::inject(
                self.ctx(),
                self.tensor(*o),
                self.tensor(*gates),
                *streams,
                self.tensor(*hyper),
            ),
            Elementwise::PleGate {
                key,
                query,
                value,
                streams,
                y,
            } => elemwise::hc::ple_gate(
                self.ctx(),
                self.tensor(*key),
                self.tensor(*query),
                self.tensor(*value),
                *streams,
                self.tensor(*y),
            ),
            // **BOTH GATE CURVES, WHICH USED TO BE ONE.** The sigmoid arm
            // answered `Unsupported` by name while the silu arm served, and
            // qwen4's GatedDeltaNet is the sigmoid one — its checkpoint's
            // `output_gate_type` says so. The shader was always templated on
            // the choice; only the instantiation and the entry's argument
            // were missing.
            Elementwise::RmsnormGated {
                x,
                gate,
                weight,
                head_dim,
                eps,
                act,
                y,
            } => elemwise::norm::rmsnorm_gated(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *head_dim,
                *eps,
                matches!(act, model_ir::GateActivation::Sigmoid),
                self.tensor(*y),
            ),
            Elementwise::RmsnormGatedBy {
                x,
                gate,
                weight,
                heads,
                eps,
                y,
            } => elemwise::norm::rmsnorm_gated_by(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *heads,
                *eps,
                self.tensor(*y),
            ),
            Elementwise::ResidualAdd { x, y, y_out: _ } => {
                elemwise::norm::residual_add(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            Elementwise::AddBias {
                bias,
                out,
                out_out: _,
            } => elemwise::norm::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),
            // **THE TOWER'S OUTPUT STANDARDIZATION** (multimodal §21.3): the
            // per-column affine `(x − bias) · scale`, in place. It sits beside
            // `add_bias` because that is its launch — same grid, same
            // threadgroup, one more plane — and it is a row of its own because
            // neither neighbour can spell it: `scale` reads ONE device-held
            // scalar, and the composed pair would round the difference to the
            // activation's element between two launches at the one tower site
            // where a bias cancels what it is subtracted from.
            Elementwise::Standardize {
                x,
                bias,
                scale,
                x_out: _,
            } => elemwise::norm::standardize(
                self.ctx(),
                self.tensor(*bias),
                self.tensor(*scale),
                self.tensor(*x),
            ),
            Elementwise::MulScalar { s, x, x_out: _ } => {
                elemwise::norm::mul_scalar(self.ctx(), *s, self.tensor(*x))
            }
            Elementwise::Scale { s, x, x_out: _ } => {
                elemwise::norm::scale(self.ctx(), self.tensor(*s), self.tensor(*x))
            }
            Elementwise::ResBlend {
                prefix,
                blocks,
                weight,
                eps,
                proj,
                y,
            } => {
                let blocks: Vec<Tensor> = blocks.iter().map(|b| self.tensor(*b)).collect();
                elemwise::norm::res_blend(
                    self.ctx(),
                    self.tensor(*prefix),
                    &blocks,
                    self.tensor(*weight),
                    *eps,
                    self.tensor(*proj),
                    self.tensor(*y),
                )
            }

            // The absorbed `rope` family (`elementwise.rope_*`), calling into
            // `kernels_metal::elemwise::rope`.
            Elementwise::RopeFull {
                q,
                k,
                positions,
                head_dim,
                theta,
                interleaved,
                q_out: _,
                k_out: _,
            } => elemwise::rope::full(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *head_dim,
                *theta,
                *interleaved,
            ),
            Elementwise::RopePartial {
                q,
                k,
                positions,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => elemwise::rope::partial(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            // The interleaved section layout forwards, as it always did; the
            // tower's BLOCKED one refuses by name, because this plane's
            // `elemwise::rope_mrope` ships one entry and a rotation that
            // handed the sections out the other way would answer plausible
            // numbers for the wrong checkpoint. One entry beside
            // `interleaved` retires the refusal.
            Elementwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Interleaved,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => elemwise::rope_mrope::interleaved(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            // **THE TOWER'S OWN LAYOUT** (multimodal §6.3): contiguous
            // sections, each restarting the frequency ladder at a denominator
            // that is `Σ sections` and not the head. Its own entry beside
            // `interleaved` for the reason the refusal it retires gave — a
            // rotation that handed the sections out the other way would answer
            // plausible numbers for the wrong checkpoint — and the two forms
            // differ in the ladder as well as the split, so one shader with a
            // mode word would be two kernels sharing a register file.
            Elementwise::RopeMrope {
                q,
                k,
                positions,
                sections,
                form: MropeForm::Blocked,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => elemwise::rope_mrope::blocked(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *sections,
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            Elementwise::RopePartialQ {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
            } => elemwise::rope::partial_q(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            Elementwise::RopePartialLast {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                interleaved,
                q_out: _,
            } => elemwise::rope::partial_last(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *interleaved,
            ),
            Elementwise::RopeYarn {
                q,
                k,
                positions,
                head_dim,
                theta,
                factor,
                beta_fast,
                beta_slow,
                attention_factor,
                original_max_position,
                interleaved,
                q_out: _,
                k_out: _,
            } => elemwise::rope::yarn(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*positions),
                *head_dim,
                *theta,
                *factor,
                *beta_fast,
                *beta_slow,
                *attention_factor,
                *original_max_position,
                *interleaved,
            ),

            // The absorbed `gate` family (`elementwise.gate_*`), calling into
            // `kernels_metal::elemwise::gate`.
            Elementwise::GateSigmoidMul { x, gate, x_out: _ } => {
                elemwise::gate::sigmoid_mul(self.ctx(), self.tensor(*gate), self.tensor(*x))
            }

            // The absorbed `hc` family (`elementwise.hc_*`), calling into
            // `kernels_metal::elemwise::hc`.
            Elementwise::HcExpand { x, streams, y } => {
                elemwise::hc::expand(self.ctx(), self.tensor(*x), *streams, self.tensor(*y))
            }
            Elementwise::HcRmsnormF32 { streams, eps, y } => {
                elemwise::hc::rmsnorm_f32(self.ctx(), self.tensor(*streams), *eps, self.tensor(*y))
            }
            Elementwise::HcProject {
                normed,
                weight,
                stream_count,
                mixes,
            } => elemwise::hc::project(
                self.ctx(),
                self.tensor(*normed),
                self.tensor(*weight),
                *stream_count,
                self.tensor(*mixes),
            ),
            Elementwise::HcGates {
                normed,
                streams,
                scale,
                base,
                stream_count,
                gate_eps,
                alpha,
                sinkhorn,
                x,
                post_mix,
                comb_mix,
            } => elemwise::hc::gates(
                self.ctx(),
                self.tensor(*normed),
                self.tensor(*streams),
                self.tensor(*scale),
                self.tensor(*base),
                *stream_count,
                *gate_eps,
                *alpha,
                *sinkhorn,
                self.tensor(*x),
                self.tensor(*post_mix),
                self.tensor(*comb_mix),
            ),
            Elementwise::HcFold {
                x,
                streams,
                post_mix,
                comb_mix,
                y,
            } => elemwise::hc::fold(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*streams),
                self.tensor(*post_mix),
                self.tensor(*comb_mix),
                self.tensor(*y),
            ),
        }
    }
}
