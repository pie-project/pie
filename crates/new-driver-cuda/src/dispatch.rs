//! The 15 `Dispatch*` impls: every arm is destructure → resolve → call
//! (decision #13), one arm per variant, matches exhaustive.
//!
//! No arm selects a kernel — dtype, lattice point, gemv-vs-dense, and smem
//! arm all live inside the `new-kernels-cuda` entries — and no arm syncs
//! (#15): a returned `Ok` means the launch is on the stream, nothing more,
//! so the same arms run identically inside a graph capture. The one routing
//! an arm does perform is *resolution*: following a plan output's declared
//! `StructKind`, or a plan slot's held kind — choices the trace already
//! wrote down, not choices made here.
//!
//! Alias outputs (`#[out(alias = x)]`) bind as `_`: the compiler folded
//! them onto their input's slot, so the input name is the one the in-place
//! kernel reads. Families whose entries refuse (`hc.collapse`,
//! `attention.prefill_sm90`) still get real resolve → call arms, so the
//! typed refusal carries the entry's own name.
//!
//! The plan-building arms are the prepare phase's whole population (#16):
//! each one runs a pure builder over the host twins in [`FireBindings`],
//! `stage`s the schedule's upload immediately — eagerly, on the stream,
//! before any capture begins — and seats the payload for the consuming
//! arms.

use new_kernels::{
    DispatchAttention, DispatchCuda, DispatchDist, DispatchGate, DispatchGemm, DispatchHc,
    DispatchIndex, DispatchLayout, DispatchMla, DispatchMlp, DispatchMoe, DispatchNorm,
    DispatchPool, DispatchRope, DispatchSsm, KernelError,
};
use new_kernels_cuda::attn::{self, fa2, fused, index, mla, plan, pool};
use new_kernels_cuda::{Tensor, dist, gate, gemm, hc, layout, mlp, moe, norm, rope, ssm};
use new_model_ir::{
    Attention, Cuda, Dist, Gate, Gemm, Hc, Index, Layout, Mla, Mlp, Moe, Norm, Pool, Rope, Ssm,
    StructKind,
};

use crate::run::{Run, StructSlot};

impl DispatchNorm for Run<'_> {
    fn dispatch(&mut self, op: &Norm) -> Result<(), KernelError> {
        match op {
            Norm::Rmsnorm { x, weight, eps, y } => norm::rmsnorm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                &mut self.tensor(*y),
            ),
            Norm::RmsnormPerHead {
                x,
                weight,
                head_dim,
                eps,
                y,
            } => norm::rmsnorm_per_head(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                &mut self.tensor(*y),
            ),
            Norm::RmsnormPlusOne { x, weight, eps, y } => norm::rmsnorm_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                &mut self.tensor(*y),
            ),
            Norm::RmsnormPerHeadPlusOne {
                x,
                weight,
                head_dim,
                eps,
                y,
            } => norm::rmsnorm_per_head_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *head_dim,
                *eps,
                &mut self.tensor(*y),
            ),
            Norm::RmsnormNoScale {
                x,
                head_dim,
                eps,
                y,
            } => norm::rmsnorm_no_scale(
                self.ctx(),
                self.tensor(*x),
                *head_dim,
                *eps,
                &mut self.tensor(*y),
            ),
            Norm::RmsnormGated {
                x,
                gate,
                weight,
                head_dim,
                eps,
                y,
            } => norm::rmsnorm_gated(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *head_dim,
                *eps,
                &mut self.tensor(*y),
            ),
            Norm::RmsnormGatedBy {
                x,
                gate,
                weight,
                heads,
                eps,
                y,
            } => norm::rmsnorm_gated_by(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*gate),
                self.tensor(*weight),
                *heads,
                *eps,
                &mut self.tensor(*y),
            ),
            Norm::ResidualAdd { x, y, y_out: _ } => {
                norm::residual_add(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Norm::AddBias {
                bias,
                out,
                out_out: _,
            } => norm::add_bias(self.ctx(), self.tensor(*bias), &mut self.tensor(*out)),
            Norm::MulScalar { s, x, x_out: _ } => {
                norm::mul_scalar(self.ctx(), *s, &mut self.tensor(*x))
            }
            Norm::Scale { s, x, x_out: _ } => {
                norm::scale(self.ctx(), self.tensor(*s), &mut self.tensor(*x))
            }
            Norm::ResBlend {
                prefix,
                blocks,
                weight,
                eps,
                proj,
                y,
            } => {
                let blocks: Vec<Tensor> = blocks.iter().map(|b| self.tensor(*b)).collect();
                norm::res_blend(
                    self.ctx(),
                    self.tensor(*prefix),
                    &blocks,
                    self.tensor(*weight),
                    *eps,
                    self.tensor(*proj),
                    &mut self.tensor(*y),
                )
            }
        }
    }
}

impl DispatchMlp for Run<'_> {
    fn dispatch(&mut self, op: &Mlp) -> Result<(), KernelError> {
        match op {
            Mlp::Swiglu {
                packed,
                intermediate,
                y,
            } => mlp::swiglu(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                &mut self.tensor(*y),
            ),
            Mlp::SwigluClamp {
                packed,
                intermediate,
                limit,
                y,
            } => mlp::swiglu_clamp(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *limit,
                &mut self.tensor(*y),
            ),
            Mlp::SwigluClampAlpha {
                packed,
                intermediate,
                limit,
                alpha,
                y,
            } => mlp::swiglu_clamp_alpha(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *limit,
                *alpha,
                &mut self.tensor(*y),
            ),
            Mlp::GegluTanh { gate, up, y } => mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                &mut self.tensor(*y),
            ),
            Mlp::GegluTanhPacked {
                packed,
                intermediate,
                y,
            } => mlp::geglu_tanh_packed(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                &mut self.tensor(*y),
            ),
            Mlp::Situ {
                packed,
                intermediate,
                beta,
                up_cap,
                y,
            } => mlp::situ(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                *beta,
                *up_cap,
                &mut self.tensor(*y),
            ),
        }
    }
}

impl DispatchGemm for Run<'_> {
    fn dispatch(&mut self, op: &Gemm) -> Result<(), KernelError> {
        match op {
            Gemm::Matmul { act, w, y } => gemm::matmul(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                &mut self.tensor(*y),
            ),
            Gemm::LmHead { act, w, y } => gemm::lm_head(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                &mut self.tensor(*y),
            ),
            Gemm::AttentionLanding { act, w, layer, y } => gemm::attention_landing(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                *layer,
                &mut self.tensor(*y),
            ),
        }
    }
}

impl DispatchDist for Run<'_> {
    fn dispatch(&mut self, op: &Dist) -> Result<(), KernelError> {
        match op {
            Dist::AllReduce { buf, buf_out: _ } => {
                dist::all_reduce(self.ctx(), &mut self.tensor(*buf))
            }
            Dist::AllGather { x, y } => {
                dist::all_gather(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Dist::ReduceScatter { x, y } => {
                dist::reduce_scatter(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
        }
    }
}

impl DispatchRope for Run<'_> {
    fn dispatch(&mut self, op: &Rope) -> Result<(), KernelError> {
        match op {
            Rope::Full {
                q,
                k,
                positions,
                head_dim,
                theta,
                interleaved,
                q_out: _,
                k_out: _,
            } => rope::full(
                self.ctx(),
                &mut self.tensor(*q),
                &mut self.tensor(*k),
                self.tensor(*positions),
                *head_dim,
                *theta,
                *interleaved,
            ),
            Rope::Partial {
                q,
                k,
                positions,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
                k_out: _,
            } => rope::partial(
                self.ctx(),
                &mut self.tensor(*q),
                &mut self.tensor(*k),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            Rope::PartialQ {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                q_out: _,
            } => rope::partial_q(
                self.ctx(),
                &mut self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
            ),
            Rope::PartialLast {
                q,
                positions,
                rotary_dim,
                head_dim,
                theta,
                interleaved,
                q_out: _,
            } => rope::partial_last(
                self.ctx(),
                &mut self.tensor(*q),
                self.tensor(*positions),
                *rotary_dim,
                *head_dim,
                *theta,
                *interleaved,
            ),
            Rope::Yarn {
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
            } => rope::yarn(
                self.ctx(),
                &mut self.tensor(*q),
                &mut self.tensor(*k),
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
        }
    }
}

impl DispatchMoe for Run<'_> {
    fn dispatch(&mut self, op: &Moe) -> Result<(), KernelError> {
        match op {
            Moe::TopkSoftmax {
                logits,
                experts,
                top_k,
                routes,
                weights,
            } => moe::topk_softmax(
                self.ctx(),
                self.tensor(*logits),
                *experts,
                *top_k,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Moe::TopkSigmoid {
                logits,
                experts,
                top_k,
                renormalize,
                scaling,
                routes,
                weights,
            } => moe::topk_sigmoid(
                self.ctx(),
                self.tensor(*logits),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
                &mut self.tensor(*routes),
                &mut self.tensor(*weights),
            ),
            Moe::TopkSqrtSoftplus {
                logits,
                bias,
                experts,
                top_k,
                renormalize,
                scaling,
                routes,
                weights,
            } => moe::topk_sqrt_softplus(
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
            Moe::MatmulSelect { x, bank, routes, y } => moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                &mut self.tensor(*y),
            ),
            // MENLO-SEAM: the IR's one `bank` id is two device planes — the
            // (codes, scales) pair the entry reads. The metal shell's
            // one-handle weight rows refused this form; here the weight
            // table seats it (`WeightRow::Planes`) and the id resolves
            // through `Run::planes`.
            Moe::MatmulSelectBias {
                x,
                bank,
                bias,
                routes,
                y,
            } => {
                let (codes, scales) = self.planes(*bank);
                moe::matmul_select_bias(
                    self.ctx(),
                    self.tensor(*x),
                    codes,
                    scales,
                    self.tensor(*bias),
                    self.tensor(*routes),
                    &mut self.tensor(*y),
                )
            }
            Moe::WeightedSum { routed, weights, y } => moe::weighted_sum(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*weights),
                &mut self.tensor(*y),
            ),
            Moe::SigmoidGateAdd {
                routed,
                shared,
                gate,
                y,
            } => moe::sigmoid_gate_add(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*shared),
                self.tensor(*gate),
                &mut self.tensor(*y),
            ),
        }
    }
}

impl DispatchGate for Run<'_> {
    fn dispatch(&mut self, op: &Gate) -> Result<(), KernelError> {
        match op {
            Gate::SigmoidMul { x, gate, x_out: _ } => {
                gate::sigmoid_mul(self.ctx(), self.tensor(*gate), &mut self.tensor(*x))
            }
        }
    }
}

impl DispatchLayout for Run<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        match op {
            Layout::Embed {
                ids,
                table,
                vocab,
                y,
            } => layout::embed(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                &mut self.tensor(*y),
            ),
            Layout::SplitQkv {
                packed,
                q_width,
                kv_width,
                q,
                k,
                v,
            } => layout::split_qkv(
                self.ctx(),
                self.tensor(*packed),
                *q_width,
                *kv_width,
                &mut self.tensor(*q),
                &mut self.tensor(*k),
                &mut self.tensor(*v),
            ),
            Layout::SplitQGate {
                packed,
                head_dim,
                q,
                gate,
            } => layout::split_q_gate(
                self.ctx(),
                self.tensor(*packed),
                *head_dim,
                &mut self.tensor(*q),
                &mut self.tensor(*gate),
            ),
            Layout::SplitRows {
                x,
                width,
                left,
                right,
            } => layout::split_rows(
                self.ctx(),
                self.tensor(*x),
                *width,
                &mut self.tensor(*left),
                &mut self.tensor(*right),
            ),
            Layout::Select {
                table,
                layer,
                width,
                y,
            } => layout::select(
                self.ctx(),
                self.tensor(*table),
                *layer,
                *width,
                &mut self.tensor(*y),
            ),
        }
    }
}

impl DispatchSsm for Run<'_> {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError> {
        match op {
            Ssm::CausalConv1d {
                x,
                weight,
                state,
                conv_width,
                y,
            } => ssm::causal_conv1d(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                self.recurrent(*state),
                *conv_width,
                &mut self.tensor(*y),
            ),
            Ssm::CausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                y,
            } => ssm::causal_conv1d_chunked(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*weight),
                self.recurrent(*state),
                *conv_width,
                &mut self.tensor(*y),
            ),
            Ssm::GdnPrep {
                ba,
                dt_bias,
                a_log,
                gates,
            } => ssm::gdn_prep(
                self.ctx(),
                self.tensor(*ba),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                &mut self.tensor(*gates),
            ),
            Ssm::GatedDelta {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => ssm::gated_delta(
                self.ctx(),
                self.tensor(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                &mut self.tensor(*y),
            ),
            Ssm::GatedDeltaChunked {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => ssm::gated_delta_chunked(
                self.ctx(),
                self.ragged(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                &mut self.tensor(*y),
            ),
            Ssm::KdaStep {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                y,
            } => ssm::kda_step(
                self.ctx(),
                self.tensor(*mixed),
                self.tensor(*f),
                self.tensor(*b),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                self.recurrent(*state),
                *heads,
                *head_dim,
                *norm_eps,
                &mut self.tensor(*y),
            ),
            Ssm::KdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                y,
            } => ssm::kda_chunked(
                self.ctx(),
                self.ragged(*mixed),
                self.tensor(*f),
                self.tensor(*b),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                self.recurrent(*state),
                *heads,
                *head_dim,
                *norm_eps,
                &mut self.tensor(*y),
            ),
        }
    }
}

impl DispatchAttention for Run<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        match op {
            // MENLO-SEAM (driver side): the op names its kv geometry as
            // device values, but the builder walks kv_indptr's CONTENTS —
            // so the arm routes through the geometry input's cache space to
            // the host twin the shell bound beside it (`Run::planning`).
            // kv_indices and last_page_len are never read at build time;
            // they ride the pool row into the launches. The occupancy fact
            // (`decode_max_grid_size`) is host prepare-phase work the
            // builder's purity keeps outside itself.
            Attention::PlanDecode {
                kv_indptr,
                kv_indices: _,
                last_page_len: _,
                plan,
            } => {
                let built = {
                    let fire = self.bindings();
                    let seat = self.planning(*kv_indptr);
                    let max_grid = fa2::decode_max_grid_size(
                        seat.shape.head_dim,
                        seat.shape.num_q_heads,
                        seat.shape.num_kv_heads,
                        &fire.device,
                    );
                    plan::plan_decode(
                        &seat.kv_indptr,
                        seat.shape,
                        seat.window,
                        fire.capture,
                        max_grid,
                        &fire.device,
                        seat.decode_grant(),
                    )?
                };
                built.stage(self.ctx())?;
                self.put(*plan, StructSlot::Decode(built));
                Ok(())
            }
            // MENLO-SEAM: as `PlanDecode`, plus the qo side — the fire's
            // shared indptr has a host twin too, and the mask pair (no op
            // names one) binds onto the plan here, for `attention.masked`
            // to find. Which builder runs is the trace's declaration: the
            // plan value's `StructKind` says fa2 or sm90, and the arm
            // follows it. `causal: true` is `attention.prefill`'s reading;
            // `attention.masked` replaces the causal bound with the mask
            // the plan carries and never consults the flag.
            Attention::PlanPrefill {
                kv_indptr,
                kv_indices: _,
                last_page_len: _,
                plan,
            } => {
                let built = {
                    let fire = self.bindings();
                    let seat = self.planning(*kv_indptr);
                    match self.declared(*plan) {
                        StructKind::AttnPrefillPlan => {
                            let built = plan::plan_prefill(
                                &fire.indptr_host,
                                &seat.kv_indptr,
                                fire.total_tokens(),
                                seat.shape,
                                seat.window,
                                true,
                                fire.capture,
                                fire.tables.mask,
                                &fire.device,
                                seat.prefill_grant(),
                            )?;
                            built.stage(self.ctx())?;
                            StructSlot::Prefill(built)
                        }
                        StructKind::AttnPrefillPlanSm90 => {
                            let built = plan::plan_prefill_sm90(
                                &fire.indptr_host,
                                &seat.kv_indptr,
                                &seat.kv_len,
                                fire.total_tokens(),
                                seat.shape,
                                true,
                                fire.capture,
                                &fire.device,
                                seat.prefill_grant(),
                            )?;
                            built.stage(self.ctx())?;
                            StructSlot::PrefillSm90(built)
                        }
                        other => panic!(
                            "`attention.plan_prefill` defines a {other:?}, which is no \
                             prefill plan kind"
                        ),
                    }
                };
                self.put(*plan, built);
                Ok(())
            }
            Attention::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
            } => attn::decode(
                self.ctx(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            // A prefill plan can hold either kind the trace declared; the
            // arm follows the slot. The sm90 launcher still answers a typed
            // refusal in its own name (`attn::prefill_sm90`).
            Attention::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
            } => match self.slot(*plan) {
                StructSlot::PrefillSm90(sm90) => attn::prefill_sm90(
                    self.ctx(),
                    self.ragged(*q),
                    sm90,
                    self.pool(*cache),
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    &mut self.tensor(*o),
                ),
                _ => attn::prefill(
                    self.ctx(),
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    self.pool(*cache),
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    &mut self.tensor(*o),
                ),
            },
            // MENLO-SEAM: the op names no mask operand; the fire mask rode
            // onto the plan at build (the `PlanPrefill` arm), and the entry
            // refuses a plan no mask rides.
            Attention::Masked {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
            } => attn::masked(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::DecodeLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse,
            } => attn::decode_lse(
                self.ctx(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
            Attention::PrefillLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                lse,
            } => attn::prefill_lse(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
            Attention::Sink {
                o,
                lse,
                sink,
                head_dim,
                o_out: _,
            } => attn::sink(
                self.ctx(),
                &mut self.tensor(*o),
                self.tensor(*lse),
                self.tensor(*sink),
                *head_dim,
            ),
            Attention::MergeLse {
                o1,
                lse1,
                o2,
                lse2,
                heads,
                head_dim,
                o,
                lse,
            } => attn::merge_lse(
                self.ctx(),
                self.tensor(*o1),
                self.tensor(*lse1),
                self.tensor(*o2),
                self.tensor(*lse2),
                *heads,
                *head_dim,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
            Attention::LogitSoftcap { x, cap, x_out: _ } => {
                attn::logit_softcap(self.ctx(), &mut self.tensor(*x), *cap)
            }
            // MENLO-SEAM: the stated kv_indices/positions pass through but
            // go unread — the appender addresses by the pool row's write
            // tables and the fire indptr riding in `k`; the shell derives
            // those tables from the same declared inputs.
            Attention::KvAppend {
                k,
                v,
                cache,
                kv_indices,
                positions,
            } => attn::kv_append(
                self.ctx(),
                self.ragged(*k),
                self.tensor(*v),
                self.pool(*cache),
                self.tensor(*kv_indices),
                self.tensor(*positions),
            ),
            Attention::KvAppendShared {
                plane,
                cache,
                kv_indices,
                positions,
            } => attn::kv_append_shared(
                self.ctx(),
                self.ragged(*plane),
                self.pool(*cache),
                self.tensor(*kv_indices),
                self.tensor(*positions),
            ),
        }
    }
}

impl DispatchMla for Run<'_> {
    fn dispatch(&mut self, op: &Mla) -> Result<(), KernelError> {
        match op {
            // MENLO-SEAM: the same host-twin routing as the attention plan
            // arms, plus `kv_len` — derived host-side, seated on the
            // planning twin. The builder's `causal` word is derived from
            // the fire's own boundaries: multi-token lanes attend causally
            // within themselves, single-token (decode) lanes have nothing
            // to order.
            Mla::Plan {
                kv_indptr,
                kv_indices: _,
                last_page_len: _,
                plan,
            } => {
                let built = {
                    let fire = self.bindings();
                    let seat = self.planning(*kv_indptr);
                    plan::plan_mla(
                        &fire.indptr_host,
                        &seat.kv_indptr,
                        &seat.kv_len,
                        seat.shape.num_requests,
                        seat.shape.num_q_heads,
                        seat.shape.head_dim,
                        fire.multi_token(),
                        &fire.device,
                        seat.mla_grant(),
                    )?
                };
                built.stage(self.ctx())?;
                self.put(*plan, StructSlot::Mla(built));
                Ok(())
            }
            Mla::Latents {
                kv_a,
                weight,
                eps,
                kv_lora_rank,
                kv_c,
                k_pe,
            } => mla::latents(
                self.ctx(),
                self.tensor(*kv_a),
                self.tensor(*weight),
                *eps,
                *kv_lora_rank,
                &mut self.tensor(*kv_c),
                &mut self.tensor(*k_pe),
            ),
            Mla::LatentsRope {
                kv_a,
                positions,
                weight,
                eps,
                kv_lora_rank,
                rope_dim,
                theta,
                kv_c,
                k_pe,
            } => mla::latents_rope(
                self.ctx(),
                self.tensor(*kv_a),
                self.tensor(*positions),
                self.tensor(*weight),
                *eps,
                *kv_lora_rank,
                *rope_dim,
                *theta,
                &mut self.tensor(*kv_c),
                &mut self.tensor(*k_pe),
            ),
            Mla::SplitQB {
                q_b,
                heads,
                nope_dim,
                rope_dim,
                q_nope,
                q_pe,
            } => mla::split_q_b(
                self.ctx(),
                self.tensor(*q_b),
                *heads,
                *nope_dim,
                *rope_dim,
                &mut self.tensor(*q_nope),
                &mut self.tensor(*q_pe),
            ),
            Mla::AbsorbQ {
                q_nope,
                kv_b,
                heads,
                kv_lora_rank,
                nope_dim,
                v_head_dim,
                q_latent,
            } => mla::absorb_q(
                self.ctx(),
                self.tensor(*q_nope),
                self.tensor(*kv_b),
                *heads,
                *kv_lora_rank,
                *nope_dim,
                *v_head_dim,
                &mut self.tensor(*q_latent),
            ),
            Mla::AbsorbOut {
                latent,
                kv_b,
                heads,
                kv_lora_rank,
                v_head_dim,
                nope_dim,
                o,
            } => mla::absorb_out(
                self.ctx(),
                self.tensor(*latent),
                self.tensor(*kv_b),
                *heads,
                *kv_lora_rank,
                *v_head_dim,
                *nope_dim,
                &mut self.tensor(*o),
            ),
            // MENLO-SEAM: as `attention.kv_append` — the stated pair goes
            // unread; the appender addresses by the pool's page tables and
            // the fire indptr riding in `kv_c`.
            Mla::KvAppend {
                kv_c,
                k_pe,
                cache,
                kv_indices,
                positions,
            } => mla::kv_append(
                self.ctx(),
                self.ragged(*kv_c),
                self.tensor(*k_pe),
                self.pool(*cache),
                self.tensor(*kv_indices),
                self.tensor(*positions),
            ),
            Mla::AttentionDecode {
                q,
                plan,
                q_pe,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => mla::attention_decode(
                self.ctx(),
                self.ragged(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Mla::AttentionPrefill {
                q,
                plan,
                q_pe,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => mla::attention_prefill(
                self.ctx(),
                self.ragged(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Mla::AttentionDecodeSelected {
                q,
                plan,
                q_pe,
                selection,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => mla::attention_decode_selected(
                self.ctx(),
                self.ragged(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                self.tensor(*selection),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Mla::AttentionPrefillSelected {
                q,
                plan,
                q_pe,
                selection,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => mla::attention_prefill_selected(
                self.ctx(),
                self.ragged(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                self.tensor(*selection),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
        }
    }
}

impl DispatchIndex for Run<'_> {
    fn dispatch(&mut self, op: &Index) -> Result<(), KernelError> {
        match op {
            Index::LayernormRope {
                k,
                positions,
                weight,
                bias,
                eps,
                rope_dim,
                theta,
                k_out: _,
            } => index::layernorm_rope(
                self.ctx(),
                &mut self.tensor(*k),
                self.tensor(*positions),
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                *rope_dim,
                *theta,
            ),
            Index::Rope {
                q,
                positions,
                heads,
                head_dim,
                rope_dim,
                theta,
                q_out: _,
            } => index::rope(
                self.ctx(),
                &mut self.tensor(*q),
                self.tensor(*positions),
                *heads,
                *head_dim,
                *rope_dim,
                *theta,
            ),
            Index::Topk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                selection,
            } => index::topk(
                self.ctx(),
                self.ragged(*q),
                self.tensor(*weights),
                self.pool(*keys),
                *heads,
                *head_dim,
                *top_k,
                &mut self.tensor(*selection),
            ),
            // MENLO-SEAM: as `attention.kv_append` — the stated pair goes
            // unread; the fire indptr rides in `k`.
            Index::KvAppend {
                k,
                keys,
                kv_indices,
                positions,
            } => index::kv_append(
                self.ctx(),
                self.ragged(*k),
                self.pool(*keys),
                self.tensor(*kv_indices),
                self.tensor(*positions),
            ),
        }
    }
}

impl DispatchPool for Run<'_> {
    fn dispatch(&mut self, op: &Pool) -> Result<(), KernelError> {
        match op {
            // MENLO-SEAM: `row_valid` (the CUDA-graph padding mask) has no
            // seat on this op — the boundary stage runs before any pool row
            // is named — so the arm binds the fire table.
            Pool::BoundaryDecode {
                positions,
                ratio,
                boundary_pos,
                boundary_req,
            } => pool::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                *ratio,
                self.bindings().tables.row_valid,
                &mut self.tensor(*boundary_pos),
                &mut self.tensor(*boundary_req),
            ),
            // MENLO-SEAM: `row_valid` as `BoundaryDecode`; the fire indptr
            // rides in `positions`.
            Pool::BoundaryPrefill {
                positions,
                ratio,
                boundary_pos,
                boundary_req,
            } => pool::boundary_prefill(
                self.ctx(),
                self.ragged(*positions),
                *ratio,
                self.bindings().tables.row_valid,
                &mut self.tensor(*boundary_pos),
                &mut self.tensor(*boundary_req),
            ),
            // MENLO-SEAM: the dsv4 compressor state (`state_kv`,
            // `state_score`, `ape`) has no IR seat; the arm binds the slabs
            // the shell staged for the pooled space (`Run::slabs`).
            Pool::Gather {
                boundary_pos,
                boundary_req,
                pages,
                head_dim,
                ratio,
                entries,
            } => {
                let slabs = self.slabs();
                pool::gather(
                    self.ctx(),
                    self.tensor(*boundary_pos),
                    self.tensor(*boundary_req),
                    self.pool(*pages),
                    *head_dim,
                    *ratio,
                    slabs.state_kv,
                    slabs.state_score,
                    slabs.ape,
                    &mut self.tensor(*entries),
                )
            }
            // MENLO-SEAM: the stated kv_indices go unread — the store
            // addresses by the pool's page tables, derived from the same
            // input.
            Pool::KvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool: into,
                kv_indices,
            } => pool::kv_append(
                self.ctx(),
                self.tensor(*entries),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*into),
                self.tensor(*kv_indices),
            ),
            // MENLO-SEAM: `request_of_token` (the owning request per token
            // row) has no IR seat; the arm binds the fire table. `entries`
            // names the compressed cache space on this plane, so it
            // resolves to a pool.
            Pool::AttentionLse {
                q,
                positions,
                entries,
                ratio,
                heads,
                head_dim,
                sm_scale,
                o,
                lse,
            } => pool::attention_lse(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                self.pool(*entries),
                *ratio,
                *heads,
                *head_dim,
                *sm_scale,
                self.bindings().tables.request_of_token,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
        }
    }
}

impl DispatchHc for Run<'_> {
    fn dispatch(&mut self, op: &Hc) -> Result<(), KernelError> {
        match op {
            Hc::Expand { x, streams, y } => {
                hc::expand(self.ctx(), self.tensor(*x), *streams, &mut self.tensor(*y))
            }
            Hc::RmsnormF32 { streams, eps, y } => hc::rmsnorm_f32(
                self.ctx(),
                self.tensor(*streams),
                *eps,
                &mut self.tensor(*y),
            ),
            Hc::Gates {
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
            } => hc::gates(
                self.ctx(),
                self.tensor(*normed),
                self.tensor(*streams),
                self.tensor(*scale),
                self.tensor(*base),
                *stream_count,
                *gate_eps,
                *alpha,
                *sinkhorn,
                &mut self.tensor(*x),
                &mut self.tensor(*post_mix),
                &mut self.tensor(*comb_mix),
            ),
            Hc::Fold {
                x,
                streams,
                post_mix,
                comb_mix,
                y,
            } => hc::fold(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*streams),
                self.tensor(*post_mix),
                self.tensor(*comb_mix),
                &mut self.tensor(*y),
            ),
            // The entry refuses (its mix plane has no producer yet — the
            // `MENLO-SEAM` in `new_kernels_cuda::hc`); the arm still
            // resolves and calls, so the typed refusal carries the entry's
            // own name.
            Hc::Collapse {
                streams,
                head_scale,
                head_base,
                stream_count,
                gate_eps,
                y,
            } => hc::collapse(
                self.ctx(),
                self.tensor(*streams),
                self.tensor(*head_scale),
                self.tensor(*head_base),
                *stream_count,
                *gate_eps,
                &mut self.tensor(*y),
            ),
        }
    }
}

impl DispatchCuda for Run<'_> {
    /// The cuda-plane fused family on its home `Run` — this is the shell
    /// the trace emitted it for, so the arm dispatches the real entry.
    /// The write side addresses by the pool row's `write_page`/
    /// `write_offset` tables (the entry's `MENLO-SEAM`), which the shell
    /// derived when it built the [`CachePool`](crate::run::CachePool) row.
    fn dispatch(&mut self, op: &Cuda) -> Result<(), KernelError> {
        match op {
            Cuda::QkvFusedQknormRopeVnormWrite {
                packed,
                positions,
                q_norm_weight,
                q_norm_eps,
                k_norm_weight,
                k_norm_eps,
                cache,
                kv_indices,
                kv_heads,
                head_dim,
                theta,
                q,
            } => fused::qkv_fused_qknorm_rope_vnorm_write(
                self.ctx(),
                self.tensor(*packed),
                self.tensor(*positions),
                self.tensor(*q_norm_weight),
                *q_norm_eps,
                self.tensor(*k_norm_weight),
                *k_norm_eps,
                self.pool(*cache),
                self.tensor(*kv_indices),
                *kv_heads,
                *head_dim,
                *theta,
                &mut self.tensor(*q),
            ),
        }
    }
}
