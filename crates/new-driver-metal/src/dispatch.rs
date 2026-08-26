//! The 15 `Dispatch*` impls: every arm is destructure → resolve → call
//! (decision #13), one arm per variant, matches exhaustive.
//!
//! No arm selects a kernel — dtype and variant choice live inside the
//! `new-kernels-metal` entries — and no arm syncs (#15): a returned `Ok`
//! means the launch is encoded, nothing more. Alias outputs
//! (`#[out(alias = x)]`) bind as `_`: the compiler folded them onto their
//! input's slot, so the input name is the one the in-place kernel reads.
//! Families the metal plane stubs as `Unsupported` still get real arms that
//! forward to the stub, so the typed refusal carries the entry's own name.

use new_kernels::{
    DispatchAttention, DispatchCuda, DispatchDist, DispatchGate, DispatchGemm, DispatchHc,
    DispatchIndex, DispatchLayout, DispatchMla, DispatchMlp, DispatchMoe, DispatchNorm,
    DispatchPool, DispatchRope, DispatchSsm, KernelError,
};
use new_kernels_metal::{
    Tensor, attn, dist, gate, gemm, hc, index, layout, mla, mlp, moe, norm, pool, rope, ssm,
};
use new_model_ir::{
    Attention, Cuda, Dist, Gate, Gemm, Hc, Index, Layout, Mla, Mlp, Moe, Norm, Operands, Pool,
    Rope, Ssm,
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
                self.tensor(*y),
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
                self.tensor(*y),
            ),
            Norm::RmsnormPlusOne { x, weight, eps, y } => norm::rmsnorm_plus_one(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                *eps,
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
            ),
            Norm::ResidualAdd { x, y, y_out: _ } => {
                norm::residual_add(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            Norm::AddBias {
                bias,
                out,
                out_out: _,
            } => norm::add_bias(self.ctx(), self.tensor(*bias), self.tensor(*out)),
            Norm::MulScalar { s, x, x_out: _ } => {
                norm::mul_scalar(self.ctx(), *s, self.tensor(*x))
            }
            Norm::Scale { s, x, x_out: _ } => {
                norm::scale(self.ctx(), self.tensor(*s), self.tensor(*x))
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
                    self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
            ),
            Mlp::GegluTanh { gate, up, y } => mlp::geglu_tanh(
                self.ctx(),
                self.tensor(*gate),
                self.tensor(*up),
                self.tensor(*y),
            ),
            Mlp::GegluTanhPacked {
                packed,
                intermediate,
                y,
            } => mlp::geglu_tanh_packed(
                self.ctx(),
                self.tensor(*packed),
                *intermediate,
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
            ),
            Gemm::LmHead { act, w, y } => gemm::lm_head(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                self.tensor(*y),
            ),
            Gemm::AttentionLanding { act, w, layer, y } => gemm::attention_landing(
                self.ctx(),
                self.tensor(*act),
                self.tensor(*w),
                *layer,
                self.tensor(*y),
            ),
        }
    }
}

impl DispatchDist for Run<'_> {
    fn dispatch(&mut self, op: &Dist) -> Result<(), KernelError> {
        match op {
            Dist::AllReduce { buf, buf_out: _ } => {
                dist::all_reduce(self.ctx(), self.tensor(*buf))
            }
            Dist::AllGather { x, y } => {
                dist::all_gather(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            Dist::ReduceScatter { x, y } => {
                dist::reduce_scatter(self.ctx(), self.tensor(*x), self.tensor(*y))
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
                self.tensor(*q),
                self.tensor(*k),
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
                self.tensor(*q),
                self.tensor(*k),
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
                self.tensor(*q),
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
                self.tensor(*q),
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
                self.tensor(*routes),
                self.tensor(*weights),
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
                self.tensor(*routes),
                self.tensor(*weights),
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
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Moe::MatmulSelect {
                x,
                bank,
                routes,
                y,
            } => moe::matmul_select(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*bank),
                self.tensor(*routes),
                self.tensor(*y),
            ),
            // MENLO-SEAM: the metal entry (`moe::matmul_select_bias`) is real
            // but reads the bank as its (codes, scales) planes; a weight id
            // resolves to one dense handle here, and the split-plane weight
            // form is the shell's binding business, not this Run's.
            Moe::MatmulSelectBias { .. } => Err(KernelError::Unsupported { op: op.name() }),
            Moe::WeightedSum {
                routed,
                weights,
                y,
            } => moe::weighted_sum(
                self.ctx(),
                self.tensor(*routed),
                self.tensor(*weights),
                self.tensor(*y),
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
                self.tensor(*y),
            ),
        }
    }
}

impl DispatchGate for Run<'_> {
    fn dispatch(&mut self, op: &Gate) -> Result<(), KernelError> {
        match op {
            Gate::SigmoidMul { x, gate, x_out: _ } => {
                gate::sigmoid_mul(self.ctx(), self.tensor(*x), self.tensor(*gate))
            }
        }
    }
}

impl DispatchLayout for Run<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        match op {
            // MENLO-SEAM: an affine-quantized table would take
            // `layout::embed_gather_mb_4bit` instead; that selection keys off
            // a split-plane weight form the table types do not yet carry, so
            // every table resolves dense here.
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
                self.tensor(*y),
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
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
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
                self.tensor(*q),
                self.tensor(*gate),
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
                self.tensor(*left),
                self.tensor(*right),
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
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*gates),
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
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
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
                self.tensor(*y),
            ),
        }
    }
}

impl DispatchAttention for Run<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        match op {
            // MENLO-SEAM (driver side): the op names kv geometry
            // (kv_indptr/kv_indices/last_page_len) the metal builder never
            // reads — the pool row carries the page tables — while the
            // tables the builder does read (positions, request_of_token,
            // mask) are no op's named inputs; they bind from fire state
            // here. The kernel side of this seam is marked in
            // `new_kernels_metal::attn`.
            Attention::PlanDecode {
                kv_indptr: _,
                kv_indices: _,
                last_page_len: _,
                plan,
            } => {
                let fire = self.bindings();
                let (positions, t) = (fire.positions, fire.tables);
                let built = attn::plan_decode(
                    self.ctx(),
                    positions,
                    t.request_of_token,
                    t.mask,
                    t.mask_enabled,
                    t.mask_stride,
                )?;
                self.put(*plan, StructSlot::Decode(built));
                Ok(())
            }
            // MENLO-SEAM: same misalignment as `PlanDecode`.
            Attention::PlanPrefill {
                kv_indptr: _,
                kv_indices: _,
                last_page_len: _,
                plan,
            } => {
                let fire = self.bindings();
                let (positions, t) = (fire.positions, fire.tables);
                let built = attn::plan_prefill(
                    self.ctx(),
                    positions,
                    t.request_of_token,
                    t.mask,
                    t.mask_enabled,
                    t.mask_stride,
                )?;
                self.put(*plan, StructSlot::Prefill(built));
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
                self.pool(*cache),
                self.decode_plan(*plan),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
            ),
            Attention::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
            } => attn::prefill(
                self.ctx(),
                self.ragged(*q),
                self.pool(*cache),
                self.prefill_plan(*plan),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
            ),
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
                self.pool(*cache),
                self.prefill_plan(*plan),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
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
                self.pool(*cache),
                self.decode_plan(*plan),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
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
                self.pool(*cache),
                self.prefill_plan(*plan),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
            ),
            Attention::Sink {
                o,
                lse,
                sink,
                head_dim,
                o_out: _,
            } => attn::sink(
                self.ctx(),
                self.tensor(*o),
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
                self.tensor(*o),
                self.tensor(*lse),
            ),
            Attention::LogitSoftcap { x, cap, x_out: _ } => {
                attn::logit_softcap(self.ctx(), self.tensor(*x), *cap)
            }
            Attention::KvAppend {
                k,
                v,
                cache,
                kv_indices,
                positions,
            } => attn::kv_append(
                self.ctx(),
                self.tensor(*k),
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
                self.tensor(*plane),
                self.pool(*cache),
                self.tensor(*kv_indices),
                self.tensor(*positions),
            ),
        }
    }
}

/// `mla.plan` refuses before a payload exists (`new_kernels_metal::mla`), so
/// no consuming arm can ever hold a live one; the unit keeps the stubs'
/// signatures satisfied while each refusal names its own entry. When the
/// builder becomes real, this constant stops making sense and the arms
/// resolve through a `Run::mla_plan` accessor instead.
const NO_MLA_PLAN: &mla::MlaPlan = &mla::MlaPlan;

impl DispatchMla for Run<'_> {
    fn dispatch(&mut self, op: &Mla) -> Result<(), KernelError> {
        match op {
            Mla::Plan {
                kv_indptr,
                kv_indices,
                last_page_len,
                plan,
            } => {
                let built = mla::plan(
                    self.ctx(),
                    self.tensor(*kv_indptr),
                    self.tensor(*kv_indices),
                    self.tensor(*last_page_len),
                )?;
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
                self.tensor(*kv_c),
                self.tensor(*k_pe),
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
                self.tensor(*kv_c),
                self.tensor(*k_pe),
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
                self.tensor(*q_nope),
                self.tensor(*q_pe),
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
                self.tensor(*q_latent),
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
                self.tensor(*o),
            ),
            Mla::KvAppend {
                kv_c,
                k_pe,
                cache,
                kv_indices,
                positions,
            } => mla::kv_append(
                self.ctx(),
                self.tensor(*kv_c),
                self.tensor(*k_pe),
                self.pool(*cache),
                self.tensor(*kv_indices),
                self.tensor(*positions),
            ),
            Mla::AttentionDecode {
                q,
                plan: _,
                q_pe,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => mla::attention_decode(
                self.ctx(),
                self.tensor(*q),
                self.pool(*cache),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
            ),
            Mla::AttentionPrefill {
                q,
                plan: _,
                q_pe,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => mla::attention_prefill(
                self.ctx(),
                self.ragged(*q),
                self.pool(*cache),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
            ),
            Mla::AttentionDecodeSelected {
                q,
                plan: _,
                q_pe,
                selection,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => mla::attention_decode_selected(
                self.ctx(),
                self.tensor(*q),
                self.pool(*cache),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                self.tensor(*selection),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
            ),
            Mla::AttentionPrefillSelected {
                q,
                plan: _,
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
                self.pool(*cache),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                self.tensor(*selection),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
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
                self.tensor(*k),
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
                self.tensor(*q),
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
                self.tensor(*q),
                self.tensor(*weights),
                self.pool(*keys),
                *heads,
                *head_dim,
                *top_k,
                self.tensor(*selection),
            ),
            Index::KvAppend {
                k,
                keys,
                kv_indices,
                positions,
            } => index::kv_append(
                self.ctx(),
                self.tensor(*k),
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
            Pool::BoundaryDecode {
                positions,
                ratio,
                boundary_pos,
                boundary_req,
            } => pool::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
            ),
            Pool::BoundaryPrefill {
                positions,
                ratio,
                boundary_pos,
                boundary_req,
            } => pool::boundary_prefill(
                self.ctx(),
                self.ragged(*positions),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
            ),
            Pool::Gather {
                boundary_pos,
                boundary_req,
                pages,
                head_dim,
                ratio,
                entries,
            } => pool::gather(
                self.ctx(),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*pages),
                *head_dim,
                *ratio,
                self.tensor(*entries),
            ),
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
                self.tensor(*entries),
                *ratio,
                *heads,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
            ),
        }
    }
}

impl DispatchHc for Run<'_> {
    fn dispatch(&mut self, op: &Hc) -> Result<(), KernelError> {
        match op {
            Hc::Expand { x, streams, y } => hc::expand(
                self.ctx(),
                self.tensor(*x),
                *streams,
                self.tensor(*y),
            ),
            Hc::RmsnormF32 { streams, eps, y } => hc::rmsnorm_f32(
                self.ctx(),
                self.tensor(*streams),
                *eps,
                self.tensor(*y),
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
                self.tensor(*x),
                self.tensor(*post_mix),
                self.tensor(*comb_mix),
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
                self.tensor(*y),
            ),
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
                self.tensor(*y),
            ),
        }
    }
}

impl DispatchCuda for Run<'_> {
    /// A cuda-plane fused family on the metal `Run` — the foreign-plane case
    /// the aggregate's doc names. Nothing resolves: the plan was traced for
    /// another backend, and the typed refusal says which op proves it.
    fn dispatch(&mut self, op: &Cuda) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}
