use kernels_wgpu::attn;
use model_exec::{DispatchAttention, KernelError};
use model_ir::{Attention, Operands, StructKind};

use crate::run::{Run, StructSlot};

impl DispatchAttention for Run<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn requests(&self) -> u32 {
        u32::try_from(self.qo_indptr_host().len().saturating_sub(1)).unwrap_or(u32::MAX)
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_scores(
        &mut self,
        q: model_ir::ValueId,
        plan: model_ir::ValueId,
        cache: model_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        lse: model_ir::ValueId,
    ) -> Result<(), kernels_wgpu::Error> {
        let Some(seat) = self.score_seat() else {
            return Ok(());
        };
        let Some(plane) = seat.plane_of(lse) else {
            return Ok(());
        };

        let lane_offset = self.window().span.lane_offset;
        let requests = self.requests();
        attn::score::capture(
            self.ctx(),
            self.ragged(q),
            self.prefill_plan(plan),
            self.pool(cache),
            window,
            head_dim,
            kv_heads,
            sm_scale,
            seat.observe,
            lane_offset,
            seat.plane_stride,
            plane,
            crate::scores::KV_MAX,
            requests,
            seat.slab,
        )
    }

    fn attention(&mut self, op: &Attention) -> Result<(), kernels_wgpu::Error> {
        match op {
            Attention::PlanDecode {
                kv_indptr: _,
                kv_indices: _,
                last_page_len: _,
                kv_len,
                q_heads: _,
                kv_heads: _,
                head_dim: _,
                window: _,
                plan,
            } => {
                match self.declared(*plan) {
                    StructKind::AttnDecodePlan => {}
                    other => panic!(
                        "`attention.plan_decode` defines a {other:?}, which is no \
                         decode plan kind"
                    ),
                }
                let kv_len = self.tensor(*kv_len);
                let fire = self.bindings();
                let (positions, t) = (fire.positions, fire.tables);

                let built = attn::plan_decode(
                    self.ctx(),
                    kv_len,
                    self.cut_rows(positions),
                    self.cut_rows(t.request_of_token),
                    self.cut_rows(t.mask),
                    self.cut_rows(t.mask_enabled),
                    t.mask_stride,
                )?;
                self.put(*plan, StructSlot::Decode(built));
                Ok(())
            }

            Attention::PlanPrefill {
                kv_indptr: _,
                kv_indices: _,
                last_page_len: _,
                kv_len,
                q_heads: _,
                kv_heads: _,
                head_dim: _,
                window: _,
                plan,
            } => {
                match self.declared(*plan) {
                    StructKind::AttnPrefillPlan => {}
                    StructKind::AttnPrefillPlanSm90 => panic!(
                        "an sm90 plan kind on a wgpu trace is a trace bug: this plane \
                         builds only the fa2-shaped `AttnPrefillPlan`"
                    ),
                    other => panic!(
                        "`attention.plan_prefill` defines a {other:?}, which is no \
                         prefill plan kind"
                    ),
                }
                let kv_len = self.tensor(*kv_len);
                let fire = self.bindings();
                let (positions, t) = (fire.positions, fire.tables);

                let built = attn::plan_prefill(
                    self.ctx(),
                    kv_len,
                    self.cut_rows(positions),
                    self.cut_rows(t.request_of_token),
                    self.cut_rows(t.mask),
                    self.cut_rows(t.mask_enabled),
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
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.sdpa_split(),
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
            } => attn::arbiter::prefill(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
                self.requests(),
                &kernels_wgpu::tuning::current(),
            ),

            Attention::Dense {
                q,
                k,
                v,
                segments,
                head_dim,
                sm_scale,
                o,
            } => attn::dense::bidirectional(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*segments),
                *head_dim,
                *sm_scale,
                self.tensor(*o),
            ),
            Attention::Masked {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
            } => attn::arbiter::masked(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.cut_rows(self.tensor(*mask)),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.requests(),
                &kernels_wgpu::tuning::current(),
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
                self.tensor(*o),
                self.tensor(*lse),
                self.sdpa_split(),
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
            } => {
                attn::arbiter::prefill_lse(
                    self.ctx(),
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    self.pool(*cache),
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    self.tensor(*o),
                    self.tensor(*lse),
                    self.requests(),
                    &kernels_wgpu::tuning::current(),
                )?;

                self.capture_scores(
                    *q, *plan, *cache, *window, *head_dim, *kv_heads, *sm_scale, *lse,
                )
            }
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
                write_page,
                write_offset,
            } => attn::kv_append(
                self.ctx(),
                self.tensor(*k),
                self.tensor(*v),
                self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::KvAppendShared {
                plane,
                cache,
                write_page,
                write_offset,
            } => attn::kv_append_shared(
                self.ctx(),
                self.tensor(*plane),
                self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),

            Attention::MlaPlan {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                heads: _,
                kv_lora_rank: _,
                plan,
            } => {
                let built = attn::mla::plan(
                    self.ctx(),
                    self.tensor(*kv_indptr),
                    self.tensor(*kv_indices),
                    self.tensor(*last_page_len),
                    self.tensor(*kv_len),
                )?;
                self.put(*plan, StructSlot::Mla(built));
                Ok(())
            }
            Attention::MlaLatents {
                kv_a,
                weight,
                eps,
                kv_lora_rank,
                kv_c,
                k_pe,
            } => attn::mla::latents(
                self.ctx(),
                self.tensor(*kv_a),
                self.tensor(*weight),
                *eps,
                *kv_lora_rank,
                self.tensor(*kv_c),
                self.tensor(*k_pe),
            ),
            Attention::MlaLatentsRope {
                kv_a,
                positions,
                weight,
                eps,
                kv_lora_rank,
                rope_dim,
                theta,
                kv_c,
                k_pe,
            } => attn::mla::latents_rope(
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
            Attention::MlaSplitQB {
                q_b,
                heads,
                nope_dim,
                rope_dim,
                q_nope,
                q_pe,
            } => attn::mla::split_q_b(
                self.ctx(),
                self.tensor(*q_b),
                *heads,
                *nope_dim,
                *rope_dim,
                self.tensor(*q_nope),
                self.tensor(*q_pe),
            ),
            Attention::MlaAbsorbQ {
                q_nope,
                kv_b,
                heads,
                kv_lora_rank,
                nope_dim,
                v_head_dim,
                q_latent,
            } => attn::mla::absorb_q(
                self.ctx(),
                self.tensor(*q_nope),
                self.tensor(*kv_b),
                *heads,
                *kv_lora_rank,
                *nope_dim,
                *v_head_dim,
                self.tensor(*q_latent),
            ),
            Attention::MlaAbsorbOut {
                latent,
                kv_b,
                heads,
                kv_lora_rank,
                v_head_dim,
                nope_dim,
                o,
            } => attn::mla::absorb_out(
                self.ctx(),
                self.tensor(*latent),
                self.tensor(*kv_b),
                *heads,
                *kv_lora_rank,
                *v_head_dim,
                *nope_dim,
                self.tensor(*o),
            ),
            Attention::MlaKvAppend {
                kv_c,
                k_pe,
                cache,
                write_page,
                write_offset,
            } => attn::mla::kv_append(
                self.ctx(),
                self.tensor(*kv_c),
                self.tensor(*k_pe),
                self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),

            Attention::MlaDecode {
                q,
                plan: _,
                q_pe,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.positions, fire.tables.request_of_token)
                };
                attn::mla::attention_decode(
                    self.ctx(),
                    self.tensor(*q),
                    self.tensor(*q_pe),
                    self.pool(*cache),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                )
            }
            Attention::MlaPrefill {
                q,
                plan: _,
                q_pe,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.positions, fire.tables.request_of_token)
                };
                attn::mla::attention_prefill(
                    self.ctx(),
                    self.ragged(*q),
                    self.tensor(*q_pe),
                    self.pool(*cache),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                )
            }
            Attention::MlaDecodeSelected {
                q,
                plan: _,
                q_pe,
                selection,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.positions, fire.tables.request_of_token)
                };
                attn::mla::attention_decode_selected(
                    self.ctx(),
                    self.tensor(*q),
                    self.tensor(*q_pe),
                    self.tensor(*selection),
                    self.pool(*cache),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                )
            }
            Attention::MlaPrefillSelected {
                q,
                plan: _,
                q_pe,
                selection,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.positions, fire.tables.request_of_token)
                };
                attn::mla::attention_prefill_selected(
                    self.ctx(),
                    self.ragged(*q),
                    self.tensor(*q_pe),
                    self.tensor(*selection),
                    self.pool(*cache),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                )
            }

            Attention::PleNgramIds {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                ngram_ids,
            } => {
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_wgpu::Error::Unsupported { op: op.name() });
                };
                attn::ple::ngram_ids(
                    self.ctx(),
                    self.tensor(*ids),
                    &self.recurrent(*state),
                    hash,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    self.tensor(*ngram_ids),
                )
            }
            Attention::PleNgramIdsChunked {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                ngram_ids,
            } => {
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_wgpu::Error::Unsupported { op: op.name() });
                };
                attn::ple::ngram_ids_chunked(
                    self.ctx(),
                    self.ragged(*ids),
                    &self.recurrent(*state),
                    hash,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    self.tensor(*ngram_ids),
                )
            }

            Attention::SsmCausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            }
            | Attention::SsmCausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } if self.rs_seat().is_some() => {
                const OP: &str = "attention.ssm_causal_conv1d_committed";
                let seat = self.rs_seat().expect("guarded");
                let ext_x = self.rs_extend(OP, &seat, *x)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                attn::ssm::causal_conv1d_committed(
                    self.ctx(),
                    ext_x,
                    self.qo_indptr(),
                    &self.rs_committed(&seat),
                    self.tensor(*weight),
                    &self.recurrent(*state),
                    *conv_width,
                    *dilation,
                    ext_y,
                )?;
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Attention::SsmGdnPrep {
                ba,
                dt_bias,
                a_log,
                gates,
            } if self.rs_seat().is_some() => {
                const OP: &str = "attention.ssm_gdn_prep";
                let seat = self.rs_seat().expect("guarded");
                let ext_ba = self.rs_extend(OP, &seat, *ba)?;
                let ext_gates = self.rs_out(OP, &seat, *gates)?;
                attn::ssm::gdn_prep(
                    self.ctx(),
                    ext_ba,
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    ext_gates,
                )?;
                self.rs_land(OP, &seat, ext_gates, *gates)
            }
            Attention::SsmGatedDelta {
                qkv,
                z: _,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            }
            | Attention::SsmGatedDeltaChunked {
                qkv,
                z: _,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } if self.rs_seat().is_some() => {
                const OP: &str = "attention.ssm_gated_delta_committed";
                let seat = self.rs_seat().expect("guarded");

                let ext_qkv = self.rs_ext_of(OP, &seat, *qkv)?;
                let ext_gates = self.rs_ext_of(OP, &seat, *gates)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                attn::ssm::gated_delta_committed(
                    self.ctx(),
                    ext_qkv,
                    self.qo_indptr(),
                    &self.rs_committed(&seat),
                    ext_gates,
                    &self.recurrent(*state),
                    seat.work,
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    ext_y,
                )?;
                self.rs_land(OP, &seat, ext_y, *y)
            }
            Attention::SsmKdaStep {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            }
            | Attention::SsmKdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } if self.rs_seat().is_some() => {
                const OP: &str = "attention.ssm_kda_committed";
                let seat = self.rs_seat().expect("guarded");
                let ext_mixed = self.rs_ext_of(OP, &seat, *mixed)?;
                let ext_f = self.rs_extend(OP, &seat, *f)?;
                let ext_b = self.rs_extend(OP, &seat, *b)?;
                let ext_y = self.rs_out(OP, &seat, *y)?;
                attn::ssm::kda_committed(
                    self.ctx(),
                    ext_mixed,
                    self.qo_indptr(),
                    &self.rs_committed(&seat),
                    ext_f,
                    ext_b,
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    &self.recurrent(*state),
                    seat.work,
                    *heads,
                    *head_dim,
                    *norm_eps,
                    *gate_floor,
                    ext_y,
                )?;
                self.rs_land(OP, &seat, ext_y, *y)
            }

            Attention::SsmCausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => attn::ssm::causal_conv1d(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                &self.recurrent(*state),
                *conv_width,
                *dilation,
                self.tensor(*y),
            ),
            Attention::SsmCausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => attn::ssm::causal_conv1d_chunked(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*weight),
                &self.recurrent(*state),
                *conv_width,
                *dilation,
                self.tensor(*y),
            ),
            Attention::SsmGdnPrep {
                ba,
                dt_bias,
                a_log,
                gates,
            } => attn::ssm::gdn_prep(
                self.ctx(),
                self.tensor(*ba),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                self.tensor(*gates),
            ),
            Attention::SsmGatedDelta {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => attn::ssm::gated_delta(
                self.ctx(),
                self.tensor(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                &self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y),
            ),
            Attention::SsmGatedDeltaChunked {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => attn::ssm::gated_delta_chunked(
                self.ctx(),
                self.ragged(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                &self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y),
            ),
            Attention::SsmKdaStep {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } => attn::ssm::kda_step(
                self.ctx(),
                self.tensor(*mixed),
                self.tensor(*f),
                self.tensor(*b),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                &self.recurrent(*state),
                *heads,
                *head_dim,
                *norm_eps,
                *gate_floor,
                self.tensor(*y),
            ),
            Attention::SsmKdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                heads,
                head_dim,
                norm_eps,
                gate_floor,
                y,
            } => attn::ssm::kda_chunked(
                self.ctx(),
                self.ragged(*mixed),
                self.tensor(*f),
                self.tensor(*b),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                &self.recurrent(*state),
                *heads,
                *head_dim,
                *norm_eps,
                *gate_floor,
                self.tensor(*y),
            ),

            Attention::IndexLayernormRope {
                k,
                positions,
                weight,
                bias,
                eps,
                rope_dim,
                theta,
                k_out: _,
            } => attn::index::layernorm_rope(
                self.ctx(),
                self.tensor(*k),
                self.tensor(*positions),
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                *rope_dim,
                *theta,
            ),
            Attention::IndexRope {
                q,
                positions,
                heads,
                head_dim,
                rope_dim,
                theta,
                q_out: _,
            } => attn::index::rope(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                *heads,
                *head_dim,
                *rope_dim,
                *theta,
            ),

            Attention::IndexTopk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                ratio,
                selection,
            } => {
                let Some(scores) = self.index_scores() else {
                    return Err(kernels_wgpu::Error::Unsupported {
                        op: "attention.index_topk",
                    });
                };
                let fire = self.bindings();
                let (positions, request_of_token) = (fire.positions, fire.tables.request_of_token);
                attn::index::topk(
                    self.ctx(),
                    self.tensor(*q),
                    self.tensor(*weights),
                    self.pool(*keys),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    scores,
                    *heads,
                    *head_dim,
                    *top_k,
                    *ratio,
                    self.tensor(*selection),
                )
            }
            Attention::IndexKvAppend {
                k,
                keys,
                write_page,
                write_offset,
            } => attn::index::kv_append(
                self.ctx(),
                self.tensor(*k),
                self.pool(*keys),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),

            Attention::PoolBoundaryDecode {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => attn::pool::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                self.tensor(*row_valid),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.tensor(*boundary_rope),
            ),
            Attention::PoolBoundaryPrefill {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => attn::pool::boundary_prefill(
                self.ctx(),
                self.ragged(*positions),
                self.tensor(*row_valid),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.tensor(*boundary_rope),
            ),

            Attention::PoolGather {
                boundary_pos,
                boundary_req,
                pages,
                ape,
                head_dim,
                ratio,
                entries,
            } => {
                let Some(state) = self.pool_state(*pages) else {
                    return Err(kernels_wgpu::Error::Unsupported {
                        op: "attention.pool_gather",
                    });
                };
                attn::pool::gather(
                    self.ctx(),
                    self.tensor(*boundary_pos),
                    self.tensor(*boundary_req),
                    self.pool(*pages),
                    *head_dim,
                    *ratio,
                    state.state_kv,
                    state.state_score,
                    ape.map(|id| self.tensor(id)),
                    self.tensor(*entries),
                )
            }

            Attention::PoolStateWrite {
                kv,
                score,
                pages,
                write_page,
                write_offset,
                head_dim,
                ratio,
            } => {
                let Some(state) = self.pool_state(*pages) else {
                    return Err(kernels_wgpu::Error::Unsupported {
                        op: "attention.pool_state_write",
                    });
                };
                attn::pool::state_write(
                    self.ctx(),
                    self.tensor(*kv),
                    self.tensor(*score),
                    self.pool(*pages),
                    self.tensor(*write_page),
                    self.tensor(*write_offset),
                    *head_dim,
                    *ratio,
                    state.state_kv,
                    state.state_score,
                )
            }
            Attention::PoolKvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool: into,
                write_page,
                write_offset,
            } => attn::pool::kv_append(
                self.ctx(),
                self.tensor(*entries),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*into),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::PoolLse {
                q,
                positions,
                request_of_token,
                entries,
                ratio,
                heads,
                head_dim,
                sm_scale,
                o,
                lse,
            } => attn::pool::attention_lse(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                self.tensor(*request_of_token),
                self.pool(*entries),
                *ratio,
                *heads,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
            ),
            Attention::PoolLseSelected {
                q,
                positions,
                request_of_token,
                selection,
                entries,
                ratio,
                top_k,
                heads,
                head_dim,
                sm_scale,
                o,
                lse,
            } => attn::pool::attention_lse_selected(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                self.tensor(*request_of_token),
                self.tensor(*selection),
                self.pool(*entries),
                *ratio,
                *top_k,
                *heads,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
            ),
        }
    }
}
