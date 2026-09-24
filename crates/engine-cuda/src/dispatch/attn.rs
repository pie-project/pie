use kernels_cuda::attn::{self, fa2, index, mla, plan, pool, selected};
use kernels_cuda::attn_dense;
use model_exec::{DispatchAttention, KernelError};
use model_ir::{Attention, Operands, StructKind};

use crate::run::{Run, StructSlot};
use kernels_cuda::RaggedTensor;
use model_ir::ValueId;

impl DispatchAttention for Run<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
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
    ) -> Result<(), kernels_cuda::Error> {
        let Some(seat) = self.bindings().scores.clone() else {
            return Ok(());
        };
        let Some(plane) = seat.plane_of(lse) else {
            return Ok(());
        };
        let lane_offset = self.window().span().lane_offset;
        let mut slab = seat.slab;
        let mut q_rows = self.ragged(q);
        q_rows.data = self.windowed(q_rows.data);
        kernels_cuda::attn_score::capture(
            self.ctx(),
            q_rows,
            self.prefill_plan(plan),
            &self.pool(cache),
            window,
            head_dim,
            kv_heads,
            sm_scale,
            seat.observe,
            lane_offset,
            seat.plane_stride,
            plane,
            crate::scores::KV_MAX,
            &mut slab,
        )
    }

    fn attention(&mut self, op: &Attention) -> Result<(), kernels_cuda::Error> {
        match op {
            Attention::PlanDecode {
                kv_indptr,
                kv_indices: _,
                last_page_len: _,
                kv_len: _,
                q_heads: _,
                kv_heads: _,
                head_dim: _,
                window: _,
                plan,
            } => {
                let built = {
                    let fire = self.bindings();
                    let seat = self.planning(*kv_indptr, *plan);
                    let max_grid = fa2::decode_max_grid_size(
                        seat.shape.head_dim,
                        seat.shape.num_q_heads,
                        seat.shape.num_kv_heads,
                        &fire.device,
                    );
                    plan::plan_decode(
                        &seat.kv_indptr,
                        &seat.kv_len,
                        seat.shape,
                        seat.live,
                        seat.window,
                        fire.capture,
                        max_grid,
                        fire.toggles,
                        &fire.device,
                        seat.workspace,
                    )?
                };
                if crate::serve::diag::on().plan_trace.is_some() {
                    let seat = self.planning(*kv_indptr, *plan);
                    eprintln!(
                        "[plan-trace] decode capture={} rows={} live={:?} window={:?} grant={:?} info={:?}",
                        self.bindings().capture,
                        seat.rows,
                        seat.live,
                        seat.window,
                        seat.workspace,
                        built.info
                    );
                }
                built.stage(self.ctx())?;
                self.put(*plan, StructSlot::Decode(built));
                Ok(())
            }
            Attention::PlanPrefill {
                kv_indptr,
                kv_indices: _,
                last_page_len: _,
                kv_len: _,
                q_heads: _,
                kv_heads: _,
                head_dim: _,
                window: _,
                plan,
            } => {
                let built = {
                    let fire = self.bindings();
                    let seat = self.planning(*kv_indptr, *plan);
                    let spans = self.mask_indptr();
                    debug_assert_eq!(seat.live.rows, self.total_tokens());
                    match self.declared(*plan) {
                        StructKind::AttnPrefillPlan => {
                            let built = plan::plan_prefill(
                                self.qo_indptr_host(),
                                &seat.kv_indptr,
                                &seat.kv_len,
                                seat.rows,
                                seat.shape,
                                seat.live,
                                seat.window,
                                true,
                                fire.capture,
                                spans,
                                &fire.device,
                                seat.workspace,
                            )?;
                            if let Some(wanted) = crate::serve::diag::on().plan_trace.as_deref()
                                && (wanted == "all" || wanted == seat.rows.to_string())
                            {
                                let ints: Vec<i32> = built
                                    .int_upload
                                    .chunks_exact(4)
                                    .take(96)
                                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                                    .collect();
                                eprintln!(
                                    "[plan-trace] capture={} rows={} live={:?} window={:?} grant={:?} info={:?} int_bytes={} float_bytes={} ints[..96]={:?}",
                                    fire.capture,
                                    seat.rows,
                                    seat.live,
                                    seat.window,
                                    seat.workspace,
                                    built.info,
                                    built.int_bytes,
                                    built.float_bytes,
                                    ints
                                );
                            }
                            built.stage(self.ctx())?;
                            StructSlot::Prefill(built)
                        }
                        StructKind::AttnPrefillPlanSm90 => {
                            let built = plan::plan_prefill_sm90(
                                self.qo_indptr_host(),
                                &seat.kv_indptr,
                                &seat.kv_len,
                                seat.rows,
                                seat.shape,
                                seat.live,
                                true,
                                fire.capture,
                                &fire.device,
                                seat.workspace,
                            )?;
                            if crate::serve::diag::on().plan_trace.is_some() {
                                eprintln!(
                                    "[plan-trace] sm90 capture={} rows={} live={:?} window={:?} grant={:?} info={:?}",
                                    fire.capture,
                                    seat.rows,
                                    seat.live,
                                    seat.window,
                                    seat.workspace,
                                    built.info
                                );
                            }
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
                self.ragged_q(*q),
                self.decode_plan(*plan),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::DecodeSelected {
                q,
                plan: _,
                selection,
                cache,
                window,
                head_dim,
                sm_scale,
                ratio,
                o,
            } => selected::attention_decode_selected(
                self.ctx(),
                self.ragged_q(*q),
                self.tensor(*selection),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *sm_scale,
                *ratio,
                &mut self.tensor(*o),
            ),
            Attention::PrefillSelected {
                q,
                plan: _,
                selection,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                ratio,
                o,
            } => selected::attention_prefill_selected(
                self.ctx(),
                self.ragged_q(*q),
                self.tensor(*selection),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                *ratio,
                &mut self.tensor(*o),
            ),
            Attention::DecodeRel {
                q,
                plan,
                cache,
                bias,
                window,
                head_dim,
                extent,
                sm_scale,
                log_floor,
                log_alpha,
                o,
            } => attn::decode_rel(
                self.ctx(),
                self.ragged_q(*q),
                self.decode_plan(*plan),
                &self.pool_absolute(*cache),
                attn::RelBias {
                    bias: self.tensor(*bias),
                    extent: *extent,
                    log_floor: *log_floor,
                    log_alpha: *log_alpha,
                },
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::PrefillRel {
                q,
                plan,
                cache,
                bias,
                window,
                head_dim,
                kv_heads,
                extent,
                sm_scale,
                log_floor,
                log_alpha,
                o,
            } => match self.slot(*plan) {
                StructSlot::PrefillSm90(_) => {
                    Err(kernels_cuda::Error::Unsupported { op: op.name() })
                }
                _ => attn::prefill_rel(
                    self.ctx(),
                    self.ragged_q(*q),
                    self.prefill_plan(*plan),
                    &self.pool_absolute(*cache),
                    attn::RelBias {
                        bias: self.tensor(*bias),
                        extent: *extent,
                        log_floor: *log_floor,
                        log_alpha: *log_alpha,
                    },
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    &mut self.tensor(*o),
                ),
            },
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
                    self.ragged_q(*q),
                    sm90,
                    &self.pool(*cache),
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    &mut self.tensor(*o),
                ),
                _ => attn::prefill(
                    self.ctx(),
                    self.ragged_q(*q),
                    self.prefill_plan(*plan),
                    &self.pool_absolute(*cache),
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    &mut self.tensor(*o),
                ),
            },
            Attention::Masked {
                kv_heads: _,
                causal: _,
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
            } => attn::masked(
                self.ctx(),
                self.ragged_q(*q),
                self.prefill_plan(*plan),
                self.tensor(*mask),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::MaskedLse {
                kv_heads: _,
                causal: _,
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse,
            } => attn::masked_lse(
                self.ctx(),
                self.ragged_q(*q),
                self.prefill_plan(*plan),
                self.tensor(*mask),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
            Attention::Dense {
                q,
                k,
                v,
                segments,
                head_dim,
                sm_scale,
                o,
            } => attn_dense::bidirectional(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*segments),
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
                self.ragged_q(*q),
                self.decode_plan(*plan),
                &self.pool_absolute(*cache),
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
            } => {
                attn::prefill_lse(
                    self.ctx(),
                    self.ragged_q(*q),
                    self.prefill_plan(*plan),
                    &self.pool_absolute(*cache),
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    &mut self.tensor(*o),
                    &mut self.tensor(*lse),
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
            Attention::KvAppend {
                k,
                v,
                cache,
                write_page,
                write_offset,
            } => attn::kv_append(
                self.ctx(),
                self.ragged_lanes(*k),
                self.tensor(*v),
                &self.pool(*cache),
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
                self.ragged_lanes(*plane),
                &self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::MlaPlan {
                kv_indptr,
                kv_indices: _,
                last_page_len: _,
                kv_len: _,
                heads: _,
                kv_lora_rank: _,
                plan,
            } => {
                let built = {
                    let fire = self.bindings();
                    let seat = self.planning(*kv_indptr, *plan);
                    plan::plan_mla(
                        self.qo_indptr_host(),
                        &seat.kv_indptr,
                        &seat.kv_len,
                        seat.rows,
                        seat.shape.num_requests,
                        seat.live,
                        seat.shape.num_q_heads,
                        seat.shape.head_dim,
                        self.multi_token(),
                        &fire.device,
                        seat.workspace,
                    )?
                };
                built.stage(self.ctx())?;
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
            } => mla::latents(
                self.ctx(),
                self.tensor(*kv_a),
                self.tensor(*weight),
                *eps,
                *kv_lora_rank,
                &mut self.tensor(*kv_c),
                &mut self.tensor(*k_pe),
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
            Attention::MlaSplitQB {
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
            Attention::MlaAbsorbQ {
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
                self.dense_or_decoded(
                    "attention.mla_absorb_q",
                    *kv_b,
                    heads * (nope_dim + v_head_dim),
                    *kv_lora_rank,
                )?,
                *heads,
                *kv_lora_rank,
                *nope_dim,
                *v_head_dim,
                &mut self.tensor(*q_latent),
            ),
            Attention::MlaAbsorbOut {
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
                self.dense_or_decoded(
                    "attention.mla_absorb_out",
                    *kv_b,
                    heads * (nope_dim + v_head_dim),
                    *kv_lora_rank,
                )?,
                *heads,
                *kv_lora_rank,
                *v_head_dim,
                *nope_dim,
                &mut self.tensor(*o),
            ),
            Attention::MlaKvAppend {
                kv_c,
                k_pe,
                cache,
                write_page,
                write_offset,
            } => mla::kv_append(
                self.ctx(),
                self.ragged(*kv_c),
                self.tensor(*k_pe),
                &self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::MlaDecode {
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
                &self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::MlaPrefill {
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
                &self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::MlaDecodeSelected {
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
                self.ragged_lanes(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                self.tensor(*selection),
                &self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::MlaPrefillSelected {
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
                self.ragged_lanes(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                self.tensor(*selection),
                &self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attention::SsmCausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => {
                if self.rs_extended() {
                    return self.conv1d_extended(
                        "attention.ssm_causal_conv1d",
                        *x,
                        *weight,
                        *state,
                        *conv_width,
                        *dilation,
                        *y,
                    );
                }
                attn::ssm::causal_conv1d(
                    self.ctx(),
                    self.tensor(*x),
                    self.tensor(*weight),
                    &self.recurrent_absolute(*state),
                    *conv_width,
                    *dilation,
                    &mut self.tensor(*y),
                )
            }
            Attention::ShortConv {
                x,
                weight,
                state,
                conv_width,
                y,
            } => attn::ssm::short_conv(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                &self.recurrent_absolute(*state),
                *conv_width,
                &mut self.tensor(*y),
            ),
            Attention::ShortConvChunked {
                x,
                weight,
                state,
                conv_width,
                y,
            } => {
                self.rs_move("attention.short_conv_chunked", *x, self.tensor(*x))?;
                let tail = self.recurrent_tail_absolute(*state);
                attn::ssm::short_conv_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    &self.recurrent_absolute(*state),
                    *conv_width,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                attn::ssm::short_conv_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    &tail,
                    *conv_width,
                    &mut self.tensor(*y),
                )
            }
            Attention::SsmCausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => {
                if self.rs_extended() {
                    return self.conv1d_extended(
                        "attention.ssm_causal_conv1d_chunked",
                        *x,
                        *weight,
                        *state,
                        *conv_width,
                        *dilation,
                        *y,
                    );
                }
                self.rs_move("attention.ssm_causal_conv1d_chunked", *x, self.tensor(*x))?;
                let tail = self.recurrent_tail_absolute(*state);
                attn::ssm::causal_conv1d_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    &self.recurrent_absolute(*state),
                    *conv_width,
                    *dilation,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                attn::ssm::causal_conv1d_chunked(
                    self.ctx(),
                    self.ragged_lanes(*x),
                    self.tensor(*weight),
                    &tail,
                    *conv_width,
                    *dilation,
                    &mut self.tensor(*y),
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
                map,
                ngram_ids,
            } => kernels_cuda::attn_ple::ngram_ids(
                self.ctx(),
                self.tensor(*ids),
                &self.recurrent(*state),
                *eos,
                mults,
                primes,
                offsets,
                *heads_per_ngram,
                map.map(|id| self.tensor(id)),
                &mut self.tensor(*ngram_ids),
            ),
            Attention::PleNgramIdsChunked {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } => {
                let tail = self.recurrent_tail_absolute(*state);
                kernels_cuda::attn_ple::ngram_ids_chunked(
                    self.ctx(),
                    self.ragged_lanes(*ids),
                    &self.recurrent_absolute(*state),
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    map.map(|id| self.tensor(id)),
                    &mut self.tensor(*ngram_ids),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                kernels_cuda::attn_ple::ngram_ids_chunked(
                    self.ctx(),
                    self.ragged_lanes(*ids),
                    &tail,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    map.map(|id| self.tensor(id)),
                    &mut self.tensor(*ngram_ids),
                )
            }
            Attention::SsmGdnPrep {
                ba,
                dt_bias,
                a_log,
                gates,
            } => {
                if self.rs_extended() {
                    const OP: &str = "attention.ssm_gdn_prep";
                    let ba_ext = self.rs_extend(OP, *ba, true)?;
                    let mut gates_ext = self.rs_out(OP, *gates)?;
                    attn::ssm::gdn_prep(
                        self.ctx(),
                        ba_ext,
                        self.tensor(*dt_bias),
                        self.tensor(*a_log),
                        &mut gates_ext,
                    )?;
                    return self.rs_land(OP, gates_ext, *gates);
                }
                self.rs_move("attention.ssm_gdn_prep", *ba, self.tensor(*ba))?;
                attn::ssm::gdn_prep(
                    self.ctx(),
                    self.tensor(*ba),
                    self.tensor(*dt_bias),
                    self.tensor(*a_log),
                    &mut self.tensor(*gates),
                )
            }
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
            } => {
                if self.rs_extended() {
                    return self.gated_delta_extended(
                        "attention.ssm_gated_delta",
                        *qkv,
                        *z,
                        *gates,
                        *state,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        *y,
                    );
                }
                attn::ssm::gated_delta(
                    self.ctx(),
                    self.tensor(*qkv),
                    self.tensor(*z),
                    self.tensor(*gates),
                    &self.recurrent_absolute(*state),
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    &mut self.tensor(*y),
                )
            }
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
            } => {
                if self.rs_extended() {
                    return self.gated_delta_extended(
                        "attention.ssm_gated_delta_chunked",
                        *qkv,
                        *z,
                        *gates,
                        *state,
                        *k_heads,
                        *v_heads,
                        *k_dim,
                        *v_dim,
                        *y,
                    );
                }
                let tail = self.recurrent_tail_absolute(*state);
                attn::ssm::gated_delta_chunked(
                    self.ctx(),
                    self.ragged_lanes(*qkv),
                    self.tensor(*z),
                    self.tensor(*gates),
                    &self.recurrent_absolute(*state),
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    &mut self.tensor(*y),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                attn::ssm::gated_delta_chunked(
                    self.ctx(),
                    self.ragged_lanes(*qkv),
                    self.tensor(*z),
                    self.tensor(*gates),
                    &tail,
                    *k_heads,
                    *v_heads,
                    *k_dim,
                    *v_dim,
                    &mut self.tensor(*y),
                )
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
                &mut self.tensor(*y),
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
                self.ragged_lanes(*mixed),
                self.tensor(*f),
                self.tensor(*b),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                &self.recurrent_absolute(*state),
                *heads,
                *head_dim,
                *norm_eps,
                *gate_floor,
                &mut self.tensor(*y),
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
            Attention::IndexRope {
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
            Attention::IndexTopk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                ratio,
                selection,
            } => index::topk(
                self.ctx(),
                self.ragged_lanes(*q),
                weights.map(|w| self.tensor(w)),
                &self.pool(*keys),
                *heads,
                *head_dim,
                *top_k,
                *ratio,
                &mut self.tensor(*selection),
            ),
            Attention::IndexKvAppend {
                k,
                keys,
                write_page,
                write_offset,
            } => index::kv_append(
                self.ctx(),
                self.ragged(*k),
                &self.pool(*keys),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::IndexBlockMean {
                boundary_pos,
                boundary_req,
                keys,
                head_dim,
                ratio,
                entries,
            } => index::block_mean(
                self.ctx(),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                &self.pool(*keys),
                *head_dim,
                *ratio,
                &mut self.tensor(*entries),
            ),
            Attention::PoolBoundaryDecode {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => pool::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                self.tensor(*row_valid),
                *ratio,
                &mut self.tensor(*boundary_pos),
                &mut self.tensor(*boundary_req),
                &mut self.tensor(*boundary_rope),
            ),
            Attention::PoolBoundaryPrefill {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => pool::boundary_prefill(
                self.ctx(),
                self.ragged(*positions),
                self.tensor(*row_valid),
                *ratio,
                &mut self.tensor(*boundary_pos),
                &mut self.tensor(*boundary_req),
                &mut self.tensor(*boundary_rope),
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
                let slabs = self.slabs(*pages);
                pool::gather(
                    self.ctx(),
                    self.tensor(*boundary_pos),
                    self.tensor(*boundary_req),
                    &self.pool(*pages),
                    *head_dim,
                    *ratio,
                    slabs.state_kv,
                    slabs.state_score,
                    ape.map_or(slabs.ape, |id| self.tensor(id)),
                    &mut self.tensor(*entries),
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
                let slabs = self.slabs(*pages);
                pool::state_write(
                    self.ctx(),
                    self.tensor(*kv),
                    self.tensor(*score),
                    &self.pool(*pages),
                    self.tensor(*write_page),
                    self.tensor(*write_offset),
                    *head_dim,
                    *ratio,
                    slabs.state_kv,
                    slabs.state_score,
                )
            }
            Attention::PoolKvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool: into,
                write_page,
                write_offset,
            } => pool::kv_append(
                self.ctx(),
                self.tensor(*entries),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                &self.pool(*into),
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
            } => pool::attention_lse(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                self.tensor(*request_of_token),
                &self.pool(*entries),
                *ratio,
                *heads,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
            Attention::PoolLseSelected { .. } => Err(kernels_cuda::Error::Unsupported {
                op: "attention.pool_lse_selected",
            }),
            Attention::Ragged {
                q,
                k,
                v,
                q_indptr,
                kv_indptr,
                head_dim,
                kv_heads: _,
                sm_scale,
                mask,
                o,
            } => {
                if self.run_index() > 0 {
                    return Ok(());
                }
                let mask = match mask {
                    model_ir::RaggedMask::None | model_ir::RaggedMask::GroupBlockDiagonal
                        if self.class_table().is_some() =>
                    {
                        let (table, count) = self.class_table().expect("checked above");
                        let classes_of = |id: model_ir::ValueId, what: &str| match self.values()
                            [id.0 as usize]
                            .def
                        {
                            model_ir::Def::Input(model_ir::RuntimeInput::Geometry {
                                kind: model_ir::GeomKind::GroupIndptr { select },
                                ..
                            }) => Ok(self.packing(id.0 as usize, select).attn_class),
                            _ => Err(kernels_cuda::Error::Backend {
                                op: "attention.ragged",
                                detail: format!(
                                    "a lane states attention classes and this attention's \
                                         {what} rows are not packed by a group selection, so \
                                         no class table applies to them"
                                ),
                            }),
                        };
                        kernels_cuda::attn_ragged::RaggedMask::ClassTable {
                            q_classes: classes_of(*q_indptr, "query")?,
                            kv_classes: classes_of(*kv_indptr, "key")?,
                            table,
                            count,
                        }
                    }
                    model_ir::RaggedMask::None | model_ir::RaggedMask::GroupBlockDiagonal => {
                        kernels_cuda::attn_ragged::RaggedMask::None
                    }
                    model_ir::RaggedMask::ReferenceSelfOnly { .. }
                    | model_ir::RaggedMask::RelativeBias { .. }
                        if self.class_table().is_some() =>
                    {
                        return Err(kernels_cuda::Error::Backend {
                            op: "attention.ragged",
                            detail: "a lane states attention classes over an attention the \
                                  model already masks (reference tags or a relative bias); \
                                  the two do not compose"
                                .to_string(),
                        });
                    }
                    model_ir::RaggedMask::ReferenceSelfOnly { q_tags, kv_tags } => {
                        kernels_cuda::attn_ragged::RaggedMask::ReferenceTags {
                            q_tags: self.fire_wide(*q_tags),
                            kv_tags: self.fire_wide(*kv_tags),
                        }
                    }
                    model_ir::RaggedMask::RelativeBias { table, max_len } => {
                        kernels_cuda::attn_ragged::RaggedMask::RelativeBias {
                            table: self.tensor(*table),
                            max_len: *max_len,
                        }
                    }
                };
                kernels_cuda::attn_ragged::ragged(
                    self.ctx(),
                    self.fire_wide(*q),
                    self.fire_wide(*k),
                    self.fire_wide(*v),
                    self.fire_wide(*q_indptr),
                    self.fire_wide(*kv_indptr),
                    *head_dim,
                    *sm_scale,
                    mask,
                    &mut self.fire_wide(*o),
                )
            }
            Attention::BlockDynConv {
                x,
                coeff,
                base,
                side,
                taps,
                group,
                y,
            } => kernels_cuda::attn::dynconv::block_dyn_conv(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*coeff),
                self.tensor(*base),
                *side,
                *taps,
                *group,
                &mut self.tensor(*y),
            ),
            Attention::SelectorWalk {
                cand,
                unary,
                hp,
                tokens,
                pred,
                succ,
                first,
                picks,
            } => kernels_cuda::attn::selector::walk(
                self.ctx(),
                self.ragged(*cand),
                self.tensor(*unary),
                hp.map(|hp| self.tensor(hp)),
                self.tensor(*tokens),
                self.tensor(*pred),
                self.tensor(*succ),
                *first,
                &mut self.tensor(*picks),
            ),
        }
    }
}

impl Run<'_> {
    pub(crate) fn dense_or_decoded(
        &self,
        op: &'static str,
        w: model_ir::ValueId,
        n: u32,
        k: u32,
    ) -> Result<kernels_cuda::Tensor, kernels_cuda::Error> {
        match self.maybe_planes(w) {
            Some((codes, scales, biases, seat)) => kernels_cuda::linear::quant::decoded_plane(
                self.ctx(),
                op,
                codes,
                scales,
                kernels_cuda::linear::quant::OffsetKind::Post,
                biases,
                model_ir::Dtype::Bf16,
                n,
                k,
                seat,
            ),
            None => Ok(self.tensor(w)),
        }
    }
}

impl Run<'_> {
    #[allow(clippy::too_many_arguments)]
    fn conv1d_extended(
        &self,
        op: &'static str,
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let x_ext = self.rs_extend(op, x, true)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        attn::ssm::causal_conv1d_chunked(
            self.ctx(),
            RaggedTensor {
                data: x_ext,
                indptr: csr,
            },
            self.tensor(weight),
            &self.recurrent_absolute(state),
            conv_width,
            dilation,
            &mut y_ext,
        )?;
        if let Some(tail) = tail {
            attn::ssm::causal_conv1d_chunked(
                self.ctx(),
                RaggedTensor {
                    data: x_ext,
                    indptr: csr,
                },
                self.tensor(weight),
                &tail,
                conv_width,
                dilation,
                &mut y_ext,
            )?;
        }
        self.rs_land(op, y_ext, y)
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_delta_extended(
        &self,
        op: &'static str,
        qkv: ValueId,
        z: ValueId,
        gates: ValueId,
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: ValueId,
    ) -> Result<(), kernels_cuda::Error> {
        let qkv_ext = self.rs_ext_of(op, qkv)?;
        let gates_ext = self.rs_ext_of(op, gates)?;
        let z_ext = self.rs_extend(op, z, false)?;
        let csr = self.rs_ext_csr(op)?;
        let mut y_ext = self.rs_out(op, y)?;
        let tail = self.recurrent_tail_absolute(state);
        attn::ssm::gated_delta_chunked(
            self.ctx(),
            RaggedTensor {
                data: qkv_ext,
                indptr: csr,
            },
            z_ext,
            gates_ext,
            &self.recurrent_absolute(state),
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            &mut y_ext,
        )?;
        if let Some(tail) = tail {
            attn::ssm::gated_delta_chunked(
                self.ctx(),
                RaggedTensor {
                    data: qkv_ext,
                    indptr: csr,
                },
                z_ext,
                gates_ext,
                &tail,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                &mut y_ext,
            )?;
        }
        self.rs_land(op, y_ext, y)?;
        self.rs_layer_done();
        Ok(())
    }
}
