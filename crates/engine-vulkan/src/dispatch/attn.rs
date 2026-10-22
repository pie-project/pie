use kernels_vulkan::attn;
use kernels_vulkan::spatial;
use poem_exec::{DispatchAttention, KernelError};
use poem_ir::{Attention, IndexScore, Operands, StructKind};

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
        q: poem_ir::ValueId,
        plan: poem_ir::ValueId,
        cache: poem_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        lse: poem_ir::ValueId,
    ) -> Result<(), kernels_vulkan::Error> {
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

    fn attention(&mut self, op: &Attention) -> Result<(), kernels_vulkan::Error> {
        match op {
            Attention::Voxel {
                q,
                k,
                v,
                grid,
                segment,
                sm_scale,
                o,
            } => spatial::attn::attention(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*grid),
                match segment {
                    poem_ir::VoxelSegment::Clip => spatial::attn::Segment::Clip,
                    poem_ir::VoxelSegment::Frames(n) => spatial::attn::Segment::Frames(*n),
                },
                *sm_scale,
                self.tensor(*o),
            ),
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
                self.tuning(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.split_scratch(),
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
                self.tuning(),
            ),

            Attention::DecodeSelected { per_head: true, .. }
            | Attention::PrefillSelected { per_head: true, .. }
            | Attention::IndexTopk {
                score: IndexScore::BlockMax,
                ..
            }
            | Attention::IndexTopk {
                score: IndexScore::PooledTokens,
                ..
            } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            // Bounded replay's floors are DeepSeek-V4.1's, served on CUDA.
            Attention::KvEnds { .. }
            | Attention::WindowSink { .. }
            | Attention::ReplayFloor { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attention::DecodeSelected {
                q,
                plan,
                selection,
                cache,
                window,
                head_dim,
                sm_scale,
                ratio,
                per_head: false,
                o,
            } => attn::decode_selected(
                self.ctx(),
                self.tuning(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.tensor(*selection),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                *ratio,
                self.tensor(*o),
            ),
            Attention::PrefillSelected {
                q,
                plan,
                selection,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                ratio,
                per_head: false,
                o,
            } => attn::prefill_selected(
                self.ctx(),
                self.tuning(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.tensor(*selection),
                self.pool(*cache),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                *ratio,
                self.tensor(*o),
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
            Attention::Ragged {
                q,
                k,
                v,
                q_indptr,
                kv_indptr,
                head_dim,
                kv_heads,
                sm_scale,
                mask,
                o,
            } => {
                if self.run_index() > 0 {
                    return Ok(());
                }
                match mask {
                    poem_ir::RaggedMask::None | poem_ir::RaggedMask::GroupBlockDiagonal => {}
                    _ => return Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
                }
                attn::ragged::forward(
                    self.ctx(),
                    self.uncut(*q),
                    self.uncut(*k),
                    self.uncut(*v),
                    self.uncut(*q_indptr),
                    self.uncut(*kv_indptr),
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    self.uncut(*o),
                )
            }
            Attention::Masked { causal: false, .. }
            | Attention::MaskedLse { causal: false, .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attention::DecodeRel { .. } | Attention::PrefillRel { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attention::Masked {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads: _,
                causal: _,
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
                self.tuning(),
            ),
            Attention::MaskedLse {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads: _,
                causal: _,
                sm_scale,
                o,
                lse,
            } => attn::arbiter::masked_lse(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.cut_rows(self.tensor(*mask)),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
                self.requests(),
                self.tuning(),
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
                self.tuning(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
                self.split_scratch(),
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
                // This backend splits no prompt's rows, so a replay floor stays 0.
                floor: _,
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
                    self.tuning(),
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

            // The per-head log-sum-exp a sink rescales by is landed on CUDA alone.
            Attention::MlaDecode { lse: Some(_), .. }
            | Attention::MlaPrefill { lse: Some(_), .. }
            | Attention::MlaDecodeSelected { lse: Some(_), .. }
            | Attention::MlaPrefillSelected { lse: Some(_), .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attention::MlaDecode {
                window: Some(_), ..
            }
            | Attention::MlaPrefill {
                window: Some(_), ..
            } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            Attention::MlaDecode {
                q,
                plan: _,
                q_pe,
                cache,
                window: None,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse: _,
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
                window: None,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse: _,
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
                lse: _,
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
                lse: _,
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
                score: IndexScore::Pooled,
                selection,
            } => {
                let Some(scores) = self.index_scores() else {
                    return Err(kernels_vulkan::Error::Unsupported {
                        op: "attention.index_topk",
                    });
                };
                let fire = self.bindings();
                let (positions, request_of_token) = (fire.positions, fire.tables.request_of_token);
                attn::index::topk(
                    self.ctx(),
                    self.tensor(*q),
                    weights.map(|w| self.tensor(w)),
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

            Attention::IndexBlockMean {
                boundary_pos,
                boundary_req,
                keys,
                head_dim,
                ratio,
                entries,
            } => attn::index::block_mean(
                self.ctx(),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*keys),
                *head_dim,
                *ratio,
                self.tensor(*entries),
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
                self.cut_rows(self.bindings().tables.request_of_token),
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
                self.cut_rows(self.bindings().tables.request_of_token),
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
                    return Err(kernels_vulkan::Error::Unsupported {
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
                    return Err(kernels_vulkan::Error::Unsupported {
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
