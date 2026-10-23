use kernels_vulkan::attn;
use kernels_vulkan::spatial;
use poem_exec::{DispatchAttn, KernelError};
use poem_ir::{Attn, IndexScore, Operands, StructKind};

use crate::run::{Run, StructSlot};

impl DispatchAttn for Run<'_> {
    fn dispatch(&mut self, op: &Attn) -> Result<(), KernelError> {
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
        kernels_slang::attn::score::capture(
            &self.slang(),
            crate::encode::slang_ragged(self.ragged(q)),
            crate::encode::slang_tensor(self.prefill_plan(plan).positions),
            &crate::encode::slang_pool(self.pool(cache)),
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
            crate::encode::slang_tensor(seat.slab),
        )
        .map_err(crate::encode::from_slang)
    }

    fn attention(&mut self, op: &Attn) -> Result<(), kernels_vulkan::Error> {
        match op {
            Attn::Voxel {
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
            Attn::PlanDecode {
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
                        "`attn.plan_decode` defines a {other:?}, which is no \
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

            Attn::PlanPrefill {
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
                        "`attn.plan_prefill` defines a {other:?}, which is no \
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
            Attn::Decode {
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

            Attn::Prefill {
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

            Attn::DecodeSelected { per_head: true, .. }
            | Attn::PrefillSelected { per_head: true, .. }
            | Attn::IndexTopk {
                score: IndexScore::BlockMax,
                ..
            }
            | Attn::IndexTopk {
                score: IndexScore::PooledTokens,
                ..
            } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            // Bounded replay's floors are DeepSeek-V4.1's, served on CUDA.
            Attn::KvEnds { .. } | Attn::WindowSink { .. } | Attn::ReplayFloor { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attn::DecodeSelected {
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
            Attn::PrefillSelected {
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
            Attn::Dense {
                q,
                k,
                v,
                segments,
                head_dim,
                sm_scale,
                o,
            } => kernels_slang::attn::dense::bidirectional(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*v)),
                crate::encode::slang_tensor(self.tensor(*segments)),
                *head_dim,
                *sm_scale,
                crate::encode::slang_tensor(self.tensor(*o)),
            )
            .map_err(crate::encode::from_slang),
            Attn::Ragged {
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
                kernels_slang::attn::dense::ragged(
                    &self.slang(),
                    crate::encode::slang_tensor(self.uncut(*q)),
                    crate::encode::slang_tensor(self.uncut(*k)),
                    crate::encode::slang_tensor(self.uncut(*v)),
                    crate::encode::slang_tensor(self.uncut(*q_indptr)),
                    crate::encode::slang_tensor(self.uncut(*kv_indptr)),
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    crate::encode::slang_tensor(self.uncut(*o)),
                )
                .map_err(crate::encode::from_slang)
            }
            Attn::Masked { causal: false, .. } | Attn::MaskedLse { causal: false, .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attn::DecodeRel { .. } | Attn::PrefillRel { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attn::Masked {
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
            Attn::MaskedLse {
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
            Attn::DecodeLse {
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
            Attn::PrefillLse {
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
            Attn::Sink {
                o,
                lse,
                sink,
                head_dim,
                o_out: _,
            } => kernels_slang::attn::merge::sink(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*o)),
                crate::encode::slang_tensor(self.tensor(*lse)),
                crate::encode::slang_tensor(self.tensor(*sink)),
                *head_dim,
            )
            .map_err(crate::encode::from_slang),
            Attn::MergeLse {
                o1,
                lse1,
                o2,
                lse2,
                heads,
                head_dim,
                o,
                lse,
            } => kernels_slang::attn::merge::merge_lse(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*o1)),
                crate::encode::slang_tensor(self.tensor(*lse1)),
                crate::encode::slang_tensor(self.tensor(*o2)),
                crate::encode::slang_tensor(self.tensor(*lse2)),
                *heads,
                *head_dim,
                crate::encode::slang_tensor(self.tensor(*o)),
                crate::encode::slang_tensor(self.tensor(*lse)),
            )
            .map_err(crate::encode::from_slang),
            Attn::LogitSoftcap { x, cap, x_out: _ } => kernels_slang::attn::merge::logit_softcap(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                *cap,
            )
            .map_err(crate::encode::from_slang),
            Attn::KvAppend {
                k,
                v,
                cache,
                write_page,
                write_offset,
            } => kernels_slang::attn::kv_append::kv_append(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*v)),
                &crate::encode::slang_pool(self.pool(*cache)),
                crate::encode::slang_tensor(self.tensor(*write_page)),
                crate::encode::slang_tensor(self.tensor(*write_offset)),
            )
            .map_err(crate::encode::from_slang),
            Attn::KvAppendShared {
                plane,
                cache,
                write_page,
                write_offset,
            } => kernels_slang::attn::kv_append::kv_append_shared(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*plane)),
                &crate::encode::slang_pool(self.pool(*cache)),
                crate::encode::slang_tensor(self.tensor(*write_page)),
                crate::encode::slang_tensor(self.tensor(*write_offset)),
            )
            .map_err(crate::encode::from_slang),

            Attn::MlaPlan {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                heads: _,
                kv_lora_rank: _,
                plan,
            } => {
                let built = kernels_slang::attn::mla::plan(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*kv_indptr)),
                    crate::encode::slang_tensor(self.tensor(*kv_indices)),
                    crate::encode::slang_tensor(self.tensor(*last_page_len)),
                    crate::encode::slang_tensor(self.tensor(*kv_len)),
                )
                .map_err(crate::encode::from_slang)?;
                self.put(*plan, StructSlot::Mla(built));
                Ok(())
            }
            Attn::MlaLatents {
                kv_a,
                weight,
                eps,
                kv_lora_rank,
                kv_c,
                k_pe,
            } => kernels_slang::attn::mla::latents(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*kv_a)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *eps,
                *kv_lora_rank,
                crate::encode::slang_tensor(self.tensor(*kv_c)),
                crate::encode::slang_tensor(self.tensor(*k_pe)),
            )
            .map_err(crate::encode::from_slang),
            Attn::MlaLatentsRope {
                kv_a,
                positions,
                weight,
                eps,
                kv_lora_rank,
                rope_dim,
                theta,
                kv_c,
                k_pe,
            } => kernels_slang::attn::mla::latents_rope(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*kv_a)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                *eps,
                *kv_lora_rank,
                *rope_dim,
                *theta,
                crate::encode::slang_tensor(self.tensor(*kv_c)),
                crate::encode::slang_tensor(self.tensor(*k_pe)),
            )
            .map_err(crate::encode::from_slang),
            Attn::MlaSplitQB {
                q_b,
                heads,
                nope_dim,
                rope_dim,
                q_nope,
                q_pe,
            } => kernels_slang::attn::mla::split_q_b(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q_b)),
                *heads,
                *nope_dim,
                *rope_dim,
                crate::encode::slang_tensor(self.tensor(*q_nope)),
                crate::encode::slang_tensor(self.tensor(*q_pe)),
            )
            .map_err(crate::encode::from_slang),
            Attn::MlaAbsorbQ {
                q_nope,
                kv_b,
                heads,
                kv_lora_rank,
                nope_dim,
                v_head_dim,
                q_latent,
            } => kernels_slang::attn::mla::absorb_q(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q_nope)),
                crate::encode::slang_tensor(self.tensor(*kv_b)),
                *heads,
                *kv_lora_rank,
                *nope_dim,
                *v_head_dim,
                crate::encode::slang_tensor(self.tensor(*q_latent)),
            )
            .map_err(crate::encode::from_slang),
            Attn::MlaAbsorbOut {
                latent,
                kv_b,
                heads,
                kv_lora_rank,
                v_head_dim,
                nope_dim,
                o,
            } => kernels_slang::attn::mla::absorb_out(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*latent)),
                crate::encode::slang_tensor(self.tensor(*kv_b)),
                *heads,
                *kv_lora_rank,
                *v_head_dim,
                *nope_dim,
                crate::encode::slang_tensor(self.tensor(*o)),
            )
            .map_err(crate::encode::from_slang),
            Attn::MlaKvAppend {
                kv_c,
                k_pe,
                cache,
                write_page,
                write_offset,
            } => kernels_slang::attn::mla::kv_append(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*kv_c)),
                crate::encode::slang_tensor(self.tensor(*k_pe)),
                &crate::encode::slang_pool(self.pool(*cache)),
                crate::encode::slang_tensor(self.tensor(*write_page)),
                crate::encode::slang_tensor(self.tensor(*write_offset)),
            )
            .map_err(crate::encode::from_slang),

            // The per-head log-sum-exp a sink rescales by is landed on CUDA alone.
            Attn::MlaDecode { lse: Some(_), .. }
            | Attn::MlaPrefill { lse: Some(_), .. }
            | Attn::MlaDecodeSelected { lse: Some(_), .. }
            | Attn::MlaPrefillSelected { lse: Some(_), .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
            Attn::MlaDecode {
                window: Some(_), ..
            }
            | Attn::MlaPrefill {
                window: Some(_), ..
            } => Err(kernels_vulkan::Error::Unsupported { op: op.name() }),
            Attn::MlaDecode {
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
                kernels_slang::attn::mla::attention_decode(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*q)),
                    crate::encode::slang_tensor(self.tensor(*q_pe)),
                    &crate::encode::slang_pool(self.pool(*cache)),
                    crate::encode::slang_tensor(self.cut_rows(positions)),
                    crate::encode::slang_tensor(self.cut_rows(request_of_token)),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    crate::encode::slang_tensor(self.tensor(*o)),
                )
                .map_err(crate::encode::from_slang)
            }
            Attn::MlaPrefill {
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
                kernels_slang::attn::mla::attention_prefill(
                    &self.slang(),
                    crate::encode::slang_ragged(self.ragged(*q)),
                    crate::encode::slang_tensor(self.tensor(*q_pe)),
                    &crate::encode::slang_pool(self.pool(*cache)),
                    crate::encode::slang_tensor(self.cut_rows(positions)),
                    crate::encode::slang_tensor(self.cut_rows(request_of_token)),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    crate::encode::slang_tensor(self.tensor(*o)),
                )
                .map_err(crate::encode::from_slang)
            }
            Attn::MlaDecodeSelected {
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
                kernels_slang::attn::mla::attention_decode_selected(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*q)),
                    crate::encode::slang_tensor(self.tensor(*q_pe)),
                    crate::encode::slang_tensor(self.tensor(*selection)),
                    &crate::encode::slang_pool(self.pool(*cache)),
                    crate::encode::slang_tensor(self.cut_rows(positions)),
                    crate::encode::slang_tensor(self.cut_rows(request_of_token)),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    crate::encode::slang_tensor(self.tensor(*o)),
                )
                .map_err(crate::encode::from_slang)
            }
            Attn::MlaPrefillSelected {
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
                kernels_slang::attn::mla::attention_prefill_selected(
                    &self.slang(),
                    crate::encode::slang_ragged(self.ragged(*q)),
                    crate::encode::slang_tensor(self.tensor(*q_pe)),
                    crate::encode::slang_tensor(self.tensor(*selection)),
                    &crate::encode::slang_pool(self.pool(*cache)),
                    crate::encode::slang_tensor(self.cut_rows(positions)),
                    crate::encode::slang_tensor(self.cut_rows(request_of_token)),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    crate::encode::slang_tensor(self.tensor(*o)),
                )
                .map_err(crate::encode::from_slang)
            }

            Attn::IndexLayernormRope {
                k,
                positions,
                weight,
                bias,
                eps,
                rope_dim,
                theta,
                k_out: _,
            } => kernels_slang::attn::index::layernorm_rope(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                crate::encode::slang_tensor(self.tensor(*weight)),
                crate::encode::slang_tensor(self.tensor(*bias)),
                *eps,
                *rope_dim,
                *theta,
            )
            .map_err(crate::encode::from_slang),
            Attn::IndexRope {
                q,
                positions,
                heads,
                head_dim,
                rope_dim,
                theta,
                q_out: _,
            } => kernels_slang::attn::index::rope(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                *heads,
                *head_dim,
                *rope_dim,
                *theta,
            )
            .map_err(crate::encode::from_slang),

            Attn::IndexTopk {
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
                        op: "attn.index_topk",
                    });
                };
                let fire = self.bindings();
                let (positions, request_of_token) = (fire.positions, fire.tables.request_of_token);
                kernels_slang::attn::index::topk(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*q)),
                    weights.map(|w| crate::encode::slang_tensor(self.tensor(w))),
                    &crate::encode::slang_pool(self.pool(*keys)),
                    crate::encode::slang_tensor(self.cut_rows(positions)),
                    crate::encode::slang_tensor(self.cut_rows(request_of_token)),
                    crate::encode::slang_tensor(scores),
                    *heads,
                    *head_dim,
                    *top_k,
                    *ratio,
                    crate::encode::slang_tensor(self.tensor(*selection)),
                )
                .map_err(crate::encode::from_slang)
            }
            Attn::IndexKvAppend {
                k,
                keys,
                write_page,
                write_offset,
            } => kernels_slang::attn::index::kv_append(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*k)),
                &crate::encode::slang_pool(self.pool(*keys)),
                crate::encode::slang_tensor(self.tensor(*write_page)),
                crate::encode::slang_tensor(self.tensor(*write_offset)),
            )
            .map_err(crate::encode::from_slang),

            Attn::IndexBlockMean {
                boundary_pos,
                boundary_req,
                keys,
                head_dim,
                ratio,
                entries,
            } => kernels_slang::attn::index::block_mean(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*boundary_pos)),
                crate::encode::slang_tensor(self.tensor(*boundary_req)),
                &crate::encode::slang_pool(self.pool(*keys)),
                *head_dim,
                *ratio,
                crate::encode::slang_tensor(self.tensor(*entries)),
            )
            .map_err(crate::encode::from_slang),
            Attn::PoolBoundaryDecode {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => kernels_slang::attn::pool::boundary_decode(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*positions)),
                crate::encode::slang_tensor(self.cut_rows(self.bindings().tables.request_of_token)),
                crate::encode::slang_tensor(self.tensor(*row_valid)),
                *ratio,
                crate::encode::slang_tensor(self.tensor(*boundary_pos)),
                crate::encode::slang_tensor(self.tensor(*boundary_req)),
                crate::encode::slang_tensor(self.tensor(*boundary_rope)),
            )
            .map_err(crate::encode::from_slang),
            Attn::PoolBoundaryPrefill {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => kernels_slang::attn::pool::boundary_prefill(
                &self.slang(),
                crate::encode::slang_ragged(self.ragged(*positions)),
                crate::encode::slang_tensor(self.cut_rows(self.bindings().tables.request_of_token)),
                crate::encode::slang_tensor(self.tensor(*row_valid)),
                *ratio,
                crate::encode::slang_tensor(self.tensor(*boundary_pos)),
                crate::encode::slang_tensor(self.tensor(*boundary_req)),
                crate::encode::slang_tensor(self.tensor(*boundary_rope)),
            )
            .map_err(crate::encode::from_slang),

            Attn::PoolGather {
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
                        op: "attn.pool_gather",
                    });
                };
                kernels_slang::attn::pool::gather(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*boundary_pos)),
                    crate::encode::slang_tensor(self.tensor(*boundary_req)),
                    &crate::encode::slang_pool(self.pool(*pages)),
                    *head_dim,
                    *ratio,
                    crate::encode::slang_tensor(state.state_kv),
                    crate::encode::slang_tensor(state.state_score),
                    ape.map(|id| crate::encode::slang_tensor(self.tensor(id))),
                    crate::encode::slang_tensor(self.tensor(*entries)),
                )
                .map_err(crate::encode::from_slang)
            }

            Attn::PoolStateWrite {
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
                        op: "attn.pool_state_write",
                    });
                };
                kernels_slang::attn::pool::state_write(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*kv)),
                    crate::encode::slang_tensor(self.tensor(*score)),
                    &crate::encode::slang_pool(self.pool(*pages)),
                    crate::encode::slang_tensor(self.tensor(*write_page)),
                    crate::encode::slang_tensor(self.tensor(*write_offset)),
                    *head_dim,
                    *ratio,
                    crate::encode::slang_tensor(state.state_kv),
                    crate::encode::slang_tensor(state.state_score),
                )
                .map_err(crate::encode::from_slang)
            }
            Attn::PoolKvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool: into,
                write_page,
                write_offset,
            } => kernels_slang::attn::pool::kv_append(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*entries)),
                crate::encode::slang_tensor(self.tensor(*boundary_pos)),
                crate::encode::slang_tensor(self.tensor(*boundary_req)),
                &crate::encode::slang_pool(self.pool(*into)),
                crate::encode::slang_tensor(self.tensor(*write_page)),
                crate::encode::slang_tensor(self.tensor(*write_offset)),
            )
            .map_err(crate::encode::from_slang),
            Attn::PoolLse {
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
            } => kernels_slang::attn::pool::attention_lse(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                crate::encode::slang_tensor(self.tensor(*request_of_token)),
                &crate::encode::slang_pool(self.pool(*entries)),
                *ratio,
                *heads,
                *head_dim,
                *sm_scale,
                crate::encode::slang_tensor(self.tensor(*o)),
                crate::encode::slang_tensor(self.tensor(*lse)),
            )
            .map_err(crate::encode::from_slang),
            Attn::PoolLseSelected {
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
            } => kernels_slang::attn::pool::attention_lse_selected(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*positions)),
                crate::encode::slang_tensor(self.tensor(*request_of_token)),
                crate::encode::slang_tensor(self.tensor(*selection)),
                &crate::encode::slang_pool(self.pool(*entries)),
                *ratio,
                *top_k,
                *heads,
                *head_dim,
                *sm_scale,
                crate::encode::slang_tensor(self.tensor(*o)),
                crate::encode::slang_tensor(self.tensor(*lse)),
            )
            .map_err(crate::encode::from_slang),
        }
    }
}
