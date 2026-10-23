use kernels_xla::attn;
use poem_exec::{DispatchAttn, KernelError};
use poem_ir::VoxelSegment;
use poem_ir::{Attn, IndexScore, Operands, StructKind};

use crate::run::{Run, StructSlot};

fn segment(segment: VoxelSegment) -> kernels_xla::attn::voxel::Segment {
    match segment {
        VoxelSegment::Clip => kernels_xla::attn::voxel::Segment::Lane,
        VoxelSegment::Frames(n) => kernels_xla::attn::voxel::Segment::Frames(n),
    }
}

impl DispatchAttn for Run<'_> {
    fn dispatch(&mut self, op: &Attn) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn requests(&self) -> u32 {
        u32::try_from(self.qo_indptr_host().len().saturating_sub(1)).unwrap_or(u32::MAX)
    }

    fn attention(&mut self, op: &Attn) -> Result<(), kernels_xla::Error> {
        match op {
            Attn::Voxel {
                q,
                k,
                v,
                grid,
                segment: how,
                sm_scale,
                o,
            } => crate::lower::lower(self.ctx(), |l| {
                let grid_v = l.read(self.tensor(*grid))?;
                let q_v = l.read(self.tensor(*q))?;
                let k_v = l.read(self.tensor(*k))?;
                let v_v = l.read(self.tensor(*v))?;
                let o_y = self.tensor(*o);
                let v = kernels_xla::attn::voxel::attention(
                    l.f(),
                    q_v,
                    k_v,
                    v_v,
                    grid_v,
                    segment(*how),
                    *sm_scale,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
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
                let (positions, t) = (fire.cache_rows, fire.tables);

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
                let (positions, t) = (fire.cache_rows, fire.tables);

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
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
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
            } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            // Bounded replay's floors are DeepSeek-V4.1's, served on CUDA.
            Attn::ReplayFloor { .. } | Attn::KvEnds { .. } | Attn::WindowSink { .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
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
                let mask = match mask {
                    // A lane's attention class table (engine-cuda's ragged
                    // `ClassTable` arm): each packed row's class, the table.
                    poem_ir::RaggedMask::None | poem_ir::RaggedMask::GroupBlockDiagonal
                        if self.class_table().is_some() =>
                    {
                        let (table, count) = self.class_table().expect("checked above");
                        let classes_of = |id: poem_ir::ValueId, what: &str| {
                            self.packed_classes(id)
                                .ok_or_else(|| kernels_xla::Error::Backend {
                                    op: "attn.ragged",
                                    detail: format!(
                                        "a lane states attention classes and this attention's \
                                     {what} rows are not packed by a group selection, so no \
                                     class table applies to them"
                                    ),
                                })
                        };
                        attn::ragged::RaggedMask::ClassTable {
                            q_classes: classes_of(*q_indptr, "query")?,
                            kv_classes: classes_of(*kv_indptr, "key")?,
                            table,
                            count,
                        }
                    }
                    poem_ir::RaggedMask::ReferenceSelfOnly { .. }
                    | poem_ir::RaggedMask::RelativeBias { .. }
                        if self.class_table().is_some() =>
                    {
                        return Err(kernels_xla::Error::Backend {
                            op: "attn.ragged",
                            detail: "a lane states attention classes over an attention the \
                                     model already masks (reference tags or a relative bias); \
                                     the two do not compose"
                                .to_string(),
                        });
                    }
                    poem_ir::RaggedMask::None | poem_ir::RaggedMask::GroupBlockDiagonal => {
                        attn::ragged::RaggedMask::Segments
                    }
                    poem_ir::RaggedMask::ReferenceSelfOnly { q_tags, kv_tags } => {
                        attn::ragged::RaggedMask::ReferenceTags {
                            q_tags: self.tensor(*q_tags),
                            kv_tags: self.tensor(*kv_tags),
                        }
                    }
                    poem_ir::RaggedMask::RelativeBias { table, max_len } => {
                        attn::ragged::RaggedMask::RelativeBias {
                            table: self.tensor(*table),
                            max_len: *max_len,
                        }
                    }
                };
                attn::ragged::forward(
                    self.ctx(),
                    self.tensor(*q),
                    self.tensor(*k),
                    self.tensor(*v),
                    self.tensor(*q_indptr),
                    self.tensor(*kv_indptr),
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    &mask,
                    self.tensor(*o),
                )
            }
            Attn::DecodeRel {
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
            } => attn::rel::decode_rel(
                self.ctx(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                attn::rel::RelBias {
                    bias: self.tensor(*bias),
                    extent: *extent,
                    log_floor: *log_floor,
                    log_alpha: *log_alpha,
                },
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
            ),
            Attn::PrefillRel {
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
            } => attn::rel::prefill_rel(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.pool(*cache),
                attn::rel::RelBias {
                    bias: self.tensor(*bias),
                    extent: *extent,
                    log_floor: *log_floor,
                    log_alpha: *log_alpha,
                },
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
            ),
            Attn::Masked {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads: _,
                causal,
                sm_scale,
                o,
            } => attn::arbiter::masked(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.cut_rows(self.tensor(*mask)),
                self.pool(*cache),
                *window,
                *causal,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.requests(),
            ),
            Attn::MaskedLse {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads: _,
                causal,
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
                *causal,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
                self.requests(),
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
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
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
            } => attn::arbiter::prefill_lse(
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
            ),
            Attn::Sink {
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
            Attn::MergeLse {
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
            Attn::LogitSoftcap { x, cap, x_out: _ } => {
                attn::logit_softcap(self.ctx(), self.tensor(*x), *cap)
            }
            Attn::KvAppend {
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
            Attn::KvAppendShared {
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

            Attn::MlaPlan {
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
            Attn::MlaLatents {
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
            Attn::MlaSplitQB {
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
            Attn::MlaAbsorbQ {
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
                self.dense_or_decoded(
                    "attn.mla_absorb_q",
                    *kv_b,
                    heads * (nope_dim + v_head_dim),
                    *kv_lora_rank,
                )?,
                *heads,
                *kv_lora_rank,
                *nope_dim,
                *v_head_dim,
                self.tensor(*q_latent),
            ),
            Attn::MlaAbsorbOut {
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
                self.dense_or_decoded(
                    "attn.mla_absorb_out",
                    *kv_b,
                    heads * (nope_dim + v_head_dim),
                    *kv_lora_rank,
                )?,
                *heads,
                *kv_lora_rank,
                *v_head_dim,
                *nope_dim,
                self.tensor(*o),
            ),
            Attn::MlaKvAppend {
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
            Attn::MlaDecode { lse: Some(_), .. }
            | Attn::MlaPrefill { lse: Some(_), .. }
            | Attn::MlaDecodeSelected { lse: Some(_), .. }
            | Attn::MlaPrefillSelected { lse: Some(_), .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
            Attn::MlaDecode {
                window: Some(_), ..
            }
            | Attn::MlaPrefill {
                window: Some(_), ..
            } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
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
                    (fire.cache_rows, fire.tables.request_of_token)
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
                    (fire.cache_rows, fire.tables.request_of_token)
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
                    (fire.cache_rows, fire.tables.request_of_token)
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
                    (fire.cache_rows, fire.tables.request_of_token)
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

            Attn::IndexLayernormRope {
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
            Attn::IndexRope {
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
                let fire = self.bindings();
                let (positions, request_of_token) = (fire.cache_rows, fire.tables.request_of_token);
                attn::index::topk(
                    self.ctx(),
                    self.tensor(*q),
                    weights.map(|w| self.tensor(w)),
                    self.pool(*keys),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *head_dim,
                    *top_k,
                    *ratio,
                    self.tensor(*selection),
                )
            }
            Attn::IndexKvAppend {
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

            Attn::IndexBlockMean {
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
            Attn::PoolBoundaryDecode {
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
            Attn::PoolBoundaryPrefill {
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

            Attn::PoolGather {
                boundary_pos,
                boundary_req,
                pages,
                ape,
                head_dim,
                ratio,
                entries,
            } => {
                let Some([state_kv, state_score]) = self.pool_state(*pages) else {
                    return Err(kernels_xla::Error::Unsupported {
                        op: "attn.pool_gather",
                    });
                };
                attn::pool::gather(
                    self.ctx(),
                    self.tensor(*boundary_pos),
                    self.tensor(*boundary_req),
                    self.pool(*pages),
                    *head_dim,
                    *ratio,
                    state_kv,
                    state_score,
                    ape.map(|id| self.tensor(id)),
                    self.tensor(*entries),
                )
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
                let Some([state_kv, state_score]) = self.pool_state(*pages) else {
                    return Err(kernels_xla::Error::Unsupported {
                        op: "attn.pool_state_write",
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
                    state_kv,
                    state_score,
                )
            }
            Attn::PoolKvAppend {
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

impl Run<'_> {
    /// `t`'s first rows, as many as `like` has, when it has more.
    pub(crate) fn first_rows(
        &self,
        t: kernels_xla::Tensor,
        like: poem_ir::ValueId,
    ) -> kernels_xla::Tensor {
        let rows = self.tensor(like).rows;
        if t.rows > rows {
            self.handles().cut(t, 0, rows)
        } else {
            t
        }
    }

    /// A weight read as a plain `[n, k]` array: its handle, or a bank
    /// decoded to bf16 into a fire temp first (engine-cuda
    /// `dense_or_decoded`).
    pub(crate) fn dense_or_decoded(
        &self,
        op: &'static str,
        w: poem_ir::ValueId,
        n: u32,
        k: u32,
    ) -> Result<kernels_xla::Tensor, kernels_xla::Error> {
        match self.banked(w) {
            Some(bank) => {
                let out = self.temp(n, k, poem_ir::Dtype::Bf16);
                kernels_xla::linear::decode::decoded_plane(self.ctx(), op, bank, out)?;
                Ok(out)
            }
            None => Ok(self.tensor(w)),
        }
    }
}
