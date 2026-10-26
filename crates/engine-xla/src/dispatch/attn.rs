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
            Attn::ScoreCapture {
                q,
                plan,
                cache,
                head_dim,
                kv_heads,
                sm_scale,
                observe,
                kv_max,
                scores,
            } => {
                let pool = *self.pool(*cache);
                let span = self.window().span;
                let page_indptr =
                    (self.handles()).cut(pool.page_indptr, span.lane_offset, span.lanes + 1);
                let q = self.ragged(*q);
                let positions = self.prefill_plan(*plan).positions;
                let scores = self.tensor(*scores);
                let requests = self.requests();
                let (head_dim, kv_heads, sm_scale, observe, kv_max) =
                    (*head_dim, *kv_heads, *sm_scale, *observe, *kv_max);
                crate::lower::lower(self.ctx(), |l| {
                    let page_indptr = l.read(page_indptr)?;
                    let page_indices = l.read(pool.page_indices)?;
                    let indptr = l.read(q.indptr)?;
                    let positions = l.read(positions)?;
                    let data = l.read(q.data)?;
                    let keys = l.read(pool.keys)?;
                    let slab = l.read(scores)?;
                    let heads = data.width / head_dim.max(1);
                    // The slab as a `kv_max` row per lane and head.
                    let slab = kernels_xla::tensor::Operand {
                        val: l.f().reshape(
                            slab.val,
                            &[i64::from(scores.rows * heads), i64::from(kv_max)],
                        )?,
                        rows: scores.rows * heads,
                        width: kv_max,
                        dtype: slab.dtype,
                    };
                    let tuning = l.tuning();
                    let v = attn::score::capture(
                        l.f(),
                        kernels_xla::tensor::RaggedTensor { data, indptr },
                        positions,
                        keys,
                        page_indptr,
                        page_indices,
                        &pool,
                        None,
                        head_dim,
                        kv_heads,
                        sm_scale,
                        observe,
                        0,
                        heads,
                        0,
                        kv_max,
                        requests,
                        slab,
                        tuning,
                    )?;
                    l.land(scores, v)
                })
            }
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
                    self.grid_base(*grid, *q),
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
                selection: None,
                bias: None,
                selected: None,
                rel: None,
                lse: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::decode(
                    self.tensor(*q),
                    self.decode_plan(*plan),
                    *window,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }

            Attn::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                mask: None,
                selection: None,
                bias: None,
                floor: None,
                causal: true,
                selected: None,
                rel: None,
                lse: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::arbiter::prefill(
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    &pool,
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    self.tensor(*o),
                    self.requests(),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }

            Attn::Decode {
                selection: Some(_),
                bias: None,
                selected: Some(poem_ir::Blocks { per_head: true, .. }),
                lse: None,
                ..
            }
            | Attn::Prefill {
                mask: None,
                selection: Some(_),
                bias: None,
                selected: Some(poem_ir::Blocks { per_head: true, .. }),
                lse: None,
                ..
            }
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
            Attn::Decode {
                q,
                plan,
                selection: Some(selection),
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                bias: None,
                selected:
                    Some(poem_ir::Blocks {
                        ratio,
                        per_head: false,
                    }),
                rel: None,
                lse: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::decode_selected(
                    self.tensor(*q),
                    self.decode_plan(*plan),
                    self.tensor(*selection),
                    *window,
                    *head_dim,
                    *sm_scale,
                    *ratio,
                    self.tensor(*o),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Prefill {
                q,
                plan,
                selection: Some(selection),
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                mask: None,
                bias: None,
                floor: None,
                causal: true,
                selected:
                    Some(poem_ir::Blocks {
                        ratio,
                        per_head: false,
                    }),
                rel: None,
                lse: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::prefill_selected(
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    self.tensor(*selection),
                    &pool,
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    *ratio,
                    self.tensor(*o),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Dense {
                q,
                k,
                v,
                segments,
                head_dim,
                sm_scale,
                o,
            } => crate::lower::attn::dense::bidirectional(
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
                let stated = self.group_mask(*q_indptr, *kv_indptr);
                let mask = match (mask, stated) {
                    (
                        poem_ir::RaggedMask::None | poem_ir::RaggedMask::GroupBlockDiagonal,
                        Some(stated),
                    ) => attn::ragged::RaggedMask::Custom {
                        bits: stated.bits,
                        indptr: stated.indptr,
                    },
                    (poem_ir::RaggedMask::RelativeBias { .. }, Some(_)) => {
                        return Err(kernels_xla::Error::Backend {
                            op: "attn.ragged",
                            detail: "a lane states an attention mask over an attention the \
                                     model already biases (a relative bias); the two do not \
                                     compose"
                                .to_string(),
                        });
                    }
                    (poem_ir::RaggedMask::None | poem_ir::RaggedMask::GroupBlockDiagonal, None) => {
                        attn::ragged::RaggedMask::Segments
                    }
                    (poem_ir::RaggedMask::RelativeBias { table, max_len }, None) => {
                        attn::ragged::RaggedMask::RelativeBias {
                            table: self.tensor(*table),
                            max_len: *max_len,
                        }
                    }
                };
                crate::lower::attn::ragged::forward(
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
            Attn::Decode {
                q,
                plan,
                cache,
                bias: Some(bias),
                window,
                head_dim,
                sm_scale,
                o,
                selection: None,
                selected: None,
                rel:
                    Some(poem_ir::Relative {
                        extent,
                        log_floor,
                        log_alpha,
                    }),
                lse: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::rel::decode_rel(
                    self.tensor(*q),
                    self.decode_plan(*plan),
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
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Prefill {
                q,
                plan,
                cache,
                bias: Some(bias),
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                mask: None,
                selection: None,
                floor: None,
                causal: true,
                selected: None,
                rel:
                    Some(poem_ir::Relative {
                        extent,
                        log_floor,
                        log_alpha,
                    }),
                lse: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::rel::prefill_rel(
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    &pool,
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
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Prefill {
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
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
                lse: None,
            } if mask.is_some() || !*causal => {
                let pool = *self.pool(*cache);
                let call = attn::arbiter::masked(
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    self.cut_rows(self.mask_bits(*mask)),
                    *window,
                    *causal,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                    self.requests(),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Prefill {
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
                lse: Some(lse),
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
            } if mask.is_some() || !*causal => {
                let pool = *self.pool(*cache);
                let call = attn::arbiter::masked_lse(
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    self.cut_rows(self.mask_bits(*mask)),
                    *window,
                    *causal,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                    self.tensor(*lse),
                    self.requests(),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse: Some(lse),
                selection: None,
                bias: None,
                selected: None,
                rel: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::decode_lse(
                    self.tensor(*q),
                    self.decode_plan(*plan),
                    *window,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                    self.tensor(*lse),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                lse: Some(lse),
                // This backend splits no prompt's rows, so a replay floor stays 0.
                floor: _,
                mask: None,
                selection: None,
                bias: None,
                causal: true,
                selected: None,
                rel: None,
            } => {
                let pool = *self.pool(*cache);
                let call = attn::arbiter::prefill_lse(
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    &pool,
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    self.tensor(*o),
                    self.tensor(*lse),
                    self.requests(),
                )?;
                crate::lower::attn::attend(self.ctx(), call, pool)
            }
            Attn::Sink {
                o,
                lse,
                sink,
                head_dim,
                o_out: _,
            } => crate::lower::attn::sink(
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
            } => crate::lower::attn::merge_lse(
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
            Attn::KvAppend {
                k,
                v: Some(v),
                cache,
                write_page,
                write_offset,
            } => crate::lower::attn::kv_append(
                self.ctx(),
                self.tensor(*k),
                self.tensor(*v),
                self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attn::KvAppend {
                k: plane,
                v: None,
                cache,
                write_page,
                write_offset,
            } => crate::lower::attn::kv_append_shared(
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
            } => crate::lower::attn::mla::latents(
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
            } => crate::lower::attn::mla::latents_rope(
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
            } => crate::lower::attn::mla::split_q_b(
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
            } => crate::lower::attn::mla::absorb_q(
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
            } => crate::lower::attn::mla::absorb_out(
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
            } => crate::lower::attn::mla::kv_append(
                self.ctx(),
                self.tensor(*kv_c),
                self.tensor(*k_pe),
                self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),

            // The per-head log-sum-exp a sink rescales by is landed on CUDA alone.
            Attn::MlaDecode {
                lse: Some(_),
                selection: None,
                ..
            }
            | Attn::MlaPrefill {
                lse: Some(_),
                selection: None,
                ..
            }
            | Attn::MlaDecode {
                lse: Some(_),
                selection: Some(_),
                ..
            }
            | Attn::MlaPrefill {
                lse: Some(_),
                selection: Some(_),
                ..
            } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Attn::MlaDecode {
                window: Some(_),
                selection: None,
                ..
            }
            | Attn::MlaPrefill {
                window: Some(_),
                selection: None,
                ..
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
                selection: None,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.cache_rows, fire.tables.request_of_token)
                };
                let call = attn::mla::attention_decode(
                    self.tensor(*q),
                    self.tensor(*q_pe),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                );
                crate::lower::attn::mla::flash(self.ctx(), call, *self.pool(*cache))
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
                selection: None,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.cache_rows, fire.tables.request_of_token)
                };
                let call = attn::mla::attention_prefill(
                    self.ragged(*q),
                    self.tensor(*q_pe),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                );
                crate::lower::attn::mla::flash(self.ctx(), call, *self.pool(*cache))
            }
            Attn::MlaDecode {
                q,
                plan: _,
                q_pe,
                selection: Some(selection),
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse: _,
                window: None,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.cache_rows, fire.tables.request_of_token)
                };
                let call = attn::mla::attention_decode_selected(
                    self.tensor(*q),
                    self.tensor(*q_pe),
                    self.tensor(*selection),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                );
                crate::lower::attn::mla::flash(self.ctx(), call, *self.pool(*cache))
            }
            Attn::MlaPrefill {
                q,
                plan: _,
                q_pe,
                selection: Some(selection),
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse: _,
                window: None,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.cache_rows, fire.tables.request_of_token)
                };
                let call = attn::mla::attention_prefill_selected(
                    self.ragged(*q),
                    self.tensor(*q_pe),
                    self.tensor(*selection),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *kv_lora_rank,
                    *sm_scale,
                    self.tensor(*o),
                );
                crate::lower::attn::mla::flash(self.ctx(), call, *self.pool(*cache))
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
            } => crate::lower::attn::index::layernorm_rope(
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
            } => crate::lower::attn::index::rope(
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
                crate::lower::attn::index::topk(
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
            } => crate::lower::attn::index::kv_append(
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
            } => crate::lower::attn::index::block_mean(
                self.ctx(),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*keys),
                *head_dim,
                *ratio,
                self.tensor(*entries),
            ),
            Attn::PoolBoundary {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
                decode: true,
            } => crate::lower::attn::pool::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                self.cut_rows(self.bindings().tables.request_of_token),
                self.tensor(*row_valid),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.tensor(*boundary_rope),
            ),
            Attn::PoolBoundary {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
                decode: false,
            } => crate::lower::attn::pool::boundary_prefill(
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
                crate::lower::attn::pool::gather(
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
                crate::lower::attn::pool::state_write(
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
                write_page: _,
                write_offset: _,
            } => crate::lower::attn::pool::kv_append(
                self.ctx(),
                self.tensor(*entries),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*into),
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
                selection: None,
                top_k: None,
            } => {
                let pool = *self.pool(*entries);
                let call = attn::pool::attention_lse(
                    self.tensor(*q),
                    self.tensor(*positions),
                    self.tensor(*request_of_token),
                    &pool,
                    *ratio,
                    *heads,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                    self.tensor(*lse),
                )?;
                crate::lower::attn::pool::lse(self.ctx(), call, pool)
            }
            Attn::PoolLse {
                q,
                positions,
                request_of_token,
                selection: Some(selection),
                entries,
                ratio,
                top_k: Some(top_k),
                heads,
                head_dim,
                sm_scale,
                o,
                lse,
            } => {
                let pool = *self.pool(*entries);
                let call = attn::pool::attention_lse_selected(
                    self.tensor(*q),
                    self.tensor(*positions),
                    self.tensor(*request_of_token),
                    self.tensor(*selection),
                    &pool,
                    *ratio,
                    *top_k,
                    *heads,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                    self.tensor(*lse),
                )?;
                crate::lower::attn::pool::lse(self.ctx(), call, pool)
            }
            // A selection, a window, a relative bias and a mask no kernel here
            // combines.
            Attn::Decode { .. }
            | Attn::Prefill { .. }
            | Attn::MlaDecode { .. }
            | Attn::MlaPrefill { .. }
            | Attn::PoolLse { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
        }
    }
}

impl Run<'_> {
    /// `t`'s first rows, as many as `like` has, when it has more.
    pub(crate) fn first_rows(
        &self,
        t: crate::handle::Tensor,
        like: poem_ir::ValueId,
    ) -> crate::handle::Tensor {
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
    ) -> Result<crate::handle::Tensor, kernels_xla::Error> {
        match self.banked(w) {
            Some(bank) => {
                let out = self.temp(n, k, poem_ir::Dtype::Bf16);
                crate::lower::linear::decoded_plane(self.ctx(), op, bank, out)?;
                Ok(out)
            }
            None => Ok(self.tensor(w)),
        }
    }
}
