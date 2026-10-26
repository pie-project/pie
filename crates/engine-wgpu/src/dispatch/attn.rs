use kernels_slang::attn::sdpa;
use poem_exec::{DispatchAttn, KernelError};
use poem_ir::{Attn, IndexScore, Operands, StructKind};

use crate::run::{Run, StructSlot};

/// Storage buffers a split decode binds (its LSE form one more); a device
/// offering fewer (SwiftShader: 10) walks each row whole.
const SPLIT_STORAGE_BUFFERS: u32 = 12;

impl DispatchAttn for Run<'_> {
    fn dispatch(&mut self, op: &Attn) -> Result<(), KernelError> {
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// How this device runs paged attention: the adapter's tier (its
    /// subgroup prefill tile where every subgroup is 32 lanes), a decode
    /// workgroup of 256 invocations (128 measured 26-61% slower through
    /// wgpu on an RTX 4090) and prefill tiles of 16 rows of 16 lanes, the
    /// 256 invocations WebGPU's default workgroup holds.
    fn sdpa_tactic(&self) -> sdpa::Tactic {
        let tuning = self.tuning();
        sdpa::Tactic {
            tier: self.tier(),
            narrow: false,
            decode_threads: 256,
            tile_rows: 16,
            tile_lanes: 16,
            split_groups: tuning.sdpa_split_groups(),
            split_min_keys: tuning.sdpa_split_min_keys(),
        }
    }

    fn prefill_tables(
        &self,
        plan: poem_ir::ValueId,
        mask: Option<kernels_slang::Tensor>,
    ) -> sdpa::Tables {
        let plan = *self.prefill_plan(plan);
        sdpa::Tables {
            mask: mask.unwrap_or(plan.mask),
            ..plan
        }
    }

    /// Refuses a pool whose row does not hold the stated kv heads.
    fn kv_heads_agree(
        op: &'static str,
        pool: &kernels_slang::KvPool,
        kv_heads: u32,
    ) -> Result<(), kernels_slang::Error> {
        if pool.head_stride == 0 || pool.seq_stride / pool.head_stride != u64::from(kv_heads) {
            return Err(kernels_slang::Error::Backend {
                op,
                detail: format!(
                    "the stated kv head count {kv_heads} is not the one the pool row's strides \
                     spell ({} / {})",
                    pool.seq_stride, pool.head_stride
                ),
            });
        }
        Ok(())
    }

    /// Prefill attention: query rows tiled a workgroup of one head where a
    /// request brings enough of them (`kernel.sdpa.tile_rows_per_request`),
    /// else each row walked as a decode row.
    #[allow(clippy::too_many_arguments)]
    fn prefill_attention(
        &self,
        q: poem_ir::ValueId,
        plan: poem_ir::ValueId,
        mask: Option<kernels_slang::Tensor>,
        cache: poem_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: kernels_slang::Tensor,
        lse: Option<kernels_slang::Tensor>,
    ) -> Result<(), kernels_slang::Error> {
        const OP: &str = "attn.prefill";
        let q = self.ragged(q).data;
        let pool = self.pool(cache);
        Self::kv_heads_agree(OP, pool, kv_heads)?;
        if let Some(mask) = mask
            && mask.dtype != poem_ir::Dtype::U8
        {
            return Err(kernels_slang::Error::Backend {
                op: OP,
                detail: format!(
                    "the mask this op states is {:?}, and the shader reads packed u8 mask planes",
                    mask.dtype
                ),
            });
        }
        let tactic = self.sdpa_tactic();
        let tables = self.prefill_tables(plan, mask);
        let (q, pool, o) = (q, *pool, o);
        let tiles = q.rows / self.requests().max(1) >= self.tuning().sdpa_tile_rows_per_request();
        if tiles {
            sdpa::prefill(
                self.ctx, tactic, q, &tables, &pool, window, head_dim, sm_scale, o, lse,
            )
        } else {
            sdpa::decode(
                self.ctx, tactic, q, &tables, &pool, window, head_dim, sm_scale, o, lse, None,
            )
        }
    }

    /// Decode attention, split into key ranges where the rows are too few
    /// and the device binds the split's planes.
    #[allow(clippy::too_many_arguments)]
    fn decode_attention(
        &self,
        q: poem_ir::ValueId,
        plan: poem_ir::ValueId,
        cache: poem_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        o: poem_ir::ValueId,
        lse: Option<poem_ir::ValueId>,
    ) -> Result<(), kernels_slang::Error> {
        let buffers = self.tuning().storage_buffers();
        let binds = SPLIT_STORAGE_BUFFERS + u32::from(lse.is_some());
        let split = self
            .sdpa_split()
            .filter(|_| buffers == 0 || buffers >= binds);
        sdpa::decode(
            self.ctx,
            self.sdpa_tactic(),
            self.tensor(q),
            self.decode_plan(plan),
            self.pool(cache),
            window,
            head_dim,
            sm_scale,
            self.tensor(o),
            lse.map(|l| self.tensor(l)),
            split,
        )
    }

    /// Attention over the key blocks a selection names.
    #[allow(clippy::too_many_arguments)]
    fn selected_attention(
        &self,
        op: &'static str,
        q: kernels_slang::Tensor,
        tables: &sdpa::Tables,
        selection: poem_ir::ValueId,
        cache: poem_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        ratio: u32,
        o: poem_ir::ValueId,
    ) -> Result<(), kernels_slang::Error> {
        if window.is_some() {
            return Err(kernels_slang::Error::Backend {
                op,
                detail: "a selection and a sliding window are two answers to which keys a row \
                         reads"
                    .to_string(),
            });
        }
        sdpa::selected(
            self.ctx,
            self.sdpa_tactic(),
            q,
            tables,
            self.pool(cache),
            head_dim,
            sm_scale,
            ratio,
            self.tensor(selection),
            self.tensor(o),
        )
    }

    fn requests(&self) -> u32 {
        u32::try_from(self.qo_indptr_host().len().saturating_sub(1)).unwrap_or(u32::MAX)
    }

    fn attention(&mut self, op: &Attn) -> Result<(), kernels_slang::Error> {
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
            } => kernels_slang::attn::score::capture(
                self.ctx,
                self.ragged(*q),
                self.prefill_plan(*plan).positions,
                self.pool(*cache),
                *head_dim,
                *kv_heads,
                *sm_scale,
                *observe,
                // The rows are the window's lanes already.
                0,
                *kv_max,
                self.requests(),
                self.tensor(*scores),
            ),
            Attn::PlanDecode {
                kv_indptr: _,
                kv_indices: _,
                last_page_len: _,
                kv_len: _,
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
                let fire = self.bindings();
                let (positions, t) = (fire.positions, fire.tables);
                let built = sdpa::Tables::plan(
                    "attn.plan_decode",
                    sdpa::Tables {
                        positions: self.cut_rows(positions),
                        request_of_token: self.cut_rows(t.request_of_token),
                        mask: self.cut_rows(t.mask),
                        mask_enabled: self.cut_rows(t.mask_enabled),
                        mask_stride: t.mask_stride,
                    },
                )?;
                self.put(*plan, StructSlot::Decode(built));
                Ok(())
            }

            Attn::PlanPrefill {
                kv_indptr: _,
                kv_indices: _,
                last_page_len: _,
                kv_len: _,
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
                let fire = self.bindings();
                let (positions, t) = (fire.positions, fire.tables);
                let built = sdpa::Tables::plan(
                    "attn.plan_prefill",
                    sdpa::Tables {
                        positions: self.cut_rows(positions),
                        request_of_token: self.cut_rows(t.request_of_token),
                        mask: self.cut_rows(t.mask),
                        mask_enabled: self.cut_rows(t.mask_enabled),
                        mask_stride: t.mask_stride,
                    },
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
            } => self.decode_attention(*q, *plan, *cache, *window, *head_dim, *sm_scale, *o, None),

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
            } => self.prefill_attention(
                *q,
                *plan,
                None,
                *cache,
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
                None,
            ),

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
            } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            // Bounded replay's floors are DeepSeek-V4.1's, served on CUDA.
            Attn::KvEnds { .. } | Attn::WindowSink { .. } | Attn::ReplayFloor { .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
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
            } => self.selected_attention(
                "attn.decode",
                self.tensor(*q),
                self.decode_plan(*plan),
                *selection,
                *cache,
                *window,
                *head_dim,
                *sm_scale,
                *ratio,
                *o,
            ),
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
                const OP: &str = "attn.prefill";
                Self::kv_heads_agree(OP, self.pool(*cache), *kv_heads)?;
                self.selected_attention(
                    OP,
                    self.ragged(*q).data,
                    &self.prefill_tables(*plan, None),
                    *selection,
                    *cache,
                    *window,
                    *head_dim,
                    *sm_scale,
                    *ratio,
                    *o,
                )
            }
            Attn::Dense {
                q,
                k,
                v,
                segments,
                head_dim,
                sm_scale,
                o,
            } => kernels_slang::attn::dense::bidirectional(
                self.ctx,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*segments),
                *head_dim,
                *sm_scale,
                self.tensor(*o),
            ),
            Attn::Voxel {
                q,
                k,
                v,
                grid,
                segment,
                sm_scale,
                o,
            } => kernels_slang::attn::voxel::attention(
                self.ctx,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*grid),
                match segment {
                    poem_ir::VoxelSegment::Clip => kernels_slang::attn::voxel::Segment::Clip,
                    poem_ir::VoxelSegment::Frames(n) => {
                        kernels_slang::attn::voxel::Segment::Frames(*n)
                    }
                },
                *sm_scale,
                self.tensor(*o),
                self.grid_base(*grid, *q),
            ),
            Attn::Prefill {
                causal: false,
                mask: Some(_),
                selection: None,
                bias: None,
                lse: None,
                ..
            }
            | Attn::Prefill {
                causal: false,
                mask: Some(_),
                selection: None,
                bias: None,
                lse: Some(_),
                ..
            } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
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
                    _ => return Err(kernels_slang::Error::Unsupported { op: op.name() }),
                }
                let stated = self.group_mask(*q_indptr, *kv_indptr).map(|stated| {
                    kernels_slang::attn::dense::CustomMask {
                        bits: stated.bits,
                        indptr: stated.indptr,
                    }
                });
                kernels_slang::attn::dense::ragged(
                    self.ctx,
                    self.uncut(*q),
                    self.uncut(*k),
                    self.uncut(*v),
                    self.uncut(*q_indptr),
                    self.uncut(*kv_indptr),
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    stated,
                    self.uncut(*o),
                )
            }
            Attn::Decode {
                selection: None,
                bias: Some(_),
                lse: None,
                ..
            }
            | Attn::Prefill {
                mask: None,
                selection: None,
                bias: Some(_),
                lse: None,
                ..
            } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Attn::Prefill {
                q,
                plan,
                mask: Some(mask),
                cache,
                window,
                head_dim,
                kv_heads,
                causal: _,
                sm_scale,
                o,
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
                lse: None,
            } => self.prefill_attention(
                *q,
                *plan,
                Some(self.cut_rows(self.tensor(*mask))),
                *cache,
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
                None,
            ),
            Attn::Prefill {
                q,
                plan,
                mask: Some(mask),
                cache,
                window,
                head_dim,
                kv_heads,
                causal: _,
                sm_scale,
                o,
                lse: Some(lse),
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
            } => self.prefill_attention(
                *q,
                *plan,
                Some(self.cut_rows(self.tensor(*mask))),
                *cache,
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
                Some(self.tensor(*lse)),
            ),
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
            } => self.decode_attention(
                *q,
                *plan,
                *cache,
                *window,
                *head_dim,
                *sm_scale,
                *o,
                Some(*lse),
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
                lse: Some(lse),
                // This backend splits no prompt's rows, so a replay floor stays 0.
                floor: _,
                mask: None,
                selection: None,
                bias: None,
                causal: true,
                selected: None,
                rel: None,
            } => self.prefill_attention(
                *q,
                *plan,
                None,
                *cache,
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
                Some(self.tensor(*lse)),
            ),
            Attn::Sink {
                o,
                lse,
                sink,
                head_dim,
                o_out: _,
            } => kernels_slang::attn::merge::sink(
                self.ctx,
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
            } => kernels_slang::attn::merge::merge_lse(
                self.ctx,
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
            } => kernels_slang::attn::kv_append::kv_append(
                self.ctx,
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
            } => kernels_slang::attn::kv_append::kv_append_shared(
                self.ctx,
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
                let built = kernels_slang::attn::mla::plan(
                    self.ctx,
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
            } => kernels_slang::attn::mla::latents(
                self.ctx,
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
            } => kernels_slang::attn::mla::latents_rope(
                self.ctx,
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
            } => kernels_slang::attn::mla::split_q_b(
                self.ctx,
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
            } => kernels_slang::attn::mla::absorb_q(
                self.ctx,
                self.tensor(*q_nope),
                self.tensor(*kv_b),
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
            } => kernels_slang::attn::mla::absorb_out(
                self.ctx,
                self.tensor(*latent),
                self.tensor(*kv_b),
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
            } => kernels_slang::attn::mla::kv_append(
                self.ctx,
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
            } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Attn::MlaDecode {
                window: Some(_),
                selection: None,
                ..
            }
            | Attn::MlaPrefill {
                window: Some(_),
                selection: None,
                ..
            } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
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
                    (fire.positions, fire.tables.request_of_token)
                };
                kernels_slang::attn::mla::attention_decode(
                    self.ctx,
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
                selection: None,
            } => {
                let (positions, request_of_token) = {
                    let fire = self.bindings();
                    (fire.positions, fire.tables.request_of_token)
                };
                kernels_slang::attn::mla::attention_prefill(
                    self.ctx,
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
                    (fire.positions, fire.tables.request_of_token)
                };
                kernels_slang::attn::mla::attention_decode_selected(
                    self.ctx,
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
                    (fire.positions, fire.tables.request_of_token)
                };
                kernels_slang::attn::mla::attention_prefill_selected(
                    self.ctx,
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
            } => kernels_slang::attn::index::layernorm_rope(
                self.ctx,
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
            } => kernels_slang::attn::index::rope(
                self.ctx,
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
                let Some(scores) = self.index_scores() else {
                    return Err(kernels_slang::Error::Unsupported {
                        op: "attn.index_topk",
                    });
                };
                let fire = self.bindings();
                let (positions, request_of_token) = (fire.positions, fire.tables.request_of_token);
                kernels_slang::attn::index::topk(
                    self.ctx,
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
            Attn::IndexKvAppend {
                k,
                keys,
                write_page,
                write_offset,
            } => kernels_slang::attn::index::kv_append(
                self.ctx,
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
            } => kernels_slang::attn::index::block_mean(
                self.ctx,
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
            } => kernels_slang::attn::pool::boundary_decode(
                self.ctx,
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
            } => kernels_slang::attn::pool::boundary_prefill(
                self.ctx,
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
                let Some(state) = self.pool_state(*pages) else {
                    return Err(kernels_slang::Error::Unsupported {
                        op: "attn.pool_gather",
                    });
                };
                kernels_slang::attn::pool::gather(
                    self.ctx,
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
                    return Err(kernels_slang::Error::Unsupported {
                        op: "attn.pool_state_write",
                    });
                };
                kernels_slang::attn::pool::state_write(
                    self.ctx,
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
            Attn::PoolKvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool: into,
                write_page,
                write_offset,
            } => kernels_slang::attn::pool::kv_append(
                self.ctx,
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
                selection: None,
                top_k: None,
            } => kernels_slang::attn::pool::attention_lse(
                self.ctx,
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
            } => kernels_slang::attn::pool::attention_lse_selected(
                self.ctx,
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
            // A selection, a window, a relative bias and a mask no kernel here
            // combines.
            Attn::Decode { .. }
            | Attn::Prefill { .. }
            | Attn::MlaDecode { .. }
            | Attn::MlaPrefill { .. }
            | Attn::PoolLse { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
        }
    }
}
