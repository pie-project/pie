use kernels_slang::attn::sdpa;
use poem_exec::{DispatchAttn, KernelError};
use poem_ir::{Attn, IndexScore, Operands, StructKind};

use crate::encode::{from_slang, slang_pool, slang_tensor};
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
    /// How this device runs paged attention: WGSL's baseline tier, a decode
    /// workgroup of 256 invocations (128 measured 26-61% slower through
    /// wgpu on an RTX 4090) and prefill tiles of 8 rows of 32 lanes, the
    /// 256 invocations WebGPU's default workgroup holds.
    fn sdpa_tactic(&self) -> sdpa::Tactic {
        let tuning = self.tuning();
        sdpa::Tactic {
            tier: kernels_slang::Tier::Baseline,
            narrow: false,
            decode_threads: 256,
            tile_rows: 8,
            split_groups: tuning.sdpa_split_groups(),
            split_min_keys: tuning.sdpa_split_min_keys(),
        }
    }

    fn prefill_tables(
        &self,
        plan: poem_ir::ValueId,
        mask: Option<kernels_wgpu::Tensor>,
    ) -> sdpa::Tables {
        let plan = *self.prefill_plan(plan);
        sdpa::Tables {
            mask: mask.map_or(plan.mask, slang_tensor),
            ..plan
        }
    }

    /// Refuses a pool whose row does not hold the stated kv heads.
    fn kv_heads_agree(
        op: &'static str,
        pool: &kernels_wgpu::KvPool,
        kv_heads: u32,
    ) -> Result<(), kernels_wgpu::Error> {
        if pool.head_stride == 0 || pool.seq_stride / pool.head_stride != u64::from(kv_heads) {
            return Err(kernels_wgpu::Error::Backend {
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
        mask: Option<kernels_wgpu::Tensor>,
        cache: poem_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: kernels_wgpu::Tensor,
        lse: Option<kernels_wgpu::Tensor>,
    ) -> Result<(), kernels_wgpu::Error> {
        const OP: &str = "attn.prefill";
        let q = self.ragged(q).data;
        let pool = self.pool(cache);
        Self::kv_heads_agree(OP, pool, kv_heads)?;
        if let Some(mask) = mask
            && mask.dtype != poem_ir::Dtype::U8
        {
            return Err(kernels_wgpu::Error::Backend {
                op: OP,
                detail: format!(
                    "the mask this op states is {:?}, and the shader reads packed u8 mask planes",
                    mask.dtype
                ),
            });
        }
        let tactic = self.sdpa_tactic();
        let tables = self.prefill_tables(plan, mask);
        let (q, pool, o) = (slang_tensor(q), slang_pool(pool), slang_tensor(o));
        let lse = lse.map(slang_tensor);
        let tiles = q.rows / self.requests().max(1) >= self.tuning().sdpa_tile_rows_per_request();
        if tiles {
            sdpa::prefill(
                &self.slang(),
                tactic,
                q,
                &tables,
                &pool,
                window,
                head_dim,
                sm_scale,
                o,
                lse,
            )
        } else {
            sdpa::decode(
                &self.slang(),
                tactic,
                q,
                &tables,
                &pool,
                window,
                head_dim,
                sm_scale,
                o,
                lse,
                None,
            )
        }
        .map_err(from_slang)
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
    ) -> Result<(), kernels_wgpu::Error> {
        let buffers = self.tuning().storage_buffers();
        let binds = SPLIT_STORAGE_BUFFERS + u32::from(lse.is_some());
        let split = self
            .sdpa_split()
            .filter(|_| buffers == 0 || buffers >= binds);
        sdpa::decode(
            &self.slang(),
            self.sdpa_tactic(),
            slang_tensor(self.tensor(q)),
            self.decode_plan(plan),
            &slang_pool(self.pool(cache)),
            window,
            head_dim,
            sm_scale,
            slang_tensor(self.tensor(o)),
            lse.map(|l| slang_tensor(self.tensor(l))),
            split,
        )
        .map_err(from_slang)
    }

    /// Attention over the key blocks a selection names.
    #[allow(clippy::too_many_arguments)]
    fn selected_attention(
        &self,
        op: &'static str,
        q: kernels_wgpu::Tensor,
        tables: &sdpa::Tables,
        selection: poem_ir::ValueId,
        cache: poem_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        ratio: u32,
        o: poem_ir::ValueId,
    ) -> Result<(), kernels_wgpu::Error> {
        if window.is_some() {
            return Err(kernels_wgpu::Error::Backend {
                op,
                detail: "a selection and a sliding window are two answers to which keys a row \
                         reads"
                    .to_string(),
            });
        }
        sdpa::selected(
            &self.slang(),
            self.sdpa_tactic(),
            slang_tensor(q),
            tables,
            &slang_pool(self.pool(cache)),
            head_dim,
            sm_scale,
            ratio,
            slang_tensor(self.tensor(selection)),
            slang_tensor(self.tensor(o)),
        )
        .map_err(from_slang)
    }

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
    ) -> Result<(), kernels_wgpu::Error> {
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
            self.prefill_plan(plan).positions,
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

    fn attention(&mut self, op: &Attn) -> Result<(), kernels_wgpu::Error> {
        match op {
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
                        positions: slang_tensor(self.cut_rows(positions)),
                        request_of_token: slang_tensor(self.cut_rows(t.request_of_token)),
                        mask: slang_tensor(self.cut_rows(t.mask)),
                        mask_enabled: slang_tensor(self.cut_rows(t.mask_enabled)),
                        mask_stride: t.mask_stride,
                    },
                )
                .map_err(from_slang)?;
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
                        positions: slang_tensor(self.cut_rows(positions)),
                        request_of_token: slang_tensor(self.cut_rows(t.request_of_token)),
                        mask: slang_tensor(self.cut_rows(t.mask)),
                        mask_enabled: slang_tensor(self.cut_rows(t.mask_enabled)),
                        mask_stride: t.mask_stride,
                    },
                )
                .map_err(from_slang)?;
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

            Attn::DecodeSelected { per_head: true, .. }
            | Attn::PrefillSelected { per_head: true, .. }
            | Attn::IndexTopk {
                score: IndexScore::BlockMax,
                ..
            }
            | Attn::IndexTopk {
                score: IndexScore::PooledTokens,
                ..
            } => Err(kernels_wgpu::Error::Unsupported { op: op.name() }),
            // Bounded replay's floors are DeepSeek-V4.1's, served on CUDA.
            Attn::KvEnds { .. } | Attn::WindowSink { .. } | Attn::ReplayFloor { .. } => {
                Err(kernels_wgpu::Error::Unsupported { op: op.name() })
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
            } => self.selected_attention(
                "attn.decode_selected",
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
            } => {
                const OP: &str = "attn.prefill_selected";
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
            Attn::Masked { causal: false, .. }
            | Attn::MaskedLse { causal: false, .. }
            | Attn::Ragged { .. }
            | Attn::Voxel { .. } => Err(kernels_wgpu::Error::Unsupported { op: op.name() }),
            Attn::DecodeRel { .. } | Attn::PrefillRel { .. } => {
                Err(kernels_wgpu::Error::Unsupported { op: op.name() })
            }
            Attn::Masked {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads,
                causal: _,
                sm_scale,
                o,
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
            Attn::MaskedLse {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads,
                causal: _,
                sm_scale,
                o,
                lse,
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
            Attn::DecodeLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse,
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
                self.prefill_attention(
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
                Err(kernels_wgpu::Error::Unsupported { op: op.name() })
            }
            Attn::MlaDecode {
                window: Some(_), ..
            }
            | Attn::MlaPrefill {
                window: Some(_), ..
            } => Err(kernels_wgpu::Error::Unsupported { op: op.name() }),
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
                    return Err(kernels_wgpu::Error::Unsupported {
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
                    return Err(kernels_wgpu::Error::Unsupported {
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
                    return Err(kernels_wgpu::Error::Unsupported {
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
