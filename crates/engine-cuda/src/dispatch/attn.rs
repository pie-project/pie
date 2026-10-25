use kernels_cuda::attn::compressed::pool;
use kernels_cuda::attn::dense;
use kernels_cuda::attn::dense::voxel;
#[cfg(not(feature = "rocm"))]
use kernels_cuda::attn::fa2;
#[cfg(feature = "rocm")]
use kernels_cuda::attn::paged;
use kernels_cuda::attn::{self, index, mla, plan, selected};
use poem_exec::{DispatchAttn, KernelError};
use poem_ir::VoxelSegment;
use poem_ir::{Attn, IndexScore, Operation, StructKind};

use crate::run::{Run, StructSlot};

fn segment(segment: VoxelSegment) -> voxel::Segment {
    match segment {
        VoxelSegment::Clip => voxel::Segment::Lane,
        VoxelSegment::Frames(n) => voxel::Segment::Frames(n),
    }
}

impl DispatchAttn for Run<'_> {
    fn dispatch(&mut self, op: &Attn) -> Result<(), KernelError> {
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_scores(
        &mut self,
        q: poem_ir::ValueId,
        plan: poem_ir::ValueId,
        cache: poem_ir::ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        lse: poem_ir::ValueId,
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
        kernels_cuda::attn::score::capture(
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
            &seat.seat_of,
        )
    }

    pub(super) fn attention(&mut self, op: &Attn) -> Result<(), kernels_cuda::Error> {
        match op {
            Attn::Voxel {
                q,
                k,
                v,
                grid,
                segment: how,
                sm_scale,
                o,
            } => voxel::attention(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*grid),
                segment(*how),
                *sm_scale,
                &mut self.tensor(*o),
                self.grid_base(*grid, *q),
            ),
            Attn::PlanDecode {
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
                    #[cfg(not(feature = "rocm"))]
                    let max_grid = fa2::decode_max_grid_size(
                        seat.shape.head_dim,
                        seat.shape.num_q_heads,
                        seat.shape.num_kv_heads,
                        &fire.device,
                    );
                    #[cfg(feature = "rocm")]
                    let max_grid =
                        paged::decode_max_grid_size(seat.shape.num_kv_heads, fire.device.num_sm);
                    plan::plan_decode(
                        seat.kv_indptr,
                        seat.kv_len,
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
            Attn::PlanPrefill {
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
                                seat.kv_indptr,
                                seat.kv_len,
                                seat.rows,
                                seat.shape,
                                seat.live,
                                seat.window,
                                true,
                                fire.capture,
                                spans,
                                &fire.device,
                                fire.toggles,
                                seat.workspace,
                            )?;
                            if let Some(wanted) = crate::serve::diag::on().plan_trace.as_deref()
                                && (wanted == "all" || wanted == seat.rows.to_string())
                            {
                                let ints: Vec<i32> = built
                                    .int_upload
                                    .as_chunks::<4>()
                                    .0
                                    .iter()
                                    .take(96)
                                    .map(|b| i32::from_le_bytes(*b))
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
                        other => panic!(
                            "`attn.plan_prefill` defines a {other:?}, which is no \
                             prefill plan kind"
                        ),
                    }
                };
                self.put(*plan, built);
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
            Attn::Decode {
                q,
                plan: _,
                selection: Some(selection),
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                bias: None,
                selected: Some(poem_ir::Blocks { ratio, per_head }),
                rel: None,
                lse: None,
            } => selected::attention_decode_selected(
                self.ctx(),
                self.ragged_q(*q),
                self.tensor(*selection),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *sm_scale,
                *ratio,
                *per_head,
                &mut self.tensor(*o),
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
                selected: Some(poem_ir::Blocks { ratio, per_head }),
                rel: None,
                lse: None,
            } => {
                #[cfg(not(feature = "rocm"))]
                let _ = plan;
                // On ROCm the rows' causal plan walks them a tile at a time,
                // the selection masking each row's keys; the per-row walk
                // takes what the tiled one does not cover.
                #[cfg(feature = "rocm")]
                if let StructSlot::Prefill(planned) = self.slot(*plan)
                    && attn::prefill_selected_tiled(
                        self.ctx(),
                        self.ragged_q(*q),
                        planned,
                        self.tensor(*selection),
                        &self.pool_absolute(*cache),
                        *window,
                        *head_dim,
                        *kv_heads,
                        *sm_scale,
                        *ratio,
                        *per_head,
                        &mut self.tensor(*o),
                    )?
                {
                    return Ok(());
                }
                selected::attention_prefill_selected(
                    self.ctx(),
                    self.ragged_q(*q),
                    self.tensor(*selection),
                    &self.pool_absolute(*cache),
                    *window,
                    *head_dim,
                    *kv_heads,
                    *sm_scale,
                    *ratio,
                    *per_head,
                    &mut self.tensor(*o),
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
            } => attn::prefill_rel(
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
            } => attn::prefill(
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
            // `causal: false` marks a wide class: its staged mask bits are unclipped.
            Attn::Prefill {
                kv_heads: _,
                causal: _,
                q,
                plan,
                mask: Some(mask),
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
                lse: None,
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
            Attn::Prefill {
                kv_heads: _,
                causal: _,
                q,
                plan,
                mask: Some(mask),
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse: Some(lse),
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
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
            Attn::Prefill {
                kv_heads: _,
                causal: false,
                q,
                plan,
                mask: None,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
                lse: None,
            } => attn::masked(
                self.ctx(),
                self.ragged_q(*q),
                self.prefill_plan(*plan),
                self.mask_bits(),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
            ),
            Attn::Prefill {
                kv_heads: _,
                causal: false,
                q,
                plan,
                mask: None,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse: Some(lse),
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
            } => attn::masked_lse(
                self.ctx(),
                self.ragged_q(*q),
                self.prefill_plan(*plan),
                self.mask_bits(),
                &self.pool_absolute(*cache),
                *window,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
            Attn::Dense {
                q,
                k,
                v,
                segments,
                head_dim,
                sm_scale,
                o,
            } => dense::bidirectional(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*segments),
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
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
                floor,
                mask: None,
                selection: None,
                bias: None,
                causal: true,
                selected: None,
                rel: None,
            } => {
                // A floor stays 0 unless the profile turns bounded replay on;
                // then only the fused kernels read it.
                if floor.is_some() && self.ctx().tuning().compressed_replay {
                    return Err(kernels_cuda::Error::Backend {
                        op: "attn.prefill",
                        detail: "the window is held above a replay floor, which only the fused \
                                 `fused.pool_attention` reads"
                            .to_string(),
                    });
                }
                if attn::compressed::decode::ds_rows(&self.pool(*cache)) {
                    return Err(kernels_cuda::Error::Backend {
                        op: "attn.prefill",
                        detail: "the cache holds DeepSeek's quantized rows, which only the fused \
                                 `fused.pool_attention` and `fused.lse_sink` read"
                            .to_string(),
                    });
                }
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
            Attn::Sink {
                o,
                lse,
                sink,
                head_dim,
                o_out: _,
            } => attn::merge::sink(
                self.ctx(),
                &mut self.tensor(*o),
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
            } => attn::merge::merge_lse(
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
            Attn::KvAppend {
                k,
                v: Some(v),
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
            Attn::KvAppend {
                k: plane,
                v: None,
                cache,
                write_page,
                write_offset,
            } => {
                let pool = self.pool(*cache);
                if attn::compressed::decode::ds_rows(&pool) {
                    // DeepSeek's quantized rows, quantized on the way in.
                    attn::compressed::kv_fp8::append(
                        self.ctx(),
                        self.ragged_lanes(*plane).data,
                        &pool,
                        self.tensor(*write_page),
                        self.tensor(*write_offset),
                    )
                } else {
                    attn::kv_append_shared(
                        self.ctx(),
                        self.ragged_lanes(*plane),
                        &pool,
                        self.tensor(*write_page),
                        self.tensor(*write_offset),
                    )
                }
            }
            Attn::MlaPlan {
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
                        seat.kv_indptr,
                        seat.kv_len,
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
                // Only the dense FA2 arms read the plan off the card; a trunk
                // whose MLA reads are all selected (the sparse or naive
                // kernels) leaves its upload (some 600 KB a fire) unstaged.
                let read = self.nodes().iter().any(|node| {
                    matches!(
                        &node.op,
                        Operation::Attn(
                            Attn::MlaDecode { plan: read, selection: None, .. }
                                | Attn::MlaPrefill { plan: read, selection: None, .. }
                        ) if read == plan
                    )
                });
                if read {
                    built.stage(self.ctx())?;
                }
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
            } => mla::latents(
                self.ctx(),
                self.tensor(*kv_a),
                self.tensor(*weight),
                *eps,
                *kv_lora_rank,
                &mut self.tensor(*kv_c),
                &mut self.tensor(*k_pe),
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
            Attn::MlaSplitQB {
                q_b,
                heads,
                nope_dim,
                rope_dim,
                q_nope,
                q_pe,
            } => mla::prep::split_q_b(
                self.ctx(),
                self.tensor(*q_b),
                *heads,
                *nope_dim,
                *rope_dim,
                &mut self.tensor(*q_nope),
                &mut self.tensor(*q_pe),
            ),
            Attn::MlaAbsorbQ {
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
                    "attn.mla_absorb_q",
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
            Attn::MlaAbsorbOut {
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
                    "attn.mla_absorb_out",
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
            Attn::MlaKvAppend {
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
            Attn::MlaDecode {
                q,
                plan,
                q_pe,
                cache,
                window,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse,
                selection: None,
            } => mla::attention_decode(
                self.ctx(),
                self.ragged(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                &self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                *window,
                &mut self.tensor(*o),
                lse.map(|lse| self.tensor(lse)).as_mut(),
            ),
            Attn::MlaPrefill {
                q,
                plan,
                q_pe,
                cache,
                window,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse,
                selection: None,
            } => mla::attention_prefill(
                self.ctx(),
                self.ragged(*q),
                self.mla_plan(*plan),
                self.tensor(*q_pe),
                &self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                *window,
                &mut self.tensor(*o),
                lse.map(|lse| self.tensor(lse)).as_mut(),
            ),
            Attn::MlaDecode {
                q,
                plan,
                q_pe,
                selection: Some(selection),
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse,
                window: None,
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
                lse.map(|lse| self.tensor(lse)).as_mut(),
            ),
            Attn::MlaPrefill {
                q,
                plan,
                q_pe,
                selection: Some(selection),
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
                lse,
                window: None,
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
                lse.map(|lse| self.tensor(lse)).as_mut(),
            ),
            Attn::ReplayFloor {
                positions,
                state,
                mark,
                floor,
            } => kernels_cuda::attn::kv_window::floor(
                self.ctx(),
                self.ragged_lanes(*positions),
                &self.recurrent_absolute(*state),
                *mark,
                &mut self.tensor(*floor),
            ),
            Attn::KvEnds {
                request_of_token,
                cache,
                ends,
            } => {
                // Each row's request against the whole fire's tables, as
                // the window that reads these ends does.
                let pool = self
                    .pool_whole(*cache)
                    .ok_or_else(|| kernels_cuda::Error::Backend {
                        op: "attn.kv_ends",
                        detail: "a cache over gathered seats".to_string(),
                    })?;
                kernels_cuda::attn::kv_window::kv_ends(
                    self.ctx(),
                    self.tensor(*request_of_token),
                    &pool,
                    &mut self.tensor(*ends),
                )
            }
            Attn::WindowSink {
                q,
                positions,
                request_of_token,
                cache,
                sink,
                window,
                head_dim,
                sm_scale,
                o,
            } => {
                const OP: &str = "attn.window_sink";
                // Each row reads its request's pages against the whole
                // fire's tables, as DeepSeek's fused decode does.
                let pool = self
                    .pool_whole(*cache)
                    .ok_or_else(|| kernels_cuda::Error::Backend {
                        op: OP,
                        detail: "a window over gathered seats".to_string(),
                    })?;
                let q = self.tensor(*q);
                kernels_cuda::attn::compressed::decode::decode(
                    self.ctx(),
                    q,
                    self.tensor(*positions),
                    self.tensor(*request_of_token),
                    None,
                    &pool,
                    *window,
                    None,
                    self.tensor(*sink),
                    q.width / (*head_dim).max(1),
                    *head_dim,
                    *sm_scale,
                    None,
                    &mut self.tensor(*o),
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
            Attn::IndexRope {
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
            Attn::IndexTopk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                ratio,
                score: IndexScore::PooledTokens,
                selection,
            } => index::topk_tokens(
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
            Attn::IndexTopk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                ratio,
                score: IndexScore::BlockMax,
                selection,
            } => {
                if weights.is_some() {
                    return Err(kernels_cuda::Error::Unsupported {
                        op: "attn.index_topk",
                    });
                }
                index::topk_block_max(
                    self.ctx(),
                    self.ragged_lanes(*q),
                    &self.pool(*keys),
                    *heads,
                    *head_dim,
                    *top_k,
                    *ratio,
                    &mut self.tensor(*selection),
                )
            }
            Attn::IndexKvAppend {
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
            Attn::IndexBlockMean {
                boundary_pos,
                boundary_req,
                keys,
                head_dim,
                ratio,
                entries,
            } => pool::block_mean(
                self.ctx(),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                &self.pool(*keys),
                *head_dim,
                *ratio,
                &mut self.tensor(*entries),
            ),
            Attn::PoolBoundary {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
                decode: true,
            } => pool::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                self.tensor(*row_valid),
                *ratio,
                &mut self.tensor(*boundary_pos),
                &mut self.tensor(*boundary_req),
                &mut self.tensor(*boundary_rope),
            ),
            Attn::PoolBoundary {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
                decode: false,
            } => pool::boundary_prefill(
                self.ctx(),
                self.ragged(*positions),
                self.tensor(*row_valid),
                *ratio,
                &mut self.tensor(*boundary_pos),
                &mut self.tensor(*boundary_req),
                &mut self.tensor(*boundary_rope),
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
            Attn::PoolStateWrite {
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
            Attn::PoolKvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool: into,
                write_page,
                write_offset,
            } => {
                let into = self.pool(*into);
                if attn::compressed::decode::ds_rows(&into) {
                    attn::compressed::kv_fp8::store_entries(
                        self.ctx(),
                        self.tensor(*entries),
                        self.tensor(*boundary_pos),
                        self.tensor(*boundary_req),
                        &into,
                    )
                } else if attn::index::fp4_keys(&into) {
                    // DeepSeek-V4.1's FP4 index keys, quantized on the way in.
                    attn::index::store_fp4(
                        self.ctx(),
                        self.tensor(*entries),
                        self.tensor(*boundary_pos),
                        self.tensor(*boundary_req),
                        &into,
                    )
                } else {
                    pool::kv_append(
                        self.ctx(),
                        self.tensor(*entries),
                        self.tensor(*boundary_pos),
                        self.tensor(*boundary_req),
                        &into,
                        self.tensor(*write_page),
                        self.tensor(*write_offset),
                    )
                }
            }
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
            } => pool::attention_lse_selected(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*positions),
                self.tensor(*request_of_token),
                self.tensor(*selection),
                &self.pool(*entries),
                *ratio,
                *top_k,
                *heads,
                *head_dim,
                *sm_scale,
                &mut self.tensor(*o),
                &mut self.tensor(*lse),
            ),
            Attn::Ragged {
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
                let packing = |id: poem_ir::ValueId| match self.values()[id.0 as usize].def {
                    poem_ir::Def::Input(poem_ir::RuntimeInput::Geometry {
                        kind: poem_ir::GeomKind::GroupIndptr { select },
                        ..
                    }) => Some(select),
                    _ => None,
                };
                let stated = packing(*q_indptr)
                    .zip(packing(*kv_indptr))
                    .and_then(|(q, kv)| self.group_mask(q, kv));
                let mask = match (mask, stated) {
                    (
                        poem_ir::RaggedMask::None | poem_ir::RaggedMask::GroupBlockDiagonal,
                        Some(stated),
                    ) => kernels_cuda::attn::ragged::RaggedMask::Custom {
                        bits: stated.bits,
                        indptr: stated.indptr,
                    },
                    (poem_ir::RaggedMask::None | poem_ir::RaggedMask::GroupBlockDiagonal, None) => {
                        kernels_cuda::attn::ragged::RaggedMask::None
                    }
                    (poem_ir::RaggedMask::RelativeBias { .. }, Some(_)) => {
                        return Err(kernels_cuda::Error::Backend {
                            op: "attn.ragged",
                            detail: "a lane states an attention mask over an attention the \
                                  model already biases (a relative bias); the two do not \
                                  compose"
                                .to_string(),
                        });
                    }
                    (poem_ir::RaggedMask::RelativeBias { table, max_len }, None) => {
                        kernels_cuda::attn::ragged::RaggedMask::RelativeBias {
                            table: self.tensor(*table),
                            max_len: *max_len,
                        }
                    }
                };
                kernels_cuda::attn::ragged::ragged(
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
            // A selection, a window, a relative bias and a mask no kernel here
            // combines.
            Attn::Decode { .. }
            | Attn::Prefill { .. }
            | Attn::MlaDecode { .. }
            | Attn::MlaPrefill { .. }
            | Attn::PoolLse { .. } => Err(kernels_cuda::Error::Unsupported {
                op: poem_ir::Operands::name(op),
            }),
        }
    }
}

impl Run<'_> {
    pub(crate) fn dense_or_decoded(
        &self,
        op: &'static str,
        w: poem_ir::ValueId,
        n: u32,
        k: u32,
    ) -> Result<kernels_cuda::Tensor, kernels_cuda::Error> {
        match self.maybe_planes(w) {
            #[cfg(not(feature = "rocm"))]
            Some((codes, ..)) if codes.dtype == poem_ir::Dtype::E4m3tensor => {
                Err(kernels_cuda::Error::Backend {
                    op,
                    detail: "a per-tensor e4m3 plane is read by the n-gram gather alone"
                        .to_string(),
                })
            }
            #[cfg(not(feature = "rocm"))]
            Some((codes, scales, biases, seat)) if codes.dtype == poem_ir::Dtype::Nvfp4 => {
                kernels_cuda::linear::dequant::nvfp4_plane(
                    self.ctx(),
                    op,
                    codes,
                    scales,
                    super::linear::tensor_scales(op, biases)?,
                    n,
                    k,
                    seat,
                )
            }
            Some((codes, scales, biases, seat)) => kernels_cuda::linear::dequant::decoded_plane(
                self.ctx(),
                op,
                codes,
                scales,
                kernels_cuda::linear::affine::OffsetKind::Post,
                biases,
                poem_ir::Dtype::Bf16,
                n,
                k,
                seat,
            ),
            None => Ok(self.tensor(w)),
        }
    }
}
