//! `Attention`: the merged attention family — the attention anchor, mla,
//! ssm, index, and pool arms, plus the plan-building prepare phase those
//! launches ride on.

use kernels_cuda::attn::{self, fa2, index, mla, plan, pool};
use kernels_cuda::attn_dense;
use model_exec::{DispatchAttention, KernelError};
use model_ir::{Attention, StructKind};

use crate::run::{Run, StructSlot};

impl DispatchAttention for Run<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// One launch, or none at all: returns `Ok(())` without touching a
    /// stream for a load with no slab, a fire no lane captured, or a
    /// `prefill_lse` node the plan's `attn.scores` seam doesn't name.
    ///
    /// # Errors
    ///
    /// Whatever [`kernels_cuda::attn_score::capture`] refuses: a sliding
    /// window (the row would not be the softmax the eviction papers define),
    /// a quantized key plane, a head this kernel is not stamped for.
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
        // This launch isn't an IR op; its grid reads q + indptr[blockIdx.x]
        // off rebased boundaries, so it needs the windowed rectangle.
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

    /// The arms themselves, in `kernels-cuda`'s error vocabulary; each stays
    /// a plain tail call with `?`, and [`kernel`](crate::error::kernel)
    /// lifts the whole family into the contract's.
    fn attention(&mut self, op: &Attention) -> Result<(), kernels_cuda::Error> {
        match op {
            // ---- attention (anchor) ----
            // The builder walks kv_indptr/kv_len's host contents via the
            // twins Run::planning bound beside them.
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
                built.stage(self.ctx())?;
                self.put(*plan, StructSlot::Decode(built));
                Ok(())
            }
            // As `PlanDecode`, plus the qo side; which builder runs follows
            // the plan value's `StructKind`. `causal: true` is
            // `attention.prefill`'s reading; `attention.masked` ignores it.
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
                    // seat.rows is the row axis's half of Run::planning's
                    // pin, raised to the bucket for row-total kinds.
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
                            built.stage(self.ctx())?;
                            StructSlot::Prefill(built)
                        }
                        StructKind::AttnPrefillPlanSm90 => {
                            let built = plan::plan_prefill_sm90(
                                self.qo_indptr_host(),
                                &seat.kv_indptr,
                                &seat.kv_len,
                                // Hash hygiene, not a live path: the launcher
                                // refuses before it ever launches an sm90
                                // prefill, so no gate exercises this.
                                seat.rows,
                                seat.shape,
                                seat.live,
                                true,
                                fire.capture,
                                &fire.device,
                                seat.workspace,
                            )?;
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
            // A prefill plan holds either kind the trace declared. Uses
            // `ragged_q` (FA2's by-value params block CSR must match q's raw
            // pointer), so cache/schedule state goes absolute with it too.
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
            // The op names its mask bits now; only their span table still
            // rides the plan (bound at build by the `PlanPrefill` arm), and
            // the entry refuses a plan no span table rides.
            Attention::Masked {
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
            // The tower's attention: no pool, no plan slot, no window.
            // `segments` is the patch axis's own indptr, cut at the patch
            // window so one fire sees only its own images.
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
                // Asks fire state for which plane this layer owns, matched
                // against the plan's `attn.scores` exports. `lane_offset`
                // maps this window's request number to a fire lane.
                self.capture_scores(*q, *plan, *cache, *window, *head_dim, *kv_heads, *sm_scale, *lse)
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
            // The op states its write geometry; the arm resolves the
            // per-token `write_page`/`write_offset` descriptors and the
            // entry lands each row in the stated cell.
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
            // ---- mla ----
            // Same host-twin routing as the attention plan arms. The op's
            // `heads`/`kv_lora_rank` ride in via the seat as
            // `num_q_heads`/`head_dim`.
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
                        // seat.rows and seat.shape.num_requests: the carved
                        // pair Run::planning raises to record::BodyKey's
                        // numbers.
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
            // The arm resolves the op's write descriptors; the entry marks
            // its own remaining seam.
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
            // ---- ssm ----
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
                &mut self.tensor(*y),
            ),
            // `x` is the conv's in-projection rows: a `RsVerb::Buffer` lane
            // scatters them into a slab and `RsVerb::FoldBuffered` gathers
            // them back over the GEMM output. 2R split: the head runs
            // `[0, n)` and folds, the tail runs `[n, rows)` from the head's
            // rolling state (a row folding a prefix can't be one call).
            Attention::SsmCausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => {
                self.rs_move("attention.ssm_causal_conv1d_chunked", *x, self.tensor(*x))?;
                // Only the chunked arms take the absolute lane door: a body
                // bakes its slot map, so `lane_offset` isn't a function of
                // the key.
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
            // N-gram hasher over the lane's trailing window; same state
            // discipline as the conv above (decode shifts unconditionally,
            // chunked advances only the committed prefix).
            Attention::PleNgramIds {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
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
                    &mut self.tensor(*ngram_ids),
                )
            }
            // `ba` is the `[b | a]` projection; the prep is its only reader,
            // so the move sits directly in front of it.
            Attention::SsmGdnPrep {
                ba,
                dt_bias,
                a_log,
                gates,
            } => {
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
                &mut self.tensor(*y),
            ),
            // Same 2R split as the chunked conv above: the head folds the
            // boundary into the bank, the tail continues from what the head
            // wrote.
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
            // ---- index ----
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
            // Pooled keys (`ratio > 1`) are read at their boundary cells and
            // published as the pool's tokens, tail first.
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
                self.tensor(*weights),
                &self.pool(*keys),
                *heads,
                *head_dim,
                *top_k,
                *ratio,
                &mut self.tensor(*selection),
            ),
            // As `attention.mla_kv_append`.
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
            // ---- pool ----
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
            // The dsv4 compressor state has no IR seat; binds the slabs the
            // store staged for the gather's own space (`Run::slabs`).
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
            // As the other appenders.
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
            // `entries` names the compressed cache space on this plane, so
            // it resolves to a pool.
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
            // `pool.cuh` has only the dense `pool_lse_paged` reader, no
            // selected twin; falling back to it would answer a different
            // attention (every compressed row, not the indexer's chosen
            // ones) under this op's name, so the arm refuses by name.
            Attention::PoolLseSelected { .. } => Err(kernels_cuda::Error::Unsupported {
                op: "attention.pool_lse_selected",
            }),
        }
    }
}

impl Run<'_> {
    /// A weight an entry reads whole: the dense handle, or an affine bank
    /// decoded to bf16 in fire scratch (`[n, k]`, resident planes only).
    fn dense_or_decoded(
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
