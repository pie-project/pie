//! The `attention` family: `impl DispatchAttention for Run<'_>`, holding the
//! paged and plan arms plus the absorbed `mla`, `ssm`, `index`, and `pool`
//! groups, in the order the merged enum lists them.

use kernels_metal::attn;
use model_exec::{DispatchAttention, KernelError};
use model_ir::{Attention, Operands, StructKind};

use crate::run::{Run, StructSlot};

/// `attention.mla_plan` refuses before a payload exists
/// (`kernels_metal::attn::mla`), so no consuming arm can ever hold a live
/// one; the unit keeps the stubs' signatures satisfied while each refusal
/// names its own entry. When the builder becomes real, this constant stops
/// making sense and the arms resolve through a `Run::mla_plan` accessor
/// instead.
const NO_MLA_PLAN: &attn::mla::MlaPlan = &attn::mla::MlaPlan;

impl DispatchAttention for Run<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// How many requests the window's rows belong to.
    ///
    /// The one fact the sdpa arbitration turns on and no operand carries: a
    /// prefill and a fleet of decodes can present the SAME row count, and
    /// what separates them is how many key spans those rows sit over. The
    /// window's qo boundaries are one entry per request plus a terminator,
    /// so the count is already here and nothing has to be recomputed from
    /// the composition.
    fn requests(&self) -> u32 {
        u32::try_from(self.qo_indptr_host().len().saturating_sub(1)).unwrap_or(u32::MAX)
    }

    /// **THE CAPTURE ARM'S OBSERVATION** — one launch, or none at all
    /// (`.wiki/alto/attn-score.md` §4).
    ///
    /// Returns `Ok(())` without touching an encoder in every case that is not
    /// an observation: a load with no slab, a fire no lane captured, and a
    /// `prefill_lse` node the plan's `attn.scores` seam does not name. That
    /// last one matters — a text may write a log-sum-exp for its own reasons,
    /// and a node the seam did not declare owns no plane.
    ///
    /// # Errors
    ///
    /// Whatever [`kernels_metal::attn::score::capture`] refuses: a sliding
    /// window (the row would not be the softmax the eviction papers define), a
    /// quantized key plane, a head this kernel is not stamped for.
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
    ) -> Result<(), kernels_metal::Error> {
        let Some(seat) = self.bindings().scores.clone() else {
            return Ok(());
        };
        let Some(plane) = seat.plane_of(lse) else {
            return Ok(());
        };
        // `lane_offset` is what turns this window's request number into a fire
        // lane, which is the slab's row space; a window in two pieces
        // therefore lands each piece on its own lanes with no lane map
        // anywhere.
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

    /// The arms themselves, in `kernels-metal`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn attention(&mut self, op: &Attention) -> Result<(), kernels_metal::Error> {
        match op {
            // MENLO-SEAM (engine side): the op names kv geometry the metal
            // builder never reads — kv_indptr/kv_indices/last_page_len stay
            // unresolved, kv_len resolves and rides through unread — because
            // the pool row carries the page tables; meanwhile the tables the
            // builder does read (positions, request_of_token, mask) are no
            // op's named inputs and bind from fire state here. The kernel
            // side of this seam is marked in `kernels_metal::attn`.
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
                // THE AMBIENT TABLES ARE CUT LIKE THE `q` THAT WILL BE READ
                // BESIDE THEM. The sdpa shaders index `position_ids[row]`,
                // `req_of_token[row]` and `attention_mask_enabled[row]` by the
                // LOCAL row of the launch, so a plan built for a windowed
                // class has to carry the window's own slice of each. What
                // stays absolute is what those tables CONTAIN — a lane id
                // into the fire-wide `page_indptr` — because slicing a vector
                // does not renumber it.
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
            // MENLO-SEAM: same misalignment as `PlanDecode`. Which plan the
            // trace declared is honored, not assumed: only the fa2 kind
            // exists on this plane, so an sm90 declaration is answered with
            // a panic, never a silently-substituted fa2 plan.
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
                        "an sm90 plan kind on a metal trace is a trace bug: this plane \
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
                // As `PlanDecode`: the window's slice of each ambient table.
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
            ),
            // **THE OP NAMES THE SHAPE, NOT THE KERNEL.** A `Prefill` is a
            // statement that these rows have keys behind them, and the tiled
            // shader is only the right answer when the rows SHARE those keys:
            // a fleet of thirty-two one-row lanes has a prefill's row count
            // and a decode's dataflow, and measures 370 tok/s tiled against
            // 728 per-row. So the arbitration is here, off the window's own
            // request count, and `kernels_metal::attn::arbiter` holds the
            // rule. See its header.
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
                &kernels_metal::tuning::current(),
            ),
            // THE TOWER'S ATTENTION. A real arm and not a refusal: this
            // plane's `attn::dense` is a written shader, so the family's rule
            // holds — the arm forwards, and if the entry declines it declines
            // by its own name.
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
                // THE OP-NAMED MASK IS CUT LIKE THE PLAN-CARRIED ONE, AND
                // THAT IS WHAT MAKES THE SEAM'S CLAIM TRUE.
                // `kernels_metal::attn::masked`'s MENLO-SEAM note says the
                // op's `mask` and `plan.mask` are "one buffer wearing two
                // names"; the plan's was built from `cut_rows` and this one
                // resolves WHOLE (`Run::cut` excludes `RuntimeInput::Mask`,
                // because a row offset is not a byte offset for a slab the IR
                // spells in bits). `cut_rows` is what knows the stride, so
                // applying it here is what makes the two names name the same
                // rows — rather than leaving them equal only while the masked
                // window happens to start at fire row zero, which today's
                // split order (`[masked, captures_scores, qo_one, rest]`)
                // makes true and no rule requires.
                self.cut_rows(self.tensor(*mask)),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.requests(),
                &kernels_metal::tuning::current(),
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
                    &kernels_metal::tuning::current(),
                )?;
                // ── **THE OBSERVATION, BESIDE THE ARM THAT ALREADY RAN**
                //    (`.wiki/alto/attn-score.md` §4: "the graph writes; the
                //    epilogue reads").
                //
                //    MENLO-SEAM (engine side): the op names `o` and `lse` and
                //    nothing else, because the per-key rectangle is not a
                //    value another node consumes — it is what this layer paid
                //    attention to, written into a slab the shell owns and the
                //    epilogue binds. So the arm asks fire state for it, the
                //    way the masked arm asks for its span table, and a load
                //    that seats no slab launches nothing at all.
                //
                //    **THE NODE IS ASKED, NOT THE OP.** Which plane this layer
                //    owns comes from the VALUE it writes, matched against the
                //    plan's `attn.scores` exports — the only reading that
                //    cannot be fooled by a text reusing `prefill_lse` somewhere
                //    the seam does not name.
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

            // The absorbed `mla` family (`attention.mla_*`), calling into
            // `kernels_metal::attn::mla`.
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
            } => attn::mla::attention_decode(
                self.ctx(),
                self.tensor(*q),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
            ),
            Attention::MlaPrefill {
                q,
                plan: _,
                q_pe,
                cache,
                heads,
                kv_lora_rank,
                sm_scale,
                o,
            } => attn::mla::attention_prefill(
                self.ctx(),
                self.ragged(*q),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
            ),
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
            } => attn::mla::attention_decode_selected(
                self.ctx(),
                self.tensor(*q),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                self.tensor(*selection),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
            ),
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
            } => attn::mla::attention_prefill_selected(
                self.ctx(),
                self.ragged(*q),
                NO_MLA_PLAN,
                self.tensor(*q_pe),
                self.tensor(*selection),
                self.pool(*cache),
                *heads,
                *kv_lora_rank,
                *sm_scale,
                self.tensor(*o),
            ),

            // The absorbed `ssm` family (`attention.ssm_*`), calling into
            // `kernels_metal::attn::ssm`.
            // qwen4's n-gram hasher and its dilated convolution: not on
            // this plane yet.
            Attention::PleNgramIds { .. } | Attention::PleNgramIdsChunked { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            Attention::SsmCausalConv1d {
                dilation: 2..,
                ..
            }
            | Attention::SsmCausalConv1dChunked {
                dilation: 2..,
                ..
            } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Attention::SsmCausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation: _,
                y,
            } => attn::ssm::causal_conv1d(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                &self.recurrent(*state),
                *conv_width,
                self.tensor(*y),
            ),
            Attention::SsmCausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation: _,
                y,
            } => attn::ssm::causal_conv1d_chunked(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*weight),
                &self.recurrent(*state),
                *conv_width,
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
                self.tensor(*y),
            ),

            // The absorbed `index` family (`attention.index_*`), calling into
            // `kernels_metal::attn::index`.
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
                selection,
            } => attn::index::topk(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*weights),
                self.pool(*keys),
                *heads,
                *head_dim,
                *top_k,
                self.tensor(*selection),
            ),
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

            // The absorbed `pool` family (`attention.pool_*`), calling into
            // `kernels_metal::attn::pool`.
            Attention::PoolBoundaryDecode {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
            } => attn::pool::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                self.tensor(*row_valid),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
            ),
            Attention::PoolBoundaryPrefill {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
            } => attn::pool::boundary_prefill(
                self.ctx(),
                self.ragged(*positions),
                self.tensor(*row_valid),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
            ),
            Attention::PoolGather {
                boundary_pos,
                boundary_req,
                pages,
                head_dim,
                ratio,
                entries,
            } => attn::pool::gather(
                self.ctx(),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*pages),
                *head_dim,
                *ratio,
                self.tensor(*entries),
            ),
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
        }
    }
}
