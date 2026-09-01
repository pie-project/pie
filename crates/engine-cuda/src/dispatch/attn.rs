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
    /// **THE CAPTURE ARM'S OBSERVATION** — one launch, or none at all.
    ///
    /// Returns `Ok(())` without touching a stream in every case that is not
    /// an observation: a load with no slab, a fire no lane captured, and a
    /// `prefill_lse` node the plan's `attn.scores` seam does not name. That
    /// last one matters — a text may write a log-sum-exp for its own reasons,
    /// and a node the seam did not declare owns no plane.
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
        // **AND THE OBSERVATION TAKES THE WINDOW BACK** (chunk 2c-b). The arm
        // above is on `crate::SHIFTED` now and may be handed the plane's
        // base; this launch is not an IR op at all, its grid is
        // `[requests, heads]`, and it reads `q + indptr[blockIdx.x]` off the
        // REBASED boundaries. So it wants the rectangle every other path
        // resolves — see `Run::windowed`.
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

    /// The arms themselves, in `kernels-cuda`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn attention(&mut self, op: &Attention) -> Result<(), kernels_cuda::Error> {
        match op {
            // ---- attention (anchor) ----
            // MENLO-SEAM (engine side): the op names its kv geometry as
            // device values, but the builder walks kv_indptr's and kv_len's
            // CONTENTS — so the arm routes through the geometry input's
            // cache space to the host twins the shell bound beside them
            // (`Run::planning`). kv_indices and last_page_len are never
            // read at build time; they ride the pool row into the launches.
            // The occupancy fact (`decode_max_grid_size`) is host
            // prepare-phase work the builder's purity keeps outside itself,
            // like the shell's one `Toggles::from_env` read riding in on
            // the bindings.
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
            // MENLO-SEAM: as `PlanDecode`, plus the qo side — the fire's
            // shared indptr has a host twin too, and the mask span table
            // (the op-named mask bits' per-request bounds, which no op
            // names) binds onto the plan here, for `attention.masked` to
            // find. It is taken at THIS NODE's window (`Run::mask_indptr`),
            // because the table is indexed by the schedule's own request
            // number; the byte offsets inside it stay absolute, because the
            // slab they address is handed over whole. Which builder runs is
            // the trace's declaration: the plan value's `StructKind` says
            // fa2 or sm90, and the arm follows it. `causal: true` is
            // `attention.prefill`'s reading; `attention.masked` replaces the
            // causal bound with its mask and never consults the flag.
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
                    // The row axis's half of `Run::planning`'s pin: what the
                    // builders CARVE at is `seat.rows`, which the ceiling
                    // raises to the bucket for every kind that reads a row
                    // total, and the staged row-total word stays on
                    // `live.rows`'s side of the split.
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
                                // `seat.rows` on this arm too, and since
                                // chunk 5 it is the BUCKET here as well:
                                // `sched_sm90` already took the carved pair
                                // (`total_num_rows`, `batch_size`) and only
                                // ever got the fire's, so freezing
                                // `same_schedule_for_all_heads` and the eight
                                // int offsets under it was a change to
                                // `Run::planning`'s clause and to nothing
                                // else. Hash hygiene and not a live path: the
                                // launcher answers a typed refusal before it
                                // launches an sm90 prefill at all
                                // (`attn::prefill_sm90`), so no gate can
                                // exercise this and none claims to.
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
            // A prefill plan can hold either kind the trace declared; the
            // arm follows the slot. The sm90 launcher still answers a typed
            // refusal in its own name (`attn::prefill_sm90`).
            //
            // **AND THE Q AXIS IS `ragged_q` AND NOT `ragged`** (bodies
            // design, chunk 2c-a). FIVE launches in this family hand FA2 a
            // by-value params block with no seat in it — `bufs.qo_indptr`
            // becomes `PrefillPagedParams::q_indptr`, and since chunk 2c-b
            // `BatchDecodeParams::q_indptr` too, and the kernel computes
            // `q + q_indptr[req] * stride` with no `win[1]` to add — so their
            // CSR has to match whatever `q`'s POINTER is, which under a plane
            // base is not the window's first row. Every other `ragged` in this
            // file feeds a SEATED kernel that adds the start itself and wants
            // the window-local reading it always had; `Run::ragged_q` argues
            // both halves.
            //
            // **AND THE LANE AXIS RIDES BESIDE IT** (chunk 2c-b): the five
            // names are on `crate::SHIFTED` now, so these regions CAN be
            // `plane_base`, and where the q pointer goes absolute the pool's
            // per-lane tables and the schedule's request numbers go with it —
            // `Run::pool_absolute` here, `Run::planning` at the plan op. All
            // five arms take that door for that reason, and nothing else in
            // this file does.
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
            // THE TOWER'S ATTENTION, AND IT TAKES NOTHING THIS FAMILY'S
            // OTHER ARMS TAKE: no pool, no plan slot, no window. `segments` is
            // the patch axis's own indptr and `self.tensor` cuts it at the
            // PATCH window, which is what makes one fire's images the ones
            // this launch sees.
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
                //    **THE NODE IS ASKED, NOT THE OP.** Which plane this
                //    layer owns comes from the VALUE it writes, matched
                //    against the plan's `attn.scores` exports — the same
                //    reading `exports::writer_classes` takes, and the only
                //    one that cannot be fooled by a text reusing
                //    `prefill_lse` somewhere the seam does not name.
                //
                //    `lane_offset` is what turns this window's request number
                //    into a fire lane, which is the slab's row space; a
                //    window in two pieces (P4's split, the capturing-prefill-
                //    beside-capturing-decode fire) therefore lands each piece
                //    on its own lanes with no lane map anywhere.
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
            // The op states its write geometry: the arm resolves the
            // per-token `write_page`/`write_offset` descriptors and the
            // entry lands each row in the stated cell (the old pool-row
            // write-table seam is closed).
            Attention::KvAppend {
                k,
                v,
                cache,
                write_page,
                write_offset,
            } => attn::kv_append(
                self.ctx(),
                self.ragged(*k),
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
                self.ragged(*plane),
                &self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            // ---- mla ----
            // MENLO-SEAM: the same host-twin routing as the attention plan
            // arms — kv_indptr and the op-named kv_len are walked as the
            // host copies on the planning twin. The builder's `causal` word
            // is derived from the fire's own boundaries: multi-token lanes
            // attend causally within themselves, single-token (decode)
            // lanes have nothing to order. The op's own `heads` and
            // `kv_lora_rank` ride in on the seat — `store::kv::probe` seats
            // them as `num_q_heads` and `head_dim`, and `head_dim` is what
            // `plan_mla` sizes its partial-output buffer at (`head_dim_o`).
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
                        // **THE CARVED PAIR** (chunk 5): `seat.rows` and
                        // `seat.shape.num_requests`, which `Run::planning`
                        // raises together to the numbers the `record::BodyKey`
                        // spells — the fire's bucket and its lane ceiling on a
                        // whole-fire body, and since the ceiling design's
                        // Option B this window's own classes' lattice rungs on
                        // a windowed one — and leaves at this window's own
                        // together everywhere else. The latent builder
                        // averages them into `cluster_size`, which is the only
                        // number of this payload that ever followed the fire.
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
                self.tensor(*kv_b),
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
                self.tensor(*kv_b),
                *heads,
                *kv_lora_rank,
                *v_head_dim,
                *nope_dim,
                &mut self.tensor(*o),
            ),
            // The arm resolves the op's write descriptors; the latent
            // writer's own remaining seam (it still re-derives the cells)
            // is marked at the entry.
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
                self.ragged(*q),
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
                self.ragged(*q),
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
            // **WHERE THE BUFFER TOUCHES THE FORWARD, HALF OF TWO** (alto
            // design §6, wave F3). `x` is the conv's in-projection rows —
            // dev's `mixed_qkv` — so a `RsVerb::Buffer` lane scatters them
            // into its slab here and a `RsVerb::FoldBuffered` lane gathers
            // them back over the GEMM output that just landed. The gather is
            // what "skipping the in-projection GEMM" means for a shell whose
            // graph is immutable: the projection still runs and is
            // overwritten, on the same stream, in front of the conv that
            // reads it.
            //
            // A fire with no buffered lane tests one `Option` and returns.
            //
            // **AND THE 2R SPLIT IS HERE, AS TWO LAUNCHES ON ONE STREAM**
            // (design §6's interior boundary, wave F3b). A row that folds a
            // prefix of the tokens it is writing cannot be one call —
            // `commit_len` truncates, so the tokens past the boundary would
            // get no outputs. So the head runs `[0, n)` and folds, and the
            // tail runs `[n, rows)` from the rolling state the head just
            // wrote and folds nothing. Both write their own share of `y`, so
            // the pair leaves exactly the rectangle one call would have. A
            // fire no row splits gets `None` and the single launch it always
            // made.
            Attention::SsmCausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => {
                self.rs_move("attention.ssm_causal_conv1d_chunked", *x, self.tensor(*x))?;
                // **THE ABSOLUTE LANE DOOR, AND ONLY THE CHUNKED ARMS TAKE
                // IT** (`Run::recurrent_absolute`): a body bakes the slot map
                // it is handed, `lane_offset` is not a function of its key, so
                // the map goes over whole and the kernel finds its lane at
                // `r + win[3]`. Off a plane base this is the sliced reading
                // the per-step conv above takes, byte for byte.
                //
                // **AND `ragged_lanes` IS THE OTHER HALF OF THE SAME
                // SENTENCE** (the grid-at-ceiling wave). This arm's grid is
                // one block per REQUEST and it counts them off the CSR's
                // length, so a body captured at one batch could serve no
                // larger one until the length became a function of the key:
                // the vector is the same rebased boundaries `ragged` hands
                // over, declared out to the ceiling the ladder spells, and
                // `win[2]` retires the requests past this fire's own. Nothing
                // in this arm's dispatch computes a grid — the count still
                // arrives as the shape of an operand, which is where every
                // other grid in this file comes from too.
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
            // The n-gram hasher: token ids against the lane's trailing
            // window, the same state discipline the convolution above keeps
            // — decode shifts unconditionally, chunked advances only over
            // the committed prefix and re-covers the tail.
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
            // The other half: `ba` is the `[b | a]` projection — dev's `a`
            // and `b` planes — and the prep is its only reader, so the move
            // stands immediately in front of it for the same reason the
            // conv's does.
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
            // The scan's half of the 2R split, and the same two launches for
            // the same reason (see the chunked conv above): the head folds
            // the boundary into the bank, the tail loads what the head wrote
            // and carries the rest of the row's outputs from it.
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
            // **THE KEY STRIDE IS WHAT THIS PLANE SERVES AND WHAT IT
            // REFUSES.** `index.cuh`'s `index_topk_paged` scans `j = 0 .. pos`
            // and reads cell `j` — one key per TOKEN, which is `ratio == 1`
            // and is glm_5's whole indexer. dsv4-flash keys one row per
            // COMPRESSED BLOCK and states its compressor's ratio; the metal
            // shader takes that stride as a number, `index.cuh` does not, and
            // serving it here would score cells nobody wrote and publish ids
            // in a space the reader does not walk. So the arm serves the
            // stride the kernel has and names the op for the rest. The day
            // `index.cuh` grows the parameter, this arm is the guard that
            // goes.
            Attention::IndexTopk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                ratio,
                selection,
            } => {
                if *ratio != 1 {
                    return Err(kernels_cuda::Error::Unsupported {
                        op: "attention.index_topk",
                    });
                }
                index::topk(
                    self.ctx(),
                    self.ragged(*q),
                    self.tensor(*weights),
                    &self.pool(*keys),
                    *heads,
                    *head_dim,
                    *top_k,
                    &mut self.tensor(*selection),
                )
            }
            // As `attention.mla_kv_append`: the arm resolves the op's write
            // descriptors, and the entry marks its own remaining seam.
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
            // MENLO-SEAM: the dsv4 compressor state (`state_kv`,
            // `state_score`) has no IR seat; the arm binds the slabs the
            // shell staged for the pooled space (`Run::slabs`).
            //
            // **`ape` HAS A SEAT NOW** — it is a checkpoint plane
            // (`attn.compressor.ape`) and not shell scratch — so the op's own
            // operand is what reaches the entry when the compressor states
            // one, and the shell's absent seat when it does not (the toy's
            // parameter-free pool, the `ape == nullptr` path).
            //
            // **AND THIS SHELL STILL STAGES ONE PLANE FOR EVERY POOLED
            // LAYER.** `engine-metal`'s reservation moved to one plane per
            // SPACE when `attention.pool_state_write` gave the slabs a
            // writer, because two pooled layers hold different projections at
            // the same paged cell. This shell's `fire.tables.pool_state` is
            // still fire-wide; a dsv4 artifact with two pooled layers reads
            // the later layer's state in the earlier layer's gather here.
            // Stated rather than hidden — the move belongs to a CUDA lane
            // that can build.
            Attention::PoolGather {
                boundary_pos,
                boundary_req,
                pages,
                ape,
                head_dim,
                ratio,
                entries,
            } => {
                let slabs = self.slabs();
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
                let slabs = self.slabs();
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
            // As the other appenders: the arm resolves the op's write
            // descriptors, and the entry marks its own remaining seam (the
            // dsv4 store still re-derives its cells).
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
            // **THE NSA FINE BRANCH REFUSES BY NAME ON THIS PLANE.**
            // `pool.cuh` carries `pool_lse_paged`, the DENSE reader over every
            // visible compressed row, and no selected twin of it — the metal
            // shell serves `attention.pool_lse_selected` off
            // `attn/pool.metal`'s `pool_lse_selected_paged`. Falling back to
            // the dense reader here would answer a different attention (every
            // compressed row instead of the indexer's chosen ones) under the
            // selected op's name, so this arm says which kernel is missing.
            // The day `pool.cuh` grows the selected walk, this arm is the one
            // line that changes.
            Attention::PoolLseSelected { .. } => Err(kernels_cuda::Error::Unsupported {
                op: "attention.pool_lse_selected",
            }),
        }
    }
}
