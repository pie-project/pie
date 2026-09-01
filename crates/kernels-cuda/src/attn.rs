//! `Attention`: paged flash attention over the fire's kv pool — the fa2
//! lattice, the appenders, and the plan machinery those launches ride on.
//! One entry per IR variant; kernel *selection* (lattice point, variant arm,
//! naive-vs-fa2, smem arm) lives below the entries, so a dispatch arm stays
//! destructure → resolve → call (decision #13).
//!
//! **The §6 substitution, stated once for the whole family.** Everywhere the
//! old path reached into implicit engine state — `Ctx.raised::<Fa2Decode>`,
//! `Ctx.raised::<MlaPlanned>`, the pool row's smuggled `qo_indptr`, the
//! raised mask/fire-table views — the new entries take explicit arguments
//! instead: the plan structs built by [`plan`]'s pure builders, the
//! [`RaggedTensor`] whose indptr is the fire's shared boundaries, and the
//! geometry tensors the IR names on the op. What has no IR seat and no
//! plan seat is an explicit argument marked `MENLO-SEAM` at its site; the
//! engine binds it from fire state, visibly.
//!
//! The families that shared the old file keep their seats: [`mla`],
//! [`index`], [`pool`], and the recurrent [`ssm`] mixers live as submodules;
//! the tier-2 fused point moved out to [`custom`](crate::custom), the
//! escape hatch's own family. `elementwise.res_blend`'s launch lives here
//! too (the old CANON row `norm.res_blend -> attn::attn_res_blend`).

pub mod fa2;

pub mod fa2_abi;

pub mod index;

pub mod kv;

pub mod mla;

pub mod plan;

pub mod pool;

pub mod sched;

pub mod sched_decode;

pub mod sched_mla;

pub mod sched_prefill;

pub mod sched_sm90;

pub mod ssm;

use crate::error::Error;
use dtype::Dtype;

use crate::attn::fa2_abi::{Buffers, make_decode_params, make_prefill_params};
use crate::attn::plan::{DecodePlan, PrefillPlan, PrefillPlanSm90};
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const BLOCK: u32 = 256;

/// The attention entries never soft-cap: `attention.logit_softcap` is its
/// own op, applied where the model says, as before.
const NO_SOFT_CAP: f32 = 0.0;

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

#[must_use]
const fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    #[must_use]
    const fn head_dim_block(head_dim: u32) -> u32 {
        const SINK_BLOCK_MAX: u32 = 128;

        const SINK_BLOCK_MIN: u32 = 32;

        if head_dim < SINK_BLOCK_MIN {
            SINK_BLOCK_MIN
        } else if head_dim > SINK_BLOCK_MAX {
            SINK_BLOCK_MAX
        } else {
            head_dim
        }
    }

    Launch::grid([rows, heads, 1], [head_dim_block(head_dim), 1, 1])
}

/// The head count a row's width spells at a stated head width.
fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if width == 0 || width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide row does not divide by the stated head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

fn attention_lands(op: &'static str, q: Tensor, o: &Tensor) {
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{op}` lands one output row per query row"
    );
}

fn lse_plane(op: &'static str, lse: &Tensor, rows: u32, heads: u32) {
    debug_assert_eq!(lse.dtype, Dtype::F32, "`{op}`'s log-sum-exp plane is f32");
    debug_assert!(
        lse.rows == rows && lse.width == heads,
        "`{op}`'s log-sum-exp plane is one f32 per head per row"
    );
}

/// **THE BOUNDARY VECTOR MUST REACH EVERY LANE THE SCHEDULE NAMES.**
///
/// BOTH SIDES ARE DRIVER-BOUND — the ragged boundary vector is one the
/// windowing rule names, the plan's shape is the engine's, and no op names
/// either — so a disagreement is refused rather than asserted (the boundary
/// rule at `refuse`). Asserted, a release build reads past the end of a
/// shorter vector and takes the kv pool's spans from whatever follows it.
///
/// **AND IT IS `>=` AND NOT `==` BECAUSE THE VECTOR HAS TWO READINGS.** A
/// launch cut to its window is handed its own `[lanes + 1]` boundaries and
/// `lane_offset` is zero, which is the equality this door always checked. A
/// launch handed the PLANE's base takes the FIRE's whole vector, whose length
/// is the fire's lanes and not this window's — so what is left to check is
/// the thing that was ever wrong: a vector too SHORT for the last request
/// `lane_offset + num_requests - 1` the schedule staged.
fn lanes_carry(
    op: &'static str,
    q: &RaggedTensor,
    lane_offset: u32,
    num_requests: u32,
) -> Result<(), Error> {
    let carried = q.indptr.rows.saturating_sub(1);
    if carried >= lane_offset.saturating_add(num_requests) {
        return Ok(());
    }
    Err(refuse(
        op,
        format!(
            "the fire's indptr spells {carried} lanes and this schedule names {num_requests} \
             requests from lane {lane_offset}"
        ),
    ))
}

fn pool_buffers(q_ptr: u64, pool: &KvPool, plan_ws: plan::Workspace, o_ptr: u64) -> Buffers {
    Buffers {
        q: q_ptr,
        k_pages: pool.keys.ptr,
        v_pages: pool.values.ptr,
        o: o_ptr,
        kv_page_indices: pool.page_indices.ptr,
        kv_page_indptr: pool.page_indptr.ptr,
        kv_last_page_lens: pool.last_page_lens.ptr,
        qo_indptr: 0,
        lse: 0,
        int_buffer: plan_ws.int_ptr,
        float_buffer: plan_ws.float_ptr,
    }
}

/// The decode launch both entries share. The plan's agreement with the
/// op's restated facts is the caller's duty ([`DecodePlan::accepts`]).
#[allow(clippy::too_many_arguments)]
fn fa2_decode(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    attention_lands(op, q.data, o);
    let window_left = plan::window_left(op, window)?;
    // **THE BOUNDARIES ARE AN ARGUMENT NOW**, because `decode.cuh` reads
    // `q + q_indptr[batch_idx] * q_stride_n` where it used to take
    // `batch_idx` itself. The two are the same number for a launch handed
    // its own rebased vector — a decode request is one query row, so entry
    // `l` of that vector IS `l` — and they part company exactly where they
    // must: `batch_idx` is `lane_offset + r` and the rows it names are the
    // plane's. Both readings are checked here, and the check is the one the
    // prefill door keeps: a vector with fewer boundaries than the schedule
    // names lets every work item past the first read whatever follows it.
    lanes_carry(op, &q, plan.shape.lane_offset, plan.shape.num_requests)?;
    // **AND AN UNSPLIT DECODE HAS NO ABSOLUTE READING OF ITS OUTPUT.**
    // `decode.cuh` writes `o + bx * ...` at the WORK ITEM, which is a row of
    // whatever `o` points at and nothing the schedule can shift; under a
    // split it writes the plan's partial plane instead and the fold behind it
    // carries the window on the staged seat. A schedule naming fire lanes is
    // one whose `o` is the plane's base, so the two cannot meet. They never
    // do — a plane base means a body, a body means `Graphs::On`, and both
    // decode planners take the split unconditionally under a graph-shaped
    // build — and this is the refusal that says so out loud rather than the
    // comment that hoped so.
    if plan.shape.lane_offset > 0 && !plan.info.split_kv {
        return Err(refuse(
            op,
            "this schedule names fire lanes and did not split kv, so its output rows would be \
             the launch's own and not the plane's",
        ));
    }

    let _ = kv::dequant_active(
        ctx,
        op,
        pool,
        stated(op, plan.shape.num_kv_heads)?,
        stated(op, head_dim)?,
    );

    let mut bufs = pool_buffers(q.data.ptr, pool, plan.workspace, o.ptr);
    bufs.qo_indptr = q.indptr.ptr;
    if let Some(lse) = &lse {
        lse_plane(op, lse, q.data.rows, plan.shape.num_q_heads);
        bufs.lse = lse.ptr;
    }
    let arm = fa2::decode_arm(plan.full_attention_variant(), window_left, NO_SOFT_CAP);
    let (params, split) =
        make_decode_params(plan, &bufs, window_left, NO_SOFT_CAP, sm_scale, false);
    fa2::decode(
        ctx,
        op,
        fa2::DecodePoint {
            head_dim: plan.shape.head_dim,
            group_size: plan.shape.group_size(),
            arm,
            padded_batch_size: params.padded_batch_size,
            num_kv_heads: plan.shape.num_kv_heads,
            device: plan.device,
        },
        &params,
    )?;
    if plan.info.split_kv {
        fa2::fold(ctx, op, &split)
    } else {
        Ok(())
    }
}

/// The prefill launch every prefill entry shares. The plan's agreement
/// with the op's restated facts is the caller's duty
/// ([`PrefillPlan::accepts`]); `mask` is `attention.masked`'s custom-mask
/// pair (bits beside their span table), riding the params instead of the
/// causal bound.
#[allow(clippy::too_many_arguments)]
fn fa2_prefill(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
    mask: Option<(Tensor, Tensor)>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    attention_lands(op, q.data, o);
    let window_left = plan::window_left(op, window)?;

    let _ = kv::dequant_active(
        ctx,
        op,
        pool,
        stated(op, plan.shape.num_kv_heads)?,
        stated(op, head_dim)?,
    );

    lanes_carry(op, &q, plan.shape.lane_offset, plan.shape.num_requests)?;
    let mut bufs = pool_buffers(q.data.ptr, pool, plan.workspace, o.ptr);
    bufs.qo_indptr = q.indptr.ptr;
    if let Some(lse) = &lse {
        lse_plane(op, lse, q.data.rows, plan.shape.num_q_heads);
        bufs.lse = lse.ptr;
    }
    let arm = match mask {
        Some(_) => fa2::prefill_custom_arm(NO_SOFT_CAP),
        None => fa2::prefill_arm(plan.full_attention_variant(), plan.causal, NO_SOFT_CAP),
    };
    let (mut params, split) = make_prefill_params(plan, &bufs, window_left, NO_SOFT_CAP, sm_scale);
    if let Some((bits, indptr)) = mask {
        params.maybe_custom_mask = bits.ptr;
        params.maybe_mask_indptr = indptr.ptr;
    }
    fa2::prefill(
        ctx,
        op,
        fa2::PrefillPoint {
            head_dim: plan.shape.head_dim,
            cta_tile_q: plan.cta_tile_q(),
            arm,
            padded_batch_size: params.padded_batch_size,
            num_kv_heads: plan.shape.num_kv_heads,
            device: plan.device,
        },
        &params,
    )?;
    if plan.info.split_kv {
        fa2::fold(ctx, op, &split)
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.decode";
    plan.accepts(OP, head_dim, window)?;
    fa2_decode(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_lse(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.decode_lse";
    plan.accepts(OP, head_dim, window)?;
    fa2_decode(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, Some(lse))
}

/// The stated kv head count must be the one the plan was carved at; the
/// boundaries ride in `q.indptr`.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    plan.accepts(OP, head_dim, Some(kv_heads), window)?;
    fa2_prefill(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None, None)
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_lse(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    plan.accepts(OP, head_dim, Some(kv_heads), window)?;
    fa2_prefill(
        ctx,
        OP,
        q,
        plan,
        pool,
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        None,
    )
}

/// Prefill against the op's `mask` (packed `u8` bits) instead of the causal
/// bound, optionally inside a sliding window.
///
/// **THE WINDOW COMPOSES WITH THE MASK, AND ALWAYS DID.** C1 recorded a third
/// blocker here — "fa2 instantiates no custom-mask + sliding-window arm" — and
/// it was a misreading of the variant's own template arguments.
/// `VariantCustom` is `flashinfer::DefaultAttention<use_custom_mask = true,
/// use_sliding_window = true, ...>`: its `REGISTER_LOGITS_MASK` ANDs the
/// custom bit with `kv_idx + qo_len + window_left >= kv_len + qo_idx`, and
/// `window_left` is `params.window_left` when that is non-negative and
/// `kv_len` — which makes the term vacuous — when it is not. So the arm the
/// unwindowed masked path has been firing since C1 IS the windowed arm, run
/// at `window_left = -1`. There was nothing to instantiate; there was a
/// refusal in front of it.
///
/// What the refusal actually feared — "a windowed schedule would discard
/// positions the mask may keep" — is the wrong way round for a model that
/// states a window. Gemma's masked reading is *causal ∧ mask ∧ window*: the
/// causal bound is already folded into the staged bits (`engine_cuda::mask`),
/// the window is the model's own statement on the node, and a key outside it
/// is dropped by the variant whether the schedule visited it or not. The
/// schedule's window is not an approximation of the mask, it is the second
/// conjunct — and it has to AGREE, because `sched_prefill` sizes its kv
/// chunking from `window_left` and the kernel's own `num_kv_chunks` recomputes
/// it from the same number. A full schedule under a windowed launch would put
/// the partials and the merge at two different counts, which is the silent
/// half of this failure and the reason [`PrefillPlan::accepts`] is asked with
/// the stated window rather than with `None`.
#[allow(clippy::too_many_arguments)]
pub fn masked(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.masked";
    debug_assert_eq!(mask.dtype, Dtype::U8, "`{OP}` reads packed u8 mask bits");
    // The op states no kv head count; the window must be the one the schedule
    // carved its kv spans for, windowed or not.
    plan.accepts(OP, head_dim, None, window)?;
    // MENLO-SEAM: the op names the mask bits, but their per-request span
    // table has no IR seat — the engine binds it onto the plan at build
    // (`plan_prefill`'s `mask_indptr`).
    let Some(mask_indptr) = plan.mask_indptr else {
        return Err(refuse(
            OP,
            "no mask span table rides this prefill plan; the engine binds one at plan build",
        ));
    };
    fa2_prefill(
        ctx,
        OP,
        q,
        plan,
        pool,
        window,
        head_dim,
        sm_scale,
        o,
        None,
        Some((mask, mask_indptr)),
    )
}

/// The sm90 plan's consumer seat. The schedule builder is real
/// ([`plan::plan_prefill_sm90`]); the launcher was never part of this
/// lattice — the old plane refused the same way.
#[allow(clippy::too_many_arguments)]
pub fn prefill_sm90(
    _ctx: &Ctx,
    _q: RaggedTensor,
    _plan: &PrefillPlanSm90,
    _pool: &KvPool,
    _window: Option<u32>,
    _head_dim: u32,
    _kv_heads: u32,
    _sm_scale: f32,
    _o: &mut Tensor,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "attention.prefill_sm90",
    })
}

/// Folds attention-sink mass into `o` using its log-sum-exp, in place on
/// `o`.
pub fn sink(
    ctx: &Ctx,
    o: &mut Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attention.sink";
    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = row_heads(OP, o.width, head_dim)?;
    lse_plane(OP, &lse, o.rows, heads);
    let rows = count(OP, "rows", o.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/attention.cuh",
            symbol(&format!("::pie::attn::sink_rescale<{t}>")),
        )
        .apply(per_head_elementwise(o.rows, heads, head_dim)),
        &[
            o.arg(),
            lse.arg(),
            sink.arg(),
            rows.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// Merges two attention outputs by their log-sum-exps.
#[allow(clippy::too_many_arguments)]
pub fn merge_lse(
    ctx: &Ctx,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.merge_lse";

    const COMBINE_BLOCK_MIN: u32 = 32;

    const COMBINE_BLOCK_MAX: u32 = 256;

    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        o1.rows == o.rows && o2.rows == o.rows,
        "the merged outputs are one row per query row"
    );
    lse_plane(OP, lse, o.rows, heads);
    let heads = count(OP, "the head count this merge states", heads)?;
    let head_dim = count(OP, "the head width this merge states", head_dim)?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/attention.cuh",
            symbol(&format!("::pie::attn::merge_lse_combine<{t}>")),
        )
        .apply(Launch::grid(
            [o.rows, heads.unsigned_abs(), 1],
            [
                head_dim
                    .unsigned_abs()
                    .clamp(COMBINE_BLOCK_MIN, COMBINE_BLOCK_MAX),
                1,
                1,
            ],
        )),
        &[
            o1.arg(),
            lse1.arg(),
            o2.arg(),
            lse2.arg(),
            o.arg(),
            lse.arg(),
            heads.arg(),
            head_dim.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// `x = cap * tanh(x / cap)`, in place on `x`.
pub fn logit_softcap(ctx: &Ctx, x: &mut Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "attention.logit_softcap";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if cap.is_nan() || cap <= 0.0 {
        return Err(refuse(OP, format!("{cap} is not a logit soft cap")));
    }
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/attention.cuh",
            symbol(&format!("::pie::attn::logit_softcap<{t}>")),
        )
        .apply(elementwise(lanes)),
        &[x.arg(), cap.arg(), n.arg()],
    )
}

/// Appends `k`/`v` rows into the pool's pages, each row landing in the cell
/// the op's `write_page`/`write_offset` descriptors state. Boundary-aware:
/// the fire's shared indptr rides in `k` (the envelope refresh's lane walk).
///
/// (History: the op used to state kv_indices + positions, which this
/// appender never read — that seam closed when the IR named the write
/// geometry itself.)
pub fn kv_append(
    ctx: &Ctx,
    k: RaggedTensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.kv_append";
    dtype_dispatch!(OP, k.data.dtype, { Bf16 => () });
    debug_assert!(
        v.rows == k.data.rows && v.width == k.data.width && v.dtype == k.data.dtype,
        "the value plane is appended beside the key plane, one rectangle"
    );
    kv::write_kv_to_pages(ctx, OP, k.data, v, k.indptr, pool, write_page, write_offset)
}

/// Appends one plane shared as both k and v.
pub fn kv_append_shared(
    ctx: &Ctx,
    plane: RaggedTensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.kv_append_shared";
    dtype_dispatch!(OP, plane.data.dtype, { Bf16 => () });
    kv::write_kv_to_pages(
        ctx,
        OP,
        plane.data,
        plane.data,
        plane.indptr,
        pool,
        write_page,
        write_offset,
    )
}

/// The residual-block blend `elementwise.res_blend` launches (the old CANON
/// row `norm.res_blend -> attn::attn_res_blend`): RMS-score the prefix and `B`
/// candidate blocks, softmax, blend — one fused pass.
///
/// The kernel walks `blocks` as `B` stacked `[rows, hidden]` planes, so the
/// block values must land adjacently; the arena hands them out that way,
/// and anything else is refused rather than blended wrong.
pub fn res_blend(
    ctx: &Ctx,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.res_blend";

    /// The device text's `kMaxBlocks`: the softmax scratch bound.
    const MAX_BLOCKS: usize = 32;

    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = count(OP, "rows", y.rows)?;
    let hidden = count(OP, "the blended row's width", y.width)?;
    if blocks.len() > MAX_BLOCKS {
        return Err(refuse(
            OP,
            format!(
                "{} candidate blocks exceed the kernel's softmax scratch bound of {MAX_BLOCKS}",
                blocks.len()
            ),
        ));
    }
    // **THE PITCH THIS ENTRY PASSES IS THE WINDOW'S, AND THE KERNEL NEEDS THE
    // PLANE'S** — the one finding that keeps `elementwise.res_blend` off
    // `engine_cuda::SHIFTED`, written here because this is where the number is
    // chosen. The kernel walks `blocks + (j * block_rows + row) * hidden`, and
    // `block_rows` is `y.rows` below: the WINDOW's row count, which is what
    // `Run::cut` hands a shifting region while its pointers stay the plane's
    // base. The stride between two stacked candidate planes is the PLANE's row
    // capacity, and the two coincide only when the window is the whole plane.
    //
    // **THIS ENTRY CANNOT DERIVE THE PLANE'S CAPACITY.** A `Tensor` carries
    // `ptr`, `rows`, `width`, `dtype` and no height; `rows` is the window's by
    // the paragraph above. The only other witness in the room is the GAP
    // between two consecutive candidate pointers — which is the true pitch,
    // and is why the refusal below can be written at all — but it exists only
    // when there are two of them, and a lone candidate never multiplies
    // `block_rows` by anything. So the gap answers the case the kernel reads
    // and says nothing in the case it does not, which is a coincidence to rest
    // an addressing rule on rather than a derivation.
    //
    // **AND THE REFUSAL BELOW IS THE HONEST GUARD MEANWHILE**: it compares the
    // pointers against the pitch this entry is about to pass, so a windowed
    // fire whose planes are taller than its window is REFUSED here rather than
    // blended off the wrong rows. What the name is owed is the plane's height,
    // from the side that knows it — the engine's arena — either as a field on
    // the handle or as a fifth seat word (the seat holds four now, and this
    // entry reads only the row pair). Neither is this crate's to add.
    let plane_bytes = u64::from(y.rows) * u64::from(y.width) * 2;
    for pair in blocks.windows(2) {
        if pair[1].ptr != pair[0].ptr.wrapping_add(plane_bytes) {
            return Err(refuse(
                OP,
                "the candidate blocks do not land as stacked planes; the kernel walks \
                 `blocks + (j * rows + t) * hidden` and cannot gather scattered slots",
            ));
        }
    }
    let first = blocks.first().map_or(prefix.ptr, |b| b.ptr);
    ctx.fire(
        OP,
        Fire::at(
            "elemwise/norm.cuh",
            symbol(&format!("::pie::elemwise::res_blend<{t}>")),
        )
        .apply(Launch::per_row(y.rows, BLOCK)),
        &[
            prefix.arg(),
            crate::jit::ArgValue::Ptr(first),
            weight.arg(),
            proj.arg(),
            y.arg(),
            stated(OP, u32::try_from(blocks.len()).unwrap_or(u32::MAX))?.arg(),
            hidden.arg(),
            rows.arg(),
            eps.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
