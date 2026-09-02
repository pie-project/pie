//! Paged flash attention over the fire's kv pool: the fa2 lattice, the
//! appenders, and the plan machinery those launches ride on. One entry per
//! IR variant; kernel selection lives below the entries so a dispatch arm
//! stays destructure → resolve → call.

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

/// The boundary vector must reach every lane the schedule names; checked
/// (refused, not asserted) because a shorter vector reads past its end in
/// release. `>=` and not `==`: a windowed launch's own rebased vector hits
/// equality, but a plane-base launch's vector is the fire's whole length.
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
    // `decode.cuh` reads `q + q_indptr[batch_idx] * q_stride_n`, checked
    // against a vector with fewer boundaries than the schedule names.
    lanes_carry(op, &q, plan.shape.lane_offset, plan.shape.num_requests)?;
    // An unsplit decode writes `o` at the work item directly, with no shift
    // for schedule-named fire lanes; refused rather than silently wrong.
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
/// The window composes with the mask: the causal bound is already folded
/// into the staged bits, the window is the model's own statement, and a key
/// outside it is dropped regardless. The schedule's window must agree with
/// the mask's, since `sched_prefill`'s kv chunking and the kernel's
/// `num_kv_chunks` both derive from `window_left` — hence [`PrefillPlan::accepts`]
/// is asked with the stated window rather than `None`.
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
/// ([`plan::plan_prefill_sm90`]); the launcher is not part of this lattice
/// and refuses.
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
            // Live-rows word when a body replay armed one, else ABSENT.
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
            // Live-rows word when a body replay armed one, else ABSENT.
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

/// The residual-block blend `elementwise.res_blend` launches: RMS-score the
/// prefix and `B` candidate blocks, softmax, blend — one fused pass.
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
    // The kernel walks `blocks + (j * block_rows + row) * hidden` where
    // `block_rows` is the window's row count (`y.rows`), but the true stride
    // between stacked candidate planes is the plane's row capacity — which a
    // `Tensor` cannot express (no height field). So a windowed fire whose
    // planes are taller than its window is refused below, checked against
    // the actual pointer gap, rather than blended off the wrong rows.
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
            // Live-rows word when a body replay armed one, else ABSENT.
            ctx.stage(),
        ],
    )
}
