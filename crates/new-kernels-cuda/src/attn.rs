//! `Attention`: paged flash attention over the fire's kv pool — the fa2
//! lattice, the appenders, and the plan machinery those launches ride on.
//! One entry per IR variant; kernel *selection* (lattice point, variant arm,
//! naive-vs-fa2, smem arm) lives below the entries, so a dispatch arm stays
//! destructure → resolve → call (decision #13).
//!
//! **The §6 substitution, stated once for the whole family.** Everywhere the
//! old path reached into implicit driver state — `Ctx.raised::<Fa2Decode>`,
//! `Ctx.raised::<MlaPlanned>`, the pool row's smuggled `qo_indptr`, the
//! raised mask/fire-table views — the new entries take explicit arguments
//! instead: the plan structs built by [`plan`]'s pure builders, the
//! [`RaggedTensor`] whose indptr is the fire's shared boundaries, and the
//! geometry tensors the IR names on the op. What has no IR seat and no
//! plan seat is an explicit argument marked `MENLO-SEAM` at its site; the
//! driver binds it from fire state, visibly.
//!
//! The families that shared the old file keep their seats: [`mla`],
//! [`index`], [`pool`], and the [`fused`] tier-2 entry live as submodules;
//! `norm.res_blend`'s launch lives here too (the old CANON row
//! `norm.res_blend -> attn::attn_res_blend`).

pub mod fa2;

pub mod fused;

pub mod index;

pub mod kv;

pub mod mla;

pub mod plan;

pub mod pool;

use new_kernels::KernelError;
use new_model_ir::Dtype;

use crate::attn::fa2::params::{Buffers, make_decode_params, make_prefill_params};
use crate::attn::plan::{DecodePlan, PrefillPlan, PrefillPlanSm90};
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
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
fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, KernelError> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if width == 0 || width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide row does not divide by the stated head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

/// The old `agrees`: a stated head width must be the one the plan was
/// carved at — plan facts are driver-supplied, so disagreement is refused,
/// not asserted.
fn planned_head_dim(op: &'static str, planned: u32, stated: u32) -> Result<(), KernelError> {
    if planned == stated {
        return Ok(());
    }
    Err(refuse(
        op,
        format!(
            "the stated head width {stated} is not the {planned} this fire's attention \
             schedule was planned at"
        ),
    ))
}

/// The old `variant_agrees`: the windowed/full reading must match the plan.
fn planned_variant(
    op: &'static str,
    planned_full: bool,
    window: Option<u32>,
) -> Result<(), KernelError> {
    if planned_full == window.is_none() {
        return Ok(());
    }
    Err(refuse(
        op,
        "the stated window is not the reading this fire's attention schedule was planned for",
    ))
}

fn planned_window(
    op: &'static str,
    planned: Option<u32>,
    stated: Option<u32>,
) -> Result<(), KernelError> {
    if planned == stated {
        return Ok(());
    }
    Err(refuse(
        op,
        format!(
            "the stated window {stated:?} is not the {planned:?} this fire's prefill \
             schedule carved its kv spans for"
        ),
    ))
}

fn attention_lands(op: &'static str, q: Tensor, o: &Tensor) {
    let _ = op;
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "the attention lands one output row per query row"
    );
}

fn lse_plane(op: &'static str, lse: &Tensor, rows: u32, heads: u32) {
    let _ = op;
    debug_assert_eq!(lse.dtype, Dtype::F32, "the log-sum-exp plane is f32");
    debug_assert!(
        lse.rows == rows && lse.width == heads,
        "the log-sum-exp plane is one f32 per head per row"
    );
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

#[allow(clippy::too_many_arguments)]
fn fa2_decode(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: Option<&mut Tensor>,
) -> Result<(), KernelError> {
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    attention_lands(op, q, o);
    planned_head_dim(op, plan.shape.head_dim, head_dim)?;
    planned_variant(op, plan.full_attention_variant(), window)?;
    let window_left = plan::window_left(op, window)?;

    let _ = kv::dequant_active(
        ctx,
        op,
        pool,
        stated(op, plan.shape.num_kv_heads)?,
        stated(op, head_dim)?,
    );

    let mut bufs = pool_buffers(q.ptr, pool, plan.workspace, o.ptr);
    if let Some(lse) = &lse {
        lse_plane(op, lse, q.rows, plan.shape.num_q_heads);
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
) -> Result<(), KernelError> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    attention_lands(op, q.data, o);
    planned_head_dim(op, plan.shape.head_dim, head_dim)?;
    planned_window(op, plan.window, window)?;
    let window_left = plan::window_left(op, window)?;

    let _ = kv::dequant_active(
        ctx,
        op,
        pool,
        stated(op, plan.shape.num_kv_heads)?,
        stated(op, head_dim)?,
    );

    debug_assert_eq!(
        q.indptr.rows,
        plan.shape.num_requests + 1,
        "the fire's indptr spells the batch this schedule was planned at"
    );
    let mut bufs = pool_buffers(q.data.ptr, pool, plan.workspace, o.ptr);
    bufs.qo_indptr = q.indptr.ptr;
    if let Some(lse) = &lse {
        lse_plane(op, lse, q.data.rows, plan.shape.num_q_heads);
        bufs.lse = lse.ptr;
    }
    let arm = fa2::prefill_arm(plan.full_attention_variant(), plan.causal, NO_SOFT_CAP);
    let (params, split) = make_prefill_params(plan, &bufs, window_left, NO_SOFT_CAP, sm_scale);
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
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), KernelError> {
    fa2_decode(
        ctx,
        "attention.decode",
        q,
        plan,
        pool,
        window,
        head_dim,
        sm_scale,
        o,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_lse(
    ctx: &Ctx,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), KernelError> {
    fa2_decode(
        ctx,
        "attention.decode_lse",
        q,
        plan,
        pool,
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
    )
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
) -> Result<(), KernelError> {
    const OP: &str = "attention.prefill";
    planned_kv_heads(OP, plan, kv_heads)?;
    fa2_prefill(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None)
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
) -> Result<(), KernelError> {
    const OP: &str = "attention.prefill_lse";
    planned_kv_heads(OP, plan, kv_heads)?;
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
    )
}

fn planned_kv_heads(
    op: &'static str,
    plan: &PrefillPlan,
    kv_heads: u32,
) -> Result<(), KernelError> {
    if plan.shape.num_kv_heads == kv_heads {
        return Ok(());
    }
    Err(refuse(
        op,
        format!(
            "the stated kv head count {kv_heads} is not the {} this fire's prefill \
             schedule was planned at",
            plan.shape.num_kv_heads
        ),
    ))
}

/// Prefill against the mask the plan carries instead of the causal bound.
/// The schedule has to cover the whole prefix — the mask rides the launch,
/// so a windowed schedule would discard positions the mask may keep.
#[allow(clippy::too_many_arguments)]
pub fn masked(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.masked";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    attention_lands(OP, q.data, o);
    planned_head_dim(OP, plan.shape.head_dim, head_dim)?;
    if plan.window.is_some() {
        return Err(refuse(
            OP,
            "this fire's masked prefill schedule was carved for a windowed reading; the \
             window rides the launch, so the schedule has to cover the whole prefix",
        ));
    }
    // MENLO-SEAM: the IR's `Masked` names no mask operand; the driver binds
    // its fire mask onto the plan at build time (`plan::Mask`).
    let Some(mask) = plan.mask else {
        return Err(refuse(
            OP,
            "no mask rides this prefill plan; the driver binds one at plan build",
        ));
    };
    let window_left = plan::window_left(OP, window)?;

    let _ = kv::dequant_active(
        ctx,
        OP,
        pool,
        stated(OP, plan.shape.num_kv_heads)?,
        stated(OP, head_dim)?,
    );

    let mut bufs = pool_buffers(q.data.ptr, pool, plan.workspace, o.ptr);
    bufs.qo_indptr = q.indptr.ptr;
    let arm = fa2::prefill_custom_arm(NO_SOFT_CAP);
    let (mut params, split) = make_prefill_params(plan, &bufs, window_left, NO_SOFT_CAP, sm_scale);
    params.maybe_custom_mask = mask.bits.ptr;
    params.maybe_mask_indptr = mask.indptr.ptr;
    fa2::prefill(
        ctx,
        OP,
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
        fa2::fold(ctx, OP, &split)
    } else {
        Ok(())
    }
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
) -> Result<(), KernelError> {
    Err(KernelError::Unsupported {
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
) -> Result<(), KernelError> {
    const OP: &str = "attention.sink";
    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = row_heads(OP, o.width, head_dim)?;
    lse_plane(OP, &lse, o.rows, heads);
    let rows = stated(OP, nonzero(OP, "rows", o.rows)?)?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/attn_sink.cuh",
            symbol(&format!("::pie::attn::attn_sink_rescale<{t}>")),
        )
        .apply(per_head_elementwise(o.rows, heads, head_dim)),
        &[
            o.arg(),
            lse.arg(),
            sink.arg(),
            rows.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
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
) -> Result<(), KernelError> {
    const OP: &str = "attention.merge_lse";

    const COMBINE_BLOCK_MIN: u32 = 32;

    const COMBINE_BLOCK_MAX: u32 = 256;

    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        o1.rows == o.rows && o2.rows == o.rows,
        "the merged outputs are one row per query row"
    );
    lse_plane(OP, lse, o.rows, heads);
    let heads = stated(OP, nonzero(OP, "the head count this merge states", heads)?)?;
    let head_dim = stated(
        OP,
        nonzero(OP, "the head width this merge states", head_dim)?,
    )?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/dsv4_compress.cuh",
            symbol(&format!("::pie::attn::combine_attn_outputs<{t}>")),
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
        ],
    )
}

/// `x = cap * tanh(x / cap)`, in place on `x`.
pub fn logit_softcap(ctx: &Ctx, x: &mut Tensor, cap: f32) -> Result<(), KernelError> {
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
            "attn/softcap.cuh",
            symbol(&format!("::pie::attn::logit_softcap<{t}>")),
        )
        .apply(elementwise(lanes)),
        &[x.arg(), cap.arg(), n.arg()],
    )
}

/// Appends `k`/`v` rows into the pool's pages. Boundary-aware: the fire's
/// shared indptr rides in `k`.
///
// MENLO-SEAM: the IR states the write geometry as kv_indices + positions;
// this appender addresses by the pool's page tables and the fire indptr
// (the driver derives both from the same inputs), so the stated pair goes
// unread — the same seam metal noted.
pub fn kv_append(
    ctx: &Ctx,
    k: RaggedTensor,
    v: Tensor,
    pool: &KvPool,
    kv_indices: Tensor,
    positions: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.kv_append";
    let _ = (kv_indices, positions);
    dtype_dispatch!(OP, k.data.dtype, { Bf16 => () });
    debug_assert!(
        v.rows == k.data.rows && v.width == k.data.width && v.dtype == k.data.dtype,
        "the value plane is appended beside the key plane, one rectangle"
    );
    kv::write_kv_to_pages(ctx, OP, k.data, v, k.indptr, pool)
}

/// Appends one plane shared as both k and v.
pub fn kv_append_shared(
    ctx: &Ctx,
    plane: RaggedTensor,
    pool: &KvPool,
    kv_indices: Tensor,
    positions: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "attention.kv_append_shared";
    let _ = (kv_indices, positions);
    dtype_dispatch!(OP, plane.data.dtype, { Bf16 => () });
    kv::write_kv_to_pages(ctx, OP, plane.data, plane.data, plane.indptr, pool)
}

/// The residual-block blend `norm.res_blend` launches (the old CANON row
/// `norm.res_blend -> attn::attn_res_blend`): RMS-score the prefix and `B`
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
) -> Result<(), KernelError> {
    const OP: &str = "norm.res_blend";

    /// The device text's `kMaxBlocks`: the softmax scratch bound.
    const MAX_BLOCKS: usize = 32;

    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = stated(OP, nonzero(OP, "rows", y.rows)?)?;
    let hidden = stated(OP, nonzero(OP, "the blended row's width", y.width)?)?;
    if blocks.len() > MAX_BLOCKS {
        return Err(refuse(
            OP,
            format!(
                "{} candidate blocks exceed the kernel's softmax scratch bound of {MAX_BLOCKS}",
                blocks.len()
            ),
        ));
    }
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
            "attn/attn_res.cuh",
            symbol(&format!("::pie::attn::attn_res_blend<{t}>")),
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
        ],
    )
}
