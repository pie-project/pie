pub mod compressed;
pub mod dense;

pub mod index;

pub mod kv;
pub mod kv_window;

pub mod mla;

pub mod paged;

pub mod plan;
pub mod qkv_append;
pub mod ragged;

pub mod rel;

pub mod selected;

pub mod sched;
pub mod score;
pub mod voxel;

pub mod sched_decode;

pub mod sched_mla;

pub mod sched_prefill;

use crate::error::Error;
use dtype::Dtype;

use crate::attn::plan::{DecodePlan, PrefillPlan};
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const BLOCK: u32 = 256;

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// A per-head elementwise pass over `rows` rows of `heads` heads: the
/// elements a thread (eight where the head width allows) and its launch,
/// a row's threads over grid y.
#[must_use]
const fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> (u32, Launch) {
    let vec = if head_dim.is_multiple_of(8) { 8 } else { 1 };
    let threads = heads * head_dim / vec;
    (
        vec,
        Launch::grid([rows, threads.div_ceil(BLOCK), 1], [BLOCK, 1, 1]),
    )
}

fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if width == 0 || !width.is_multiple_of(head_dim) {
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
    rel: Option<RelBias>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    attention_lands(op, q.data, o);
    let window_left = plan::window_left(op, window)?;
    lanes_carry(op, &q, plan.shape.lane_offset, plan.shape.num_requests)?;

    if let Some(rel) = rel {
        if lse.is_some() {
            return Err(Error::Unsupported { op });
        }
        // The matrix-core decode where it takes the group and width.
        if paged::rel_walks(plan.shape.group_size(), head_dim) {
            return paged::decode(
                ctx,
                op,
                q,
                plan,
                pool,
                window_left,
                sm_scale,
                o,
                0,
                Some(rel.walked(q.data.rows, plan.shape.num_q_heads)?),
            );
        }
        return rel::walk(
            ctx,
            op,
            q,
            pool,
            rel,
            window_left,
            head_dim,
            plan.shape.num_kv_heads,
            sm_scale,
            o,
        );
    }
    let lse_ptr = match &lse {
        Some(lse) => {
            lse_plane(op, lse, q.data.rows, plan.shape.num_q_heads);
            lse.ptr
        }
        None => 0,
    };
    paged::decode(
        ctx,
        op,
        q,
        plan,
        pool,
        window_left,
        sm_scale,
        o,
        lse_ptr,
        None,
    )
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
    mask: Option<(Tensor, Tensor)>,
    rel: Option<RelBias>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    attention_lands(op, q.data, o);
    let window_left = plan::window_left(op, window)?;

    lanes_carry(op, &q, plan.shape.lane_offset, plan.shape.num_requests)?;
    if let Some(rel) = rel {
        if lse.is_some() || mask.is_some() {
            return Err(Error::Unsupported { op });
        }
        // The tiled prefill where it takes the group and width.
        if paged::rel_walks(plan.shape.group_size(), head_dim) {
            return paged::prefill(
                ctx,
                op,
                q,
                plan,
                pool,
                window_left,
                sm_scale,
                o,
                0,
                Some(rel.walked(q.data.rows, plan.shape.num_q_heads)?),
            );
        }
        return rel::walk(
            ctx,
            op,
            q,
            pool,
            rel,
            window_left,
            head_dim,
            plan.shape.num_kv_heads,
            sm_scale,
            o,
        );
    }
    if mask.is_some() {
        return Err(Error::Unsupported { op });
    }
    let lse_ptr = match &lse {
        Some(lse) => {
            lse_plane(op, lse, q.data.rows, plan.shape.num_q_heads);
            lse.ptr
        }
        None => 0,
    };
    paged::prefill(
        ctx,
        op,
        q,
        plan,
        pool,
        window_left,
        sm_scale,
        o,
        lse_ptr,
        None,
    )
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
    fa2_decode(
        ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None, None,
    )
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
    fa2_decode(
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
    fa2_prefill(
        ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None, None, None,
    )
}

/// A selected (DSA / QSA) prefill on the tiled causal walk of `plan`, each
/// row's keys masked to the pooled blocks its selection row names and its
/// open block: the request's queries share every key read, where the
/// per-row walk (`selected::attention_prefill_selected`) streams each row's
/// keys alone. `Ok(false)` where it does not cover the shape (a window, a
/// quantized pool, a GQA group under 9 heads, a context past
/// `paged::SEL_BLOCKS` pooled blocks), which the per-row walk then takes.
#[allow(clippy::too_many_arguments)]
pub fn prefill_selected_tiled(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    ratio: u32,
    per_head: bool,
    o: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "attention.prefill_selected";
    if window.is_some()
        || q.data.dtype != Dtype::Bf16
        || pool.keys.dtype != Dtype::Bf16
        || pool.key_scales.ptr != 0
        || !matches!(head_dim, 128 | 256)
        || !plan.causal
        || plan.info.cta_tile_q != i64::from(paged::PREFILL_TILE_Q)
        || selection.ptr == 0
        || selection.dtype != Dtype::I32
        || selection.rows != o.rows
        || ratio == 0
        || kv_heads == 0
        || plan.accepts(OP, head_dim, Some(kv_heads), window).is_err()
    {
        return Ok(false);
    }
    let q_heads = q.data.width / head_dim;
    if !q_heads.is_multiple_of(kv_heads) {
        return Ok(false);
    }
    let group = q_heads / kv_heads;
    // A 128-row tile spans at most 127 / G + 2 tokens.
    if group == 0 || 127 / group + 2 > paged::SEL_TOKENS {
        return Ok(false);
    }
    let context = i64::from(pool.max_pages_per_request) * i64::from(pool.page_size);
    if context <= 0 || context / i64::from(ratio) > i64::from(paged::SEL_BLOCKS) {
        return Ok(false);
    }
    attention_lands(OP, q.data, o);
    lanes_carry(OP, &q, plan.shape.lane_offset, plan.shape.num_requests)?;
    // Per head, the row holds one run of blocks for each kv head.
    let runs = if per_head { kv_heads } else { 1 };
    if selection.width == 0 || !selection.width.is_multiple_of(runs) {
        return Ok(false);
    }
    let sel = paged::PrefillSel {
        selection: selection.ptr,
        top_k: stated(OP, selection.width / runs)?,
        ratio: stated(OP, ratio)?,
        per_head: i32::from(per_head),
    };
    paged::prefill_selected(ctx, OP, q, plan, pool, sm_scale, o, sel)?;
    Ok(true)
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
        None,
    )
}

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
    plan.accepts(OP, head_dim, None, window)?;
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
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn masked_lse(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.masked_lse";
    debug_assert_eq!(mask.dtype, Dtype::U8, "`{OP}` reads packed u8 mask bits");
    plan.accepts(OP, head_dim, None, window)?;
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
        Some(lse),
        Some((mask, mask_indptr)),
        None,
    )
}

#[derive(Clone, Copy, Debug)]
pub struct RelBias {
    pub bias: Tensor,
    pub extent: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
}

impl RelBias {
    /// The bias as the matrix-core walks take it.
    fn walked(self, rows: u32, heads: u32) -> Result<paged::Rel, Error> {
        const OP: &str = "attention.rel_bias";
        if self.bias.dtype != Dtype::F32
            || self.bias.ptr == 0
            || self.bias.rows != rows
            || u64::from(self.bias.width) != u64::from(heads) * u64::from(self.extent)
        {
            return Err(refuse(
                OP,
                format!(
                    "the bias is {:?} [{}, {}] and the rows want f32 [{rows}, {heads} x {}]",
                    self.bias.dtype, self.bias.rows, self.bias.width, self.extent
                ),
            ));
        }
        Ok(paged::Rel {
            bias: self.bias.ptr,
            extent: stated(OP, self.extent)?,
            log_floor: stated(OP, self.log_floor)?,
            log_alpha: self.log_alpha,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_rel(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &DecodePlan,
    pool: &KvPool,
    rel: RelBias,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.decode_rel";
    plan.accepts(OP, head_dim, window)?;
    fa2_decode(
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
        Some(rel),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_rel(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    rel: RelBias,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_rel";
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
        None,
        None,
        Some(rel),
    )
}

pub fn sink(
    ctx: &Ctx,
    o: &mut Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attention.sink";
    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    // The sinks are the scores' own element, or f32 as a checkpoint keeps them.
    let s = match sink.dtype {
        Dtype::F32 => "float",
        held if held == o.dtype => t,
        held => {
            return Err(refuse(
                OP,
                format!("the sinks are {held:?} and the scores {:?}", o.dtype),
            ));
        }
    };
    let heads = row_heads(OP, o.width, head_dim)?;
    lse_plane(OP, &lse, o.rows, heads);
    let rows = count(OP, "rows", o.rows)?;
    let (vec, launch) = per_head_elementwise(o.rows, heads, head_dim);
    ctx.fire(
        OP,
        Fire::at(
            "attn/attention.hip",
            symbol(&format!("::pie::attn::sink_rescale<{t}, {s}, {vec}>")),
        )
        .apply(launch),
        &[
            o.arg(),
            lse.arg(),
            sink.arg(),
            rows.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            ctx.stage(),
        ],
    )
}

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

    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        o1.rows == o.rows && o2.rows == o.rows,
        "the merged outputs are one row per query row"
    );
    lse_plane(OP, lse, o.rows, heads);
    let (vec, launch) = per_head_elementwise(o.rows, heads, head_dim);
    let heads = count(OP, "the head count this merge states", heads)?;
    let head_dim = count(OP, "the head width this merge states", head_dim)?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/attention.hip",
            symbol(&format!("::pie::attn::merge_lse_combine<{t}, {vec}>")),
        )
        .apply(launch),
        &[
            o1.arg(),
            lse1.arg(),
            o2.arg(),
            lse2.arg(),
            o.arg(),
            lse.arg(),
            heads.arg(),
            head_dim.arg(),
            ctx.stage(),
        ],
    )
}

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
            "attn/attention.hip",
            symbol(&format!("::pie::attn::logit_softcap<{t}>")),
        )
        .apply(elementwise(lanes)),
        &[x.arg(), cap.arg(), n.arg()],
    )
}

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
    let plane_bytes = u64::from(y.rows) * u64::from(y.width) * 2;
    let stacked = blocks
        .windows(2)
        .all(|pair| pair[1].ptr == pair[0].ptr.wrapping_add(plane_bytes));
    // The kernel walks `blocks + (j * rows + t) * hidden`. The planner does not
    // place the closed blocks side by side (they are residual snapshots taken
    // layers apart), so scattered ones are staged into one stacked scratch.
    let first = if stacked {
        blocks.first().map_or(prefix.ptr, |b| b.ptr)
    } else {
        let span = usize::try_from(plane_bytes)
            .ok()
            .and_then(|p| p.checked_mul(blocks.len()))
            .ok_or_else(|| refuse(OP, "the stacked blocks do not fit a usize"))?;
        let stack = ctx.scratch(OP, "elemwise.res_blend_blocks", span)? as usize as u64;
        for (j, block) in blocks.iter().enumerate() {
            stage_plane(
                ctx,
                OP,
                stack + j as u64 * plane_bytes,
                block.ptr,
                plane_bytes,
            )?;
        }
        stack
    };
    ctx.fire(
        OP,
        Fire::at(
            "elemwise/norm.hip",
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
            ctx.stage(),
        ],
    )
}

/// One device-to-device plane copy on the context's stream.
fn stage_plane(ctx: &Ctx, op: &'static str, dst: u64, src: u64, bytes: u64) -> Result<(), Error> {
    #[cfg(feature = "rocm")]
    {
        let bytes = usize::try_from(bytes).map_err(|_| refuse(op, "a plane wider than usize"))?;
        // SAFETY: both planes are device ranges the caller sized at `bytes`.
        unsafe {
            crate::hip::memcpy_async(
                dst as usize as *mut core::ffi::c_void,
                src as usize as *const core::ffi::c_void,
                bytes,
                crate::hip::Direction::DeviceToDevice,
                ctx.stream(),
            )
        }
        .map_err(|code| {
            refuse(
                op,
                format!("`hipMemcpyAsync` answered {code} stacking a block"),
            )
        })
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (ctx, dst, src, bytes);
        Err(crate::jit::runtimeless(op))
    }
}
