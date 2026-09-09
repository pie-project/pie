pub mod dynconv;

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

pub mod selector;

pub mod ssm;

use crate::error::Error;
use dtype::Dtype;

use crate::attn::fa2_abi::{
    Buffers, DecodeRelParams, PrefillRelParams, make_decode_params, make_prefill_params,
};
use crate::attn::plan::{DecodePlan, PrefillPlan, PrefillPlanSm90};
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const BLOCK: u32 = 256;

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
    let (params, split) =
        make_decode_params(plan, &bufs, window_left, NO_SOFT_CAP, sm_scale, false);
    let point = |arm| fa2::DecodePoint {
        head_dim: plan.shape.head_dim,
        group_size: plan.shape.group_size(),
        arm,
        padded_batch_size: params.padded_batch_size,
        num_kv_heads: plan.shape.num_kv_heads,
        device: plan.device,
    };
    match rel {
        None => {
            let arm = fa2::decode_arm(plan.full_attention_variant(), window_left, NO_SOFT_CAP);
            fa2::decode(ctx, op, point(arm), &params)?;
        }
        Some(rel) => {
            rel_table(op, &rel.bias, q.data.rows, plan.shape.num_q_heads, rel.extent)?;
            let arm = fa2::decode_rel_arm(plan.full_attention_variant(), window_left);
            let params = DecodeRelParams {
                base: params,
                rel_bias: rel.bias.ptr,
                rel_extent: rel.extent,
                rel_heads: plan.shape.num_q_heads,
                log_floor: rel.log_floor,
                log_alpha: rel.log_alpha,
            };
            fa2::decode(ctx, op, point(arm), &params)?;
        }
    }
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
    mask: Option<(Tensor, Tensor)>,
    rel: Option<RelBias>,
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
    let (mut params, split) = make_prefill_params(plan, &bufs, window_left, NO_SOFT_CAP, sm_scale);
    if let Some((bits, indptr)) = &mask {
        params.maybe_custom_mask = bits.ptr;
        params.maybe_mask_indptr = indptr.ptr;
    }
    let point = |arm| fa2::PrefillPoint {
        head_dim: plan.shape.head_dim,
        cta_tile_q: plan.cta_tile_q(),
        arm,
        padded_batch_size: params.padded_batch_size,
        num_kv_heads: plan.shape.num_kv_heads,
        device: plan.device,
    };
    match rel {
        None => {
            let arm = match mask {
                Some(_) => fa2::prefill_custom_arm(NO_SOFT_CAP),
                None => fa2::prefill_arm(plan.full_attention_variant(), plan.causal, NO_SOFT_CAP),
            };
            fa2::prefill(ctx, op, point(arm), &params)?;
        }
        Some(rel) => {
            if mask.is_some() || !plan.causal {
                return Err(refuse(
                    op,
                    "the relative-bias arm is causal and takes no custom mask",
                ));
            }
            rel_table(op, &rel.bias, q.data.rows, plan.shape.num_q_heads, rel.extent)?;
            let arm = fa2::prefill_rel_arm(plan.full_attention_variant(), window_left);
            let params = PrefillRelParams {
                base: params,
                rel_bias: rel.bias.ptr,
                rel_extent: rel.extent,
                rel_heads: plan.shape.num_q_heads,
                log_floor: rel.log_floor,
                log_alpha: rel.log_alpha,
            };
            fa2::prefill(ctx, op, point(arm), &params)?;
        }
    }
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
    fa2_decode(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None, None)
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
    fa2_decode(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, Some(lse), None)
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
    fa2_prefill(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None, None, None)
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

#[derive(Clone, Copy, Debug)]
pub struct RelBias {
    pub bias: Tensor,
    pub extent: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
}

fn rel_table(op: &'static str, bias: &Tensor, rows: u32, heads: u32, extent: u32) -> Result<(), Error> {
    if bias.dtype != Dtype::F32 {
        return Err(refuse(op, format!("the relative-bias table is {:?}, and the score adds f32", bias.dtype)));
    }
    let width = u64::from(heads) * u64::from(extent);
    if bias.rows != rows || u64::from(bias.width) != width {
        return Err(refuse(
            op,
            format!(
                "the relative-bias table is [{}, {}] and the fire's queries want [{rows}, {heads} x {extent}]",
                bias.rows, bias.width
            ),
        ));
    }
    Ok(())
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
    fa2_decode(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None, Some(rel))
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
    fa2_prefill(ctx, OP, q, plan, pool, window, head_dim, sm_scale, o, None, None, Some(rel))
}

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
            "attn/attention.cuh",
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
            ctx.stage(),
        ],
    )
}
