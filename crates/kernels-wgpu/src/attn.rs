#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{
    Arg, ArgValue, Ctx, Fire, Grid, dtype_dispatch, elementwise, head_grid, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};
use crate::tuning::DeviceInfo;

pub(crate) fn even_lanes(op: &'static str, what: &str, width: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "the {what} is {width} wide, and a bf16 plane moves as whole words (an even width)"
            ),
        ));
    }
    Ok(width / 2)
}

pub mod arbiter;
pub mod dense;
pub mod merge;
pub mod ple;
pub mod score;
pub mod ssm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub positions: Tensor,

    pub request_of_token: Tensor,

    pub mask: Tensor,

    pub mask_enabled: Tensor,

    pub mask_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    pub positions: Tensor,

    pub request_of_token: Tensor,

    pub mask: Tensor,

    pub mask_enabled: Tensor,

    pub mask_stride: u32,
}

fn tables_agree(
    op: &'static str,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
) -> Result<(), Error> {
    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's position table is {:?}, and this plan carries i32 positions",
                positions.dtype
            ),
        ));
    }
    if request_of_token.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's owning-request table is {:?}, and this plan carries an i32 \
                 request per token",
                request_of_token.dtype
            ),
        ));
    }
    if mask.dtype != Dtype::U8 {
        return Err(refuse(
            op,
            format!(
                "the fire's mask planes are {:?}, and this plan carries a packed u8 mask",
                mask.dtype
            ),
        ));
    }
    if mask_enabled.dtype != Dtype::U8 {
        return Err(refuse(
            op,
            format!(
                "the fire's mask-enabled flags are {:?}, and this plan carries one u8 \
                 per request",
                mask_enabled.dtype
            ),
        ));
    }
    if positions.rows != request_of_token.rows {
        return Err(refuse(
            op,
            format!(
                "the fire tables disagree: {} positions beside {} owning requests, and \
                 both are one entry per token",
                positions.rows, request_of_token.rows
            ),
        ));
    }
    Ok(())
}

pub fn plan_decode(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<DecodePlan, Error> {
    let _ = (ctx, kv_len);
    tables_agree(
        "attention.plan_decode",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(DecodePlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

pub fn plan_prefill(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<PrefillPlan, Error> {
    let _ = (ctx, kv_len);
    tables_agree(
        "attention.plan_prefill",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(PrefillPlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

const FILE: &str = "attn/sdpa_paged.wgsl";

pub(crate) const SDPA_TILE: u32 = 16;
const SDPA_LANES: u32 = 16;

const SDPA_WIDTHS: [u32; 4] = [64, 128, 256, 512];

const SDPA_DECODE: [&str; 4] = [
    "sdpa_paged_decode_bfloat16_d_64",
    "sdpa_paged_decode_bfloat16_d_128",
    "sdpa_paged_decode_bfloat16_d_256",
    "sdpa_paged_decode_bfloat16_d_512",
];

const SDPA_TILED: [&str; 4] = [
    "sdpa_paged_tiled_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_128",
    "sdpa_paged_tiled_bfloat16_d_256",
    "sdpa_paged_tiled_bfloat16_d_512",
];

const SDPA_DECODE_LSE: [&str; 4] = [
    "sdpa_paged_decode_lse_bfloat16_d_64",
    "sdpa_paged_decode_lse_bfloat16_d_128",
    "sdpa_paged_decode_lse_bfloat16_d_256",
    "sdpa_paged_decode_lse_bfloat16_d_512",
];

const SDPA_TILED_LSE: [&str; 4] = [
    "sdpa_paged_tiled_lse_bfloat16_d_64",
    "sdpa_paged_tiled_lse_bfloat16_d_128",
    "sdpa_paged_tiled_lse_bfloat16_d_256",
    "sdpa_paged_tiled_lse_bfloat16_d_512",
];

const SDPA_SPLIT: [&str; 4] = [
    "sdpa_paged_split_bfloat16_d_64",
    "sdpa_paged_split_bfloat16_d_128",
    "sdpa_paged_split_bfloat16_d_256",
    "sdpa_paged_split_bfloat16_d_512",
];

const SPLIT_FILE: &str = "attn/sdpa_split.wgsl";
const SPLIT_FOLD: &str = "sdpa_split_fold_bf16";
const SPLIT_FOLD_LSE: &str = "sdpa_split_fold_lse_bf16";

#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub o: Tensor,

    pub lse: Tensor,

    pub keys: u32,
}

#[must_use]
pub fn splits_for(q_heads: u32, rows: u32, keys: u32, info: DeviceInfo) -> u32 {
    let tuning = crate::tuning::current();
    let occupied = q_heads.saturating_mul(rows).max(1);
    if tuning.sdpa_split_max <= 1 || occupied >= info.cores.max(1) {
        return 1;
    }

    let room = info.cores.max(1).div_ceil(occupied);
    let afford = keys / tuning.sdpa_split_min_keys.max(1);
    room.min(afford).min(tuning.sdpa_split_max).max(1)
}

fn fits(part: Split, splits: u32, shape: &Paged, head_dim: u32, o: Tensor) -> bool {
    let want = u64::from(splits) * u64::from(shape.rows);
    part.o.dtype == o.dtype
        && part.lse.dtype == Dtype::F32
        && u64::from(part.o.rows) >= want
        && u64::from(part.lse.rows) >= want
        && part.o.width >= shape.q_heads.saturating_mul(head_dim)
        && part.lse.width >= shape.q_heads
}

#[allow(clippy::too_many_arguments)]
fn split_decode(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &DecodePlan,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    part: Split,
    splits: u32,
    shape: &Paged,
    head_dim: u32,
) -> Result<(), Error> {
    let rows = splits.saturating_mul(shape.rows);
    let partial_o = Tensor::new(
        part.o.buf,
        rows,
        shape.q_heads.saturating_mul(head_dim),
        o.dtype,
    );
    let partial_lse = Tensor::new(part.lse.buf, rows, shape.q_heads, Dtype::F32);

    let mut args = sdpa_args(
        ctx,
        op,
        q,
        pool,
        plan.positions,
        plan.request_of_token,
        plan.mask,
        plan.mask_enabled,
        plan.mask_stride,
        shape,
        sm_scale,
        partial_o,
    )?;
    args.push(partial_lse.arg_mut());
    ctx.fire(
        Fire::at(FILE, SDPA_SPLIT[shape.at])
            .groups([shape.q_heads, shape.rows, splits])
            .group([SDPA_LANES, SDPA_TILE, 1]),
        &args,
    )?;

    let entry = match lse {
        None => SPLIT_FOLD,
        Some(_) => SPLIT_FOLD_LSE,
    };
    let pairs = even_lanes(op, "head", head_dim)?;
    let lanes = head_grid(op, pairs, shape.q_heads, shape.rows)?;
    let mut fold = vec![partial_o.arg(), partial_lse.arg(), o.arg_mut()];
    if let Some(lse) = lse {
        lse_plane(op, lse, shape);
        fold.push(lse.arg_mut());
    }
    fold.push(stated(op, head_dim)?.arg());
    fold.push(stated(op, shape.q_heads)?.arg());
    fold.push(stated(op, shape.rows)?.arg());
    fold.push(stated(op, splits)?.arg());
    ctx.fire(
        Fire::at(SPLIT_FILE, entry).apply(Grid::of(lanes, [SPLIT_GROUP, 1, 1])),
        &fold,
    )
}

const SPLIT_GROUP: u32 = 256;

fn head_point(op: &'static str, head_dim: u32) -> Result<usize, Error> {
    SDPA_WIDTHS
        .iter()
        .position(|&p| p == head_dim)
        .ok_or_else(|| {
            refuse(
                op,
                format!("no sdpa shader is instantiated at head width {head_dim}"),
            )
        })
}

fn window_extent(op: &'static str, window: Option<u32>) -> Result<i32, Error> {
    match window {
        None => Ok(0),
        Some(w) => {
            nonzero(op, "the sliding extent this attention states", w)?;
            stated(op, w)
        }
    }
}

fn pool_heads(op: &'static str, pool: &KvPool, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if pool.head_stride != u64::from(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the stated head width {head_dim} is not the pool row's head stride {}",
                pool.head_stride
            ),
        ));
    }
    if pool.seq_stride == 0 || !pool.seq_stride.is_multiple_of(pool.head_stride) {
        return Err(refuse(
            op,
            format!(
                "the pool's sequence stride {} is not a whole number of {head_dim}-wide kv heads",
                pool.seq_stride
            ),
        ));
    }
    u32::try_from(pool.seq_stride / pool.head_stride).map_err(|_| {
        refuse(
            op,
            "the kv head count this pool row's strides spell does not fit the shader's int",
        )
    })
}

pub(crate) fn kv_heads_agree(
    op: &'static str,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
) -> Result<(), Error> {
    let spelled = pool_heads(op, pool, head_dim)?;
    if kv_heads != spelled {
        return Err(refuse(
            op,
            format!(
                "the stated kv head count {kv_heads} is not the {spelled} the pool row's strides spell"
            ),
        ));
    }
    Ok(())
}

fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the {width}-wide query row does not divide by the stated head width {head_dim}"
            ),
        ));
    }
    Ok(width / head_dim)
}

fn pairs(op: &'static str, head_dim: u32) -> Result<u32, Error> {
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the head width {head_dim} is odd; the bf16 kernels write whole pairs"),
        ));
    }
    Ok(head_dim / 2)
}

pub(crate) struct Paged {
    pub(crate) q_heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) gqa: u32,
    pub(crate) window: i32,
    pub(crate) rows: u32,
    pub(crate) at: usize,
}

impl Paged {
    pub(crate) fn of(
        op: &'static str,
        q: Tensor,
        pool: &KvPool,
        window: Option<u32>,
        head_dim: u32,
    ) -> Result<Self, Error> {
        if pool.page_size <= 0 {
            return Err(refuse(op, "the kv page size is zero"));
        }
        let kv_heads = pool_heads(op, pool, head_dim)?;
        let q_heads = row_heads(op, q.width, head_dim)?;
        if q_heads % kv_heads != 0 {
            return Err(refuse(
                op,
                format!(
                    "the {q_heads} query heads this row divides into are not a whole number \
                     of the pool row's {kv_heads} kv heads"
                ),
            ));
        }
        Ok(Self {
            q_heads,
            kv_heads,
            gqa: q_heads / kv_heads,
            window: window_extent(op, window)?,
            rows: nonzero(op, "rows", q.rows)?,
            at: head_point(op, head_dim)?,
        })
    }
}

pub(crate) fn lse_plane(op: &'static str, lse: Tensor, shape: &Paged) {
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{op}` lands an f32 log-sum-exp plane"
    );
    debug_assert!(
        lse.rows == shape.rows && lse.width == shape.q_heads,
        "`{op}`'s log-sum-exp plane is one f32 per head per row"
    );
}

fn sdpa_args(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
    shape: &Paged,
    sm_scale: f32,
    o: Tensor,
) -> Result<Vec<ArgValue>, Error> {
    Ok(vec![
        q.arg(),
        pool.keys.arg(),
        pool.values.arg(),
        o.arg_mut(),
        positions.arg(),
        request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        mask.arg(),
        mask_enabled.arg(),
        ctx.absent()?,
        stated(op, shape.gqa)?.arg(),
        pool.page_size.arg(),
        stated(op, shape.kv_heads)?.arg(),
        sm_scale.arg(),
        mask_stride.arg(),
        shape.window.arg(),
    ])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vector(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &DecodePlan,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    split: Option<Split>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "the attention lands one output row per query row"
    );
    let shape = Paged::of(op, q, pool, window, head_dim)?;
    if let Some(part) = split {
        let splits = splits_for(
            shape.q_heads,
            shape.rows,
            part.keys,
            crate::tuning::device(),
        );
        if splits > 1 && fits(part, splits, &shape, head_dim, o) {
            return split_decode(
                ctx, op, q, pool, plan, sm_scale, o, lse, part, splits, &shape, head_dim,
            );
        }
    }
    let entry = match lse {
        None => SDPA_DECODE[shape.at],
        Some(_) => SDPA_DECODE_LSE[shape.at],
    };
    let mut args = sdpa_args(
        ctx,
        op,
        q,
        pool,
        plan.positions,
        plan.request_of_token,
        plan.mask,
        plan.mask_enabled,
        plan.mask_stride,
        &shape,
        sm_scale,
        o,
    )?;
    if let Some(lse) = lse {
        lse_plane(op, lse, &shape);
        args.push(lse.arg_mut());
    }
    ctx.fire(
        Fire::at(FILE, entry)
            .groups([shape.q_heads, shape.rows, 1])
            .group([SDPA_LANES, SDPA_TILE, 1]),
        &args,
    )
}

pub(crate) fn tiled(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "the attention lands one output row per query row"
    );
    let shape = Paged::of(op, q, pool, window, head_dim)?;
    let entry = match lse {
        None => SDPA_TILED[shape.at],
        Some(_) => SDPA_TILED_LSE[shape.at],
    };
    let mut args = sdpa_args(
        ctx,
        op,
        q,
        pool,
        plan.positions,
        plan.request_of_token,
        mask,
        plan.mask_enabled,
        plan.mask_stride,
        &shape,
        sm_scale,
        o,
    )?;
    args.push(stated(op, shape.rows)?.arg());
    if let Some(lse) = lse {
        lse_plane(op, lse, &shape);
        args.push(lse.arg_mut());
    }
    ctx.fire(
        Fire::at(FILE, entry)
            .groups([shape.q_heads, shape.rows.div_ceil(SDPA_TILE), 1])
            .group([SDPA_LANES, SDPA_TILE, 1]),
        &args,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    split: Option<Split>,
) -> Result<(), Error> {
    vector(
        ctx,
        "attention.decode",
        q,
        pool,
        plan,
        window,
        head_dim,
        sm_scale,
        o,
        None,
        split,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_lse(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
    split: Option<Split>,
) -> Result<(), Error> {
    vector(
        ctx,
        "attention.decode_lse",
        q,
        pool,
        plan,
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        split,
    )
}

pub fn prefill(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    tiled(
        ctx, OP, q.data, pool, plan, plan.mask, window, head_dim, sm_scale, o, None,
    )
}

pub fn prefill_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    tiled(
        ctx,
        OP,
        q.data,
        pool,
        plan,
        plan.mask,
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
    )
}

pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.masked";
    if mask.dtype != Dtype::U8 {
        return Err(refuse(
            OP,
            format!(
                "the mask this op states is {:?}, and the shader reads packed u8 mask planes",
                mask.dtype
            ),
        ));
    }
    tiled(
        ctx, OP, q.data, pool, plan, mask, window, head_dim, sm_scale, o, None,
    )
}

pub fn sink(
    ctx: &Ctx<'_>,
    o: Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attention.sink";
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "attn_sink_rescale_bfloat16" });
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 log-sum-exp plane"
    );
    let heads = row_heads(OP, o.width, head_dim)?;
    debug_assert!(
        lse.rows == o.rows && lse.width == heads,
        "`{OP}`'s log-sum-exp plane is one f32 per head per row"
    );
    let lanes = head_grid(OP, pairs(OP, head_dim)?, heads, o.rows)?;
    ctx.fire(
        Fire::at("attn/attn_sink.wgsl", entry).apply(Grid::of(lanes, [64, 1, 1])),
        &[
            o.arg(),
            o.arg_mut(),
            lse.arg(),
            sink.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
        ],
    )
}

pub fn merge_lse(
    ctx: &Ctx<'_>,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    merge::merge_lse(ctx, o1, lse1, o2, lse2, heads, head_dim, o, lse)
}

pub fn logit_softcap(ctx: &Ctx<'_>, x: Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "attention.logit_softcap";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "logit_softcap_bfloat16" });
    let lanes = elementwise(OP, x.width, x.rows)?;
    let n = lanes[0];

    let groups = n.div_ceil(2).div_ceil(256);
    let fold = groups.div_ceil(MAX_GROUPS_PER_AXIS);
    ctx.fire(
        Fire::at("attn/logit_softcap.wgsl", entry)
            .groups([groups.div_ceil(fold), fold, 1])
            .group([256, 1, 1]),
        &[x.arg(), x.arg_mut(), cap.arg(), stated(OP, n)?.arg()],
    )
}

const MAX_GROUPS_PER_AXIS: u32 = 65535;

fn head_split(op: &'static str, pool: &KvPool, row: u32) -> Result<(u32, u32), Error> {
    let head_dim = u32::try_from(pool.head_stride)
        .ok()
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "the pool row's head stride {} spells no head width",
                    pool.head_stride
                ),
            )
        })?;
    if row == 0 || !row.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the {row}-wide appended row does not divide by the pool's head stride {head_dim}"
            ),
        ));
    }
    let heads = row / head_dim;
    if pool.seq_stride != u64::from(heads) * u64::from(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the pool's sequence stride {} is not the page row this appender writes",
                pool.seq_stride
            ),
        ));
    }
    Ok((head_dim, heads))
}

fn append_paged(
    ctx: &Ctx<'_>,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, k.dtype, { Bf16 => "kv_append_paged_bfloat16" });
    if pool.page_size <= 0 {
        return Err(refuse(op, "the kv page size is zero"));
    }
    debug_assert!(
        v.rows == k.rows && v.width == k.width && v.dtype == k.dtype,
        "the value plane is appended beside the key plane, one rectangle"
    );
    debug_assert!(
        write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
        "the write tables are u32: one destination page and one in-page slot per lane"
    );
    let (head_dim, heads) = head_split(op, pool, k.width)?;
    let lanes = head_grid(op, pairs(op, head_dim)?, heads, k.rows)?;
    ctx.fire(
        Fire::at("attn/kv_write.wgsl", entry).apply(Grid::of(lanes, [64, 1, 1])),
        &[
            k.arg(),
            v.arg(),
            pool.keys.arg_mut(),
            pool.values.arg_mut(),
            write_page.arg(),
            write_offset.arg(),
            stated(op, head_dim)?.arg(),
            pool.page_size.arg(),
            stated(op, heads)?.arg(),
            0_i32.arg(),
        ],
    )
}

pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append_paged(
        ctx,
        "attention.kv_append",
        k,
        v,
        pool,
        write_page,
        write_offset,
    )
}

pub fn kv_append_shared(
    ctx: &Ctx<'_>,
    plane: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append_paged(
        ctx,
        "attention.kv_append_shared",
        plane,
        plane,
        pool,
        write_page,
        write_offset,
    )
}

pub mod mla {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    const FILE: &str = "attn/mla.wgsl";

    const PREP_THREADS: u32 = 256;

    const LANES: u32 = 32;

    const MAX_CKV: u32 = 16 * LANES;
    const MAX_KPE: u32 = 4 * LANES;

    const ABSORB_GROUP: u32 = 64;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MlaPlan;

    pub fn plan(
        _ctx: &Ctx<'_>,
        _kv_indptr: Tensor,
        _kv_indices: Tensor,
        _last_page_len: Tensor,
        _kv_len: Tensor,
    ) -> Result<MlaPlan, Error> {
        Ok(MlaPlan)
    }

    pub fn latents(
        ctx: &Ctx<'_>,
        kv_a: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        split_kv_a_norm(
            ctx,
            "attention.mla_latents",
            kv_a,
            weight,
            eps,
            kv_lora_rank,
            kv_c,
            k_pe,
        )
    }

    pub fn latents_rope(
        ctx: &Ctx<'_>,
        kv_a: Tensor,
        positions: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        split_kv_a_norm(
            ctx,
            "attention.mla_latents_rope",
            kv_a,
            weight,
            eps,
            kv_lora_rank,
            kv_c,
            k_pe,
        )?;
        crate::elemwise::rope::partial_q(ctx, k_pe, positions, rope_dim, rope_dim, theta)
    }

    fn split_kv_a_norm(
        ctx: &Ctx<'_>,
        op: &'static str,
        kv_a: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        let entry = dtype_dispatch!(op, kv_a.dtype, { Bf16 => "mla_latents_bf16" });
        debug_assert!(
            kv_c.width == kv_lora_rank && kv_c.rows == kv_a.rows,
            "the latent output is the stated rank wide, one row per source row"
        );
        debug_assert!(
            k_pe.rows == kv_a.rows,
            "the rope tail is one row per source row"
        );
        let kv_lora = stated(op, nonzero(op, "the latent rank", kv_lora_rank)?)?;
        let rope = stated(op, k_pe.width)?;
        let src_row_stride = stated(op, kv_a.width)?;
        if kv_a.width < kv_lora_rank + k_pe.width {
            return Err(refuse(
                op,
                format!(
                    "the {}-wide source row does not hold the {kv_lora_rank}-wide latent beside \
                     the {}-wide rope tail",
                    kv_a.width, k_pe.width
                ),
            ));
        }
        let rows = nonzero(op, "rows", kv_a.rows)?;
        super::even_lanes(op, "latent rank", kv_lora_rank)?;
        super::even_lanes(op, "source row", kv_a.width)?;
        if k_pe.width != 0 {
            super::even_lanes(op, "rope tail", k_pe.width)?;
        }
        ctx.fire(
            Fire::at(FILE, entry)
                .apply(Grid::of([PREP_THREADS * rows, 1, 1], [PREP_THREADS, 1, 1])),
            &[
                kv_a.arg(),
                weight.arg(),
                kv_c.arg_mut(),
                k_pe.arg_mut(),
                kv_lora.arg(),
                rope.arg(),
                src_row_stride.arg(),
                eps.arg(),
            ],
        )
    }

    pub fn split_q_b(
        ctx: &Ctx<'_>,
        q_b: Tensor,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        q_nope: Tensor,
        q_pe: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_split_q_b";
        let entry = dtype_dispatch!(OP, q_b.dtype, { Bf16 => "mla_split_q_b_bf16" });
        let heads = stated(OP, nonzero(OP, "heads", heads)?)?;
        let nope = stated(OP, nope_dim)?;
        let rope = stated(OP, rope_dim)?;
        let per = i64::from(nope) + i64::from(rope);
        let total = i64::from(q_b.rows) * i64::from(heads) * per;
        let total = i32::try_from(total).map_err(|_| {
            refuse(
                OP,
                format!("{total} split elements do not fit the shader's int"),
            )
        })?;
        let lanes = nonzero(OP, "split elements", u32::try_from(total).unwrap_or(0))?;
        if !nope_dim.is_multiple_of(2) || !rope_dim.is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!(
                    "a {nope_dim} + {rope_dim} head splits in bf16 pairs, so both widths are even"
                ),
            ));
        }
        let lanes = lanes.div_ceil(2);
        debug_assert!(
            q_nope.width == u32::try_from(i64::from(heads) * i64::from(nope)).unwrap_or(u32::MAX)
                && q_pe.width
                    == u32::try_from(i64::from(heads) * i64::from(rope)).unwrap_or(u32::MAX),
            "the nope and rope planes are the per-head cut of q_b's row"
        );
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([lanes, 1, 1], [256, 1, 1])),
            &[
                q_b.arg(),
                q_nope.arg_mut(),
                q_pe.arg_mut(),
                total.arg(),
                heads.arg(),
                nope.arg(),
                rope.arg(),
            ],
        )
    }

    pub fn absorb_q(
        ctx: &Ctx<'_>,
        q_nope: Tensor,
        kv_b: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_absorb_q";
        let entry = dtype_dispatch!(OP, q_nope.dtype, { Bf16 => "mla_absorb_q_bf16" });
        let heads_i = stated(OP, nonzero(OP, "heads", heads)?)?;
        let rank = stated(OP, nonzero(OP, "the latent rank", kv_lora_rank)?)?;
        let nope = stated(OP, nope_dim)?;
        let v_dim = stated(OP, v_head_dim)?;
        let rows = nonzero(OP, "rows", q_nope.rows)?;
        let pairs = super::even_lanes(OP, "latent rank", kv_lora_rank)?;
        super::even_lanes(OP, "nope width", nope_dim)?;
        debug_assert!(
            q_latent.width == heads * kv_lora_rank && q_latent.rows == q_nope.rows,
            "the absorbed q is `heads · rank` wide, one row per token"
        );
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([pairs, heads, rows], [ABSORB_GROUP, 1, 1])),
            &[
                q_nope.arg(),
                kv_b.arg(),
                q_latent.arg_mut(),
                heads_i.arg(),
                rank.arg(),
                nope.arg(),
                v_dim.arg(),
            ],
        )
    }

    pub fn absorb_out(
        ctx: &Ctx<'_>,
        latent: Tensor,
        kv_b: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        o: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_absorb_out";
        let entry = dtype_dispatch!(OP, latent.dtype, { Bf16 => "mla_absorb_out_bf16" });
        let heads_i = stated(OP, nonzero(OP, "heads", heads)?)?;
        let rank = stated(OP, nonzero(OP, "the latent rank", kv_lora_rank)?)?;
        let v_dim = stated(OP, nonzero(OP, "the value head dim", v_head_dim)?)?;
        let nope = stated(OP, nope_dim)?;
        let rows = nonzero(OP, "rows", latent.rows)?;
        let pairs = super::even_lanes(OP, "value head width", v_head_dim)?;
        super::even_lanes(OP, "latent rank", kv_lora_rank)?;
        debug_assert!(
            latent.width == heads * kv_lora_rank,
            "the latent reading is `heads · rank` wide, one row per token"
        );
        debug_assert!(
            o.width == heads * v_head_dim && o.rows == latent.rows,
            "the value-space output is `heads · v_dim` wide, one row per token"
        );
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([pairs, heads, rows], [ABSORB_GROUP, 1, 1])),
            &[
                latent.arg(),
                kv_b.arg(),
                o.arg_mut(),
                heads_i.arg(),
                rank.arg(),
                v_dim.arg(),
                nope.arg(),
            ],
        )
    }

    pub fn kv_append(
        ctx: &Ctx<'_>,
        kv_c: Tensor,
        k_pe: Tensor,
        pool: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_kv_append";
        let entry = dtype_dispatch!(OP, kv_c.dtype, { Bf16 => "mla_kv_append_bf16" });
        if pool.page_size <= 0 {
            return Err(refuse(OP, "the kv page size is zero"));
        }
        debug_assert!(
            k_pe.rows == kv_c.rows,
            "the rope plane is appended beside the latent plane, one row each"
        );
        debug_assert!(
            write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
            "the write tables are u32: one destination page and one in-page slot per row"
        );
        let rows = nonzero(OP, "rows", kv_c.rows)?;
        let width = nonzero(OP, "the appended latent width", kv_c.width.max(k_pe.width))?;
        super::even_lanes(OP, "appended latent", kv_c.width)?;
        if k_pe.width != 0 {
            super::even_lanes(OP, "appended rope tail", k_pe.width)?;
        }
        let width = width.div_ceil(2);

        let values = if k_pe.width == 0 {
            pool.keys
        } else {
            pool.values
        };
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([width, rows, 1], [256, 1, 1])),
            &[
                kv_c.arg(),
                k_pe.arg(),
                pool.keys.arg_mut(),
                values.arg_mut(),
                write_page.arg(),
                write_offset.arg(),
                pool.page_size.arg(),
                stated(OP, kv_c.width)?.arg(),
                stated(OP, k_pe.width)?.arg(),
                stated(OP, rows)?.arg(),
            ],
        )
    }

    fn flash(
        ctx: &Ctx<'_>,
        op: &'static str,
        q: Tensor,
        q_pe: Tensor,
        selection: Option<Tensor>,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        let entry = match selection {
            None => dtype_dispatch!(op, q.dtype, { Bf16 => "mla_naive_paged_bf16" }),
            Some(_) => dtype_dispatch!(op, q.dtype, { Bf16 => "mla_naive_paged_selected_bf16" }),
        };
        if pool.page_size <= 0 {
            return Err(refuse(op, "the kv page size is zero"));
        }
        let heads = nonzero(op, "the head count this attention states", heads)?;
        let rows = nonzero(op, "rows", q.rows)?;
        let ckv = latent_strip(op, "latent rank", kv_lora_rank, MAX_CKV)?;
        if !q_pe.width.is_multiple_of(heads) {
            return Err(refuse(
                op,
                format!(
                    "the {}-wide rotated q plane does not divide by the {heads} heads",
                    q_pe.width
                ),
            ));
        }

        let (kpe, q_pe, values) = if q_pe.width == 0 {
            (0, q, pool.keys)
        } else {
            (
                latent_strip(op, "rope width", q_pe.width / heads, MAX_KPE)?,
                q_pe,
                pool.values,
            )
        };
        debug_assert!(
            positions.dtype == Dtype::I32 && request_of_token.dtype == Dtype::I32,
            "the fire's position and owning-request tables are i32, one entry per row"
        );
        debug_assert!(
            o.rows == q.rows && o.width == heads * kv_lora_rank,
            "the latent reading is `heads · rank` wide, one row per query row"
        );
        let mut args = vec![
            q.arg(),
            q_pe.arg(),
            pool.keys.arg(),
            values.arg(),
            o.arg_mut(),
            positions.arg(),
            request_of_token.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.page_size.arg(),
            stated(op, heads)?.arg(),
            ckv.arg(),
            kpe.arg(),
            sm_scale.arg(),
        ];
        if let Some(selection) = selection {
            debug_assert!(
                selection.dtype == Dtype::I32,
                "the selection is an i32 key-index row"
            );
            if selection.rows != o.rows {
                return Err(refuse(
                    op,
                    "the selection does not carry one row per query row",
                ));
            }
            args.push(selection.arg());
            args.push(stated(op, nonzero(op, "the selection budget", selection.width)?)?.arg());
        }
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([heads * LANES, rows, 1], [LANES, 1, 1])),
            &args,
        )
    }

    fn latent_strip(op: &'static str, what: &str, width: u32, max: u32) -> Result<i32, Error> {
        if width == 0 || !width.is_multiple_of(2 * LANES) || width > max {
            return Err(refuse(
                op,
                format!(
                    "the {what} {width} is not one this kernel can lane-split in pairs \
                     (a nonzero multiple of {}, at most {max})",
                    2 * LANES
                ),
            ));
        }
        stated(op, width)
    }

    pub fn attention_decode(
        ctx: &Ctx<'_>,
        q: Tensor,
        q_pe: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_decode",
            q,
            q_pe,
            None,
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    pub fn attention_prefill(
        ctx: &Ctx<'_>,
        q: RaggedTensor,
        q_pe: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_prefill",
            q.data,
            q_pe,
            None,
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    pub fn attention_decode_selected(
        ctx: &Ctx<'_>,
        q: Tensor,
        q_pe: Tensor,
        selection: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_decode_selected",
            q,
            q_pe,
            Some(selection),
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    pub fn attention_prefill_selected(
        ctx: &Ctx<'_>,
        q: RaggedTensor,
        q_pe: Tensor,
        selection: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_prefill_selected",
            q.data,
            q_pe,
            Some(selection),
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    #[must_use]
    pub fn selected_sweep(selection: &[i32], j_end: i32) -> Vec<i32> {
        let mut keys = Vec::with_capacity(selection.len());
        for &j in selection {
            if j < 0 || j >= j_end {
                continue;
            }
            keys.push(j);
        }
        keys
    }

    #[must_use]
    pub fn flash_reading(scores: &[f32], values: &[f32], width: usize) -> Vec<f32> {
        let mut acc = vec![0.0f32; width];
        let mut m = -3.0e38f32;
        let mut lsum = 0.0f32;
        for (n, &score) in scores.iter().enumerate() {
            let m_new = m.max(score);
            let corr = (m - m_new).exp();
            let p = (score - m_new).exp();
            lsum = lsum * corr + p;
            for (i, a) in acc.iter_mut().enumerate() {
                *a = *a * corr + p * values[n * width + i];
            }
            m = m_new;
        }
        let inv = if lsum > 0.0 { 1.0 / lsum } else { 0.0 };
        for a in &mut acc {
            *a *= inv;
        }
        acc
    }
}

pub mod index {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, Tensor};

    const FILE: &str = "attn/index.wgsl";

    const K_BLOCK: u32 = 256;

    const Q_ROPE_GROUP: u32 = 32;

    const MAX_ROPE_DIM: u32 = 256;

    fn rotated(op: &'static str, rope_dim: u32, head_dim: u32) -> Result<i32, Error> {
        if !rope_dim.is_multiple_of(2) {
            return Err(refuse(
                op,
                format!("the rotated prefix {rope_dim} is odd, and this rotation turns pairs"),
            ));
        }
        if rope_dim > head_dim {
            return Err(refuse(
                op,
                format!(
                    "the rotated prefix {rope_dim} is wider than the {head_dim}-wide row it rotates"
                ),
            ));
        }
        if rope_dim > MAX_ROPE_DIM {
            return Err(refuse(
                op,
                format!(
                    "the rotated prefix {rope_dim} is above the {MAX_ROPE_DIM} this indexer rotates"
                ),
            ));
        }
        stated(op, rope_dim)
    }

    fn pool_pitch(op: &'static str, pool: &KvPool, row: u32) -> Result<(), Error> {
        if row == 0 {
            return Err(refuse(op, "the index key row is zero-wide"));
        }
        if pool.seq_stride != u64::from(row) {
            return Err(refuse(
                op,
                format!(
                    "the pool's token pitch {} is not the {row}-wide row this index writes",
                    pool.seq_stride
                ),
            ));
        }
        Ok(())
    }

    pub fn layernorm_rope(
        ctx: &Ctx<'_>,
        k: Tensor,
        positions: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_layernorm_rope";
        let entry = dtype_dispatch!(OP, k.dtype, { Bf16 => "index_knorm_rope_bf16" });
        debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
        let head_dim = nonzero(OP, "the index key row's width", k.width)?;
        super::even_lanes(OP, "index key row", head_dim)?;
        let rope_dim = rotated(OP, rope_dim, head_dim)?;
        let rows = nonzero(OP, "rows", k.rows)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([K_BLOCK * rows, 1, 1], [K_BLOCK, 1, 1])),
            &[
                k.arg_mut(),
                weight.arg(),
                bias.arg(),
                positions.arg(),
                stated(OP, head_dim)?.arg(),
                rope_dim.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }

    pub fn rope(
        ctx: &Ctx<'_>,
        q: Tensor,
        positions: Tensor,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_rope";
        let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "index_q_rope_bf16" });
        debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
        let n_heads = nonzero(OP, "the head count this rotation states", heads)?;
        let head_dim = nonzero(OP, "the head width this rotation states", head_dim)?;
        let rope_dim = rotated(OP, rope_dim, head_dim)?;
        let rows = nonzero(OP, "rows", q.rows)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([n_heads, rows, 1], [Q_ROPE_GROUP, 1, 1])),
            &[
                q.arg_mut(),
                positions.arg(),
                stated(OP, n_heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                rope_dim.arg(),
                theta.arg(),
            ],
        )
    }

    pub fn kv_append(
        ctx: &Ctx<'_>,
        k: Tensor,
        keys: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_kv_append";
        dtype_dispatch!(OP, k.dtype, { Bf16 => () });
        pool_pitch(OP, keys, k.width)?;
        let no_rope = Tensor::new(k.buf, k.rows, 0, k.dtype);
        super::mla::kv_append(ctx, k, no_rope, keys, write_page, write_offset)
    }

    pub fn topk(
        ctx: &Ctx<'_>,
        q: Tensor,
        weights: Tensor,
        keys: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        scores: Tensor,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        ratio: u32,
        selection: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_topk";
        let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "index_topk_paged_bf16" });
        debug_assert_eq!(
            selection.dtype,
            Dtype::I32,
            "`{OP}` writes i32 cached positions"
        );
        debug_assert_eq!(scores.dtype, Dtype::F32, "`{OP}` bisects an f32 score slab");
        debug_assert!(
            positions.dtype == Dtype::I32 && request_of_token.dtype == Dtype::I32,
            "the fire's position and owning-request tables are i32, one entry per row"
        );
        let heads = nonzero(OP, "the head count this ranking states", heads)?;
        let head_dim = nonzero(OP, "the key width this ranking states", head_dim)?;
        super::even_lanes(OP, "index key", head_dim)?;
        let top_k = nonzero(OP, "the selection budget this ranking states", top_k)?;
        let ratio = nonzero(OP, "the key stride this ranking states", ratio)?;
        if keys.page_size <= 0 {
            return Err(refuse(OP, "the index cache page size is zero"));
        }
        pool_pitch(OP, keys, head_dim)?;
        if q.width != heads.saturating_mul(head_dim) {
            return Err(refuse(
                OP,
                format!(
                    "the {}-wide index query does not divide by the stated head count and width",
                    q.width
                ),
            ));
        }
        if weights.width != heads {
            return Err(refuse(
                OP,
                "the index head weights are not one per stated head",
            ));
        }
        if selection.width != top_k {
            return Err(refuse(
                OP,
                "the selection this statement allocated is not the budget it stated",
            ));
        }
        let rows = nonzero(OP, "rows", selection.rows)?;
        if scores.rows < rows {
            return Err(refuse(
                OP,
                format!(
                    "the score slab seats {} rows and this ranking launches {rows}",
                    scores.rows
                ),
            ));
        }
        let stride = nonzero(OP, "the score slab's key stride", scores.width)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([K_BLOCK * rows, 1, 1], [K_BLOCK, 1, 1])),
            &[
                q.arg(),
                weights.arg(),
                keys.keys.arg(),
                positions.arg(),
                request_of_token.arg(),
                keys.page_indices.arg(),
                keys.page_indptr.arg(),
                scores.arg_mut(),
                selection.arg_mut(),
                stated(OP, heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                keys.page_size.arg(),
                stated(OP, stride)?.arg(),
                stated(OP, top_k)?.arg(),
                stated(OP, ratio)?.arg(),
            ],
        )
    }

    #[must_use]
    pub fn bisect_select(scores: &[f32], topk: usize) -> Vec<i32> {
        let nkeys = scores.len();
        if nkeys <= topk {
            return (0..topk)
                .map(|n| if n < nkeys { n as i32 } else { -1 })
                .collect();
        }
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for s in scores {
            lo = lo.min(*s);
            hi = hi.max(*s);
        }
        let mut thr = hi;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let cnt = scores.iter().filter(|s| **s >= mid).count();
            if cnt > topk {
                lo = mid;
            } else {
                hi = mid;
            }
            thr = hi;
        }
        let mut out = Vec::with_capacity(topk);
        for (j, s) in scores.iter().enumerate() {
            if out.len() == topk {
                break;
            }
            if *s >= thr {
                out.push(j as i32);
            }
        }
        out.resize(topk, -1);
        out
    }

    #[must_use]
    pub fn pooled_select(scores: &[f32], total: usize, ratio: usize, topk: usize) -> Vec<i32> {
        let stride = ratio.max(1);
        let npools = (total / stride).min(scores.len());
        let tail = (total - (total / stride) * stride).min(topk);
        let pool_budget = (topk - tail) / stride;
        let mut out: Vec<i32> = (0..tail)
            .map(|i| ((total / stride) * stride + i) as i32)
            .collect();
        if npools <= pool_budget {
            for n in 0..topk - tail {
                let j = n / stride;
                out.push(if j < npools {
                    (j * stride + n % stride) as i32
                } else {
                    -1
                });
            }
            return out;
        }
        let pools = &scores[..npools];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for s in pools {
            lo = lo.min(*s);
            hi = hi.max(*s);
        }
        let mut thr = hi;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let cnt = pools.iter().filter(|s| **s >= mid).count();
            if cnt > pool_budget {
                lo = mid;
            } else {
                hi = mid;
            }
            thr = hi;
        }
        for (j, s) in pools.iter().enumerate() {
            if out.len() + stride > topk {
                break;
            }
            if *s >= thr {
                out.extend((0..stride).map(|i| (j * stride + i) as i32));
            }
        }
        out.resize(topk, -1);
        out
    }
}

pub mod pool {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    const FILE: &str = "attn/pool.wgsl";

    const META_BLOCK: u32 = 128;

    const ATTN_BLOCK: u32 = 128;

    const POOL_HEAD_MAX: u32 = 512;

    const LANE_GROUP: u32 = 256;

    const fn compressor_coff(ratio: u32) -> i32 {
        if ratio == 4 { 2 } else { 1 }
    }

    fn boundary_tables(op: &'static str, boundary_pos: &Tensor, boundary_req: &Tensor) {
        debug_assert_eq!(
            boundary_pos.dtype,
            Dtype::I32,
            "`{op}` reads i32 boundary positions"
        );
        debug_assert_eq!(
            boundary_req.dtype,
            Dtype::I32,
            "`{op}` reads i32 boundary requests"
        );
        debug_assert_eq!(
            boundary_pos.rows, boundary_req.rows,
            "`{op}`'s boundary tables are one entry per token row"
        );
    }

    fn boundary_rope_table(op: &'static str, boundary_pos: &Tensor, boundary_rope: &Tensor) {
        debug_assert_eq!(
            boundary_rope.dtype,
            Dtype::I32,
            "`{op}` writes i32 compressed-row rope positions"
        );
        debug_assert_eq!(
            boundary_pos.rows, boundary_rope.rows,
            "`{op}`'s rope column is one entry per token row"
        );
    }

    pub fn boundary_decode(
        ctx: &Ctx<'_>,
        positions: Tensor,
        row_valid: Tensor,
        ratio: u32,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        boundary_rope: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_boundary_decode";
        boundary_tables(OP, &boundary_pos, &boundary_req);
        boundary_rope_table(OP, &boundary_pos, &boundary_rope);
        let n = stated(OP, nonzero(OP, "rows", boundary_pos.rows)?)?;
        let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
        ctx.fire(
            Fire::at(FILE, "pool_boundary_decode")
                .apply(Grid::of([boundary_pos.rows, 1, 1], [META_BLOCK, 1, 1])),
            &[
                positions.arg(),
                boundary_pos.arg_mut(),
                boundary_req.arg_mut(),
                boundary_rope.arg_mut(),
                n.arg(),
                ratio.arg(),
                row_valid.arg(),
            ],
        )
    }

    pub fn boundary_prefill(
        ctx: &Ctx<'_>,
        positions: RaggedTensor,
        row_valid: Tensor,
        ratio: u32,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        boundary_rope: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_boundary_prefill";
        boundary_tables(OP, &boundary_pos, &boundary_req);
        boundary_rope_table(OP, &boundary_pos, &boundary_rope);
        let n = stated(OP, nonzero(OP, "rows", boundary_pos.rows)?)?;
        let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
        let num_requests = stated(
            OP,
            nonzero(OP, "requests", positions.indptr.rows.saturating_sub(1))?,
        )?;
        ctx.fire(
            Fire::at(FILE, "pool_boundary_prefill")
                .apply(Grid::of([boundary_pos.rows, 1, 1], [META_BLOCK, 1, 1])),
            &[
                positions.data.arg(),
                positions.indptr.arg(),
                boundary_pos.arg_mut(),
                boundary_req.arg_mut(),
                boundary_rope.arg_mut(),
                n.arg(),
                num_requests.arg(),
                ratio.arg(),
                row_valid.arg(),
            ],
        )
    }

    pub fn state_write(
        ctx: &Ctx<'_>,
        kv: Tensor,
        score: Tensor,
        pages: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
        head_dim: u32,
        ratio: u32,
        state_kv: Tensor,
        state_score: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_state_write";
        let entry = dtype_dispatch!(OP, kv.dtype, { Bf16 => "pool_state_write_bf16" });
        debug_assert_eq!(
            score.dtype, kv.dtype,
            "`{OP}` writes both projections in one element"
        );
        debug_assert!(
            state_kv.dtype == kv.dtype && state_score.dtype == kv.dtype,
            "`{OP}` lands the state plane in the projections' own element"
        );
        debug_assert!(
            write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
            "the write tables are u32: one destination page and one in-page slot per row"
        );
        if pages.page_size <= 0 {
            return Err(refuse(OP, "the source pool's page size is zero"));
        }
        let head_dim = nonzero(OP, "the head width this compressor states", head_dim)?;
        let ratio = nonzero(OP, "the pooling ratio", ratio)?;
        let width = nonzero(OP, "the compressor's row width", kv.width)?;
        if (width != head_dim && width != 2 * head_dim) || score.width != width {
            return Err(refuse(
                OP,
                format!(
                    "a ratio-{ratio} compressor projects head width {head_dim} or twice it; the \
                     pair handed over is {} and {}",
                    kv.width, score.width
                ),
            ));
        }
        let coff = width / head_dim;
        debug_assert_eq!(
            score.rows, kv.rows,
            "the two projections are one row per token row"
        );
        let pitch = state_kv.width;
        if state_score.width != pitch {
            return Err(refuse(
                OP,
                format!(
                    "`state_kv` is {pitch} wide and `state_score` is {} — the two state slabs \
                     are one plane laid at one pitch",
                    state_score.width
                ),
            ));
        }
        if pitch < width {
            return Err(refuse(
                OP,
                format!(
                    "the state slabs are {pitch} wide and this compressor writes coff {coff} x \
                     head width {head_dim} = {width} columns of every row"
                ),
            ));
        }
        let rows = nonzero(OP, "rows", kv.rows)?;
        let pairs = super::even_lanes(OP, "compressor row", width)?;
        super::even_lanes(OP, "state pitch", pitch)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([pairs, rows, 1], [LANE_GROUP, 1, 1])),
            &[
                kv.arg(),
                score.arg(),
                state_kv.arg_mut(),
                state_score.arg_mut(),
                write_page.arg(),
                write_offset.arg(),
                stated(OP, width)?.arg(),
                pages.page_size.arg(),
                stated(OP, pitch)?.arg(),
                stated(OP, rows)?.arg(),
            ],
        )
    }

    pub fn gather(
        ctx: &Ctx<'_>,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        pages: &KvPool,
        head_dim: u32,
        ratio: u32,
        state_kv: Tensor,
        state_score: Tensor,
        ape: Option<Tensor>,
        entries: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_gather";
        let entry = dtype_dispatch!(OP, entries.dtype, { Bf16 => "pool_gather_paged_bf16" });
        boundary_tables(OP, &boundary_pos, &boundary_req);
        debug_assert_eq!(
            state_kv.dtype, entries.dtype,
            "`{OP}` pools the state plane into an entry of its own element"
        );
        debug_assert_eq!(
            state_score.dtype, entries.dtype,
            "`{OP}` reads the gate logits in the state plane's element"
        );
        if pages.page_size <= 0 {
            return Err(refuse(OP, "the pooled space's page size is zero"));
        }
        let head_dim = nonzero(OP, "the head width this gather states", head_dim)?;
        if head_dim != entries.width {
            return Err(refuse(
                OP,
                format!(
                    "the stated head width {head_dim} is not the {}-wide entry it sized",
                    entries.width
                ),
            ));
        }
        let rows = nonzero(OP, "rows", boundary_pos.rows)?;
        let ratio = nonzero(OP, "the pooling ratio", ratio)?;
        let coff = match ape {
            None => compressor_coff(ratio),
            Some(ape) if ape.width == head_dim => 1,
            Some(ape) if ape.width == 2 * head_dim => 2,
            Some(ape) => {
                return Err(refuse(
                    OP,
                    format!(
                        "an ape {} wide is neither one head width ({head_dim}) nor two",
                        ape.width
                    ),
                ));
            }
        };
        let width = head_dim.saturating_mul(coff.unsigned_abs());
        let pitch = state_kv.width;
        if state_score.width != pitch {
            return Err(refuse(
                OP,
                format!(
                    "`state_kv` is {pitch} wide and `state_score` is {} — the two state slabs \
                     are one plane laid at one pitch",
                    state_score.width
                ),
            ));
        }
        if pitch < width {
            return Err(refuse(
                OP,
                format!(
                    "the state slabs are {pitch} wide and this gather reads coff {coff} x head \
                     width {head_dim} = {width} columns of every row"
                ),
            ));
        }
        if let Some(ape) = ape {
            if ape.dtype != Dtype::F32 {
                return Err(refuse(OP, "the absolute-position plane is read as f32"));
            }
            if ape.width != width || ape.rows != ratio {
                return Err(refuse(
                    OP,
                    format!(
                        "the absolute-position plane is {} x {} and this gather reads it at \
                         [ratio {ratio}, {width}]",
                        ape.rows, ape.width
                    ),
                ));
            }
        }
        let pairs = super::even_lanes(OP, "pooled head", head_dim)?;
        super::even_lanes(OP, "state pitch", pitch)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([pairs, rows, 1], [LANE_GROUP, 1, 1])),
            &[
                state_kv.arg(),
                state_score.arg(),
                ape.unwrap_or(state_score).arg(),
                boundary_pos.arg(),
                boundary_req.arg(),
                pages.page_indices.arg(),
                pages.page_indptr.arg(),
                entries.arg_mut(),
                stated(OP, head_dim)?.arg(),
                stated(OP, ratio)?.arg(),
                coff.arg(),
                pages.page_size.arg(),
                i32::from(ape.is_some()).arg(),
                stated(OP, pitch)?.arg(),
                stated(OP, rows)?.arg(),
            ],
        )
    }

    pub fn kv_append(
        ctx: &Ctx<'_>,
        entries: Tensor,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        pool: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_kv_append";
        let _ = (write_page, write_offset);
        let entry = dtype_dispatch!(OP, entries.dtype, { Bf16 => "pool_store_entries_bf16" });
        boundary_tables(OP, &boundary_pos, &boundary_req);
        if pool.page_size <= 0 {
            return Err(refuse(OP, "the compressed cache page size is zero"));
        }
        let head_dim = nonzero(OP, "the pooled entry's width", entries.width)?;
        let rows = nonzero(OP, "rows", entries.rows)?;
        let pairs = super::even_lanes(OP, "pooled entry", head_dim)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([pairs, rows, 1], [LANE_GROUP, 1, 1])),
            &[
                entries.arg(),
                pool.keys.arg_mut(),
                boundary_pos.arg(),
                boundary_req.arg(),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                stated(OP, head_dim)?.arg(),
                pool.page_size.arg(),
                stated(OP, rows)?.arg(),
            ],
        )
    }

    fn reader_head(op: &'static str, heads: u32, head_dim: u32) -> Result<(i32, i32), Error> {
        let num_q_heads = nonzero(op, "heads", heads)?;
        let head_dim = nonzero(op, "the head width", head_dim)?;
        super::even_lanes(op, "head", head_dim)?;
        if head_dim > POOL_HEAD_MAX {
            return Err(refuse(
                op,
                format!(
                    "the head width {head_dim} is above the {POOL_HEAD_MAX} this flash reader tiles in workgroup memory"
                ),
            ));
        }
        Ok((stated(op, num_q_heads)?, stated(op, head_dim)?))
    }

    pub fn attention_lse(
        ctx: &Ctx<'_>,
        q: Tensor,
        positions: Tensor,
        request_of_token: Tensor,
        entries: &KvPool,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Tensor,
        lse: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_lse";
        dtype_dispatch!(OP, q.dtype, { Bf16 => () });
        debug_assert_eq!(
            lse.dtype,
            Dtype::F32,
            "`{OP}` lands an f32 log-sum-exp plane"
        );
        debug_assert_eq!(
            request_of_token.dtype,
            Dtype::I32,
            "`{OP}` reads an i32 owning request per token"
        );
        if entries.page_size <= 0 {
            return Err(refuse(OP, "the compressed cache page size is zero"));
        }
        let (num_q_heads, head_dim) = reader_head(OP, heads, head_dim)?;
        let rows = nonzero(OP, "rows", o.rows)?;
        let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
        ctx.fire(
            Fire::at(FILE, "pool_lse_paged")
                .apply(Grid::of([ATTN_BLOCK, rows, heads], [ATTN_BLOCK, 1, 1])),
            &[
                q.arg(),
                entries.keys.arg(),
                o.arg_mut(),
                lse.arg_mut(),
                positions.arg(),
                entries.page_indices.arg(),
                entries.page_indptr.arg(),
                request_of_token.arg(),
                num_q_heads.arg(),
                head_dim.arg(),
                ratio.arg(),
                entries.page_size.arg(),
                sm_scale.arg(),
            ],
        )
    }

    pub fn attention_lse_selected(
        ctx: &Ctx<'_>,
        q: Tensor,
        positions: Tensor,
        request_of_token: Tensor,
        selection: Tensor,
        entries: &KvPool,
        ratio: u32,
        top_k: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Tensor,
        lse: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_lse_selected";
        dtype_dispatch!(OP, q.dtype, { Bf16 => () });
        debug_assert_eq!(
            lse.dtype,
            Dtype::F32,
            "`{OP}` lands an f32 log-sum-exp plane"
        );
        debug_assert_eq!(
            selection.dtype,
            Dtype::I32,
            "`{OP}` walks i32 compressed-row ids"
        );
        debug_assert_eq!(
            request_of_token.dtype,
            Dtype::I32,
            "`{OP}` reads an i32 owning request per token"
        );
        if entries.page_size <= 0 {
            return Err(refuse(OP, "the compressed cache page size is zero"));
        }
        let (num_q_heads, head_dim) = reader_head(OP, heads, head_dim)?;
        let rows = nonzero(OP, "rows", o.rows)?;
        let ratio = nonzero(OP, "the pooling ratio", ratio)?;
        let top_k = nonzero(OP, "the selection budget this reader states", top_k)?;
        if selection.width != top_k {
            return Err(refuse(
                OP,
                format!(
                    "the selection is {} wide and this reader walks the {top_k} ids it states",
                    selection.width
                ),
            ));
        }
        if selection.rows < rows {
            return Err(refuse(
                OP,
                format!(
                    "the selection seats {} rows and this reader launches {rows}",
                    selection.rows
                ),
            ));
        }
        ctx.fire(
            Fire::at(FILE, "pool_lse_selected_paged")
                .apply(Grid::of([ATTN_BLOCK, rows, heads], [ATTN_BLOCK, 1, 1])),
            &[
                q.arg(),
                entries.keys.arg(),
                selection.arg(),
                o.arg_mut(),
                lse.arg_mut(),
                positions.arg(),
                entries.page_indices.arg(),
                entries.page_indptr.arg(),
                request_of_token.arg(),
                num_q_heads.arg(),
                head_dim.arg(),
                stated(OP, ratio)?.arg(),
                stated(OP, top_k)?.arg(),
                entries.page_size.arg(),
                sm_scale.arg(),
            ],
        )
    }

    #[must_use]
    pub fn selected_cells(selection: &[i32], num_visible: i32, ratio: i32) -> Vec<i32> {
        selection
            .iter()
            .filter(|c| **c >= 0 && **c < num_visible)
            .map(|c| (c + 1) * ratio - 1)
            .collect()
    }

    #[must_use]
    pub fn compressed_rope_pos(closing_pos: i32, ratio: i32) -> i32 {
        (closing_pos / ratio) * ratio
    }
}
