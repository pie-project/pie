use super::{Paged, PrefillPlan};
use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};
use dtype::Dtype;

/// The slices `rows` rows split into, aiming at `target` slices
/// (`kernel.sdpa.split`) over the rows' 8-row groups.
pub fn split_count(rows: u32, target: u32) -> u32 {
    target
        .div_ceil(rows.div_ceil(8).max(1))
        .clamp(1, target.max(1))
}
pub fn workspace_words(splits: u32, heads: u32, head_dim: u32) -> Option<u32> {
    heads
        .checked_mul(8)?
        .checked_mul((head_dim / 2).checked_add(splits.checked_mul(head_dim + 2)?)?)
}

macro_rules! split_points {
    ($($gh:literal),+) => {
        fn point(gqa: u32, bf16: bool) -> Option<(&'static str, &'static str, &'static str, &'static str)> {
            match gqa {
                $($gh => Some((
                    concat!("pie_q8_batch_pack_g", $gh),
                    if bf16 {
                        concat!("pie_bf16_batch_split_g", $gh)
                    } else {
                        concat!("pie_q8_batch_split_g", $gh)
                    },
                    concat!("pie_q8_batch_reduce_g", $gh),
                    concat!("PIE_Q8_BATCH(", $gh, ")"),
                )),)+
                _ => None,
            }
        }
    };
}
split_points!(1, 2, 3, 4, 5, 6, 7, 8);

/// The 128-wide bf16 entries (a drafter's heads).
fn point_d128(gqa: u32) -> Option<(&'static str, &'static str, &'static str, &'static str)> {
    macro_rules! at {
        ($gh:literal) => {
            Some((
                concat!("pie_q8_batch_pack_g", $gh, "_d128"),
                concat!("pie_bf16_batch_split_g", $gh, "_d128"),
                concat!("pie_q8_batch_reduce_g", $gh, "_d128"),
                concat!("PIE_BF16_BATCH_D128(", $gh, ")"),
            ))
        };
    }
    match gqa {
        1 => at!(1),
        2 => at!(2),
        4 => at!(4),
        6 => at!(6),
        8 => at!(8),
        _ => None,
    }
}

/// The fire's per-row tables the split reads, from a prefill or a decode plan.
pub struct Tables {
    pub positions: Tensor,
    pub request_of_token: Tensor,
    pub mask: Tensor,
    pub mask_enabled: Tensor,
    pub mask_stride: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn try_paged(
    ctx: &Ctx<'_>,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    scale: f32,
    out: Tensor,
    requests: u32,
    partials: &dyn Fn(u32, u32) -> Option<Tensor>,
    target: u32,
    bf16: bool,
) -> Result<bool, Error> {
    try_tables(
        ctx,
        q,
        pool,
        &Tables {
            positions: plan.positions,
            request_of_token: plan.request_of_token,
            mask,
            mask_enabled: plan.mask_enabled,
            mask_stride: plan.mask_stride,
        },
        window,
        causal,
        head_dim,
        scale,
        out,
        requests,
        partials,
        target,
        bf16,
    )
}

/// The split over any fire's tables: verify and draft rows (a decode plan) take it too.
/// A sliding or wide window, and a block that attends both ways (a drafter's), ride in the
/// kernel's per-row walk.
#[allow(clippy::too_many_arguments)]
pub fn try_tables(
    ctx: &Ctx<'_>,
    q: Tensor,
    pool: &KvPool,
    plan: &Tables,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    scale: f32,
    out: Tensor,
    requests: u32,
    partials: &dyn Fn(u32, u32) -> Option<Tensor>,
    target: u32,
    bf16: bool,
) -> Result<bool, Error> {
    let mask = plan.mask;
    // a q8 pool, or (where `kernel.sdpa.split_bf16` says) a bf16 pool stored as it was
    // written, whose cells sit where a q8 pool's do
    let q8 = super::q8::enabled(pool, head_dim);
    let plain = bf16
        && !pool.q8
        && pool.head_stride == u64::from(head_dim)
        && pool.keys.width == pool.values.width
        && u64::from(pool.keys.width) == pool.seq_stride;
    if !q8 && !plain {
        return Ok(false);
    }
    if requests == 0
        || !(8..=128).contains(&q.rows)
        || !(head_dim == 256 || (head_dim == 128 && !q8))
        || pool.page_size != 32
        || q.dtype != Dtype::Bf16
        || pool.keys.dtype != Dtype::Bf16
        || pool.values.dtype != Dtype::Bf16
        || out.dtype != Dtype::Bf16
        || out.rows != q.rows
        || out.width != q.width
        || mask.dtype != Dtype::U8
        || plan.positions.dtype != Dtype::I32
        || plan.request_of_token.dtype != Dtype::I32
        || plan.mask_enabled.dtype != Dtype::U8
    {
        return Ok(false);
    }
    let shape = Paged::of("attn.split", q, pool, window, causal, head_dim)?;
    let entries = if head_dim == 128 {
        point_d128(shape.gqa)
    } else {
        point(shape.gqa, !q8)
    };
    let Some((pack, split, reduce, stamp)) = entries else {
        return Ok(false);
    };
    // as many history slices as the partials plane seats for one tile, up to the target
    let mut splits = split_count(q.rows, target);
    while splits > 1
        && workspace_words(splits, shape.q_heads, head_dim)
            .and_then(|w| partials(1, w))
            .is_none()
    {
        splits /= 2;
    }
    let Some(per) = workspace_words(splits, shape.q_heads, head_dim) else {
        return Ok(false);
    };
    let total = q.rows.div_ceil(8);
    let available = (1..=total).rev().find_map(|tiles| {
        let words = tiles.checked_mul(per)?;
        let workspace = partials(1, words)?;
        (workspace.dtype == Dtype::F32
            && u64::from(workspace.rows) * u64::from(workspace.width) >= u64::from(words))
        .then_some((tiles, workspace))
    });
    let Some((capacity, workspace)) = available else {
        return Ok(false);
    };
    const FILE: &str = "attn/q8_split.metal";
    for tile_base in (0..total).step_by(capacity as usize) {
        let tiles = capacity.min(total - tile_base);
        ctx.fire(
            Fire::at(FILE, pack).stamp(stamp).apply(Grid::of(
                [tiles * 8 * shape.q_heads * head_dim, 1, 1],
                [head_dim, 1, 1],
            )),
            &[
                q.arg(),
                workspace.arg_mut(),
                q.rows.arg(),
                splits.arg(),
                tile_base.arg(),
                shape.q_heads.arg(),
            ],
        )?;
        ctx.fire(
            Fire::at(FILE, split)
                .stamp(stamp)
                .apply(Grid::of([shape.kv_heads * 256, splits, tiles], [256, 1, 1])),
            &[
                q.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                out.arg_mut(),
                (shape.gqa as i32).arg(),
                plan.positions.arg(),
                plan.request_of_token.arg(),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                pool.page_size.arg(),
                (shape.kv_heads as i32).arg(),
                scale.arg(),
                mask.arg(),
                plan.mask_stride.arg(),
                plan.mask_enabled.arg(),
                shape.window.arg(),
                ctx.absent()?,
                workspace.arg_mut(),
                splits.arg(),
                q.rows.arg(),
                tile_base.arg(),
            ],
        )?;
        ctx.fire(
            Fire::at(FILE, reduce).stamp(stamp).apply(Grid::of(
                [tiles * 8 * shape.q_heads * head_dim, 1, 1],
                [head_dim, 1, 1],
            )),
            &[
                workspace.arg(),
                out.arg_mut(),
                splits.arg(),
                q.rows.arg(),
                tile_base.arg(),
                shape.q_heads.arg(),
            ],
        )?;
    }
    Ok(true)
}
