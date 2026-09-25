use super::{Paged, PrefillPlan};
use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};
use dtype::Dtype;

pub fn split_count(rows: u32) -> u32 {
    32u32.div_ceil(rows.div_ceil(8).max(1)).clamp(1, 32)
}
pub fn workspace_words(splits: u32, heads: u32) -> Option<u32> {
    heads
        .checked_mul(8)?
        .checked_mul(128u32.checked_add(splits.checked_mul(258)?)?)
}

macro_rules! split_points {
    ($($gh:literal),+) => {
        fn point(gqa: u32) -> Option<(&'static str, &'static str, &'static str, &'static str)> {
            match gqa {
                $($gh => Some((
                    concat!("pie_q8_batch_pack_g", $gh),
                    concat!("pie_q8_batch_split_g", $gh),
                    concat!("pie_q8_batch_reduce_g", $gh),
                    concat!("PIE_Q8_BATCH(", $gh, ")"),
                )),)+
                _ => None,
            }
        }
    };
}
split_points!(1, 2, 3, 4, 5, 6, 7, 8);

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
) -> Result<bool, Error> {
    if !super::q8::enabled(pool, head_dim) {
        return Ok(false);
    }
    if requests == 0
        || !(8..=128).contains(&q.rows)
        || head_dim != 256
        || pool.page_size != 32
        || window.is_some()
        || !causal
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
    let shape = Paged::of("attention.split", q, pool, window, causal, head_dim)?;
    let Some((pack, split, reduce, stamp)) = point(shape.gqa) else {
        return Ok(false);
    };
    let splits = split_count(q.rows);
    let Some(per) = workspace_words(splits, shape.q_heads) else {
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
                [tiles * 8 * shape.q_heads * 256, 1, 1],
                [256, 1, 1],
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
                0i32.arg(),
                ctx.absent()?,
                workspace.arg_mut(),
                splits.arg(),
                q.rows.arg(),
                tile_base.arg(),
            ],
        )?;
        ctx.fire(
            Fire::at(FILE, reduce).stamp(stamp).apply(Grid::of(
                [tiles * 8 * shape.q_heads * 256, 1, 1],
                [256, 1, 1],
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
