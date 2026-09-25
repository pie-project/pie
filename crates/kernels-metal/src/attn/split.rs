use super::{Paged, PrefillPlan};
use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};
use dtype::Dtype;

pub fn split_count(rows: u32) -> u32 {
    match rows.div_ceil(8) {
        1 => 28,
        2 => 16,
        _ => 8,
    }
}
pub fn workspace_words(splits: u32) -> u32 {
    24576 + splits * 192 * 258
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
) -> Result<bool, Error> {
    if !super::q8::enabled(pool, head_dim) {
        return Ok(false);
    }
    if !(1..=16).contains(&requests)
        || !(8..=128).contains(&q.rows)
        || (requests == 1 && q.rows != 8)
        || !(4..=16).contains(&(q.rows / requests))
        || q.width != 24 * 256
        || pool.keys.width != 4 * 256
        || pool.values.width != 4 * 256
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
    if shape.gqa != 6 || shape.kv_heads != 4 {
        return Ok(false);
    }
    let splits = split_count(q.rows);
    let per = workspace_words(splits);
    let total = q.rows.div_ceil(8);
    let available = (1..=total).rev().find_map(|tiles| {
        let workspace = partials(1, tiles * per)?;
        (workspace.dtype == Dtype::F32
            && u64::from(workspace.rows) * u64::from(workspace.width) >= u64::from(tiles * per))
        .then_some((tiles, workspace))
    });
    let Some((capacity, workspace)) = available else {
        return Ok(false);
    };
    const FILE: &str = "attn/q8_split.metal";
    for tile_base in (0..total).step_by(capacity as usize) {
        let tiles = capacity.min(total - tile_base);
        ctx.fire(
            Fire::at(FILE, "pie_q8_batch_pack").apply(Grid::of([tiles * 49152, 1, 1], [256, 1, 1])),
            &[
                q.arg(),
                workspace.arg_mut(),
                q.rows.arg(),
                splits.arg(),
                tile_base.arg(),
            ],
        )?;
        ctx.fire(
            Fire::at(FILE, "pie_q8_batch_split")
                .apply(Grid::of([1024, splits, tiles], [256, 1, 1])),
            &[
                q.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                out.arg_mut(),
                6i32.arg(),
                plan.positions.arg(),
                plan.request_of_token.arg(),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                pool.page_size.arg(),
                4i32.arg(),
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
            Fire::at(FILE, "pie_q8_batch_reduce")
                .apply(Grid::of([tiles * 49152, 1, 1], [256, 1, 1])),
            &[
                workspace.arg(),
                out.arg_mut(),
                splits.arg(),
                q.rows.arg(),
                tile_base.arg(),
            ],
        )?;
    }
    Ok(true)
}
