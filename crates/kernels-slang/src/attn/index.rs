//! The sparse-attention indexer (`attn.index_*`) from
//! `kernels/attn/index.slang`: the index keys' norm and rope, the index
//! query's rope, the paged append, and the per-row top-k selection.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};

const INDEX: &str = "attn/index.slang";

/// The block mean lives with the pool it averages.
const POOL: &str = "attn/pool.slang";

const POOL_GROUP: u32 = 256;

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
    const OP: &str = "attn.index_layernorm_rope";
    let entry = dtype_dispatch!(OP, k.dtype, { Bf16 => "attn_index_knorm_rope_bf16" });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let head_dim = nonzero(OP, "the index key row's width", k.width)?;
    super::even_lanes(OP, "index key row", head_dim)?;
    let rope_dim = rotated(OP, rope_dim, head_dim)?;
    let rows = nonzero(OP, "rows", k.rows)?;
    ctx.fire(
        Fire::at(INDEX, entry).apply(Grid::of([K_BLOCK * rows, 1, 1], [K_BLOCK, 1, 1])),
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
    const OP: &str = "attn.index_rope";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "attn_index_q_rope_bf16" });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let n_heads = nonzero(OP, "the head count this rotation states", heads)?;
    let head_dim = nonzero(OP, "the head width this rotation states", head_dim)?;
    let rope_dim = rotated(OP, rope_dim, head_dim)?;
    let rows = nonzero(OP, "rows", q.rows)?;
    ctx.fire(
        Fire::at(INDEX, entry).apply(Grid::of([n_heads, rows, 1], [Q_ROPE_GROUP, 1, 1])),
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
    const OP: &str = "attn.index_kv_append";
    dtype_dispatch!(OP, k.dtype, { Bf16 => () });
    pool_pitch(OP, keys, k.width)?;
    let no_rope = Tensor::new(k.buf, k.rows, 0, k.dtype);
    super::mla::kv_append(ctx, k, no_rope, keys, write_page, write_offset)
}

/// Each closed block's index keys averaged, for the indexer to rank.
pub fn block_mean(
    ctx: &Ctx<'_>,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    keys: &KvPool,
    head_dim: u32,
    ratio: u32,
    entries: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.index_block_mean";
    let entry = dtype_dispatch!(OP, entries.dtype, { Bf16 => "attn_pool_block_mean_bf16" });
    debug_assert!(
        boundary_pos.dtype == Dtype::I32 && boundary_req.dtype == Dtype::I32,
        "`{OP}` reads the i32 block tables `attn.pool_boundary_*` published"
    );
    if keys.page_size <= 0 {
        return Err(refuse(OP, "the index cache page size is zero"));
    }
    let head_dim = nonzero(OP, "the key width this mean states", head_dim)?;
    super::even_lanes(OP, "index key", head_dim)?;
    pool_pitch(OP, keys, head_dim)?;
    if entries.width != head_dim {
        return Err(refuse(
            OP,
            format!(
                "the stated head width {head_dim} is not the {}-wide entry it sized",
                entries.width
            ),
        ));
    }
    let ratio = nonzero(OP, "the block width this mean pools over", ratio)?;
    let rows = nonzero(OP, "rows", boundary_pos.rows)?;
    ctx.fire(
        Fire::at(POOL, entry).apply(Grid::of(
            [head_dim.div_ceil(2), rows, 1],
            [POOL_GROUP, 1, 1],
        )),
        &[
            keys.keys.arg(),
            entries.arg_mut(),
            boundary_pos.arg(),
            boundary_req.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, ratio)?.arg(),
            keys.page_size.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}

pub fn topk(
    ctx: &Ctx<'_>,
    q: Tensor,
    weights: Option<Tensor>,
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
    const OP: &str = "attn.index_topk";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "attn_index_topk_bf16" });
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
    if weights.is_some_and(|w| w.width != heads) {
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
        Fire::at(INDEX, entry).apply(Grid::of([K_BLOCK * rows, 1, 1], [K_BLOCK, 1, 1])),
        &[
            q.arg(),
            weights.unwrap_or(q).arg(),
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
            i32::from(weights.is_some()).arg(),
        ],
    )
}
