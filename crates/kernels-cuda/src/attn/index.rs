use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/index.cuh";

const K_BLOCK: u32 = 256;

const POOL_BLOCK: u32 = 256;

#[must_use]
fn q_rope_block(n_heads: i32) -> u32 {
    (n_heads.unsigned_abs().div_ceil(32) * 32).max(32)
}

fn pool_pitch(op: &'static str, pool: &KvPool, row: i32) -> Result<(), Error> {
    if pool.layout != 0 {
        return Err(refuse(
            op,
            "a contiguous index-key row cannot land in an HND pool: a token step there is \
             one head wide and the row would have to be scattered",
        ));
    }
    if row <= 0 {
        return Err(refuse(op, "the index key row is zero-wide"));
    }
    if pool.seq_stride != i64::from(row) {
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

#[allow(clippy::too_many_arguments)]
pub fn layernorm_rope(
    ctx: &Ctx,
    k: &mut Tensor,
    positions: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "attention.index_layernorm_rope";
    dtype_dispatch!(OP, k.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let head_dim = count(OP, "the index key row's width", k.width)?;
    let rope_dim = stated(OP, rope_dim)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::index_knorm_rope<::pie::bf16>")
            .apply(Launch::per_row(k.rows, K_BLOCK)),
        &[
            k.arg(),
            weight.arg(),
            bias.arg(),
            positions.arg(),
            head_dim.arg(),
            rope_dim.arg(),
            theta.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

pub fn rope(
    ctx: &Ctx,
    q: &mut Tensor,
    positions: Tensor,
    heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "attention.index_rope";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let n_heads = count(OP, "the head count this rotation states", heads)?;
    let head_dim = stated(OP, head_dim)?;
    let rope_dim = stated(OP, rope_dim)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::index_q_rope<::pie::bf16>")
            .apply(Launch::per_row(q.rows, q_rope_block(n_heads))),
        &[
            q.arg(),
            positions.arg(),
            n_heads.arg(),
            head_dim.arg(),
            rope_dim.arg(),
            theta.arg(),
            ctx.stage(),
        ],
    )
}

pub fn kv_append(
    ctx: &Ctx,
    k: RaggedTensor,
    keys: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_kv_append";
    let _ = (write_page, write_offset);
    dtype_dispatch!(OP, k.data.dtype, { Bf16 => () });
    pool_pitch(OP, keys, stated(OP, k.data.width)?)?;
    kv::write_mla_to_pages(
        ctx,
        OP,
        k.data,
        ArgValue::ABSENT,
        ArgValue::ABSENT,
        k.indptr,
        keys,
        stated(OP, k.data.width)?,
        0,
    )
}

pub fn block_mean(
    ctx: &Ctx,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    keys: &KvPool,
    head_dim: u32,
    ratio: u32,
    entries: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_block_mean";
    dtype_dispatch!(OP, entries.dtype, { Bf16 => () });
    debug_assert!(
        boundary_pos.dtype == Dtype::I32 && boundary_req.dtype == Dtype::I32,
        "`{OP}` reads the i32 block tables `attention.pool_boundary_*` published"
    );
    let head_dim = count(OP, "the key width this mean states", head_dim)?;
    pool_pitch(OP, keys, head_dim)?;
    if stated(OP, entries.width)? != head_dim {
        return Err(refuse(
            OP,
            "the stated head width is not the entry this mean sized",
        ));
    }
    let ratio = count(OP, "the block width this mean pools over", ratio)?;
    let rows = count(OP, "rows", boundary_pos.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            "attn/pool.cuh",
            "::pie::attn::index_block_mean_paged<::pie::bf16>",
        )
        .apply(Launch::per_row(boundary_pos.rows, POOL_BLOCK)),
        &[
            keys.keys.arg(),
            entries.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            head_dim.arg(),
            ratio.arg(),
            keys.page_size.arg(),
            ctx.stage(),
        ],
    )?;
    let _ = rows;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn topk(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    selection: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_topk";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(selection.dtype, Dtype::I32, "`{OP}` writes i32 page ids");
    let heads = count(OP, "the head count this ranking states", heads)?;
    let head_dim = count(OP, "the key width this ranking states", head_dim)?;
    let top_k = count(OP, "the selection budget this ranking states", top_k)?;

    pool_pitch(OP, keys, head_dim)?;
    let q_width = stated(OP, q.data.width)?;
    if q_width != heads.saturating_mul(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {q_width}-wide index query does not divide by the stated head count \
                 and width"
            ),
        ));
    }
    if let Some(w) = weights
        && stated(OP, w.width)? != heads
    {
        return Err(refuse(
            OP,
            "the index head weights are not one per stated head",
        ));
    }
    if stated(OP, selection.width)? != top_k {
        return Err(refuse(
            OP,
            "the selection this statement allocated is not the budget it stated",
        ));
    }
    let num_requests = kv::lanes_of(OP, q.indptr)?;

    let max_kv = keys
        .max_pages_per_request
        .checked_mul(keys.page_size)
        .filter(|bound| *bound > 0)
        .ok_or_else(|| refuse(OP, "the page budget this fire's pool row states is zero"))?;
    let scores = ctx.scratch(
        OP,
        "attn.index_topk_scores",
        (selection.rows as usize)
            .saturating_mul(max_kv as usize)
            .saturating_mul(core::mem::size_of::<f32>()),
    )?;

    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::index_topk_paged<::pie::bf16>")
            .apply(Launch::per_row(selection.rows, K_BLOCK)),
        &[
            q.data.arg(),
            ArgValue::Ptr(weights.map_or(0, |w| w.ptr)),
            keys.keys.arg(),
            q.indptr.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            keys.last_page_lens.arg(),
            ArgValue::Ptr(scores as usize as u64),
            selection.arg(),
            num_requests.arg(),
            heads.arg(),
            head_dim.arg(),
            keys.page_size.arg(),
            max_kv.arg(),
            top_k.arg(),
            count(OP, "the pooling ratio this ranking states", ratio.max(1))?.arg(),
            ctx.stage(),
        ],
    )
}
