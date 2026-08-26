//! `Index`: the sparse-attention indexer — a small key cache scored against
//! queries to select which pages the main attention will read. One entry
//! per IR variant, transcribed from the old INDEX claims.

use new_kernels::KernelError;
use new_model_ir::Dtype;

use crate::attn::kv;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/dsa_indexer.cuh";

const K_BLOCK: u32 = 256;

#[must_use]
fn q_rope_block(n_heads: i32) -> u32 {
    let rounded = ((n_heads.max(0) + 31) / 32) * 32;
    #[allow(clippy::cast_sign_loss)]
    let block = rounded as u32;
    if block < 32 { 32 } else { block }
}

/// The index pool stores whole key rows contiguously; its strides must
/// spell exactly that, and an HND pool cannot.
fn pool_pitch(op: &'static str, pool: &KvPool, row: i32) -> Result<(), KernelError> {
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

/// Layernorms the index key row and ropes its tail, in place on `k`.
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
) -> Result<(), KernelError> {
    const OP: &str = "index.layernorm_rope";
    dtype_dispatch!(OP, k.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let head_dim = stated(OP, nonzero(OP, "the index key row's width", k.width)?)?;
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
        ],
    )
}

/// Ropes the index query's tail per head, in place on `q`.
pub fn rope(
    ctx: &Ctx,
    q: &mut Tensor,
    positions: Tensor,
    heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), KernelError> {
    const OP: &str = "index.rope";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let n_heads = stated(
        OP,
        nonzero(OP, "the head count this rotation states", heads)?,
    )?;
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
        ],
    )
}

/// Appends index key rows into the pool's pages: the mla latent writer with
/// a null rope plane.
///
// MENLO-SEAM: as `attention.kv_append` — the stated kv_indices/positions go
// unread; the fire indptr rides in `k`.
pub fn kv_append(
    ctx: &Ctx,
    k: RaggedTensor,
    keys: &KvPool,
    kv_indices: Tensor,
    positions: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "index.kv_append";
    let _ = (kv_indices, positions);
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

/// Scores `q` against the cached keys and writes the top-k page ids per
/// row. The score scratch is a process-global slab (an entry may not
/// allocate per fire).
#[allow(clippy::too_many_arguments)]
pub fn topk(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Tensor,
    keys: &KvPool,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    selection: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "index.topk";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(selection.dtype, Dtype::I32, "`{OP}` writes i32 page ids");
    let heads = stated(
        OP,
        nonzero(OP, "the head count this ranking states", heads)?,
    )?;
    let head_dim = stated(
        OP,
        nonzero(OP, "the key width this ranking states", head_dim)?,
    )?;
    let top_k = stated(
        OP,
        nonzero(OP, "the selection budget this ranking states", top_k)?,
    )?;

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
    if stated(OP, weights.width)? != heads {
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
        "attn.dsa_index_scores",
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
            weights.arg(),
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
        ],
    )
}
