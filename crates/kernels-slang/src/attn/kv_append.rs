//! Paged KV appends (`attn.kv_append`, `attn.kv_append_shared`) from
//! `kernels/attn/kv_append.slang`.

use dtype::Dtype;

use super::pairs;
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, head_grid, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};

const KV_APPEND: &str = "attn/kv_append.slang";

/// The head width and the head count of a `row`-wide appended row, as the
/// pool's strides state them.
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

fn append(
    ctx: &Ctx<'_>,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, k.dtype, { Bf16 => "attn_kv_append_bf16" });
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
        Fire::at(KV_APPEND, entry).apply(Grid::of(lanes, [64, 1, 1])),
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

/// The new key and value rows written into their pages.
pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append(ctx, "attn.kv_append", k, v, pool, write_page, write_offset)
}

/// One plane written as both the key and the value.
pub fn kv_append_shared(
    ctx: &Ctx<'_>,
    plane: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append(
        ctx,
        "attn.kv_append_shared",
        plane,
        plane,
        pool,
        write_page,
        write_offset,
    )
}
