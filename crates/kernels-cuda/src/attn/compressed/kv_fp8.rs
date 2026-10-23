//! A compressed pool's fp8 rows (`attn/compressed/kv_fp8.cu`): the append
//! and the entry store.

use crate::attn::compressed::decode::HEAD_DIM;

use crate::error::Error;
use dtype::Dtype;

use crate::attn::compressed::decode::Rows;
use crate::attn::compressed::decode::ds_pool;
use crate::jit::{Arg, Ctx, Fire, Launch, refuse, stated, symbol};
use crate::tensor::{KvPool, Tensor};

const KV_FP8: &str = "attn/compressed/kv_fp8.cu";

/// The window's plane rows (bf16, 512 wide) quantized into DeepSeek's fp8
/// rows at each row's page and offset: `kv_append_shared` onto an fp8 pool.
pub fn append(
    ctx: &Ctx,
    rows: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.kv_append_shared";
    let (fmt, div) = ds_pool(OP, pool)?;
    if div != 1 {
        return Err(refuse(OP, "the window's plane holds a row a position"));
    }
    if rows.dtype != Dtype::Bf16 || rows.width != HEAD_DIM {
        return Err(refuse(OP, "an fp8 row is written from a bf16 row of 512"));
    }
    if rows.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(KV_FP8, rows_symbol("append", fmt))
            .apply(Launch::grid([rows.rows.div_ceil(4), 1, 1], [128, 1, 1])),
        &[
            rows.arg(),
            pool.keys.arg(),
            write_page.arg(),
            write_offset.arg(),
            pool.row_valid.arg(),
            stated(OP, rows.rows)?.arg(),
            pool.page_size.arg(),
            ctx.stage(),
        ],
    )
}

/// The compressed entries (bf16, 512 wide) quantized into DeepSeek's fp8
/// rows at their boundaries' positions: `pool_kv_append` onto an fp8 pool.
pub fn store_entries(
    ctx: &Ctx,
    entries: Tensor,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pool: &KvPool,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_kv_append";
    let (fmt, div) = ds_pool(OP, pool)?;
    if entries.dtype != Dtype::Bf16 || entries.width != HEAD_DIM {
        return Err(refuse(OP, "an fp8 row is written from a bf16 row of 512"));
    }
    if entries.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(KV_FP8, rows_symbol("store_entries", fmt))
            .apply(Launch::grid([entries.rows.div_ceil(4), 1, 1], [128, 1, 1])),
        &[
            entries.arg(),
            pool.keys.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            stated(OP, entries.rows)?.arg(),
            pool.page_size.arg(),
            div.arg(),
        ],
    )
}

/// A row writer's symbol: the format's instance of the generic one.
fn rows_symbol(what: &str, fmt: Rows) -> &'static str {
    symbol(&format!(
        "::pie::attn::compressed::rows_{what}<{}>",
        fmt.code()
    ))
}
