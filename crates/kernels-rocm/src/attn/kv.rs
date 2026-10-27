use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, kernel, refuse, stated};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/kv.hip";
const ENVELOPE: &str = "attn/kv_envelope.hip";

const BLOCK: u32 = 256;

#[must_use]
pub fn native_bf16(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::Bf16
}

/// Whether the pool's pages are e4m3 under a unit scale, read as they are
/// (vLLM's `fp8` KV): OCP e4m3fn codes, which `attn/kv.hip` reads on
/// gfx942's fnuz converters.
#[must_use]
pub fn native_e4m3(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::E4m3 && pool.values.dtype == Dtype::E4m3 && pool.scheme_byte == 0
}

#[must_use]
pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
    if page_size <= 0 {
        return 0;
    }
    (total_tokens + page_size - 1) / page_size + num_requests
}

pub(crate) fn head_split(
    op: &'static str,
    pool: &KvPool,
    row_width: u32,
) -> Result<(i32, i32), Error> {
    let wide = if pool.layout != 0 {
        pool.seq_stride
    } else {
        pool.head_stride
    };
    let head_dim = i32::try_from(wide).ok().filter(|d| *d > 0).ok_or_else(|| {
        refuse(
            op,
            format!("the pool row's strides spell no head width ({wide})"),
        )
    })?;
    let row = stated(op, row_width)?;
    if row <= 0 || row % head_dim != 0 {
        return Err(refuse(
            op,
            format!(
                "the {row}-wide appended row does not divide by the pool's head width {head_dim}"
            ),
        ));
    }
    Ok((row / head_dim, head_dim))
}

pub(crate) fn lanes_of(op: &'static str, indptr: Tensor) -> Result<i32, Error> {
    if indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's indptr is {:?}, and this entry walks an i32 boundary vector",
                indptr.dtype
            ),
        ));
    }
    let lanes = indptr.rows.saturating_sub(1);
    if lanes == 0 {
        return Err(refuse(op, "the fire's indptr spells no requests"));
    }
    stated(op, lanes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_kv_to_pages(
    ctx: &Ctx,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    indptr: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    let (kv_heads, head_dim) = head_split(op, pool, k.width)?;
    let fp8 = native_e4m3(pool);
    if !native_bf16(pool) && !fp8 {
        return Err(refuse(op, "the pool's pages are neither bf16 nor e4m3"));
    }
    let hnd = pool.layout != 0;
    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            &format!("::pie::attn::kv_append_explicit<{hnd}, {fp8}>"),
        ))
        .apply(Launch::per_row(k.rows, BLOCK)),
        &[
            k.arg(),
            v.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            write_page.arg(),
            write_offset.arg(),
            pool.row_valid.arg(),
            stated(op, k.rows)?.arg(),
            pool.page_size.arg(),
            kv_heads.arg(),
            head_dim.arg(),
            ctx.stage(),
        ],
    )?;

    if pool.has_envelopes && !hnd && !fp8 && k.rows > 0 {
        let num_requests = lanes_of(op, indptr)?;
        let _ = envelope_update_appended(
            ctx,
            op,
            pool,
            indptr,
            num_requests,
            max_touched_pages(stated(op, k.rows)?, num_requests, pool.page_size),
            kv_heads,
            head_dim,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn envelope_update_appended(
    ctx: &Ctx,
    op: &'static str,
    pool: &KvPool,
    indptr: Tensor,
    num_requests: i32,
    max_touched: i32,
    kv_heads: i32,
    head_dim: i32,
) -> Result<(), Error> {
    const fn threads_for(head_dim: i32) -> u32 {
        if head_dim < 256 {
            head_dim.unsigned_abs()
        } else {
            256
        }
    }

    ctx.fire(
        op,
        Fire::at(kernel(ENVELOPE, "::pie::layout::update_appended")).apply(Launch::grid(
            [max_touched.unsigned_abs(), kv_heads.unsigned_abs(), 1],
            [threads_for(head_dim), 1, 1],
        )),
        &[
            pool.keys.arg(),
            indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.env_min.arg(),
            pool.env_max.arg(),
            num_requests.arg(),
            pool.page_size.arg(),
            kv_heads.arg(),
            head_dim.arg(),
            ctx.stage(),
        ],
    )
}

/// Whether the latent pool holds vLLM's standard fp8 rows: one u8 plane, a
/// 512 latent's and 64 or no rotated dims' e4m3 codes a cell under a unit
/// scale (`attn/mla.hip`'s `FP8`).
#[must_use]
pub fn mla_fp8_rows(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::U8 && matches!(pool.keys.width, 512 | 576)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_mla_to_pages(
    ctx: &Ctx,
    op: &'static str,
    ckv: Tensor,
    kpe: ArgValue,
    kpe_pages: ArgValue,
    indptr: Tensor,
    pool: &KvPool,
    kv_lora_rank: i32,
    rope_dim: i32,
) -> Result<(), Error> {
    const MLA_WRITE_BLOCK: u32 = 256;

    let num_requests = lanes_of(op, indptr)?;
    let fp8 = mla_fp8_rows(pool);
    ctx.fire(
        op,
        Fire::at(kernel(
            "attn/mla.hip",
            &format!("::pie::attn::mla_kv_append<{fp8}>"),
        ))
        .apply(Launch::per_row(ckv.rows, MLA_WRITE_BLOCK)),
        &[
            ckv.arg(),
            kpe,
            pool.keys.arg(),
            kpe_pages,
            indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.row_valid.arg(),
            num_requests.arg(),
            pool.page_size.arg(),
            kv_lora_rank.arg(),
            rope_dim.arg(),
            ctx.stage(),
        ],
    )
}
