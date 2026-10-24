//! K/V rows written into a paged pool (`attn/kv_append.cu`): bf16, and fp8
//! or fp4 with their scales.

use crate::error::Error;

use crate::attn::kv::FP8_E4M3;
use crate::attn::kv::KvScheme;
use crate::attn::kv::fp4_block_size;
use crate::attn::kv::head_split;
use crate::attn::kv::lanes_of;
use crate::attn::kv::max_touched_pages;
use crate::attn::kv::native_bf16;
use crate::attn::kv::native_e4m3;
use crate::attn::kv::scheme_of;
use crate::attn::kv_envelope::envelope_update_appended;
use crate::jit::{Arg, Ctx, Fire, Launch, kernel, refuse, stated};
use crate::tensor::{KvPool, Tensor};

const KV_APPEND: &str = "attn/kv_append.cu";

const BLOCK: u32 = 256;

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
    if native_bf16(pool) || native_e4m3(pool) {
        write_kv_bf16(
            ctx,
            op,
            k,
            v,
            indptr,
            pool,
            write_page,
            write_offset,
            kv_heads,
            head_dim,
        )
    } else {
        let _ = (write_page, write_offset);
        write_kv_quantised(ctx, op, k, v, indptr, pool, kv_heads, head_dim)
    }
}

#[allow(clippy::too_many_arguments)]
fn write_kv_bf16(
    ctx: &Ctx,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    indptr: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
    kv_heads: i32,
    head_dim: i32,
) -> Result<(), Error> {
    let hnd = pool.layout != 0;
    let layout = if hnd {
        "::pie::true_type::value"
    } else {
        "::pie::false_type::value"
    };
    let pages = if native_e4m3(pool) {
        "__nv_fp8_e4m3"
    } else {
        "::pie::bf16"
    };
    ctx.fire(
        op,
        Fire::at(kernel(
            KV_APPEND,
            &format!("::pie::attn::kv_append_explicit<{layout}, {pages}>"),
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

    if pool.has_envelopes && !hnd && k.rows > 0 && native_bf16(pool) {
        let num_requests = lanes_of(op, indptr)?;
        envelope_update_appended(
            ctx,
            op,
            pool,
            indptr,
            num_requests,
            max_touched_pages(stated(op, k.rows)?, num_requests, pool.page_size),
            kv_heads,
            head_dim,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_kv_quantised(
    ctx: &Ctx,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    indptr: Tensor,
    pool: &KvPool,
    kv_heads: i32,
    head_dim: i32,
) -> Result<(), Error> {
    let num_requests = lanes_of(op, indptr)?;
    let tokens = k.rows;
    let heads = kv_heads.unsigned_abs();

    match scheme_of(op, pool)? {
        KvScheme::Fp8PerTensor => ctx.fire(
            op,
            Fire::at(kernel(KV_APPEND, "::pie::attn::kv_append_fp8_per_tensor"))
                .apply(Launch::per_row(tokens, BLOCK)),
            &[
                k.arg(),
                v.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                indptr.arg(),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                pool.last_page_lens.arg(),
                num_requests.arg(),
                pool.page_size.arg(),
                kv_heads.arg(),
                head_dim.arg(),
                FP8_E4M3.arg(),
                ctx.stage(),
            ],
        ),
        scheme @ (KvScheme::Int8PerTokenHead | KvScheme::Fp8PerTokenHead) => {
            let instantiation = if scheme == KvScheme::Fp8PerTokenHead {
                "::pie::attn::kv_append_per_token_head<::pie::true_type::value>"
            } else {
                "::pie::attn::kv_append_per_token_head<::pie::false_type::value>"
            };

            let smem = 2 * (BLOCK / 32) * (core::mem::size_of::<f32>() as u32);
            ctx.fire(
                op,
                Fire::at(kernel(KV_APPEND, instantiation))
                    .apply(Launch::grid([tokens, heads, 1], [BLOCK, 1, 1]).smem(smem)),
                &[
                    k.arg(),
                    v.arg(),
                    pool.keys.arg(),
                    pool.values.arg(),
                    pool.key_scales.arg(),
                    pool.value_scales.arg(),
                    indptr.arg(),
                    pool.page_indices.arg(),
                    pool.page_indptr.arg(),
                    pool.last_page_lens.arg(),
                    num_requests.arg(),
                    pool.page_size.arg(),
                    kv_heads.arg(),
                    head_dim.arg(),
                    ctx.stage(),
                ],
            )
        }
        KvScheme::Fp4Block => {
            let block_size = fp4_block_size(pool.block_size);
            let blocks =
                head_dim.div_euclid(block_size) + i32::from(head_dim.rem_euclid(block_size) != 0);
            ctx.fire(
                op,
                Fire::at(kernel(KV_APPEND, "::pie::attn::kv_append_fp4_block")).apply(
                    Launch::grid([tokens, heads, blocks.unsigned_abs()], [32, 1, 1]),
                ),
                &[
                    k.arg(),
                    v.arg(),
                    pool.keys.arg(),
                    pool.values.arg(),
                    pool.key_scales.arg(),
                    pool.value_scales.arg(),
                    indptr.arg(),
                    pool.page_indices.arg(),
                    pool.page_indptr.arg(),
                    pool.last_page_lens.arg(),
                    num_requests.arg(),
                    pool.page_size.arg(),
                    kv_heads.arg(),
                    head_dim.arg(),
                    block_size.arg(),
                    ctx.stage(),
                ],
            )
        }
        KvScheme::Native => Err(refuse(
            op,
            "the pool stores no bf16 pages yet its scheme byte says Native",
        )),
    }
}
