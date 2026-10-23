//! A quantized pool's live pages widened to bf16 before attention
//! (`attn/kv_dequant.cu`).

use crate::error::Error;

use crate::attn::kv::FP8_E4M3;
use crate::attn::kv::KvScheme;
use crate::attn::kv::fp4_block_size;
use crate::attn::kv::native_bf16;
use crate::attn::kv::native_e4m3;
use crate::attn::kv::scheme_of;
#[cfg(feature = "cuda")]
use crate::jit::ArgValue;
use crate::jit::{Arg, Ctx, Fire, Launch, refuse};
use crate::tensor::KvPool;
#[cfg(feature = "cuda")]
use crate::tensor::Tensor;
#[cfg(feature = "cuda")]
use dtype::Dtype;

const KV_DEQUANT: &str = "attn/kv_dequant.cu";

const BLOCK: u32 = 256;

/// An e4m3 pool's pages this fire reads, widened to a bf16 shadow in its
/// page list's order, as a pool of its own (its list `0..pages`), for a
/// kernel that reads bf16 pages alone; `None` where the shadow would pass
/// `cap` bytes.
#[cfg(feature = "cuda")]
pub(crate) fn widen_e4m3(
    ctx: &Ctx,
    op: &'static str,
    pool: &KvPool,
    cap: usize,
) -> Result<Option<KvPool>, Error> {
    const SHADOW_K: &str = "attn.kv_e4m3_shadow_k";
    const SHADOW_V: &str = "attn.kv_e4m3_shadow_v";
    const SHADOW_PAGES: &str = "attn.kv_e4m3_shadow_pages";
    let pages = pool.page_indices.rows;
    let page = u32::try_from(pool.page_size).unwrap_or(0);
    let width = pool.keys.width;
    let row_bytes = u64::from(page) * u64::from(width);
    if pages == 0 || pool.layout != 0 || !row_bytes.is_multiple_of(16) {
        return Ok(None);
    }
    let bytes = usize::try_from(u64::from(pages) * row_bytes * 2).unwrap_or(usize::MAX);
    if bytes > cap {
        return Ok(None);
    }
    let k = ctx.scratch_stream(op, SHADOW_K, bytes)? as u64;
    let v = ctx.scratch_stream(op, SHADOW_V, bytes)? as u64;
    let ids = ctx.scratch_stream(op, SHADOW_PAGES, pages as usize * 4)? as u64;
    let chunks = u64::from(pages) * row_bytes / 16;
    ctx.fire(
        op,
        Fire::at(KV_DEQUANT, "::pie::attn::widen_e4m3_pages").apply(Launch::flat(
            u32::try_from(chunks).unwrap_or(u32::MAX),
            BLOCK,
        )),
        &[
            pool.keys.arg(),
            pool.values.arg(),
            ArgValue::Ptr(k),
            ArgValue::Ptr(v),
            pool.page_indices.arg(),
            ArgValue::Ptr(ids),
            i64::try_from(chunks).unwrap_or(i64::MAX).arg(),
            i32::try_from(row_bytes / 16).unwrap_or(i32::MAX).arg(),
        ],
    )?;
    let cells = pages * page;
    Ok(Some(KvPool {
        keys: Tensor::new(k, cells, width, Dtype::Bf16),
        values: Tensor::new(v, cells, pool.values.width, Dtype::Bf16),
        page_indices: Tensor::new(ids, pages, 1, Dtype::I32),
        ..*pool
    }))
}

fn active_geometry(
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
    num_pages_in_batch: i32,
) -> (i64, i32, Launch) {
    let page_elems = page_size * num_kv_heads * head_dim;
    let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
    let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    (
        logical_n,
        page_elems,
        Launch::grid([blocks as u32, 1, 1], [BLOCK, 1, 1]),
    )
}

pub(crate) fn dequant_active(
    ctx: &Ctx,
    op: &'static str,
    pool: &KvPool,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Error> {
    // Pages attention reads as they are: nothing to widen.
    if native_bf16(pool) || native_e4m3(pool) {
        return Ok(());
    }
    let (logical_n, page_elems, launch) =
        active_geometry(pool.page_size, num_kv_heads, head_dim, pool.pages_in_batch);

    match scheme_of(op, pool)? {
        KvScheme::Fp8PerTensor => ctx.fire(
            op,
            Fire::at(KV_DEQUANT, "::pie::attn::dequant_fp8_pages_active").apply(launch),
            &[
                pool.keys.arg(),
                pool.values.arg(),
                pool.bf16_keys.arg(),
                pool.bf16_values.arg(),
                pool.page_indices.arg(),
                logical_n.arg(),
                page_elems.arg(),
                FP8_E4M3.arg(),
            ],
        ),
        KvScheme::Fp8PerTokenHead => ctx.fire(
            op,
            Fire::at(
                KV_DEQUANT,
                "::pie::attn::dequant_fp8_per_token_head_pages_active<::pie::bf16>",
            )
            .apply(launch),
            &[
                pool.keys.arg(),
                pool.values.arg(),
                pool.key_scales.arg(),
                pool.value_scales.arg(),
                pool.bf16_keys.arg(),
                pool.bf16_values.arg(),
                pool.page_indices.arg(),
                logical_n.arg(),
                pool.page_size.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ],
        ),
        KvScheme::Int8PerTokenHead => ctx.fire(
            op,
            Fire::at(
                KV_DEQUANT,
                "::pie::attn::dequant_int8_per_token_head_pages_active<::pie::bf16>",
            )
            .apply(launch),
            &[
                pool.keys.arg(),
                pool.values.arg(),
                pool.key_scales.arg(),
                pool.value_scales.arg(),
                pool.bf16_keys.arg(),
                pool.bf16_values.arg(),
                pool.page_indices.arg(),
                logical_n.arg(),
                pool.page_size.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ],
        ),
        KvScheme::Fp4Block => ctx.fire(
            op,
            Fire::at(
                KV_DEQUANT,
                "::pie::attn::dequant_fp4_pages_active<::pie::bf16>",
            )
            .apply(launch),
            &[
                pool.keys.arg(),
                pool.values.arg(),
                pool.key_scales.arg(),
                pool.value_scales.arg(),
                pool.bf16_keys.arg(),
                pool.bf16_values.arg(),
                pool.page_indices.arg(),
                logical_n.arg(),
                pool.page_size.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                fp4_block_size(pool.block_size).arg(),
            ],
        ),
        KvScheme::Native => Err(refuse(
            op,
            "the pool stores no bf16 pages yet its scheme byte says Native",
        )),
    }
}
