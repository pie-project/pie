use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse, stated, symbol};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/kv.cuh";

const BLOCK: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvScheme {
    Native = 0,
    Fp8PerTensor = 1,
    Int8PerTokenHead = 2,
    Fp8PerTokenHead = 3,
    Fp4Block = 4,
}

impl KvScheme {
    #[must_use]
    pub const fn of_byte(byte: i32) -> Option<Self> {
        match byte {
            0 => Some(Self::Native),
            1 => Some(Self::Fp8PerTensor),
            2 => Some(Self::Int8PerTokenHead),
            3 => Some(Self::Fp8PerTokenHead),
            4 => Some(Self::Fp4Block),
            _ => None,
        }
    }
}

fn scheme_of(op: &'static str, pool: &KvPool) -> Result<KvScheme, Error> {
    KvScheme::of_byte(pool.scheme_byte).ok_or_else(|| {
        refuse(
            op,
            format!("no kv scheme is named by byte {}", pool.scheme_byte),
        )
    })
}

const FP8_E4M3: u32 = 0;

const fn fp4_block_size(block_size: i32) -> i32 {
    if block_size > 0 { block_size } else { 16 }
}

#[must_use]
pub fn native_bf16(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::Bf16
}

/// Whether the pool's pages are e4m3 under a unit scale, read as they are
/// (vLLM's `fp8` KV): no planes of scales, no bf16 copy.
#[must_use]
pub fn native_e4m3(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::E4m3
        && pool.values.dtype == Dtype::E4m3
        && KvScheme::of_byte(pool.scheme_byte) == Some(KvScheme::Native)
}

#[must_use]
pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
    if page_size <= 0 {
        return 0;
    }
    (total_tokens + page_size - 1) / page_size + num_requests
}

/// An e4m3 pool's pages this fire reads, widened to a bf16 shadow in its
/// page list's order, as a pool of its own (its list `0..pages`), for a
/// kernel that reads bf16 pages alone; `None` where the shadow would pass
/// `cap` bytes.
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
        Fire::at(FILE, "::pie::attn::widen_e4m3_pages").apply(Launch::flat(
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
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::kv_append_explicit<{layout}, {pages}>"
            )),
        )
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
        Fire::at(
            "layout/envelope.cuh",
            "::pie::layout::update_appended<::pie::bf16>",
        )
        .apply(Launch::grid(
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
            Fire::at(FILE, "::pie::attn::kv_append_fp8_per_tensor")
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
                Fire::at(FILE, instantiation)
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
                Fire::at(FILE, "::pie::attn::kv_append_fp4_block").apply(Launch::grid(
                    [tokens, heads, blocks.unsigned_abs()],
                    [32, 1, 1],
                )),
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
    if native_bf16(pool) {
        return Ok(());
    }
    let (logical_n, page_elems, launch) =
        active_geometry(pool.page_size, num_kv_heads, head_dim, pool.pages_in_batch);

    match scheme_of(op, pool)? {
        KvScheme::Fp8PerTensor => ctx.fire(
            op,
            Fire::at(FILE, "::pie::attn::dequant_fp8_pages_active").apply(launch),
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
                FILE,
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
                FILE,
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
            Fire::at(FILE, "::pie::attn::dequant_fp4_pages_active<::pie::bf16>").apply(launch),
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

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
/// The latent rows `ckv` and their rotated dims `kpe` (64 a row) landed in
/// `pool` as standard fp8 rows (`attn/mla.cuh`'s `mla_kv_append_fp8`).
pub(crate) fn write_mla_fp8_rows(
    ctx: &Ctx,
    op: &'static str,
    ckv: Tensor,
    kpe: Tensor,
    indptr: Tensor,
    pool: &KvPool,
) -> Result<(), Error> {
    let rope = pool.keys.width.saturating_sub(512);
    if ckv.width != 512 || (rope > 0 && (kpe.width != rope || kpe.rows != ckv.rows)) {
        return Err(refuse(
            op,
            format!(
                "fp8 latent rows hold 512 latent and {rope} rotated dims a position, and this \
                 append brings {} x {} and {} x {}",
                ckv.rows, ckv.width, kpe.rows, kpe.width
            ),
        ));
    }
    let num_requests = lanes_of(op, indptr)?;
    ctx.fire(
        op,
        Fire::at("attn/mla.cuh", "::pie::attn::mla_kv_append_fp8")
            .apply(Launch::per_row(ckv.rows, 160)),
        &[
            ckv.arg(),
            kpe.arg(),
            pool.keys.arg(),
            indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            pool.row_valid.arg(),
            num_requests.arg(),
            pool.page_size.arg(),
            i32::try_from(rope).unwrap_or(0).arg(),
        ],
    )
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
    ctx.fire(
        op,
        Fire::at("attn/mla.cuh", "::pie::attn::mla_kv_append")
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
        ],
    )
}
