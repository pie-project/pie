//! The paged kv writers and the quantized-page maintenance around them: the
//! bf16 and quantized appenders, the envelope update that shadows appended
//! keys, the active-page dequant prelude the fa2 entries run, and the mla
//! latent writer the mla/index appends share.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, refuse, stated};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/kv.cuh";

const BLOCK: u32 = 256;

/// The quantization schemes the pool row's `scheme_byte` can spell.
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

/// The `__nv_fp8_interpretation_t` the fp8 paths read: e4m3.
///
// MENLO-SEAM: the erased pool spells fp8 storage as `keys.dtype == U8` plus
// the scheme byte; the old `storage_dtype` field that could say e5m2 is
// retired, so this plane always states e4m3 — the only kind the old drivers
// ever configured.
const FP8_E4M3: u32 = 0;

const fn fp4_block_size(block_size: i32) -> i32 {
    if block_size > 0 { block_size } else { 16 }
}

/// Whether the pool stores native bf16 pages — the reading that used to be
/// the `native_bf16` field, now spelled by the storage handle's dtype.
#[must_use]
pub fn native_bf16(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::Bf16
}

/// An upper bound on the pages an append of `total_tokens` rows can touch.
#[must_use]
pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
    if page_size <= 0 {
        return 0;
    }
    (total_tokens + page_size - 1) / page_size + num_requests
}

/// The `(kv_heads, head_dim)` split the pool row's strides spell for an
/// appended row. Strides are engine facts the validator never sees, so
/// disagreement is refused, not asserted.
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

/// The lane count an indptr spells: `rows - 1`, refused when degenerate. The
/// boundary vector is driver-assembled, not an operand the validator sees, so
/// a wrong dtype is refused on the same footing as a degenerate length, not
/// asserted (the boundary rule at [`refuse`]).
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

/// Appends `k`/`v` rows into the pool's pages, addressed by the op's
/// per-token write descriptors (`write_page`/`write_offset`). Dispatches on
/// the pool's storage: native bf16 (with the envelope shadow when the
/// scheme keeps one), or the quantized writer the scheme byte names.
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
    if native_bf16(pool) {
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
        // MENLO-SEAM: the quantized writers predate the explicit write
        // descriptors — they still re-derive each token's cell from the
        // read-side page tables, so the stated pair goes unread on these
        // schemes until the device text grows explicit quantized writers.
        let _ = (write_page, write_offset);
        write_kv_quantised(ctx, op, k, v, indptr, pool, kv_heads, head_dim)
    }
}

/// The general explicit-descriptor write (`kv_append_explicit`): each token
/// row lands in the ONE `(write_page[t], write_offset[t])` cell the op
/// states — never a position→(page, offset) derivation, which cannot spell
/// a fresh-page write that is not the page-run tail. `indptr` is only the
/// envelope refresh's lane walk.
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
    let instantiation = if hnd {
        "::pie::attn::kv_append_explicit<::pie::true_type::value>"
    } else {
        "::pie::attn::kv_append_explicit<::pie::false_type::value>"
    };
    ctx.fire(
        op,
        Fire::at(FILE, instantiation).apply(Launch::per_row(k.rows, BLOCK)),
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
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )?;

    if pool.has_envelopes && !hnd && k.rows > 0 {
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

/// Refreshes the bf16 key envelopes over the pages an append touched.
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
                // The staged-geometry seat: the region's live-rows word when a
                // body replay armed one, and the null seat (`ABSENT`) otherwise.
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
                    // The staged-geometry seat: the region's live-rows word when a
                    // body replay armed one, and the null seat (`ABSENT`) otherwise.
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
                    // The staged-geometry seat: the region's live-rows word when a
                    // body replay armed one, and the null seat (`ABSENT`) otherwise.
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

/// Dequantizes this fire's active pages into the bf16 shadow the fa2
/// kernels read. A no-op on native pools; on quantized ones the fa2
/// entries fire it best-effort before attending, as the old plane did.
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

/// Writes latent rows (`ckv` beside its rope plane) into an mla-shaped
/// pool. Shared by `attention.mla_kv_append` and `attention.index_kv_append`
/// (which passes a null rope plane of zero width).
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
