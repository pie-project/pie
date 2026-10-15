//! DeepSeek-V4's attention at decode widths in one launch
//! (`attn/dsv4_decode.cuh`): a row's sliding window of the shared key/value
//! plane, its compressed rows (all it sees, or the ones an indexer picked),
//! the two folded and the sinks applied, as `prefill_lse`, `pool_lse` (or
//! `pool_lse_selected`), `merge_lse` and `sink` land them in four.
//!
//! And DeepSeek's fp8 key/value rows (`attn/dsv4_kv.cuh`), FlashMLA's
//! `fp8_ds_mla` as vLLM serves V4: a page of `ps` positions is `ps` rows of
//! [`DS_DATA`] bytes (448 e4m3 codes, then the 64 rotated dims in bf16), then
//! `ps` rows of [`DS_SCALES`] bytes (seven ue8m0 exponents, one a 64 codes,
//! and a zero pad). A pool holding them is a u8 plane [`DS_ROW`] wide.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, refuse, stated, symbol};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/dsv4_decode.cuh";
const KV_FILE: &str = "attn/dsv4_kv.cuh";

/// The bytes of one position's fp8 row: data, then its share of the page's
/// scale footer.
pub const DS_ROW: u32 = 584;
/// The data bytes of a position: 448 e4m3 codes and 64 bf16 dims.
pub const DS_DATA: u32 = 576;
/// The scale bytes of a position: seven ue8m0 exponents and a pad.
pub const DS_SCALES: u32 = 8;
/// The codes a scale covers.
pub const DS_TILE: u32 = 64;
/// The dims stored as e4m3 codes (the rest, the rotated ones, as bf16).
pub const DS_NOPE: u32 = 448;

const HEADS: u32 = 64;
const GROUP: u32 = 16;
const HEAD_DIM: u32 = 512;
const THREADS: u32 = 256;
const KEYS: u32 = 32;
/// The most blocks a row's keys split over (one portable cluster).
const MAX_RANKS: u32 = 8;

/// Whether `pool` holds DeepSeek's fp8 rows: a u8 plane [`DS_ROW`] wide.
#[must_use]
pub fn fp8_ds(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::U8 && pool.seq_stride == i64::from(DS_ROW) && pool.layout == 0
}

fn plain_bf16(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::Bf16 && pool.layout == 0 && pool.seq_stride == i64::from(HEAD_DIM)
}

/// A row's compressed rows: the pool, its ratio, and the indexer's picks
/// (`top_k` ids a row) where it selects.
#[derive(Clone, Copy, Debug)]
pub struct Pooled<'a> {
    pub entries: &'a KvPool,
    pub ratio: u32,
    pub selection: Option<(Tensor, u32)>,
}

/// Whether [`decode`] takes this shape: 64 heads at 512 over bf16 planes, or
/// over fp8 rows (window and pool alike).
#[must_use]
pub fn takes(heads: u32, head_dim: u32, window: &KvPool, pooled: Option<&Pooled<'_>>) -> bool {
    let fp8 = fp8_ds(window);
    let alike = |p: &KvPool| {
        if fp8 {
            fp8_ds(p) && p.page_size % 2 == 0
        } else {
            plain_bf16(p)
        }
    };
    heads == HEADS
        && head_dim == HEAD_DIM
        && alike(window)
        && pooled.is_none_or(|p| alike(p.entries) && p.ratio > 0)
}

/// How many blocks split a row's keys: enough that the rows' four head
/// groups fill the chip, a power of two up to [`MAX_RANKS`].
fn ranks(ctx: &Ctx, rows: u32) -> u32 {
    let sms = ctx.multiprocessors().unwrap_or(148);
    let blocks = rows.saturating_mul(HEADS / GROUP);
    let mut r = 1;
    while r < MAX_RANKS && blocks.saturating_mul(r * 2) <= sms {
        r *= 2;
    }
    r
}

fn smem(fp8: bool) -> u32 {
    let ld = HEAD_DIM + 8;
    let tile = KEYS * ld * 2;
    // Tiles in flight: six of fp8 rows (and the one they widen into), four
    // of bf16.
    let ring = if fp8 {
        6 * KEYS * DS_ROW + tile
    } else {
        4 * tile
    };
    GROUP * ld * 2 + ring + 2 * GROUP * (KEYS + 8) * 4
}

#[allow(clippy::too_many_arguments)]
pub fn decode(
    ctx: &Ctx,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    window_pool: &KvPool,
    window: u32,
    pooled: Option<Pooled<'_>>,
    sinks: Tensor,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.dsv4_decode";
    if !takes(heads, head_dim, window_pool, pooled.as_ref()) {
        return Err(refuse(
            OP,
            format!(
                "no fused DeepSeek-V4 attention for {heads} heads at {head_dim} over these planes"
            ),
        ));
    }
    if q.dtype != Dtype::Bf16 || o.dtype != Dtype::Bf16 || q.width != heads * head_dim {
        return Err(refuse(
            OP,
            "the queries and output are bf16 rows of every head",
        ));
    }
    let sink_ty = match sinks.dtype {
        Dtype::F32 => "float",
        Dtype::Bf16 => "::pie::bf16",
        held => return Err(refuse(OP, format!("the sinks are {held:?}"))),
    };
    let rows = o.rows;
    if rows == 0 {
        return Ok(());
    }
    let fp8 = fp8_ds(window_pool);
    let window = count(OP, "the sliding window", window)?;
    let ranks = ranks(ctx, rows);
    let launch = Launch::grid([ranks, HEADS / GROUP, rows], [THREADS, 1, 1]).smem(smem(fp8));
    let launch = if ranks > 1 {
        launch.cluster(ranks)
    } else {
        launch
    };
    let (entries, ratio, selection, top_k) = match pooled {
        Some(p) => {
            let (sel, top_k) = p
                .selection
                .map_or((ArgValue::ABSENT, 0), |(s, k)| (s.arg(), k));
            if let Some((s, k)) = p.selection
                && (s.width != k || s.dtype != Dtype::I32 || s.rows < rows)
            {
                return Err(refuse(
                    OP,
                    "the selection is not the i32 ids a row it states",
                ));
            }
            (
                Some(p.entries),
                count(OP, "the pooling ratio", p.ratio)?,
                sel,
                i32::try_from(top_k).map_err(|_| refuse(OP, "top_k does not fit an int"))?,
            )
        }
        None => (None, 0, ArgValue::ABSENT, 0),
    };
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::dsv4::decode<{HEAD_DIM}, {sink_ty}, {ranks}, {fp8}>"
            )),
        )
        .apply(launch),
        &[
            q.arg(),
            window_pool.keys.arg(),
            window_pool.page_indices.arg(),
            window_pool.page_indptr.arg(),
            window_pool.seq_stride.arg(),
            window_pool.page_size.arg(),
            window.arg(),
            entries.map_or(ArgValue::ABSENT, |e| e.keys.arg()),
            entries.map_or(ArgValue::ABSENT, |e| e.page_indices.arg()),
            entries.map_or(ArgValue::ABSENT, |e| e.page_indptr.arg()),
            entries.map_or(0, |e| e.seq_stride).arg(),
            entries.map_or(1, |e| e.page_size).arg(),
            ratio.arg(),
            selection,
            top_k.arg(),
            positions.arg(),
            request_of_token.arg(),
            sinks.arg(),
            o.arg(),
            sm_scale.arg(),
            ctx.stage(),
        ],
    )
}

fn ds_pool(op: &'static str, pool: &KvPool) -> Result<(), Error> {
    if !fp8_ds(pool) || pool.page_size % 2 != 0 {
        return Err(refuse(
            op,
            format!(
                "the pool is {:?} at {} a position, not DeepSeek's {DS_ROW}-byte fp8 rows",
                pool.keys.dtype, pool.seq_stride
            ),
        ));
    }
    Ok(())
}

/// The window's plane rows (bf16, 512 wide) quantized into DeepSeek's fp8
/// rows at each row's page and offset: `kv_append_shared` onto an fp8 pool.
pub fn append(
    ctx: &Ctx,
    rows: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.kv_append_shared";
    ds_pool(OP, pool)?;
    if rows.dtype != Dtype::Bf16 || rows.width != HEAD_DIM {
        return Err(refuse(OP, "an fp8 row is written from a bf16 row of 512"));
    }
    if rows.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(KV_FILE, "::pie::attn::dsv4::ds_append")
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
    const OP: &str = "attention.pool_kv_append";
    ds_pool(OP, pool)?;
    if entries.dtype != Dtype::Bf16 || entries.width != HEAD_DIM {
        return Err(refuse(OP, "an fp8 row is written from a bf16 row of 512"));
    }
    if entries.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(KV_FILE, "::pie::attn::dsv4::ds_store_entries")
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
        ],
    )
}
