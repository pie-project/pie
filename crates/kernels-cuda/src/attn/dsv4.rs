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
//!
//! And DeepSeek-V4.1's rows ([`Rows`]), laid out alike: its window's plane
//! in MXFP8 (all 512 dims e4m3 under one ue8m0 a 32, 528 bytes) and its
//! compressed rows in NVFP4 (e2m1 under one e4m3 a 16, 288 bytes), as its
//! reference quantizes them.

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

/// The quantized rows a DeepSeek pool may hold, each a u8 plane of its
/// own width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// V4's fp8 rows, [`DS_ROW`] bytes a position.
    Ds,
    /// V4.1's window: 512 e4m3 codes and 16 ue8m0 exponents.
    Mx8,
    /// V4.1's compressed rows: 256 bytes of e2m1 pairs and 32 e4m3 scales.
    Nv4,
}

impl Rows {
    /// The bytes of a position.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Ds => DS_ROW,
            Self::Mx8 => 528,
            Self::Nv4 => 288,
        }
    }

    /// The kernels' format constant (`attn/dsv4_kv.cuh`).
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Ds => 1,
            Self::Mx8 => 2,
            Self::Nv4 => 3,
        }
    }
}

/// The quantized rows `pool` holds, by its width: none where it is not a u8
/// plane of one of [`Rows`]' widths.
#[must_use]
pub fn rows_of(pool: &KvPool) -> Option<Rows> {
    if pool.keys.dtype != Dtype::U8 || pool.layout != 0 {
        return None;
    }
    [Rows::Ds, Rows::Mx8, Rows::Nv4]
        .into_iter()
        .find(|r| pool.seq_stride == i64::from(r.bytes()))
}

/// The quantized rows `pool` holds and how many of its positions' rows one
/// entry spans: 1 a row a position, `ratio` a compact pool's (its entries a
/// row each, `page_size / ratio` a page).
#[must_use]
pub fn rows_packed(pool: &KvPool) -> Option<(Rows, i32)> {
    if pool.keys.dtype != Dtype::U8 || pool.layout != 0 || pool.seq_stride <= 0 {
        return None;
    }
    for div in [1i64, 2, 4, 8, 16] {
        for rows in [Rows::Ds, Rows::Mx8, Rows::Nv4] {
            if pool.seq_stride * div == i64::from(rows.bytes())
                && i64::from(pool.page_size) % div == 0
            {
                return Some((rows, i32::try_from(div).unwrap_or(1)));
            }
        }
    }
    None
}

/// `pool` as its compressed entries read at `ratio`: a compact pool's view,
/// an entry a row of its format's width and `page_size / ratio` a page, and
/// whether it is one.
#[must_use]
pub fn entries_view(pool: &KvPool, ratio: u32, row: i64) -> (KvPool, bool) {
    if !crate::attn::pool::compact(pool, ratio, row) {
        return (*pool, false);
    }
    let r = i64::from(ratio);
    let cells = u32::try_from(i64::from(pool.keys.rows) / r).unwrap_or(0);
    let width = u32::try_from(row).unwrap_or(0);
    (
        KvPool {
            keys: Tensor::new(pool.keys.ptr, cells, width, pool.keys.dtype),
            values: Tensor::new(pool.values.ptr, cells, width, pool.values.dtype),
            page_size: pool.page_size / i32::try_from(ratio).unwrap_or(1),
            seq_stride: row,
            head_stride: row,
            ..*pool
        },
        true,
    )
}

/// The natural width of an entry `pool` holds: its quantized rows' bytes,
/// or a bf16 head of 512.
fn entry_row(pool: &KvPool) -> i64 {
    rows_packed(pool).map_or(i64::from(HEAD_DIM), |(rows, _)| i64::from(rows.bytes()))
}

/// Whether `pool` holds DeepSeek's quantized rows of any format, a row a
/// position or compact.
#[must_use]
pub fn ds_rows(pool: &KvPool) -> bool {
    rows_packed(pool).is_some()
}

/// Whether `pool` holds V4's fp8 rows: a u8 plane [`DS_ROW`] wide.
#[must_use]
pub fn fp8_ds(pool: &KvPool) -> bool {
    rows_of(pool) == Some(Rows::Ds)
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
    let fp8 = ds_rows(window);
    let alike = |p: &KvPool| {
        if fp8 {
            ds_rows(p) && p.page_size % 2 == 0
        } else {
            plain_bf16(p)
        }
    };
    heads == HEADS
        && head_dim == HEAD_DIM
        && alike(window)
        && pooled.is_none_or(|p| {
            p.ratio > 0 && alike(&entries_view(p.entries, p.ratio, entry_row(p.entries)).0)
        })
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

fn smem(rows: Option<(Rows, Rows)>) -> u32 {
    let ld = HEAD_DIM + 8;
    let tile = KEYS * ld * 2;
    // Tiles in flight: six of quantized rows (and the one they widen into),
    // four of bf16. A raw key holds the wider format's data and scales.
    let ring = if let Some((w, p)) = rows {
        let split = |r: Rows| match r {
            Rows::Ds => (DS_DATA, DS_SCALES),
            Rows::Mx8 => (512, 16),
            Rows::Nv4 => (256, 32),
        };
        let ((wd, ws), (pd, ps)) = (split(w), split(p));
        6 * KEYS * (wd.max(pd) + ws.max(ps)) + tile
    } else {
        4 * tile
    };
    GROUP * ld * 2 + ring + 2 * GROUP * (KEYS + 8) * 4
}

/// The inverse rotation of an attention output's last `dim` dims, landed
/// with the output (DeepSeek-V4 turns its attention output back by each
/// row's position before its output projection): `rope_partial_last`'s
/// interleaved turn with the angles negated.
#[derive(Clone, Copy, Debug)]
pub struct Turn {
    pub dim: u32,
    pub theta: f32,
    pub yarn: Option<crate::elemwise::rope::Yarn>,
}

/// `pie::attn::dsv4::Turn`, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct TurnArgs {
    dim: i32,
    theta: f32,
    factor: f32,
    low: f32,
    high: f32,
}

impl TurnArgs {
    pub(crate) const NONE: TurnArgs = TurnArgs {
        dim: 0,
        theta: 0.0,
        factor: 1.0,
        low: 0.0,
        high: 0.0,
    };

    pub(crate) fn of(turn: Option<Turn>) -> TurnArgs {
        let Some(t) = turn else {
            return TurnArgs::NONE;
        };
        let dim = i32::try_from(t.dim).unwrap_or(0);
        let (factor, low, high) = match t.yarn {
            Some(y) => {
                let (low, high) = crate::elemwise::rope::ramp_bounds(
                    dim,
                    t.theta,
                    y.beta_fast,
                    y.beta_slow,
                    i32::try_from(y.original_max_position).unwrap_or(i32::MAX),
                );
                (y.factor, low, high)
            }
            None => (1.0, 0.0, 0.0),
        };
        TurnArgs {
            dim,
            theta: t.theta,
            factor,
            low,
            high,
        }
    }
}

/// DeepSeek-V4's window, pool and sinks in one launch, a row's window
/// held above its replay floor where `floors` (i32 a row) states one.
#[allow(clippy::too_many_arguments)]
pub fn decode(
    ctx: &Ctx,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    floors: Option<Tensor>,
    window_pool: &KvPool,
    window: u32,
    pooled: Option<Pooled<'_>>,
    sinks: Tensor,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    turn: Option<Turn>,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.dsv4_decode";
    let turn = TurnArgs::of(turn);
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
    // The window's and the pool's row formats (a window-only layer reads
    // no pool; its format stands in).
    // A compact pool read through its view: an entry a row, `page_size /
    // ratio` a page.
    let (view, pool_compact) = pooled.as_ref().map_or((None, false), |p| {
        let (v, compact) = entries_view(p.entries, p.ratio, entry_row(p.entries));
        (Some(v), compact)
    });
    let formats = rows_of(window_pool).map(|w| (w, view.as_ref().and_then(rows_of).unwrap_or(w)));
    let (wf, pf) = formats.map_or((0, 0), |(w, p)| (w.code(), p.code()));
    let window = count(OP, "the sliding window", window)?;
    let ranks = ranks(ctx, rows);
    let launch = Launch::grid([ranks, HEADS / GROUP, rows], [THREADS, 1, 1]).smem(smem(formats));
    let launch = if ranks > 1 {
        launch.cluster(ranks)
    } else {
        launch
    };
    let (entries, ratio, selection, top_k) = match pooled {
        Some(p) => {
            let p = Pooled {
                entries: view.as_ref().unwrap_or(p.entries),
                ..p
            };
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
                "::pie::attn::dsv4::decode<{HEAD_DIM}, {sink_ty}, {ranks}, {wf}, {pf}>"
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
            floors.map_or(ArgValue::ABSENT, |f| f.arg()),
            sinks.arg(),
            o.arg(),
            sm_scale.arg(),
            ArgValue::of(&turn),
            i32::from(pool_compact).arg(),
            ctx.stage(),
        ],
    )
}

fn ds_pool(op: &'static str, pool: &KvPool) -> Result<(Rows, i32), Error> {
    match rows_packed(pool) {
        Some((rows, div)) if pool.page_size % 2 == 0 => Ok((rows, div)),
        _ => Err(refuse(
            op,
            format!(
                "the pool is {:?} at {} a position, not DeepSeek's quantized rows",
                pool.keys.dtype, pool.seq_stride
            ),
        )),
    }
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
        Fire::at(KV_FILE, rows_symbol("append", fmt))
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
    let (fmt, div) = ds_pool(OP, pool)?;
    if entries.dtype != Dtype::Bf16 || entries.width != HEAD_DIM {
        return Err(refuse(OP, "an fp8 row is written from a bf16 row of 512"));
    }
    if entries.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(KV_FILE, rows_symbol("store_entries", fmt))
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
    symbol(&format!("::pie::attn::dsv4::rows_{what}<{}>", fmt.code()))
}
