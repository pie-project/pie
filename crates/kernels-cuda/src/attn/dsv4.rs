//! DeepSeek-V4's attention at decode widths in one launch
//! (`attn/dsv4_decode.cuh`): a row's sliding window of the shared key/value
//! plane, its compressed rows (all it sees, or the ones an indexer picked),
//! the two folded and the sinks applied, as `prefill_lse`, `pool_lse` (or
//! `pool_lse_selected`), `merge_lse` and `sink` land them in four.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, refuse, symbol};
use crate::tensor::{KvPool, Tensor};

const FILE: &str = "attn/dsv4_decode.cuh";

const HEADS: u32 = 64;
const HEAD_DIM: u32 = 512;
const THREADS: u32 = 256;
const KEYS: u32 = 32;
const STAGES: u32 = 3;
/// The most blocks a row's keys split over (one portable cluster).
const MAX_RANKS: u32 = 8;

/// A row's compressed rows: the pool, its ratio, and the indexer's picks
/// (`top_k` ids a row) where it selects.
#[derive(Clone, Copy, Debug)]
pub struct Pooled<'a> {
    pub entries: &'a KvPool,
    pub ratio: u32,
    pub selection: Option<(Tensor, u32)>,
}

/// Whether [`decode`] takes this shape: 64 heads at 512 over bf16 planes.
#[must_use]
pub fn takes(heads: u32, head_dim: u32, window: &KvPool, pooled: Option<&Pooled<'_>>) -> bool {
    let plain = |p: &KvPool| {
        p.keys.dtype == Dtype::Bf16 && p.layout == 0 && p.seq_stride == i64::from(HEAD_DIM)
    };
    heads == HEADS
        && head_dim == HEAD_DIM
        && plain(window)
        && pooled.is_none_or(|p| plain(p.entries) && p.ratio > 0)
}

/// How many blocks split a row's keys: enough that the rows fill the chip,
/// a power of two up to [`MAX_RANKS`].
fn ranks(ctx: &Ctx, rows: u32) -> u32 {
    let sms = ctx.multiprocessors().unwrap_or(148);
    let mut r = 1;
    while r < MAX_RANKS && rows.saturating_mul(r * 2) <= sms {
        r *= 2;
    }
    r
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
            format!("no fused DeepSeek-V4 attention for {heads} heads at {head_dim}"),
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
    let window = count(OP, "the sliding window", window)?;
    let ranks = ranks(ctx, rows);
    let ld = HEAD_DIM + 8;
    let smem = HEADS * ld * 2 + STAGES * KEYS * ld * 2 + 8 * 16 * KEYS * 4;
    let launch = Launch::grid([ranks, rows, 1], [THREADS, 1, 1]).smem(smem);
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
            symbol(&format!("::pie::attn::dsv4::decode<{HEAD_DIM}, {sink_ty}>")),
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
