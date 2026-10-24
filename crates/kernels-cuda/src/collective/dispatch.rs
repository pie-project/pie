use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, kernel, refuse};
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "collective/dispatch.cu";

/// `peer::kMaxRanks`.
pub const EP_MAX_RANKS: u32 = 8;

/// Bytes of one rank's EP signal block (`peer::Signals`, then the inbox at
/// `kInboxAt`), zeroed when the group opens and never cleared after.
pub const EP_SIGNAL_BYTES: u64 = 4096;

/// `kCountThreads`.
const COUNT_THREADS: u32 = 1024;

/// `kSendWarps`.
const SEND_WARPS: u32 = 8;

/// A rank's view of an expert-parallel group whose ranks map one another's
/// memory: every rank's EP signal block, in rank order. Every rank fires
/// every dispatch and combine, in the same order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ep {
    pub rank: u32,
    pub world: u32,
    pub signals: [u64; EP_MAX_RANKS as usize],
}

/// Where each logical expert lives: `[experts][stride]` i32 on the device,
/// a replica count (1 or more), then that many physical slots, each
/// `rank * slots + local slot`. A route picks its replica by (rank, token,
/// route), so one placement lays the same rows on any rank count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub table: u64,
    pub stride: u32,
    pub experts: u32,
    pub slots: u32,
}

/// One rank's landing buffers, which every peer writes into: `world * T`
/// rows (`T` the tokens every rank brings), each a token's row, its routes
/// as this rank's local slots (-1 for a route held elsewhere), their
/// weights, and `src_rank * T + t`. Rows past those received are padding,
/// with `origin` and every `local` -1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Landing {
    pub rows: Tensor,
    pub local: Tensor,
    pub row_weights: Option<Tensor>,
    pub origin: Tensor,
}

/// This rank's tokens: `[T, H]` rows, their routes (`[T, top_k]` i32 logical
/// experts) and, where the experts consume them, the routes' weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tokens {
    pub x: Tensor,
    pub routes: Tensor,
    pub weights: Option<Tensor>,
}

/// `pie::collective::DispatchParams`, field for field.
#[repr(C)]
struct Params {
    x: u64,
    routes: u64,
    weights: u64,
    win: u64,
    place: u64,
    pos: u64,
    offs: u64,
    counts: u64,
    rows: [u64; EP_MAX_RANKS as usize],
    local: [u64; EP_MAX_RANKS as usize],
    row_weights: [u64; EP_MAX_RANKS as usize],
    origin: [u64; EP_MAX_RANKS as usize],
    group: [u64; EP_MAX_RANKS as usize],
    self_signals: u64,
    rank: i32,
    ranks: i32,
    tokens: i32,
    packs: i32,
    top_k: i32,
    experts: i32,
    slots: i32,
    stride: i32,
}

/// Sends this rank's live `Tokens` (under the armed window) to the ranks
/// holding their routes' experts, into each one's [`Landing`], and counts
/// the rows this rank received per local slot into `counts` (`slots` i32).
/// `landing` is every rank's, in rank order.
pub fn dispatch(
    ctx: &Ctx,
    ep: &Ep,
    place: Placement,
    Tokens { x, routes, weights }: Tokens,
    landing: &[Landing],
    counts: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "collective.dispatch";
    let (world, tokens, top_k) = (ep.world, x.rows, routes.width);
    if !(1..=EP_MAX_RANKS).contains(&world) || ep.rank >= world || landing.len() != world as usize {
        return Err(refuse(
            OP,
            "the group is 1 to 8 ranks, each with its landing",
        ));
    }
    let row = u64::from(x.width) * x.dtype.bytes_ceil();
    if !aligned16(x.ptr) || !row.is_multiple_of(16) || row == 0 {
        return Err(refuse(OP, "a token row is whole 16-byte packs"));
    }
    if routes.dtype != Dtype::I32 || routes.rows != tokens || !(1..=32).contains(&top_k) {
        return Err(refuse(OP, "the routes are i32 [tokens, 1..=32]"));
    }
    if weights.is_some_and(|w| w.dtype != Dtype::F32 || w.rows != tokens || w.width != top_k) {
        return Err(refuse(OP, "the weights are f32 [tokens, top_k]"));
    }
    if place.slots == 0 || place.stride < 2 {
        return Err(refuse(
            OP,
            "a placement holds slots and at least one replica",
        ));
    }
    if counts.dtype != Dtype::I32 || counts.elements() != u64::from(place.slots) {
        return Err(refuse(OP, "the counts are one i32 a local slot"));
    }
    let cap = world
        .checked_mul(tokens)
        .filter(|c| i32::try_from(u64::from(*c) * u64::from(top_k)).is_ok())
        .ok_or_else(|| refuse(OP, "the landing rows past i32"))?;
    for l in landing {
        let fits = l.rows.dtype == x.dtype
            && l.rows.rows == cap
            && l.rows.width == x.width
            && aligned16(l.rows.ptr)
            && l.local.dtype == Dtype::I32
            && l.local.rows == cap
            && l.local.width == top_k
            && l.origin.dtype == Dtype::I32
            && l.origin.elements() == u64::from(cap)
            && l.row_weights.is_some() == weights.is_some()
            && l.row_weights
                .is_none_or(|w| w.dtype == Dtype::F32 && w.rows == cap && w.width == top_k);
        if !fits {
            return Err(refuse(
                OP,
                "a landing is [world * tokens] rows of the token, routes, weights and origin",
            ));
        }
    }

    let pos = ctx.scratch_stream(OP, "dispatch_pos", tokens as usize * world as usize * 4)?;
    let offs = ctx.scratch_stream(OP, "dispatch_offs", EP_MAX_RANKS as usize * 4)?;
    let by_rank = |f: &dyn Fn(&Landing) -> u64| {
        let mut out = [0u64; EP_MAX_RANKS as usize];
        for (o, l) in out.iter_mut().zip(landing) {
            *o = f(l);
        }
        out
    };
    let params = Params {
        x: x.ptr,
        routes: routes.ptr,
        weights: weights.map_or(0, |w| w.ptr),
        win: match ctx.stage() {
            ArgValue::Ptr(p) => p,
            _ => 0,
        },
        place: place.table,
        pos: pos as usize as u64,
        offs: offs as usize as u64,
        counts: counts.ptr,
        rows: by_rank(&|l| l.rows.ptr),
        local: by_rank(&|l| l.local.ptr),
        row_weights: by_rank(&|l| l.row_weights.map_or(0, |w| w.ptr)),
        origin: by_rank(&|l| l.origin.ptr),
        group: ep.signals,
        self_signals: ep.signals[ep.rank as usize],
        rank: ep.rank as i32,
        ranks: world as i32,
        tokens: tokens as i32,
        packs: (row / 16) as i32,
        top_k: top_k as i32,
        experts: place.experts as i32,
        slots: place.slots as i32,
        stride: place.stride as i32,
    };
    let one = Launch::grid([1, 1, 1], [COUNT_THREADS, 1, 1]);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::collective::dispatch_count")).apply(one),
        &[ArgValue::of(&params)],
    )?;
    if tokens > 0 {
        ctx.fire(
            OP,
            Fire::at(kernel(FILE, "::pie::collective::dispatch_send")).apply(Launch::grid(
                [tokens.div_ceil(SEND_WARPS), 1, 1],
                [32 * SEND_WARPS, 1, 1],
            )),
            &[ArgValue::of(&params)],
        )?;
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::collective::dispatch_settle")).apply(one),
        &[ArgValue::of(&params)],
    )
}
