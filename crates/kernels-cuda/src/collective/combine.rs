use super::dispatch::{EP_MAX_RANKS, Ep, Landing};
use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, kernel, refuse};
use crate::tensor::Tensor;

const FILE: &str = "collective/combine.cu";

pub(super) const KERNELS: [(&str, &str); 2] = [
    (FILE, "::pie::collective::combine_send"),
    (FILE, "::pie::collective::combine_settle"),
];

/// `kSendWarps`.
const SEND_WARPS: u32 = 8;

/// A grid that strides over the landing rather than covering it.
const MAX_BLOCKS: u32 = 1024;

/// `pie::collective::CombineParams`, field for field.
#[repr(C)]
struct Params {
    y: u64,
    origin: u64,
    local: u64,
    routed: [u64; EP_MAX_RANKS as usize],
    group: [u64; EP_MAX_RANKS as usize],
    self_signals: u64,
    rank: i32,
    ranks: i32,
    tokens: i32,
    packs: i32,
    top_k: i32,
}

/// Sends the expert outputs `y` (`[world * T * top_k, W]`, a row per
/// landed route) back to the ranks and tokens `landing` (this rank's, from
/// the dispatch) says they came from: route j of token t lands at row
/// `t * top_k + j` of its rank's `routed`. `routed` is every rank's
/// `[T * top_k, W]`, in rank order.
pub fn combine(
    ctx: &Ctx,
    ep: &Ep,
    landing: &Landing,
    y: Tensor,
    routed: &[Tensor],
) -> Result<(), Error> {
    const OP: &str = "collective.combine";
    let world = ep.world;
    if !(1..=EP_MAX_RANKS).contains(&world) || ep.rank >= world || routed.len() != world as usize {
        return Err(refuse(
            OP,
            "the group is 1 to 8 ranks, each with its routed rows",
        ));
    }
    let (cap, top_k) = (landing.local.rows, landing.local.width);
    if !cap.is_multiple_of(world) {
        return Err(refuse(OP, "a landing is world * tokens rows"));
    }
    let tokens = cap / world;
    let row = u64::from(y.width) * y.dtype.bytes_ceil();
    if !aligned16(y.ptr) || !row.is_multiple_of(16) || row == 0 {
        return Err(refuse(OP, "an output row is whole 16-byte packs"));
    }
    if y.rows != cap * top_k {
        return Err(refuse(OP, "the outputs are a row per landed route"));
    }
    let fits = |r: &Tensor| {
        r.dtype == y.dtype && r.width == y.width && r.rows == tokens * top_k && aligned16(r.ptr)
    };
    if !routed.iter().all(fits) {
        return Err(refuse(
            OP,
            "every rank's routed rows are [tokens * top_k, W] of the outputs",
        ));
    }
    let mut each = [0u64; EP_MAX_RANKS as usize];
    for (o, r) in each.iter_mut().zip(routed) {
        *o = r.ptr;
    }
    let params = Params {
        y: y.ptr,
        origin: landing.origin.ptr,
        local: landing.local.ptr,
        routed: each,
        group: ep.signals,
        self_signals: ep.signals[ep.rank as usize],
        rank: ep.rank as i32,
        ranks: world as i32,
        tokens: tokens as i32,
        packs: (row / 16) as i32,
        top_k: top_k as i32,
    };
    let blocks = cap.div_ceil(SEND_WARPS).clamp(1, MAX_BLOCKS);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, KERNELS[0].1))
            .apply(Launch::grid([blocks, 1, 1], [32 * SEND_WARPS, 1, 1])),
        &[ArgValue::of(&params)],
    )?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, KERNELS[1].1)).apply(Launch::grid([1, 1, 1], [32, 1, 1])),
        &[ArgValue::of(&params)],
    )
}
