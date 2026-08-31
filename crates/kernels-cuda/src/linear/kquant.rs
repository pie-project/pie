//! GGUF K-quant gemm points: `q4_k`/`q6_k` super-blocks read as stored,
//! dequantized inside the dot. The mandatory pair — Q4_K_M mixes are the
//! most-distributed quant artifacts there are, and their `output.weight`
//! is q6_k. Filled by the QNF wave (wiki alto/next.md §J2, priority 3).
//!
//! **THE WEIGHT ARRIVES AS ONE BYTE PLANE, AND THE ROW'S BYTE WIDTH NAMES
//! THE SCHEME.** A K-quant carries its scales INSIDE the super-block, so
//! there is no second plane to bind and no dtype to dispatch on: a `[n, k]`
//! weight is `n` rows of `k / 256` consecutive super-blocks, and a
//! super-block is 144 bytes at `q4_k` and 210 at `q6_k`. The two products
//! `144·k/256` and `210·k/256` are distinct for every legal `k`, so the
//! discrimination is total — [`scheme`] answers one of the two or refuses,
//! and no width is ever ambiguous between them.
//!
//! **A PLANE-FORM VARIANT MAY SUPERSEDE THESE ENTRY SHAPES.** Serving AS
//! STORED is the ruling these entries answer (§J); the canonical `.zt`
//! container is leaf-per-plane and k-group-major, so a later import wave may
//! re-seat these weights as separate code and scale planes and want entries
//! that take them apart. The decode arithmetic in `linear/kquant.cuh` is the
//! format's and would not move — only the addressing would.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/kquant.cuh";

const WARP: u32 = 32;

/// Elements in one K-quant super-block, both schemes.
const SUPER: u32 = 256;

/// `block_q4_K`: `d`, `dmin`, twelve packed scale/min bytes, 128 of nibbles.
const Q4K_BYTES: u32 = 144;

/// `block_q6_K`: 128 low nibbles, 64 high pairs, sixteen i8 scales, `d`.
const Q6K_BYTES: u32 = 210;

/// Weight rows one warp folds at a time, and the lanes a block carries —
/// `quant.rs`' dense affine geometry, kept so the two decode-in-dot points
/// have one grid shape between them.
const ROWS_PER_WARP: u32 = 4;
const BLOCK_LANES: u32 = 128;

/// Which K-quant a weight row's byte width names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scheme {
    Q4K,
    Q6K,
}

impl Scheme {
    /// The kernel this scheme's super-block decode lives in.
    const fn point(self) -> &'static str {
        match self {
            Self::Q4K => "matmul_q4k",
            Self::Q6K => "matmul_q6k",
        }
    }
}

/// **HOW A STORED K-QUANT ROW SAYS WHAT IT IS.** `k` fixes the super-block
/// count; the row's byte width then divides out to 144 or 210 and to nothing
/// else. A row that is neither is refused rather than read at a guess — the
/// two schemes disagree about every byte after the first, so a misread would
/// decode to plausible garbage instead of failing.
fn scheme(op: &'static str, k: u32, row_bytes: u32) -> Result<Scheme, Error> {
    let blocks = k / SUPER;
    if row_bytes == blocks * Q4K_BYTES {
        return Ok(Scheme::Q4K);
    }
    if row_bytes == blocks * Q6K_BYTES {
        return Ok(Scheme::Q6K);
    }
    Err(refuse(
        op,
        format!(
            "a {row_bytes}-byte weight row is neither {} bytes of q4_k nor {} of q6_k \
             over a {k}-wide contraction ({blocks} super-blocks)",
            blocks * Q4K_BYTES,
            blocks * Q6K_BYTES
        ),
    ))
}

/// `linear.matmul` over a weight the store seats as GGUF K-quant
/// super-blocks, read as stored. The scheme is the row's own, see [`scheme`].
pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    kquant(ctx, "linear.matmul", act, w, y)
}

/// [`matmul`] under the head's own op name. Not a courtesy pairing here: a
/// Q4_K_M mix stores `output.weight` at q6_k, so the head is the busiest
/// consumer this file has.
pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    kquant(ctx, "linear.lm_head", act, w, y)
}

/// The one launch behind both entries. A fire with no rows is the same
/// silent no-op the dense gemm keeps, and for the same capture reason.
fn kquant(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(w.dtype, Dtype::U8, "a stored K-quant plane binds as bytes");
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    debug_assert_eq!(w.rows, n, "one weight row per column this projection lands");
    if !k.is_multiple_of(SUPER) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {SUPER}-element K-quant super-blocks"),
        ));
    }
    let scheme = scheme(op, k, w.width)?;
    if y.rows == 0 {
        return Ok(());
    }
    let tile = (BLOCK_LANES / WARP) * ROWS_PER_WARP;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::{}<{t}, ::pie::i32({ROWS_PER_WARP})>",
                scheme.point()
            )),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [BLOCK_LANES, 1, 1],
        )),
        &[
            act.arg(),
            w.arg(),
            y.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
