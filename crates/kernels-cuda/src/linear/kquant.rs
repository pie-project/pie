//! GGUF K-quant gemm points: the whole K family — `q2_k`, `q3_k`, `q4_k`,
//! `q5_k`, `q6_k` — read as stored, dequantized inside the dot. `q4_k` and
//! `q6_k` were the mandatory pair (Q4_K_M mixes are the most-distributed
//! quant artifacts there are, and their `output.weight` is q6_k); `q3_k`
//! has a real user in this tree's own import history, `q5_k` is a
//! top-distributed mix in Q5_K_M, and `q2_k` is the family's floor. Filled
//! by the QNF wave (wiki alto/next.md §J2, priority 3), which asked for the
//! pair first and these three after.
//!
//! **THE WEIGHT ARRIVES AS ONE BYTE PLANE, AND THE ROW'S BYTE WIDTH NAMES
//! THE SCHEME.** A K-quant carries its scales INSIDE the super-block, so
//! there is no second plane to bind and no dtype to dispatch on: a `[n, k]`
//! weight is `n` rows of `k / 256` consecutive super-blocks, and a
//! super-block is 84 bytes at `q2_k`, 110 at `q3_k`, 144 at `q4_k`, 176 at
//! `q5_k` and 210 at `q6_k`.
//!
//! **THE DISCRIMINATION IS TOTAL, AND THE ARGUMENT IS ONE LINE.** `k` is
//! checked to be a whole number of super-blocks BEFORE any width is looked
//! at, so `blocks = k / 256` is a fixed positive integer and the five
//! candidate row widths are `blocks·{84, 110, 144, 176, 210}` — one common
//! positive factor against five pairwise-distinct widths. A common positive
//! factor preserves distinctness, so the five candidates are pairwise
//! distinct for every `blocks ≥ 1`, which is every legal `k`. [`scheme`]
//! answers one of the five or refuses by name, and no width is ever
//! ambiguous between two schemes.
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

/// Elements in one K-quant super-block, all five schemes.
const SUPER: u32 = 256;

/// `block_q2_K`: sixteen packed scale/min bytes, 64 of 2-bit codes, then
/// `d` and `dmin` — the super-scales after the payload, not before it.
const Q2K_BYTES: u32 = 84;

/// `block_q3_K`: 32 bytes of third-bit mask, 64 of 2-bit codes, twelve
/// packed six-bit scale bytes, `d`. Symmetric, so no `dmin`.
const Q3K_BYTES: u32 = 110;

/// `block_q4_K`: `d`, `dmin`, twelve packed scale/min bytes, 128 of nibbles.
const Q4K_BYTES: u32 = 144;

/// `block_q5_K`: `q4_k`'s head, 32 bytes of fifth-bit plane, 128 of nibbles.
const Q5K_BYTES: u32 = 176;

/// `block_q6_K`: 128 low nibbles, 64 high pairs, sixteen i8 scales, `d`.
const Q6K_BYTES: u32 = 210;

/// Weight rows one warp folds at a time, and the lanes a block carries —
/// `quant.rs`' dense affine geometry, kept so the five decode-in-dot points
/// have one grid shape between them.
const ROWS_PER_WARP: u32 = 4;
const BLOCK_LANES: u32 = 128;

/// Which K-quant a weight row's byte width names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scheme {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Scheme {
    /// The kernel this scheme's super-block decode lives in. One per scheme:
    /// the five agree on the super-block size and on nothing else, so there
    /// is no shared body to select an arm inside.
    const fn point(self) -> &'static str {
        match self {
            Self::Q2K => "matmul_q2k",
            Self::Q3K => "matmul_q3k",
            Self::Q4K => "matmul_q4k",
            Self::Q5K => "matmul_q5k",
            Self::Q6K => "matmul_q6k",
        }
    }

    /// How a refusal spells this scheme — GGUF's own name for it.
    const fn name(self) -> &'static str {
        match self {
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
        }
    }
}

/// The family, ascending by super-block width — the ladder [`scheme`] walks
/// and the order a refusal names them in.
const FAMILY: [(u32, Scheme); 5] = [
    (Q2K_BYTES, Scheme::Q2K),
    (Q3K_BYTES, Scheme::Q3K),
    (Q4K_BYTES, Scheme::Q4K),
    (Q5K_BYTES, Scheme::Q5K),
    (Q6K_BYTES, Scheme::Q6K),
];

/// **HOW A STORED K-QUANT ROW SAYS WHAT IT IS.** `k` fixes the super-block
/// count; the row's byte width then divides out to exactly one of [`FAMILY`]
/// and to nothing else (the module doc carries the argument). A row that
/// matches none of the five is refused rather than read at a guess — the
/// schemes disagree about every byte after the first, so a misread would
/// decode to plausible garbage instead of failing, and the refusal names all
/// five widths so the caller can see which conversion its artifact wants.
fn scheme(op: &'static str, k: u32, row_bytes: u32) -> Result<Scheme, Error> {
    let blocks = k / SUPER;
    for (width, scheme) in FAMILY {
        if row_bytes == blocks * width {
            return Ok(scheme);
        }
    }
    let mut ladder = String::new();
    for (at, (width, scheme)) in FAMILY.iter().enumerate() {
        if at > 0 {
            ladder.push_str(", ");
        }
        ladder.push_str(&format!("{} ({})", blocks * width, scheme.name()));
    }
    Err(refuse(
        op,
        format!(
            "a {row_bytes}-byte weight row is none of the five K-quant widths over a \
             {k}-wide contraction ({blocks} super-blocks): {ladder}"
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
