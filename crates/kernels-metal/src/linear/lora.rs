//! `Lora`: the correction class. One entry, [`correct`], for `linear.lora_correct` — `y += B[a]*(A[a]*x)` over the rows `routes` gives an adapter.
//!
//! One launch, unlike CUDA's two: the `rows x rank` waist lives in threadgroup memory inside one threadgroup's own two halves, since a kernel entry here has no scratch slab to name.
//! `alpha/r` is folded into the up bank's contents at registration, so this entry states shapes and nothing else. Rank diversity is bucketed by bank: an adapter shorter than its bank's rank was registered zero-padded.
//! No segments: `engine_metal::window` serves `Fallback::Split`, so every row of the rectangle handed here is a row of the correction.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

/// One threadgroup per token row: eight simdgroups over the rank rows of the
/// projection, then 256 lanes striding the output columns of the accumulate.
const GROUP: u32 = 256;

/// The threadgroup array `linear/lora.metal` stages the waist in, stated here
/// because a rank past it is a refusal and not a silent truncation.
const MAX_RANK: u32 = 128;

/// `y += B[a]·(A[a]·x)` over the rows `routes` gives an adapter.
///
/// `x` is `[rows, in]`, `y` is `[rows, out]`, `bank_a` is `[adapters, rank*in]`, `bank_b` is `[adapters, out*rank]`. The rank is not an argument: it is `bank_a.width / x.width`, checked against `bank_b`.
/// `routes` is `[rows, 1]` i32; `-1` is the base model, whose row is skipped and `y` keeps the trunk's value. In place: `y` is read, added to, and written back.
///
/// Errs [`Error::Backend`] for banks whose widths don't divide into a common rank, mismatched adapter counts, a rank past the shader's threadgroup array, or a degenerate extent; [`Error::DtypeUnsupported`] otherwise.
pub fn correct(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank_a: Tensor,
    bank_b: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.lora_correct";

    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "lora_correct" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 adapter ids");
    debug_assert_eq!(
        bank_a.dtype, x.dtype,
        "the adapter bank rides the activation's dtype"
    );
    debug_assert_eq!(
        bank_b.dtype, x.dtype,
        "the adapter bank rides the activation's dtype"
    );
    debug_assert_eq!(y.rows, x.rows, "a correction lands one row per input row");
    debug_assert_eq!(routes.rows, x.rows, "one adapter id per token row");
    debug_assert_eq!(routes.width, 1, "a correction routes one adapter per row");

    let rows = nonzero(OP, "rows", x.rows)?;
    let in_width = nonzero(OP, "the correction's input width", x.width)?;
    let out_width = nonzero(OP, "the correction's output width", y.width)?;
    if bank_a.width % in_width != 0 {
        return Err(refuse(
            OP,
            format!(
                "the down bank is {} wide over an input of {in_width}, which is not a \
                 whole number of ranks",
                bank_a.width
            ),
        ));
    }
    let rank = nonzero(OP, "the adapter bank's rank", bank_a.width / in_width)?;
    if bank_b.width != out_width.saturating_mul(rank) {
        return Err(refuse(
            OP,
            format!(
                "the up bank is {} wide where {out_width} x {rank} is {}; the two \
                 planes of one bank state two ranks",
                bank_b.width,
                out_width.saturating_mul(rank),
            ),
        ));
    }
    if bank_a.rows != bank_b.rows {
        return Err(refuse(
            OP,
            format!(
                "the bank's two planes seat {} and {} adapters",
                bank_a.rows, bank_b.rows
            ),
        ));
    }
    if rank > MAX_RANK {
        return Err(refuse(
            OP,
            format!(
                "the bank's rank is {rank}, above the {MAX_RANK} `linear/lora.metal` \
                 stages the waist in"
            ),
        ));
    }

    ctx.fire(
        Fire::at("linear/lora.metal", entry).apply(Grid::of([GROUP, rows, 1], [GROUP, 1, 1])),
        &[
            x.arg(),
            bank_a.arg(),
            bank_b.arg(),
            routes.arg(),
            y.arg_mut(),
            stated(OP, in_width)?.arg(),
            stated(OP, out_width)?.arg(),
            stated(OP, rank)?.arg(),
        ],
    )
}
