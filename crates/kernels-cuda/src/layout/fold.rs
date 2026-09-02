//! `Fold`: the patch axis's row folds — two ops that read `side²` consecutive
//! patch rows and answer one. [`pool_rows`] folds by averaging (gemma4's
//! `pooling_kernel_size`); [`merge_rows`] folds by concatenating (qwen's
//! `spatial_merge_size`). Same rectangle, same axis, same tail rule.
//!
//! The pool is a reduction and not a GEMM: a baked pooling matrix would
//! materialize an `O(patches²)` constant for an `O(patches)` reduction, once
//! the ordering statute below is stated it is a mean over `side²` consecutive
//! rows.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/fold.cuh";

const BLOCK: u32 = 256;

/// The spatial pool: `y[j]` is the mean of rows `[j·side², (j+1)·side²)` of
/// `x`, over `x.rows / side²` output rows.
///
/// Requires pool-block-major patch order: each `side × side` square of an
/// image's grid must be contiguous. No position stream or image indptr
/// needed — image runs are whole numbers of blocks by the preprocessor's
/// resize rule, so a block never straddles two images.
///
/// # The tail
///
/// Floors `x.rows / side²` whole blocks and reads nothing past them (not a
/// refusal): the dropped tail is provably rung padding.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// zero `side`, a zero-wide row, a destination whose rows are a different
/// width, a rectangle with fewer rows than one block (which would launch
/// nothing and leave `y` unwritten), and a destination too short to hold the
/// blocks the source has.
pub fn pool_rows(ctx: &Ctx, x: Tensor, side: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "layout.pool_rows";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` pools into the element it reads");

    let (block, out) = fold_extent(OP, x, side)?;
    let width = stated(OP, x.width)?;
    if y.width != x.width {
        return Err(refuse(
            OP,
            format!(
                "the source rows are {} wide and the destination's are {}; a pool folds rows \
                 and never a row",
                x.width, y.width
            ),
        ));
    }
    if y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{} source rows fold into {out} pooled rows and the destination holds {}",
                x.rows, y.rows
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::layout::pool_rows<{t}>")))
            .apply(Launch::per_row(out, BLOCK)),
        &[x.arg(), y.arg(), width.arg(), stated(OP, block)?.arg()],
    )
}

/// The merging fold: `y[j]` is rows `[j·side², (j+1)·side²)` of `x` laid end
/// to end — `side²` rows of `width` becoming one row of `side²·width`.
/// qwen's spatial merger (`merger.linear_fc1.weight` is `[4608, 4608]` on the
/// 1152-wide tower).
///
/// Asks exactly what [`pool_rows`] asks: same merge-block-major patch order,
/// nothing from the fire, same tail rule.
///
/// On a dense rectangle this is the identity copy — same bytes, same order —
/// but the launch exists because the IR gives one value one type.
///
/// # Errors
///
/// As [`pool_rows`], plus a refusal for a destination whose width is not
/// `side²` times the source's — which is the one shape mistake this op can
/// make that the pool cannot.
pub fn merge_rows(ctx: &Ctx, x: Tensor, side: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "layout.merge_rows";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` merges into the element it reads");

    let (block, out) = fold_extent(OP, x, side)?;
    let merged = block.checked_mul(x.width).ok_or_else(|| {
        refuse(
            OP,
            format!(
                "{block} rows of {} do not concatenate into a row that fits a u32",
                x.width
            ),
        )
    })?;
    if y.width != merged {
        return Err(refuse(
            OP,
            format!(
                "{block} rows of {} concatenate into {merged}, and the destination's rows are {} \
                 wide",
                x.width, y.width
            ),
        ));
    }
    if y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{} source rows fold into {out} merged rows and the destination holds {}",
                x.rows, y.rows
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::layout::merge_rows<{t}>")))
            .apply(Launch::per_row(out, BLOCK)),
        &[
            x.arg(),
            y.arg(),
            stated(OP, x.width)?.arg(),
            stated(OP, block)?.arg(),
        ],
    )
}

/// The two folds' shared arithmetic: how many rows one block is, and how many
/// whole blocks the source has. Floors; see [`pool_rows`]'s tail note.
fn fold_extent(op: &'static str, x: Tensor, side: u32) -> Result<(u32, u32), Error> {
    nonzero(op, "the folding square's side", side)?;
    nonzero(op, "the folded row's width", x.width)?;
    let block = side.checked_mul(side).ok_or_else(|| {
        refuse(
            op,
            format!("a {side}-wide folding square has no row count that fits a u32"),
        )
    })?;
    if x.rows < block {
        return Err(refuse(
            op,
            format!(
                "{} rows do not fill one {side}x{side} fold, and a fold with no whole block \
                 would leave the destination unwritten",
                x.rows
            ),
        ));
    }
    Ok((block, x.rows / block))
}
