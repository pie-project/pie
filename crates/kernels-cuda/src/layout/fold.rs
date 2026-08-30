//! `Fold`: **THE PATCH AXIS'S ROW FOLDS** — the two ops that read `side²`
//! consecutive patch rows and answer one (`.wiki/alto/multimodal.md` §7.4,
//! §8.3).
//!
//! One file because they are one question asked twice.
//! [`pool_rows`] folds by AVERAGING (gemma4's `pooling_kernel_size: 3`);
//! [`merge_rows`] folds by CONCATENATING (qwen's `spatial_merge_size: 2`).
//! Same rectangle, same axis, same statute, same tail rule — §8.4's own
//! reading, and the reason landing one and deferring the other would have
//! been building the door twice.
//!
//! **"FOLD" NAMES THE FILE AND NEITHER ENTRY**, deliberately: a fold is what
//! both do to `side²` rows, and each entry is named for what it folds them
//! WITH. §8 calls the concatenating one `fold_rows`; it is `merge_rows` here
//! because `merger` and `spatial_merge_size` are the checkpoint's own words
//! and because one word may not mean both the family and a member of it.
//!
//! **THIS FILE IS `src/layout/fold.rs` AND ITS MODULE PATH IS
//! `kernels_cuda::layout_fold`**, for `attn_dense`'s reason and no other: a
//! child module can only be declared by its parent, and the campaign's
//! conflict map closes `src/layout.rs`. The `#[path]` declaration in `lib.rs`
//! says so, and one line inside `layout.rs` retires it.
//!
//! # Why the pool is a reduction and not a GEMM
//!
//! §6.5 reads gemma4's `vision_soft_tokens_per_image` / `pooling_kernel_size`
//! pair as a baked pooling matrix, `[soft_tokens, patches] · [patches, 768]`,
//! and calls it a row-mixing GEMM the vocabulary does not have. The matrix
//! describes the arithmetic correctly and expresses it badly: it materializes
//! an `O(patches²)` constant for an `O(patches)` reduction, it would be a
//! different constant per rung, and its operand has a symbolic dim no weight
//! declaration in this IR carries. `Gemma4VisionPooler._avg_pool_by_positions`
//! builds exactly that one-hot at runtime — `F.one_hot(kernel_idxs, length) /
//! k_squared`, then a matmul — which is a legible way to write it in torch and
//! a wasteful way to run it.
//!
//! What is left, once the ordering statute below is stated, is a mean over
//! `side²` consecutive rows.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/fold.cuh";

const BLOCK: u32 = 256;

/// **THE SPATIAL POOL**: `y[j]` is the mean of rows `[j·side², (j+1)·side²)`
/// of `x`, over `x.rows / side²` output rows.
///
/// # What this asks of the submission, and what it asks of the fire
///
/// Of the submission: **POOL-BLOCK-MAJOR PATCH ORDER** — an image's patches
/// are ordered so that each `side × side` square of its grid is contiguous.
/// That is multimodal §2's merge-block-major statute at `side` instead of 2,
/// and it is what turns a 2-D pool into this one. Of the fire: nothing. No
/// position stream, no image indptr, no per-image grid width — image runs are
/// whole numbers of blocks by the preprocessor's own resize rule
/// (`get_aspect_ratio_preserving_size` rounds an image's height and width DOWN
/// to a multiple of `pooling_kernel_size · patch_size`), so a block never
/// straddles two images and two images pool as one concatenation.
///
/// # The tail
///
/// `x.rows` is a RUNG count, and a patch ladder's rungs are not multiples of
/// nine. So this pools `x.rows / side²` whole blocks and reads nothing past
/// them — floor, not refuse, because the tail it drops is provably rung
/// padding: every image's run is a whole number of blocks and the real rows
/// are a prefix of the rectangle, so the leading `Σ patches_i / side²` output
/// rows are exactly the real ones and everything after is zeros folded with
/// zeros.
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

/// **THE MERGING FOLD**: `y[j]` is rows `[j·side², (j+1)·side²)` of `x` laid
/// end to end — `side²` rows of `width` becoming one row of `side²·width`.
///
/// qwen's spatial merger, and the op §8.1 found the vocabulary missing:
/// `Qwen3_5VisionPatchMerger.forward` opens `x.view(-1, hidden_size ·
/// spatial_merge_size²)`, which is why `merger.linear_fc1.weight` is
/// `[4·hidden, 4·hidden]` on the 768-wide tower and `[4608, 4608]` on the
/// 1152-wide one.
///
/// # It asks exactly what [`pool_rows`] asks
///
/// The same MERGE-BLOCK-MAJOR patch order — which for this op is not a new
/// statute at all but the one multimodal §2 already made and
/// `qwen_patchify_hwc` already honours — the same nothing from the fire, and
/// the same tail rule: `x.rows / side²` rows written, the rest of `y`
/// untouched.
///
/// **AND ON A DENSE RECTANGLE IT IS THE IDENTITY COPY.** `[rows, width]` and
/// `[rows/block, block·width]` are the same bytes in the same order; the
/// launch exists because the IR gives one value one type, so a second type
/// needs a second value and a node to define it. Recorded so that a later
/// wave can retire it into a placement alias in the compiler without anyone
/// wondering whether the arithmetic changed. It did not; there is none.
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
                "{block} rows of {} concatenate into {merged}, and the destination's rows are                  {} wide",
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
/// whole blocks the source has.
///
/// **FLOOR, AND THE TAIL IT DROPS IS RUNG PADDING** — see [`pool_rows`]'s own
/// note. Both entries take it from here so the two can never disagree about
/// where a fold stops.
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
