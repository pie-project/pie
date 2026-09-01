//! The gather that concatenates (`layout.embed_concat`, qwen4's PLE n-gram
//! read): `heads` table rows per token, side by side, `heads` read off the
//! id rectangle's width.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/embed_concat.cuh";

const BLOCK: u32 = 256;

pub fn embed_concat(
    ctx: &Ctx,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 rows");
    let t = dtype_dispatch!(OP, table.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = nonzero(OP, "the hashed head count", ids.width)?;
    let width = nonzero(OP, "the table row width", table.width)?;
    debug_assert_eq!(
        y.width,
        heads * width,
        "the output row is every head's table row, side by side"
    );
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    let rows = nonzero(OP, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(heads) * u64::from(width);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::layout::embed_concat<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            table.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, vocab)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// The gather over an AFFINE-LANDED table: MLX 4-bit codes under bf16
/// scales and zero points, dequantized for exactly the rows a token touches.
/// The group width is recovered from the factor plane's own rectangle — the
/// n-gram table groups by thirty-two where the rest of the tree groups by
/// sixty-four, and the plane already says so.
#[allow(clippy::too_many_arguments)]
pub fn embed_concat_mlxu4(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    vocab: u32,
    seat: crate::linear::moe::GroupSeat,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 rows");
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let Some(biases) = biases else {
        return Err(refuse(
            OP,
            "an mxfp4 table has no gather point on this plane; the affine \
             triplet is the packed landing this gather reads",
        ));
    };
    let heads = nonzero(OP, "the hashed head count", ids.width)?;
    if y.width == 0 || !y.width.is_multiple_of(heads) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide output is not a whole number of {heads} head slices",
                y.width
            ),
        ));
    }
    let width = y.width / heads;
    // The factors plane is `[vocab, width / group]` bf16, bound as its BYTE
    // rectangle: `bytes / rows / 2` factors a row, and the group divides the
    // row width by construction.
    let factor_rows = nonzero(OP, "the factor plane's rows", scales.rows)?;
    let per_row = scales.width / 2;
    if factor_rows != vocab || per_row == 0 || !width.is_multiple_of(per_row) {
        return Err(refuse(
            OP,
            format!(
                "a [{factor_rows} x {per_row}] factor plane does not group a \
                 [{vocab} x {width}] table"
            ),
        ));
    }
    let group = width / per_row;
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    let rows = nonzero(OP, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(heads) * u64::from(width);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::layout::embed_concat_mlxu4<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, group)?.arg(),
            stated(OP, vocab)?.arg(),
            crate::jit::ArgValue::Ptr(seat.cell),
            crate::jit::ArgValue::Ptr(seat.hits),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// The PLAIN embedding read (`layout.embed`) over an affine-landed table —
/// the concatenating gather at one head, which is what a token embedding
/// is. The bit width is the codes plane's own rectangle: a `[vocab, width]`
/// table stores `width` bytes a row at eight bits and `width / 2` at four;
/// the group is the factor plane's, exactly as the concat entry reads it.
#[allow(clippy::too_many_arguments)]
pub fn embed_mlx_affine(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    vocab: u32,
    seat: crate::linear::moe::GroupSeat,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let Some(biases) = biases else {
        return Err(refuse(
            OP,
            "an mxfp4 table has no gather point on this plane; the affine \
             triplet is the packed landing this gather reads",
        ));
    };
    let width = nonzero(OP, "the embedded row's width", y.width)?;
    let entry = if codes.width == width {
        "embed_concat_mlxu8"
    } else if codes.width * 2 == width {
        "embed_concat_mlxu4"
    } else {
        return Err(refuse(
            OP,
            format!(
                "a {}-byte code row stores a {width}-wide row at neither four \
                 nor eight bits",
                codes.width
            ),
        ));
    };
    let factor_rows = nonzero(OP, "the factor plane's rows", scales.rows)?;
    let per_row = scales.width / 2;
    if factor_rows != vocab || per_row == 0 || !width.is_multiple_of(per_row) {
        return Err(refuse(
            OP,
            format!(
                "a [{factor_rows} x {per_row}] factor plane does not group a \
                 [{vocab} x {width}] table"
            ),
        ));
    }
    let group = width / per_row;
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    let rows = nonzero(OP, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(width);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::layout::{entry}<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, 1u32)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, group)?.arg(),
            stated(OP, vocab)?.arg(),
            crate::jit::ArgValue::Ptr(seat.cell),
            crate::jit::ArgValue::Ptr(seat.hits),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
