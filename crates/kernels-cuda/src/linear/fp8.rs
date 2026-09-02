//! FP8 weight-only gemm points: `e4m3` codes against bf16/f16 activations,
//! dequantized inside the dot — for the rows cuBLAS cannot serve, since its
//! fp8 lanes want both operands narrow.
//!
//! Two stored forms, four entries. `matmul`/`lm_head` read `gr_e4m3_f32_n`:
//! one f32 scale per output row. `matmul_tile`/`lm_head_tile` read
//! `g128x128_e4m3_f32_n`: a `[ceil(n/128), ceil(k/128)]` f32 rectangle, one
//! factor per 128-row band per 128-wide contraction tile.
//!
//! The form is the entry, not the rectangle: at `n <= 128` and `k <= 128`
//! the tile plane and the row plane are shape-identical bytes with different
//! dots, so discriminating by shape is not decidable. Each entry refuses a
//! plane that is not its own form's rectangle.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/fp8.cuh";

const WARP: u32 = 32;

/// The scale rectangle's tile edge, both axes — the stored form's own name
/// carries it (`g128x128_...`).
const TILE: u32 = 128;

/// Weight rows a warp folds at once, and the lanes a block spends on them:
/// the dense affine point's geometry, kept so the two stored forms tile the
/// same way.
const ROWS_PER_WARP: u32 = 4;
const BLOCK_LANES: u32 = 128;

/// `linear.matmul` over a weight the store seats as `gr_e4m3_f32_n` — e4m3
/// codes at `k` bytes a row, one f32 scale per output row.
pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", Form::Row, act, codes, scales, y)
}

/// [`matmul`] under the head's own op name, `linear::gemm`'s pairing kept.
pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", Form::Row, act, codes, scales, y)
}

/// `linear.matmul` over `g128x128_e4m3_f32_n` — the same codes under a
/// `[ceil(n/128), ceil(k/128)]` f32 scale rectangle.
pub fn matmul_tile(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", Form::Tile, act, codes, scales, y)
}

/// [`matmul_tile`] under the head's own op name.
pub fn lm_head_tile(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", Form::Tile, act, codes, scales, y)
}

/// Which stored form the caller declared by the entry it called; the scales
/// plane's rectangle is checked against it, never inferred from it.
#[derive(Clone, Copy)]
enum Form {
    Row,
    Tile,
}

impl Form {
    const fn point(self) -> &'static str {
        match self {
            Self::Row => "matmul_fp8_row",
            Self::Tile => "matmul_fp8_tile",
        }
    }

    /// The scale row this form states for an `n x k` weight, in bytes: one
    /// f32 per output row, or one per 128-wide contraction tile.
    const fn scale_row(self, k: u32) -> u32 {
        match self {
            Self::Row => 4,
            Self::Tile => 4 * k.div_ceil(TILE),
        }
    }

    /// How many scale rows the plane holds: one per output row, or one per
    /// 128-row band.
    const fn scale_rows(self, n: u32) -> u32 {
        match self {
            Self::Row => n,
            Self::Tile => n.div_ceil(TILE),
        }
    }

    const fn spelling(self) -> &'static str {
        match self {
            Self::Row => "gr_e4m3_f32_n",
            Self::Tile => "g128x128_e4m3_f32_n",
        }
    }
}

/// The one launch behind all four entries. A fire with no rows is the same
/// silent no-op the dense gemm keeps, and for the same capture reason.
fn fire(
    ctx: &Ctx,
    op: &'static str,
    form: Form,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    // An e4m3 code is one byte, so the code plane's row is the contraction.
    if codes.width != k {
        return Err(refuse(
            op,
            format!("a {}-byte code row does not store a {k}-wide row of e4m3 bytes", codes.width),
        ));
    }
    let want = form.scale_row(k);
    if scales.width != want {
        return Err(refuse(
            op,
            format!(
                "a {}-byte scale row is not {}'s {want}-byte row over a {n}x{k} weight",
                scales.width,
                form.spelling()
            ),
        ));
    }
    let rows = form.scale_rows(n);
    if scales.rows != rows {
        return Err(refuse(
            op,
            format!(
                "a {}-row scale plane is not {}'s {rows} rows over a {n}x{k} weight",
                scales.rows,
                form.spelling()
            ),
        ));
    }
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
                form.point()
            )),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [BLOCK_LANES, 1, 1],
        )),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            y.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            // Live-rows word when a body replay armed one, else `ABSENT`.
            ctx.stage(),
        ],
    )
}
