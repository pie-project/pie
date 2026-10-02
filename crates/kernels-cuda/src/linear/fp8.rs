use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/fp8.cuh";

const WARP: u32 = 32;

const TILE: u32 = 128;

const ROWS_PER_WARP: u32 = 4;
const BLOCK_LANES: u32 = 128;

pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", Form::Row, act, codes, scales, y)
}

pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", Form::Row, act, codes, scales, y)
}

pub fn matmul_tile(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", Form::Tile, act, codes, scales, y)
}

pub fn lm_head_tile(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", Form::Tile, act, codes, scales, y)
}

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

    const fn scale_row(self, k: u32) -> u32 {
        match self {
            Self::Row => 4,
            Self::Tile => 4 * k.div_ceil(TILE),
        }
    }

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
    if codes.width != k {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row does not store a {k}-wide row of e4m3 bytes",
                codes.width
            ),
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
            ctx.stage(),
        ],
    )
}
