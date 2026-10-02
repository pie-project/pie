use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/nvfp4.cuh";

const WARP: u32 = 32;

const GROUP: u32 = 16;

const ROWS_PER_WARP: u32 = 4;
const BLOCK_LANES: u32 = 128;

pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", act, codes, scales, tensor_scale, y)
}

pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", act, codes, scales, tensor_scale, y)
}

fn fire(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
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
    if k % GROUP != 0 {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {GROUP}-code nvfp4 groups"),
        ));
    }
    if codes.width != k / 2 {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row does not store a {k}-wide row of e2m1 nibbles",
                codes.width
            ),
        ));
    }
    if scales.width != k / GROUP {
        return Err(refuse(
            op,
            format!(
                "a {}-byte scale row is not one e4m3 per {GROUP} codes over a {k}-wide row",
                scales.width
            ),
        ));
    }
    if scales.rows != n {
        return Err(refuse(
            op,
            format!(
                "a {}-row scale plane is not {n} rows over a {n}x{k} weight",
                scales.rows
            ),
        ));
    }
    if !tensor_scale.is_finite() {
        return Err(refuse(
            op,
            format!("the tensor scale is {tensor_scale}, which every output would carry"),
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
                "::pie::linear::matmul_nvfp4<{t}, ::pie::i32({ROWS_PER_WARP})>"
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
            tensor_scale.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ctx.stage(),
        ],
    )
}
