use crate::encode::{Arg, Ctx, Fire, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "quant/nvfp4.wgsl";

const POINT: &str = "nvfp4_qmv_bf16";

const GROUP_CODES: u32 = 16;

const ROWS_PER_GROUP: u32 = 4;
const GROUP: [u32; 3] = [32, 2, 1];

pub fn matmul(
    ctx: &Ctx<'_>,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", act, codes, scales, tensor_scale, y)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", act, codes, scales, tensor_scale, y)
}

fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    let (m, n, k) = (y.rows, y.width, act.width);
    if !k.is_multiple_of(GROUP_CODES) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {GROUP_CODES}-code nvfp4 groups"),
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
    if scales.width != k / GROUP_CODES {
        return Err(refuse(
            op,
            format!(
                "a {}-byte scale row is not one e4m3 per {GROUP_CODES} codes over a \
                 {k}-wide row",
                scales.width
            ),
        ));
    }
    if codes.rows != n || scales.rows != n {
        return Err(refuse(
            op,
            format!(
                "the planes have {} code rows and {} scale rows over an {n}-column \
                 projection; both are one row per column",
                codes.rows, scales.rows
            ),
        ));
    }
    if !n.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "this projection lands {n} columns; the point writes bf16 pairs and \
                 an odd column count would tear the last word"
            ),
        ));
    }

    if !tensor_scale.is_finite() {
        return Err(refuse(
            op,
            format!("the tensor scale is {tensor_scale}, which every output would carry"),
        ));
    }
    if m == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, POINT)
            .groups([n.div_ceil(ROWS_PER_GROUP), m, 1])
            .group(GROUP),
        &[
            codes.arg(),
            scales.arg(),
            act.arg(),
            y.arg_mut(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            stated(op, m)?.arg(),
            tensor_scale.arg(),
        ],
    )
}
