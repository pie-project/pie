//! DFlash2's two-tap grouped dynamic convolution along a request's block
//! rows — `Attention::BlockDynConv`. One kernel, `attn/block_dyn_conv.metal`.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Tensor};

const FILE: &str = "attn/block_dyn_conv.metal";

/// Threads per threadgroup along the channel axis.
const GROUP: u32 = 256;

/// `y[i] = Σ_t (base[side, t] + δ[i, side, t, g]) ⊙ x[i − t]` within each
/// request's rows of `x`, `x` before the span being zero.
///
/// `coeff` is `[rows, 2·taps·groups]` laid `(side, tap, group)`; `base` is
/// `[2·taps, channels]`; `group` channels share one correction.
///
/// # Errors
///
/// Refuses a dtype the kernel is not stamped for, a `side` past the two the
/// projection carries, a channel count `group` does not divide, and a
/// `coeff` or `base` whose width is not the one those numbers imply.
#[allow(clippy::too_many_arguments)]
pub fn block_dyn_conv(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    coeff: Tensor,
    base: Tensor,
    side: u32,
    taps: u32,
    group: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.block_dyn_conv";
    let entry = dtype_dispatch!(OP, x.data.dtype, { Bf16 => "block_dyn_conv_bfloat16" });
    if coeff.dtype != x.data.dtype || base.dtype != x.data.dtype || y.dtype != x.data.dtype {
        return Err(refuse(
            OP,
            format!(
                "x is {:?} but coeff / base / y are {:?} / {:?} / {:?}; the kernel reads one element type",
                x.data.dtype, coeff.dtype, base.dtype, y.dtype
            ),
        ));
    }
    if x.indptr.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!("the request CSR's boundaries are {:?}, and this walk reads an i32 indptr", x.indptr.dtype),
        ));
    }
    let lanes = match x.indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => lanes,
        _ => return Err(refuse(OP, "the request CSR this fire names spans no request")),
    };
    let channels = nonzero(OP, "the convolution's channel count", x.data.width)?;
    let taps_n = nonzero(OP, "the tap count this statement states", taps)?;
    let group_n = nonzero(OP, "the channels sharing one correction", group)?;
    if side > 1 {
        return Err(refuse(OP, format!("side {side} is stated, and the projection carries two")));
    }
    if channels % group_n != 0 {
        return Err(refuse(
            OP,
            format!("{channels} channels are not a whole number of groups of {group_n}"),
        ));
    }
    let groups = channels / group_n;
    let coeff_width = 2 * taps_n * groups;
    if coeff.width != coeff_width || coeff.rows != x.data.rows {
        return Err(refuse(
            OP,
            format!(
                "the coefficients are [{}, {}], and two sides of {taps_n} taps over {groups} groups \
                 for {} rows are [{}, {coeff_width}]",
                coeff.rows, coeff.width, x.data.rows, x.data.rows
            ),
        ));
    }
    if base.rows != 2 * taps_n || base.width != channels {
        return Err(refuse(
            OP,
            format!(
                "the base kernel is [{}, {}], and two sides of {taps_n} taps over {channels} channels \
                 are [{}, {channels}]",
                base.rows, base.width, 2 * taps_n
            ),
        ));
    }
    debug_assert!(
        y.rows == x.data.rows && y.width == x.data.width,
        "the convolution lands the rows it convolves"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            [channels, lanes, 1],
            [channels.min(GROUP), 1, 1],
        )),
        &[
            x.data.arg(),
            x.indptr.arg(),
            coeff.arg(),
            base.arg(),
            y.arg_mut(),
            stated(OP, channels)?.arg(),
            stated(OP, side)?.arg(),
            stated(OP, taps_n)?.arg(),
            stated(OP, group_n)?.arg(),
        ],
    )
}
