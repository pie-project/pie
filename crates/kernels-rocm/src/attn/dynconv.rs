use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::{RaggedTensor, Tensor};

const FILE: &str = "attn/block_dyn_conv.cuh";

const BLOCK: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn block_dyn_conv(
    ctx: &Ctx,
    x: RaggedTensor,
    coeff: Tensor,
    base: Tensor,
    side: u32,
    taps: u32,
    group: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.block_dyn_conv";
    let t = dtype_dispatch!(OP, x.data.dtype, { Bf16 => "::pie::bf16" });
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
            format!(
                "the request CSR's boundaries are {:?}, and this walk reads an i32 indptr",
                x.indptr.dtype
            ),
        ));
    }
    let lanes = match x.indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => lanes,
        _ => {
            return Err(refuse(
                OP,
                "the request CSR this fire names spans no request",
            ));
        }
    };
    let channels = nonzero(OP, "the convolution's channel count", x.data.width)?;
    let taps_n = nonzero(OP, "the tap count this statement states", taps)?;
    let group_n = nonzero(OP, "the channels sharing one correction", group)?;
    if side > 1 {
        return Err(refuse(
            OP,
            format!("side {side} is stated, and the projection carries two"),
        ));
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
                base.rows,
                base.width,
                2 * taps_n
            ),
        ));
    }
    debug_assert!(
        y.rows == x.data.rows && y.width == x.data.width,
        "the convolution lands the rows it convolves"
    );
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::attn::block_dyn_conv<{t}>"))).apply(Launch::grid(
            [channels.div_ceil(BLOCK), lanes, 1],
            [channels.min(BLOCK), 1, 1],
        )),
        &[
            x.data.arg(),
            x.indptr.arg(),
            coeff.arg(),
            base.arg(),
            y.arg(),
            stated(OP, channels)?.arg(),
            stated(OP, side)?.arg(),
            stated(OP, taps_n)?.arg(),
            stated(OP, group_n)?.arg(),
            ctx.stage(),
        ],
    )
}
