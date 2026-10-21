use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Tensor};

const FILE: &str = "attn/block_dyn_conv.metal";

const GROUP: u32 = 256;

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
    convolve(ctx, x, coeff, base, side, taps, group, y, None)
}

/// [`block_dyn_conv`] over at most eight rows that lays the register tile's
/// input table for the projections of `y` in `table` as it writes them, with
/// `counters` arrival counters zeroed: bit for bit the table the tile lays
/// over `y`.
#[allow(clippy::too_many_arguments)]
pub fn block_dyn_conv_table(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    coeff: Tensor,
    base: Tensor,
    side: u32,
    taps: u32,
    group: u32,
    y: Tensor,
    table: Tensor,
    counters: u32,
) -> Result<(), Error> {
    convolve(ctx, x, coeff, base, side, taps, group, y, Some((table, counters)))
}

#[allow(clippy::too_many_arguments)]
fn convolve(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    coeff: Tensor,
    base: Tensor,
    side: u32,
    taps: u32,
    group: u32,
    y: Tensor,
    table: Option<(Tensor, u32)>,
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
    let rows = nonzero(OP, "the rows the convolution walks", x.data.rows)?;
    let args = [
        x.data.arg(),
        x.indptr.arg(),
        coeff.arg(),
        base.arg(),
        y.arg_mut(),
        stated(OP, channels)?.arg(),
        stated(OP, side)?.arg(),
        stated(OP, taps_n)?.arg(),
        stated(OP, group_n)?.arg(),
        stated(OP, lanes)?.arg(),
    ];
    match table {
        // A threadgroup a row of the table, which stages at most 8192 channels.
        Some((table, counters)) => {
            if rows > 8 || channels % 64 != 0 || channels > 8192 {
                return Err(refuse(
                    OP,
                    format!(
                        "a register table holds at most 8 rows of whole 64-input groups, and \
                         this one stages at most 8192, not {rows} x {channels}"
                    ),
                ));
            }
            let mut args = args.to_vec();
            args.extend([table.arg_mut(), rows.arg(), counters.arg()]);
            ctx.fire(
                Fire::at(FILE, "block_dyn_conv_table").apply(Grid::of(
                    [channels.min(1024), 8, 1],
                    [channels.min(1024), 1, 1],
                )),
                &args,
            )
        }
        // A thread per channel and row, so a lane's rows convolve side by side.
        None => ctx.fire(
            Fire::at(FILE, entry)
                .apply(Grid::of([channels, rows, 1], [channels.min(GROUP), 1, 1])),
            &args,
        ),
    }
}
