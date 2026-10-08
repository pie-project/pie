use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

pub const FILE: &str = "linear/ane.metal";
const PACK: &str = "ane_pack";
const SPLIT_JOIN: &str = "ane_join";

const GROUP: u32 = 256;

#[must_use]
pub fn fence_of(entrypoint: &str, args: &[ArgValue]) -> Option<(u64, bool)> {
    let (at, signal) = match entrypoint {
        PACK => (6, true),
        SPLIT_JOIN => (7, false),
        _ => return None,
    };
    match args.get(at) {
        Some(ArgValue::U32(stamp)) if *stamp != 0 => Some((u64::from(*stamp), signal)),
        _ => None,
    }
}

pub fn rotate(
    ctx: &Ctx<'_>,
    x: Tensor,
    sign: Tensor,
    rotated: Tensor,
    token_scale: Tensor,
) -> Result<(), Error> {
    ctx.fire(
        Fire::at(FILE, "ane_rotate").apply(Grid::of([x.rows * GROUP, 1, 1], [GROUP, 1, 1])),
        &[
            x.arg(),
            sign.arg(),
            rotated.arg_mut(),
            token_scale.arg_mut(),
            x.width.arg(),
        ],
    )
}

pub struct Pack {
    pub channel: u32,
    pub width: u32,
    pub stride: u32,
    pub stamp: u32,
}

pub fn pack(ctx: &Ctx<'_>, rotated: Tensor, packed: Tensor, at: &Pack) -> Result<(), Error> {
    let tiles = rotated.rows.div_ceil(32);
    ctx.fire(
        Fire::at(FILE, PACK).apply(Grid::of([tiles * 32, at.width / 32 * 8, 1], [32, 8, 1])),
        &[
            rotated.arg(),
            packed.arg_mut(),
            rotated.width.arg(),
            at.channel.arg(),
            at.stride.arg(),
            rotated.rows.arg(),
            at.stamp.arg(),
        ],
    )
}

pub struct Rows {
    pub first: u32,
    pub rows: u32,
    pub input: u32,
    pub span: u32,
    pub intermediate: bool,
}

fn planes(bank: &Bank) -> Result<[ArgValue; 3], Error> {
    let biases = bank.biases.ok_or(Error::Unsupported {
        op: "linear.ane_weights",
    })?;
    Ok([bank.codes.arg(), bank.scales.arg(), biases.arg()])
}

pub fn row_scale(
    ctx: &Ctx<'_>,
    bank: &Bank,
    at: &Rows,
    sign: Tensor,
    scale: Tensor,
) -> Result<(), Error> {
    let entry = if at.intermediate {
        "ane_row_scale_intermediate"
    } else {
        "ane_row_scale_inputs"
    };
    let [codes, scales, biases] = planes(bank)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([at.rows / 8 * GROUP, 1, 1], [GROUP, 1, 1])),
        &[
            codes,
            scales,
            biases,
            scale.arg_mut(),
            sign.arg(),
            bank.codes.width.arg(),
            at.first.arg(),
            at.input.arg(),
            at.span.arg(),
        ],
    )
}

pub struct Target {
    pub weights: Tensor,
    pub stride: u32,
    pub scale: Tensor,
    pub scale_stride: u32,
}

pub fn weights(
    ctx: &Ctx<'_>,
    bank: &Bank,
    at: &Rows,
    sign: Tensor,
    row_scale: Tensor,
    target: &Target,
) -> Result<(), Error> {
    let (entry, block) = if at.intermediate {
        ("ane_weights_intermediate", 512)
    } else {
        ("ane_weights_inputs", 128)
    };
    let [codes, scales, biases] = planes(bank)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            [at.rows / 8 * GROUP, at.span / block, 1],
            [GROUP, 1, 1],
        )),
        &[
            codes,
            scales,
            biases,
            row_scale.arg(),
            target.weights.arg_mut(),
            target.scale.arg_mut(),
            sign.arg(),
            bank.codes.width.arg(),
            at.first.arg(),
            at.input.arg(),
            target.stride.arg(),
            target.scale_stride.arg(),
        ],
    )
}

pub struct Join {
    pub stride: u32,
    pub stamp: u32,
}

pub fn split_join(
    ctx: &Ctx<'_>,
    out: Tensor,
    partial: Tensor,
    token_scale: Tensor,
    status: Tensor,
    at: &Join,
) -> Result<(), Error> {
    let tiles = out.rows.div_ceil(32);
    ctx.fire(
        Fire::at(FILE, SPLIT_JOIN).apply(Grid::of([tiles * 32, out.width / 32 * 8, 1], [32, 8, 1])),
        &[
            out.arg_mut(),
            partial.arg(),
            token_scale.arg(),
            status.arg_mut(),
            out.width.arg(),
            at.stride.arg(),
            out.rows.arg(),
            at.stamp.arg(),
        ],
    )
}
