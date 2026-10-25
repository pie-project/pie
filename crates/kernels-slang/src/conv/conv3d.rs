//! The 3-D convolution over clip tables (`conv.conv3d`), from
//! `kernels/conv/conv3d.slang` (a strip a channel) and
//! `kernels/conv/conv3d_tiled.slang` (register-tiled, when `c_in` is a
//! multiple of 16).

use dtype::Dtype;

use super::{clip_pair, even_row, row_grid};
use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const CONV3D: &str = "conv/conv3d.slang";

const CONV3D_TILED: &str = "conv/conv3d_tiled.slang";

const OP: &str = "conv.conv3d";

/// Four 32-lane strips, a strip an output channel.
const THREADS: u32 = 128;

const STRIPS: u32 = THREADS / 32;

/// The tiled kernel's tile, rows by channels.
const TILE: u32 = 64;

/// The tiled kernel's contraction step; a step stays inside one tap.
const STEP: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimePad {
    Zero,
    Replicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3d {
    pub k: [u32; 3],
    pub stride: [u32; 3],
    pub pad: [u32; 3],
    pub pad_back: [u32; 3],
    pub causal_t: bool,
    pub time_pad: TimePad,
}

impl Conv3d {
    #[must_use]
    pub const fn taps(&self) -> u32 {
        self.k[0] * self.k[1] * self.k[2]
    }

    /// The scalars both kernels read first: channels, kernel, stride, pad,
    /// the time rules, clips, rows.
    fn scalars(
        &self,
        c_in: u32,
        c_out: u32,
        clips: u32,
        rows: u32,
    ) -> Result<Vec<ArgValue>, Error> {
        let mut v = vec![stated(OP, c_in)?.arg(), stated(OP, c_out)?.arg()];
        for axis in [self.k, self.stride, self.pad] {
            for n in axis {
                v.push(stated(OP, n)?.arg());
            }
        }
        v.push(i32::from(self.causal_t).arg());
        v.push(i32::from(self.time_pad == TimePad::Replicate).arg());
        v.push(stated(OP, clips)?.arg());
        v.push(stated(OP, rows)?.arg());
        Ok(v)
    }
}

/// `y` (bf16, `[rows_out, c_out]`) the convolution of `x` (bf16, `[rows,
/// c_in]`) over the clips of `grid`, the output voxels those of `o_grid`;
/// `w` is `[c_out, taps * c_in]`, taps-major, `bias` an f32 per output
/// channel. `bases`: the rows `grid` and `o_grid` address `x`'s and `y`'s
/// row 0 at (`[x, y]`), nonzero in a window of a segmented fire whose grids
/// hold fire-wide rows.
#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    conv: Conv3d,
    o_grid: Tensor,
    y: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    for t in [x, w, y] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
    }
    let clips = clip_pair(OP, grid, o_grid)?;
    let c_in = x.width;
    let c_out = y.width;
    if c_in == 0 || c_out == 0 {
        return Err(refuse(
            OP,
            format!("a convolution from {c_in} channels into {c_out} computes nothing"),
        ));
    }
    even_row(OP, "output row", c_out)?;
    let taps = conv.taps();
    if w.rows != c_out || w.width != taps * c_in {
        return Err(refuse(
            OP,
            format!(
                "the weight is {} x {} and this convolution contracts {taps} tap(s) of \
                 {c_in} channel(s) into {c_out}",
                w.rows, w.width
            ),
        ));
    }
    if let Some(bias) = bias
        && (bias.dtype != Dtype::F32
            || u64::from(bias.rows) * u64::from(bias.width) < u64::from(c_out))
    {
        return Err(refuse(
            OP,
            format!(
                "the bias is {} x {} {:?}, and this convolution adds one f32 per \
                 output channel of {c_out}",
                bias.rows, bias.width, bias.dtype
            ),
        ));
    }
    let rows_out = y.rows;
    if rows_out == 0 {
        return Ok(());
    }
    let mut args = vec![
        x.arg(),
        grid.arg(),
        w.arg(),
        match bias {
            Some(bias) => bias.arg(),
            None => ctx.absent()?,
        },
        y.arg_mut(),
        o_grid.arg(),
    ];
    args.extend(conv.scalars(c_in, c_out, clips, rows_out)?);
    args.push(u32::from(bias.is_some()).arg());
    let bases = [stated(OP, bases[0])?.arg(), stated(OP, bases[1])?.arg()];
    if c_in.is_multiple_of(STEP) {
        args.extend(bases);
        let tiles = rows_out.div_ceil(TILE);
        if tiles > 65_535 {
            return Err(refuse(
                OP,
                format!("the grid will not launch: {rows_out} output rows"),
            ));
        }
        return ctx.fire(
            Fire::at(CONV3D_TILED, "conv_conv3d_tiled_bf16").apply(Grid::of(
                [c_out.div_ceil(TILE) * (TILE / 4), tiles * (TILE / 4), 1],
                [TILE / 4, TILE / 4, 1],
            )),
            &args,
        );
    }
    let columns = c_out.div_ceil(STRIPS);
    let lanes = columns.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {c_out} output channels"),
        )
    })?;
    let (shape, span) = row_grid(lanes, rows_out, THREADS);
    args.push(span.arg());
    args.extend(bases);
    ctx.fire(Fire::at(CONV3D, "conv_conv3d_bf16").apply(shape), &args)
}
