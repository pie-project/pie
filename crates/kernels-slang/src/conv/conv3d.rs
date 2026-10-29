//! The 3-D convolution over clip tables (`conv.conv3d`), from
//! `kernels/conv/conv3d.slang` (a strip a channel) and
//! `kernels/conv/conv3d_tiled.slang` (register-tiled, when `c_in` is a
//! multiple of 16 and the channels are not convolved alone; on SPIR-V from
//! 16 channels out, its 8-row lanes).

use dtype::Dtype;

use super::{clip_pair, row_grid};
use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, Target, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const CONV3D: &str = "conv/conv3d.slang";

const CONV3D_TILED: &str = "conv/conv3d_tiled.slang";

const OP: &str = "conv.conv3d";

/// Four 32-lane strips, a strip an output channel.
const THREADS: u32 = 128;

const STRIPS: u32 = THREADS / 32;

/// The tiled kernel's lanes.
const TILED_THREADS: u32 = 256;

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
    pub dilation: [u32; 3],
    pub depthwise: bool,
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
/// channel. `bases` (`[x, y]`): `x`'s and `y`'s row 0 in the grids' fire-wide rows.
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
    // An odd output row's words straddle rows: the flat arm lands them.
    let flat = !c_out.is_multiple_of(2);
    let taps = conv.taps();
    if conv.depthwise && c_in != c_out {
        return Err(refuse(
            OP,
            format!(
                "a depthwise convolution keeps its channels, and this one maps {c_in} into {c_out}"
            ),
        ));
    }
    let per = if conv.depthwise { 1 } else { c_in };
    if w.rows != c_out || w.width != taps * per {
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
    let dilation = conv
        .dilation
        .iter()
        .map(|&d| stated(OP, d).map(Arg::arg))
        .collect::<Result<Vec<_>, _>>()?;
    let bases = [stated(OP, bases[0])?.arg(), stated(OP, bases[1])?.arg()];
    if !flat && !conv.depthwise && c_in.is_multiple_of(STEP) {
        // `(entry, channels a tile, channels a lane, rows a lane)`. On SPIR-V,
        // sixty-four channels a tile at eight a lane, or thirty-two at four
        // when that leaves none idle; fewer than sixteen channels leave most
        // of either idle, and WGSL's workgroup memory holds only the 4 x 4
        // lane.
        let (entry, cols, per, lane_rows) = if ctx.target() == Target::Spirv && c_out >= 16 {
            if c_out.is_multiple_of(32) && !c_out.is_multiple_of(64) {
                ("conv_conv3d_tiled_bm_256_bn_32_bf16", 32, 4, 8)
            } else {
                ("conv_conv3d_tiled_bm_256_bf16", 64, 8, 8)
            }
        } else {
            ("conv_conv3d_tiled_bf16", 64, 4, 4)
        };
        let tx = cols / per;
        let ty = TILED_THREADS / tx;
        let rows = ty * lane_rows;
        let tiles = rows_out.div_ceil(rows);
        if tiles > 65_535 {
            return Err(refuse(
                OP,
                format!("the grid will not launch: {rows_out} output rows"),
            ));
        }
        args.extend(bases);
        args.extend(dilation);
        return ctx.fire(
            Fire::at(CONV3D_TILED, entry).apply(Grid::of(
                [c_out.div_ceil(cols) * tx, tiles * ty, 1],
                [tx, ty, 1],
            )),
            &args,
        );
    }
    if flat {
        let groups = (u64::from(rows_out) * u64::from(c_out)).div_ceil(u64::from(STRIPS));
        let groups = u32::try_from(groups).map_err(|_| {
            refuse(
                OP,
                format!("the grid will not launch: {rows_out} rows of {c_out} channels"),
            )
        })?;
        let (shape, span) = row_grid(THREADS, groups, THREADS);
        args.push(span.arg());
        args.extend(bases);
        args.extend(dilation);
        args.push(i32::from(conv.depthwise).arg());
        return ctx.fire(
            Fire::at(CONV3D, "conv_conv3d_flat_bf16").apply(shape),
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
    args.extend(dilation);
    args.push(i32::from(conv.depthwise).arg());
    ctx.fire(Fire::at(CONV3D, "conv_conv3d_bf16").apply(shape), &args)
}
