use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "spatial/conv.slang";

const OP: &str = "spatial.conv3d";

const THREADS: u32 = 128;

const STRIPS: u32 = THREADS / 32;

/// A 16-wide K block of the tiled point stays inside one tap when `c_in % 16 == 0`.
const RT_FILE: &str = "spatial/conv_rt.slang";

const RT_TILE: u32 = 64;

const RT_BK: u32 = 16;

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
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    cache: Option<Tensor>,
    conv: Conv3d,
    o_grid: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    for t in [x, w, y] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
    }
    let clips = super::clip_pair(OP, grid, o_grid)?;
    let c_in = x.width;
    let c_out = y.width;
    if c_in == 0 || c_out == 0 {
        return Err(refuse(
            OP,
            format!("a convolution from {c_in} channels into {c_out} computes nothing"),
        ));
    }
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
    if let Some(cache) = cache
        && (cache.dtype != Dtype::Bf16 || cache.width != c_in)
    {
        return Err(refuse(
            OP,
            format!(
                "the frame cache is {} x {} {:?}, and this convolution reads {c_in} \
                     bf16 channels a row",
                cache.rows, cache.width, cache.dtype
            ),
        ));
    }
    let rows_out = y.rows;
    if rows_out == 0 {
        return Ok(());
    }
    if cache.is_none() && c_in.is_multiple_of(RT_BK) {
        let rows = rows_out.div_ceil(RT_TILE);
        if rows > 65_535 {
            return Err(refuse(
                OP,
                format!("the grid will not launch: {rows_out} output rows"),
            ));
        }
        return ctx.fire(
            Fire::at(RT_FILE, "spatial_conv3d_rt_bf16").apply(Grid::of(
                [
                    c_out.div_ceil(RT_TILE) * (RT_TILE / 4),
                    rows * (RT_TILE / 4),
                    1,
                ],
                [RT_TILE / 4, RT_TILE / 4, 1],
            )),
            &[
                x.arg(),
                grid.arg(),
                w.arg(),
                match bias {
                    Some(bias) => bias.arg(),
                    None => ctx.absent()?,
                },
                y.arg_mut(),
                o_grid.arg(),
                stated(OP, c_in)?.arg(),
                stated(OP, c_out)?.arg(),
                stated(OP, conv.k[0])?.arg(),
                stated(OP, conv.k[1])?.arg(),
                stated(OP, conv.k[2])?.arg(),
                stated(OP, conv.stride[0])?.arg(),
                stated(OP, conv.stride[1])?.arg(),
                stated(OP, conv.stride[2])?.arg(),
                stated(OP, conv.pad[0])?.arg(),
                stated(OP, conv.pad[1])?.arg(),
                stated(OP, conv.pad[2])?.arg(),
                i32::from(conv.causal_t).arg(),
                i32::from(conv.time_pad == TimePad::Replicate).arg(),
                stated(OP, clips)?.arg(),
                stated(OP, rows_out)?.arg(),
                u32::from(bias.is_some()).arg(),
            ],
        );
    }
    let columns = c_out.div_ceil(STRIPS);
    let lanes = columns.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {c_out} output channels"),
        )
    })?;
    let (grid_of, span) = super::row_grid(lanes, rows_out, THREADS);
    ctx.fire(
        Fire::at(FILE, "spatial_conv3d_bf16").apply(grid_of),
        &[
            x.arg(),
            grid.arg(),
            w.arg(),
            match bias {
                Some(bias) => bias.arg(),
                None => ctx.absent()?,
            },
            match cache {
                Some(cache) => cache.arg(),
                None => ctx.absent()?,
            },
            y.arg_mut(),
            o_grid.arg(),
            stated(OP, c_in)?.arg(),
            stated(OP, c_out)?.arg(),
            stated(OP, conv.k[0])?.arg(),
            stated(OP, conv.k[1])?.arg(),
            stated(OP, conv.k[2])?.arg(),
            stated(OP, conv.stride[0])?.arg(),
            stated(OP, conv.stride[1])?.arg(),
            stated(OP, conv.stride[2])?.arg(),
            stated(OP, conv.pad[0])?.arg(),
            stated(OP, conv.pad[1])?.arg(),
            stated(OP, conv.pad[2])?.arg(),
            i32::from(conv.causal_t).arg(),
            i32::from(conv.time_pad == TimePad::Replicate).arg(),
            stated(OP, clips)?.arg(),
            stated(OP, rows_out)?.arg(),
            i32::from(cache.is_some()).arg(),
            u32::from(bias.is_some()).arg(),
            span.arg(),
        ],
    )
}
