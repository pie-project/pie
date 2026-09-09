use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "spatial/conv.metal";

const OP: &str = "spatial.conv3d";

const THREADS: u32 = 128;

const SIMDS: u32 = THREADS / 32;

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
    for (what, t) in [("input", x), ("weight", w), ("answer", y)] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
        let _ = what;
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
    let columns = c_out.div_ceil(SIMDS);
    let lanes = columns.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {c_out} output channels"),
        )
    })?;
    let entry = "spatial_conv3d_bfloat16";
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([lanes, rows_out, 1], [THREADS, 1, 1])),
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
        ],
    )
}
