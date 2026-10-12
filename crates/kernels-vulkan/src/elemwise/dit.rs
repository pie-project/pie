//! The elementwise points a diffusion transformer runs: timestep sinusoids,
//! adaLN modulation and its gated residual, layer norm with no affine,
//! multi-axis rotary positions, and the plain activations and sum.

use dtype::Dtype;

use crate::encode::{
    Arg, ArgValue, Ctx, Fire, Grid, elementwise, elementwise_rows, nonzero, refuse,
};
use crate::error::Error;
use crate::tensor::Tensor;

const GROUP: [u32; 3] = [256, 1, 1];

fn stamp(op: &'static str, dtype: Dtype) -> Result<&'static str, Error> {
    match dtype {
        Dtype::Bf16 => Ok("bf16"),
        Dtype::F32 => Ok("f32"),
        other => Err(Error::DtypeUnsupported { op, dtype: other }),
    }
}

fn entry(op: &'static str, name: &str, dtype: Dtype) -> Result<&'static str, Error> {
    Ok(crate::encode::symbol(&format!(
        "{name}_{}",
        stamp(op, dtype)?
    )))
}

fn activation(
    ctx: &Ctx<'_>,
    op: &'static str,
    name: &str,
    x: Tensor,
    o: Tensor,
) -> Result<(), Error> {
    let lanes = elementwise(op, o.width, o.rows)?;
    ctx.fire(
        Fire::at("dit/pointwise.slang", entry(op, name, o.dtype)?).apply(Grid::of(lanes, GROUP)),
        &[x.arg(), o.arg_mut(), lanes[0].arg()],
    )
}

pub fn silu(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    activation(ctx, "elementwise.silu", "silu", x, o)
}

pub fn tanh(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    activation(ctx, "elementwise.tanh", "tanh", x, o)
}

pub fn add(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add";
    if x.dtype != z.dtype || y.dtype != z.dtype {
        return Err(refuse(
            OP,
            format!(
                "{:?} + {:?} into {:?}; a sum rides one element",
                x.dtype, y.dtype, z.dtype
            ),
        ));
    }
    let lanes = elementwise(OP, z.width, z.rows)?;
    ctx.fire(
        Fire::at("dit/pointwise.slang", entry(OP, "add", z.dtype)?).apply(Grid::of(lanes, GROUP)),
        &[x.arg(), y.arg(), z.arg_mut(), lanes[0].arg()],
    )
}

pub fn sinusoid(
    ctx: &Ctx<'_>,
    t: Tensor,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.sinusoid";
    if t.dtype != Dtype::F32 || y.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the timesteps are {:?} and the embedding {:?}; both are f32",
                t.dtype, y.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    if dim < 2 || y.width != dim || t.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "a {dim}-wide embedding into a {}-wide plane from {} timesteps for {rows} rows",
                y.width, t.rows
            ),
        ));
    }
    ctx.fire(
        Fire::at("dit/sinusoid.slang", "sinusoid_f32")
            .apply(Grid::of(elementwise_rows(OP, dim / 2, rows)?, GROUP)),
        &[
            t.arg(),
            y.arg_mut(),
            dim.arg(),
            max_period.arg(),
            u32::from(flip_sin_cos).arg(),
            scale.arg(),
        ],
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    ScaleShift,
    Scale,
    TanhGate,
}

impl Form {
    const fn word(self) -> i32 {
        match self {
            Self::ScaleShift => 0,
            Self::Scale => 1,
            Self::TanhGate => 2,
        }
    }

    const fn vectors(self) -> u32 {
        match self {
            Self::ScaleShift => 2,
            Self::Scale | Self::TanhGate => 1,
        }
    }
}

fn lane_map(
    ctx: &Ctx<'_>,
    op: &'static str,
    lanes: Option<Tensor>,
    rows: u32,
) -> Result<(ArgValue, u32), Error> {
    let Some(map) = lanes else {
        return Ok((ctx.absent()?, 0));
    };
    if map.dtype != Dtype::I32 || u64::from(map.rows) * u64::from(map.width) < u64::from(rows) {
        return Err(refuse(
            op,
            format!(
                "a {:?} lane map of {} x {} for {rows} rows",
                map.dtype, map.rows, map.width
            ),
        ));
    }
    Ok((map.arg(), 1))
}

fn modulation(
    op: &'static str,
    name: &str,
    x: Dtype,
    m: Tensor,
    width: u32,
) -> Result<&'static str, Error> {
    let stamp = match (x, m.dtype) {
        (Dtype::Bf16, Dtype::Bf16) => "bf16",
        (Dtype::Bf16, Dtype::F32) => "bf16_f32",
        (Dtype::F32, Dtype::F32) => "f32",
        (activation, plane) => {
            return Err(refuse(
                op,
                format!("a {plane:?} modulation of {activation:?} rows"),
            ));
        }
    };
    if m.width < width {
        return Err(refuse(
            op,
            format!("a {}-wide modulation of {width}-wide rows", m.width),
        ));
    }
    Ok(crate::encode::symbol(&format!("{name}_{stamp}")))
}

pub fn modulate(
    ctx: &Ctx<'_>,
    form: Form,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.modulate";
    let width = nonzero(OP, "the modulated width", o.width)?;
    let rows = nonzero(OP, "rows", o.rows)?;
    let entry = modulation(OP, "modulate", x.dtype, m, form.vectors() * width)?;
    if lane_of_row.is_none() && m.rows < rows {
        return Err(refuse(
            OP,
            format!("{} modulation vectors for {rows} rows", m.rows),
        ));
    }
    let (map, has_lanes) = lane_map(ctx, OP, lane_of_row, rows)?;
    ctx.fire(
        Fire::at("dit/modulate.slang", entry)
            .apply(Grid::of(elementwise_rows(OP, width, rows)?, GROUP)),
        &[
            x.arg(),
            m.arg(),
            map,
            o.arg_mut(),
            width.arg(),
            (form.vectors() * width).arg(),
            form.word().arg(),
            has_lanes.arg(),
        ],
    )
}

pub fn gated_residual_add(
    ctx: &Ctx<'_>,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    lane_of_row: Option<Tensor>,
    r_out: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gated_residual_add";
    let width = nonzero(OP, "the folded width", r_out.width)?;
    let rows = nonzero(OP, "rows", r_out.rows)?;
    let entry = modulation(OP, "gated_residual_add", r.dtype, g, width)?;
    if lane_of_row.is_none() && g.rows < rows {
        return Err(refuse(OP, format!("{} gates for {rows} rows", g.rows)));
    }
    let (map, has_lanes) = lane_map(ctx, OP, lane_of_row, rows)?;
    ctx.fire(
        Fire::at("dit/modulate.slang", entry)
            .apply(Grid::of(elementwise_rows(OP, width, rows)?, GROUP)),
        &[
            r.arg(),
            g.arg(),
            y.arg(),
            map,
            r_out.arg_mut(),
            width.arg(),
            has_lanes.arg(),
        ],
    )
}

pub fn layernorm_no_scale(ctx: &Ctx<'_>, x: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm_no_scale";
    let rows = nonzero(OP, "rows", x.rows)?;
    let width = nonzero(OP, "width", x.width)?;
    ctx.fire(
        Fire::at(
            "dit/layernorm_no_scale.slang",
            entry(OP, "layernorm_no_scale", x.dtype)?,
        )
        .apply(Grid::of([rows * 256, 1, 1], GROUP)),
        &[x.arg(), y.arg_mut(), eps.arg(), width.arg()],
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    Interleaved,
    Neox,
    Split,
    SplitLadder,
}

impl RopeForm {
    const fn word(self) -> i32 {
        match self {
            Self::Interleaved => 0,
            Self::Neox => 1,
            Self::Split => 2,
            Self::SplitLadder => 3,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rope_axes(
    ctx: &Ctx<'_>,
    x: Tensor,
    positions: Tensor,
    dims: [u32; 4],
    thetas: [f32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_axes";
    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!("{:?} positions; a rotation reads f32", positions.dtype),
        ));
    }
    let rows = nonzero(OP, "rows", o.rows)?;
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    if !o.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("a {}-wide row of {head_dim}-wide heads", o.width),
        ));
    }
    let heads = o.width / head_dim;
    let axes = u32::try_from(dims.iter().take_while(|d| **d != 0).count()).unwrap_or(0);
    if axes == 0 || dims[..axes as usize].iter().any(|d| d % 2 != 0) {
        return Err(refuse(OP, format!("axis widths {dims:?}")));
    }
    let span: u32 = dims[..axes as usize].iter().sum();
    let fits = if form == RopeForm::SplitLadder {
        rotary_dim != 0
            && rotary_dim <= head_dim
            && span <= heads * rotary_dim
            && (heads * rotary_dim - span).is_multiple_of(2)
    } else {
        span == rotary_dim && rotary_dim != 0 && rotary_dim <= head_dim
    };
    if !fits || positions.rows < rows || positions.width < axes {
        return Err(refuse(
            OP,
            format!(
                "axes {dims:?} over {rotary_dim} of {head_dim} channels, {} x {} positions \
                 for {rows} rows",
                positions.rows, positions.width
            ),
        ));
    }
    let angles = rotary_dim / 2;
    let tail = if x.buf == o.buf {
        0
    } else {
        head_dim - rotary_dim
    };
    let lanes = heads * (angles + tail);
    ctx.fire(
        Fire::at("dit/rope_axes.slang", entry(OP, "rope_axes", o.dtype)?)
            .apply(Grid::of([lanes, rows, 1], GROUP)),
        &[
            x.arg(),
            positions.arg(),
            o.arg_mut(),
            dims[0].arg(),
            dims[1].arg(),
            dims[2].arg(),
            dims[3].arg(),
            thetas[0].arg(),
            thetas[1].arg(),
            thetas[2].arg(),
            thetas[3].arg(),
            axes.arg(),
            rotary_dim.arg(),
            head_dim.arg(),
            heads.arg(),
            form.word().arg(),
            u32::from(tail != 0).arg(),
        ],
    )
}
