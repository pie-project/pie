use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, elementwise_rows, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "elemwise/modulate.slang";

const GROUP: [u32; 3] = [256, 1, 1];

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
        Fire::at(FILE, entry).apply(Grid::of(elementwise_rows(OP, width, rows)?, GROUP)),
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
        Fire::at(FILE, entry).apply(Grid::of(elementwise_rows(OP, width, rows)?, GROUP)),
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
