//! Per-row modulation (`elemwise.modulate`, `elemwise.gated_residual_add`)
//! from `kernels/elemwise/modulate.slang`, a pair of columns a lane: each row
//! reads its own modulation row, or the one a lane map names.

use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, elementwise_rows, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const MODULATE: &str = "elemwise/modulate.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// What a modulation row does to its row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// `x * (1 + scale) + shift`, the two vectors side by side.
    ScaleShift,
    /// `x * (1 + scale)`.
    Scale,
    /// `tanh(gate) * x`.
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

/// The row and modulation elements, as the entry suffix that pairs them.
fn stamp(op: &'static str, x: Dtype, m: Tensor, width: u32, needs: u32) -> Result<usize, Error> {
    let at = match (x, m.dtype) {
        (Dtype::Bf16, Dtype::Bf16) => 0,
        (Dtype::Bf16, Dtype::F32) => 1,
        (Dtype::F32, Dtype::F32) => 2,
        (activation, plane) => {
            return Err(refuse(
                op,
                format!("a {plane:?} modulation of {activation:?} rows"),
            ));
        }
    };
    if x == Dtype::Bf16 && !width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("a {width}-wide bf16 row is odd; rows land two values a word"),
        ));
    }
    if m.width < needs {
        return Err(refuse(
            op,
            format!("a {}-wide modulation of {width}-wide rows", m.width),
        ));
    }
    Ok(at)
}

/// `o` is `x` modulated by `form`, row `r` by modulation row `r` or
/// `lane_of_row[r]`.
pub fn modulate(
    ctx: &Ctx<'_>,
    form: Form,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.modulate";
    let width = nonzero(OP, "the modulated width", o.width)?;
    let rows = nonzero(OP, "rows", o.rows)?;
    let m_width = form.vectors() * width;
    let entry = [
        "elemwise_modulate_bf16",
        "elemwise_modulate_bf16_f32",
        "elemwise_modulate_f32",
    ][stamp(OP, x.dtype, m, width, m_width)?];
    if lane_of_row.is_none() && m.rows < rows {
        return Err(refuse(
            OP,
            format!("{} modulation vectors for {rows} rows", m.rows),
        ));
    }
    let (map, has_lanes) = lane_map(ctx, OP, lane_of_row, rows)?;
    ctx.fire(
        Fire::at(MODULATE, entry).apply(Grid::of(
            elementwise_rows(OP, width.div_ceil(2), rows)?,
            [GROUP, 1, 1],
        )),
        &[
            x.arg(),
            m.arg(),
            map,
            o.arg_mut(),
            width.arg(),
            m_width.arg(),
            form.word().arg(),
            has_lanes.arg(),
        ],
    )
}

/// `r += g * y` in place, row `i` gated by row `i` of `g` or
/// `lane_of_row[i]`.
pub fn gated_residual_add(
    ctx: &Ctx<'_>,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    lane_of_row: Option<Tensor>,
) -> Result<(), Error> {
    const OP: &str = "elemwise.gated_residual_add";
    let width = nonzero(OP, "the folded width", r.width)?;
    let rows = nonzero(OP, "rows", r.rows)?;
    let entry = [
        "elemwise_gated_residual_add_bf16",
        "elemwise_gated_residual_add_bf16_f32",
        "elemwise_gated_residual_add_f32",
    ][stamp(OP, r.dtype, g, width, width)?];
    if lane_of_row.is_none() && g.rows < rows {
        return Err(refuse(OP, format!("{} gates for {rows} rows", g.rows)));
    }
    let (map, has_lanes) = lane_map(ctx, OP, lane_of_row, rows)?;
    ctx.fire(
        Fire::at(MODULATE, entry).apply(Grid::of(
            elementwise_rows(OP, width.div_ceil(2), rows)?,
            [GROUP, 1, 1],
        )),
        &[
            r.arg_mut(),
            g.arg(),
            y.arg(),
            map,
            width.arg(),
            has_lanes.arg(),
        ],
    )
}
