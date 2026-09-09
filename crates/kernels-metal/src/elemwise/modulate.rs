use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, elementwise_rows, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "elemwise/modulate.metal";

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
    lane_of_row: Option<Tensor>,
    rows: u32,
) -> Result<(ArgValue, u32), Error> {
    let Some(map) = lane_of_row else {
        return Ok((ctx.absent()?, 0));
    };
    if map.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the lane map is {:?}, and this modulation reads an i32 lane per row",
                map.dtype
            ),
        ));
    }
    if u64::from(map.rows) * u64::from(map.width) < u64::from(rows) {
        return Err(refuse(
            op,
            format!(
                "the lane map is {} x {}, and this modulation reads one lane for each of \
                 {rows} rows",
                map.rows, map.width
            ),
        ));
    }
    Ok((map.arg(), 1))
}

fn modulation(
    op: &'static str,
    m: Tensor,
    width: u32,
    vectors: u32,
    dtype: Dtype,
) -> Result<&'static str, Error> {
    let stamp = match (dtype, m.dtype) {
        (Dtype::Bf16, Dtype::Bf16) => "bfloat16",
        (Dtype::Bf16, Dtype::F32) => "bfloat16_f32",
        (Dtype::F32, Dtype::F32) => "float32",
        (Dtype::Bf16 | Dtype::F32, other) => {
            return Err(refuse(
                op,
                format!(
                    "the modulation plane is {other:?} and the rows it modulates are \
                     {dtype:?}; a vector rides the activation's element or stays f32"
                ),
            ));
        }
        (other, _) => return Err(Error::DtypeUnsupported { op, dtype: other }),
    };
    if m.width != vectors * width {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {} wide, and this form reads {vectors} x {width}",
                m.width
            ),
        ));
    }
    Ok(stamp)
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
    debug_assert!(
        x.rows == o.rows && x.width == o.width,
        "`{OP}` writes the rectangle it reads"
    );
    let width = nonzero(OP, "the modulated width", o.width)?;
    let rows = nonzero(OP, "rows", o.rows)?;
    let stamp = modulation(OP, m, width, form.vectors(), x.dtype)?;
    if lane_of_row.is_none() && m.rows < rows {
        return Err(refuse(
            OP,
            format!("{} per-token modulation vectors for {rows} rows", m.rows),
        ));
    }
    let entry = match stamp {
        "bfloat16" => "modulate_bfloat16",
        "bfloat16_f32" => "modulate_bfloat16_f32",
        _ => "modulate_float32",
    };
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
    debug_assert!(
        r.rows == r_out.rows && r.width == r_out.width && y.rows == r.rows && y.width == r.width,
        "`{OP}` folds one rectangle"
    );
    let width = nonzero(OP, "the folded width", r_out.width)?;
    let rows = nonzero(OP, "rows", r_out.rows)?;
    let stamp = modulation(OP, g, width, 1, r.dtype)?;
    if lane_of_row.is_none() && g.rows < rows {
        return Err(refuse(
            OP,
            format!("{} per-token gates for {rows} rows", g.rows),
        ));
    }
    let entry = match stamp {
        "bfloat16" => "gated_residual_add_bfloat16",
        "bfloat16_f32" => "gated_residual_add_bfloat16_f32",
        _ => "gated_residual_add_float32",
    };
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
