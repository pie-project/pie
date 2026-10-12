//! A VAE's ops over clip tables.

pub mod attn;
pub mod conv;
pub mod norm;
pub mod resample;
pub mod rule;

use crate::encode::{Grid, refuse};
use crate::error::Error;
use crate::tensor::Tensor;
use dtype::Dtype;

/// Rows past 65535 groups ride the z axis: row = `group.y + group.z * span`.
const ROW_SPAN: u32 = 65_535;

pub(crate) fn row_grid(lanes: u32, rows: u32, group: u32) -> (Grid, u32) {
    let span = rows.clamp(1, ROW_SPAN);
    (
        Grid::of([lanes, span, rows.div_ceil(span)], [group, 1, 1]),
        span,
    )
}

pub(crate) fn clip_pair(op: &'static str, a: Tensor, b: Tensor) -> Result<u32, Error> {
    for (what, t) in [("input", a), ("output", b)] {
        if t.dtype != Dtype::I32 || t.width != 4 {
            return Err(refuse(
                op,
                format!(
                    "the {what} clip table is {} x {} {:?}, and a box table is \
                     [clips, 4] i32",
                    t.rows, t.width, t.dtype
                ),
            ));
        }
    }
    if a.rows != b.rows || a.rows == 0 {
        return Err(refuse(
            op,
            format!(
                "the input clip table names {} clip(s) and the output {}",
                a.rows, b.rows
            ),
        ));
    }
    Ok(a.rows)
}

pub(crate) fn clips_of(op: &'static str, grid: Tensor) -> Result<u32, Error> {
    if grid.dtype != Dtype::I32 || grid.width != 4 || grid.rows == 0 {
        return Err(refuse(
            op,
            format!(
                "the clip table is {} x {} {:?}, and a box table is [clips, 4] i32",
                grid.rows, grid.width, grid.dtype
            ),
        ));
    }
    Ok(grid.rows)
}
