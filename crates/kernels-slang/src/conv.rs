//! Convolutions and resampling over clip tables, from `kernels/conv/`. A
//! clip table is `[clips, 4]` i32, a row `(t, h, w, off)` a clip; an op
//! that reshapes reads its input table and the output one `conv::rule`
//! derived.

use dtype::Dtype;

use crate::encode::{Grid, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

pub mod conv3d;
pub mod group_norm;
pub mod resample;
pub mod rule;

/// Rows past 65535 workgroups ride the z axis: row = `group.y + group.z *
/// span`.
const ROW_SPAN: u32 = 65_535;

/// The grid of `lanes` a row over `rows` rows, and the span the kernel
/// folds its row index by.
pub(crate) fn row_grid(lanes: u32, rows: u32, group: u32) -> (Grid, u32) {
    let span = rows.clamp(1, ROW_SPAN);
    (
        Grid::of([lanes, span, rows.div_ceil(span)], [group, 1, 1]),
        span,
    )
}

/// The clips an input and an output table name, which must agree.
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

/// The clips a table names.
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

/// A bf16 row of `width` lands whole words: `width` must be even.
pub(crate) fn even_row(op: &'static str, what: &str, width: u32) -> Result<(), Error> {
    if !width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the {what} is {width} wide, and a bf16 row lands whole words (an even width)"),
        ));
    }
    Ok(())
}
