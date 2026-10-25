use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "conv/rule.metal";

const OP: &str = "conv.grid";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridRule {
    Conv {
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
    },
    Upsample {
        factor: [u32; 3],
        keep_first_frame: bool,
    },
    Shuffle {
        r: [u32; 3],
        trim_t: u32,
    },
    Unshuffle {
        r: [u32; 3],
    },
    AvgDown {
        factor: [u32; 3],
    },
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Geom {
    kind: i32,
    a: [i32; 3],
    b: [i32; 3],
    c: [i32; 3],
    d: [i32; 3],
    flag: i32,
    clips: i32,
}

fn triple(v: [u32; 3]) -> Result<[i32; 3], Error> {
    Ok([stated(OP, v[0])?, stated(OP, v[1])?, stated(OP, v[2])?])
}

pub fn derive_grid(
    ctx: &Ctx<'_>,
    grid: Tensor,
    rule: GridRule,
    o_grid: Tensor,
) -> Result<(), Error> {
    let clips = super::clip_pair(OP, grid, o_grid)?;
    let geom = match rule {
        GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        } => Geom {
            kind: 0,
            a: triple(k)?,
            b: triple(stride)?,
            c: triple(pad)?,
            d: triple(pad_back)?,
            flag: i32::from(causal_t),
            clips: stated(OP, clips)?,
        },
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => Geom {
            kind: 1,
            a: triple(factor)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: i32::from(keep_first_frame),
            clips: stated(OP, clips)?,
        },
        GridRule::Shuffle { r, trim_t } => Geom {
            kind: 2,
            a: triple(r)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: stated(OP, trim_t)?,
            clips: stated(OP, clips)?,
        },
        GridRule::Unshuffle { r } => Geom {
            kind: 3,
            a: triple(r)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: 0,
            clips: stated(OP, clips)?,
        },
        GridRule::AvgDown { factor } => Geom {
            kind: 4,
            a: triple(factor)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: 0,
            clips: stated(OP, clips)?,
        },
    };
    ctx.fire(
        Fire::at(FILE, "grid_rule").apply(Grid::of([1, 1, 1], [1, 1, 1])),
        &[
            grid.arg(),
            o_grid.arg_mut(),
            geom.kind.arg(),
            geom.a[0].arg(),
            geom.a[1].arg(),
            geom.a[2].arg(),
            geom.b[0].arg(),
            geom.b[1].arg(),
            geom.b[2].arg(),
            geom.c[0].arg(),
            geom.c[1].arg(),
            geom.c[2].arg(),
            geom.d[0].arg(),
            geom.d[1].arg(),
            geom.d[2].arg(),
            geom.flag.arg(),
            geom.clips.arg(),
        ],
    )
}

/// Each token row's cell (`ops::conv::cells`): a row of a clip of `tgrid`
/// its patch's `(t, h, w)` there, any other row `positions[row]` in column
/// `axis`; zeros elsewhere. `base` is the row `tgrid` addresses `y`'s row 0
/// at, nonzero in a window of a segmented fire.
pub fn cells(
    ctx: &Ctx<'_>,
    tgrid: Tensor,
    positions: Tensor,
    axis: u32,
    y: Tensor,
    base: u32,
) -> Result<(), Error> {
    const OP: &str = "conv.cells";
    let clips = super::clips_of(OP, tgrid)?;
    if positions.dtype != Dtype::I32 || y.dtype != Dtype::F32 || y.width != axis.max(2) + 1 {
        return Err(refuse(
            OP,
            "the positions are i32 and the cells three f32 columns",
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, "cells").apply(Grid::of([y.rows, 1, 1], [256.min(y.rows), 1, 1])),
        &[
            tgrid.arg(),
            positions.arg(),
            y.arg_mut(),
            stated(OP, clips)?.arg(),
            stated(OP, axis)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, base)?.arg(),
        ],
    )
}

/// Each row's count of `csr`'s rows in its lane's group
/// (`ops::gather::group_rows`).
pub fn group_rows(
    ctx: &Ctx<'_>,
    csr: Tensor,
    groups: Tensor,
    lanes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.group_rows";
    if [csr.dtype, groups.dtype, lanes.dtype] != [Dtype::I32; 3]
        || y.dtype != Dtype::F32
        || y.width != 1
        || csr.rows == 0
    {
        return Err(refuse(
            OP,
            "the tables are i32 and the counts one f32 column",
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, "group_rows").apply(Grid::of([y.rows, 1, 1], [256.min(y.rows), 1, 1])),
        &[
            csr.arg(),
            groups.arg(),
            lanes.arg(),
            y.arg_mut(),
            stated(OP, groups.rows)?.arg(),
            stated(OP, csr.rows - 1)?.arg(),
            stated(OP, y.rows)?.arg(),
        ],
    )
}
