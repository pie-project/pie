//! The voxel-table ops a denoiser's graph runs around its token rows:
//! (un)patchify between a clip's voxels and its token rows, and each token
//! row's cell (or a peer lane's row count) for its positions. Host phases:
//! index moves over small tables, no fabric work.

use dtype::Dtype;

use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::phase::HostOp;
use crate::tensor::Tensor;

fn table(op: &'static str, what: &str, grid: Tensor) -> Result<u32, Error> {
    if grid.dtype != Dtype::I32 || grid.width != 4 {
        return Err(refuse(
            op,
            format!(
                "the {what} clip table is {}x{} {:?}; a box table is [clips, 4] i32",
                grid.rows, grid.width, grid.dtype
            ),
        ));
    }
    Ok(grid.rows)
}

/// Each token row's cell (`conv.cells`): a row of a clip of `tgrid` its
/// patch's `(t, h, w)` there, any other row `positions[row]` in column
/// `axis`; zeros elsewhere.
pub fn cells(
    ctx: &Ctx<'_>,
    tgrid: Tensor,
    positions: Tensor,
    axis: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.cells";
    let clips = table(OP, "token", tgrid)?;
    expect(OP, positions, &[Dtype::I32])?;
    if y.dtype != Dtype::F32
        || y.width != axis.max(2) + 1
        || positions.elements() < u64::from(y.rows)
    {
        return Err(refuse(
            OP,
            format!(
                "the cells are {}x{} {:?} over {} positions; max(3, {axis} + 1) f32 columns, a \
                 position a row",
                y.rows,
                y.width,
                y.dtype,
                positions.elements()
            ),
        ));
    }
    let (rows, width) = (y.rows, y.width);
    ctx.emit(&mut |cx| {
        let gb = cx.read(tgrid)?;
        let pb = cx.read(positions)?;
        let yb = cx.write(y)?;
        cx.host(HostOp::Cells {
            tgrid: gb.name,
            positions: pb.name,
            y: yb.name,
            clips,
            rows,
            axis,
            width,
        });
        Ok(())
    })
}

/// Each row's count of the rows `csr` packs in its lane's group
/// (`gather.group_rows`).
pub fn group_rows(
    ctx: &Ctx<'_>,
    csr: Tensor,
    groups: Tensor,
    lanes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.group_rows";
    for t in [csr, groups, lanes] {
        expect(OP, t, &[Dtype::I32])?;
    }
    if y.dtype != Dtype::F32 || y.width != 1 || csr.elements() == 0 {
        return Err(refuse(OP, "the counts are one f32 column over an indptr"));
    }
    let rows = y.rows;
    let (csr_len, lane_count) = (csr.rows, groups.rows);
    ctx.emit(&mut |cx| {
        let cb = cx.read(csr)?;
        let gb = cx.read(groups)?;
        let lb = cx.read(lanes)?;
        let yb = cx.write(y)?;
        cx.host(HostOp::GroupRows {
            csr: cb.name,
            groups: gb.name,
            lanes: lb.name,
            y: yb.name,
            csr_len,
            lane_count,
            rows,
        });
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn patch(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    o_grid: Tensor,
    y: Tensor,
    fold: bool,
) -> Result<(), Error> {
    let clips = table(op, "input", grid)?;
    if table(op, "output", o_grid)? != clips {
        return Err(refuse(
            op,
            "the input and output clip tables name other clip counts",
        ));
    }
    let vol = p[0] * p[1] * p[2];
    let (narrow, wide) = if fold {
        (x.width, y.width)
    } else {
        (y.width, x.width)
    };
    if vol == 0 || wide != narrow * vol || x.dtype != y.dtype {
        return Err(refuse(
            op,
            format!(
                "a patch of {p:?} moves {narrow} channels to {} and was handed {} {:?} into {} {:?}",
                narrow * vol,
                x.width,
                x.dtype,
                y.width,
                y.dtype
            ),
        ));
    }
    let rows = y.rows;
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let gb = cx.read(grid)?;
        let ob = cx.read(o_grid)?;
        let yb = cx.write(y)?;
        cx.host(HostOp::Patch {
            x: xb.name,
            grid: gb.name,
            y: yb.name,
            o_grid: ob.name,
            clips,
            rows,
            channels: narrow,
            p,
            fold,
        });
        Ok(())
    })
}

/// A clip's voxels folded into its token rows by `p` (`conv.patchify`):
/// kernels-cuda's pixel unshuffle onto the token grid.
pub fn patchify(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    tgrid: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    patch(ctx, "conv.patchify", x, grid, p, tgrid, y, true)
}

/// Token rows spread back into their clip's voxels (`conv.unpatchify`):
/// kernels-cuda's pixel shuffle from the token grid.
pub fn unpatchify(
    ctx: &Ctx<'_>,
    x: Tensor,
    tgrid: Tensor,
    p: [u32; 3],
    grid: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    patch(ctx, "conv.unpatchify", x, tgrid, p, grid, y, false)
}

/// `xs` side by side into `y` (`ops::gather::concat_cols`), one host copy an
/// input into its column range.
pub fn concat_cols(ctx: &Ctx<'_>, xs: &[Tensor], y: Tensor) -> Result<(), Error> {
    const OP: &str = "gather.concat_cols";
    let mut at = 0u32;
    let mut spans = Vec::new();
    for x in xs {
        if x.dtype != y.dtype || x.rows < y.rows {
            return Err(refuse(OP, "the inputs share the output's element and rows"));
        }
        spans.push((*x, at));
        at += x.width;
    }
    if at != y.width {
        return Err(refuse(OP, "the inputs' widths sum to the output's"));
    }
    let rows = y.rows;
    for (x, at) in spans {
        ctx.emit(&mut |cx| {
            let xb = cx.read(x)?;
            let yb = cx.write(y)?;
            cx.host(HostOp::ConcatCols {
                x: xb.name,
                y: yb.name,
                rows,
                width: x.width,
                stride: y.width,
                at,
            });
            Ok(())
        })?;
    }
    Ok(())
}
