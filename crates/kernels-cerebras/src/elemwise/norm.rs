//! Row normalisation and residual sums.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect, tile_split};
use crate::error::{Error, refuse};
use crate::program::Shard;
use crate::tensor::Tensor;

/// `y = rmsnorm(x) * weight`: each row of `x` scaled by
/// `1 / sqrt(mean(x²) + eps)`, then by the gain.
pub fn rmsnorm(ctx: &Ctx<'_>, x: Tensor, weight: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm",
        x,
        weight,
        eps,
        y,
        x.width,
        false,
    )
}

/// `rmsnorm` with gain `1 + weight`.
pub fn rmsnorm_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_plus_one",
        x,
        weight,
        eps,
        y,
        x.width,
        true,
    )
}

/// `rmsnorm_plus_one` over every `head_dim`-wide run of a row, with a
/// `head_dim`-wide weight.
pub fn rmsnorm_per_head_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_per_head_plus_one",
        x,
        weight,
        eps,
        y,
        head_dim,
        true,
    )
}

/// `rmsnorm_plus_one` over every `group`-wide run of a row, the gain a
/// bank as wide as the row (one gain per column, `1 + weight`).
pub fn rmsnorm_grouped_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    group: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_grouped_plus_one";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, weight, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    let width = x.width;
    if group == 0 || !width.is_multiple_of(group) {
        return Err(refuse(
            OP,
            format!("x is {width} wide, normalised over runs of {group}"),
        ));
    }
    if weight.elements() != u64::from(width) {
        return Err(refuse(
            OP,
            format!(
                "the weight bank holds {} elements and the row it gains is {width} wide",
                weight.elements()
            ),
        ));
    }
    if y.width != width || y.rows != x.rows {
        return Err(refuse(
            OP,
            format!("y is {}x{}, x is {}x{}", y.rows, y.width, x.rows, width),
        ));
    }
    // Rows past a PE spread over row groups, a row past a PE over whole
    // groups, the gain bank split with the columns.
    let (rg, cg) = tile_split(OP, y.rows, width, 2 * u64::from(width), 1, 0, group)?;
    let pes = rg * cg;
    let (rows, width_local) = (y.rows / rg, width / cg);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let wb = cx.read(weight)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            let tile = Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            };
            cx.shard(&xb, tile);
            cx.shard(&yb, tile);
            if cg > 1 {
                // The bank's block goes by its shape: a column `[width][1]`
                // by rows, a row `[1][width]` by columns; every row group
                // holds it.
                let gain = if wb.rows == 1 {
                    Shard::ColsBy {
                        parts: cg,
                        period: 1,
                    }
                } else {
                    Shard::RowsBy {
                        parts: cg,
                        period: 1,
                    }
                };
                cx.shard(&wb, gain);
            }
        }
        cx.library("k_rmsnorm_banked");
        cx.call(
            "k_rmsnorm_banked",
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(wb),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(width_local)),
                Arg::Int(i64::from(group)),
                Arg::Float(eps),
                Arg::Bool(true),
            ],
        );
        Ok(())
    })
}

fn rms_row(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: Tensor,
    axis: u32,
    plus_one: bool,
) -> Result<(), Error> {
    expect(op, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, y, &[Dtype::Bf16, Dtype::F32])?;
    if axis == 0 || !x.width.is_multiple_of(axis) {
        return Err(refuse(
            op,
            format!("x is {} wide, normalised over runs of {axis}", x.width),
        ));
    }
    if weight.elements() != u64::from(axis) {
        return Err(refuse(
            op,
            format!(
                "weight holds {} elements for runs of {axis}",
                weight.elements()
            ),
        ));
    }
    if y.width != x.width || y.rows > x.rows {
        return Err(refuse(
            op,
            format!("y is {}x{}, x is {}x{}", y.rows, y.width, x.rows, x.width),
        ));
    }
    let width = x.width;
    // Rows past a PE spread over row groups; a row past a PE over column
    // groups too: per run when the run is a head (the gain rides whole),
    // else by columns of the one run, the gain split with them and the
    // sum of squares gathered from every block first.
    let per_head = axis < width;
    let (rg, cg) = if x.rows == y.rows {
        if per_head {
            tile_split(
                op,
                y.rows,
                width,
                2 * u64::from(width),
                0,
                weight.elements(),
                axis,
            )?
        } else {
            tile_split(op, y.rows, width, 2 * u64::from(width), 1, 0, 1)?
        }
    } else {
        (1, 1)
    };
    let (rows_local, width_local) = (y.rows / rg, width / cg);
    let pes = rg * cg;
    if per_head || cg == 1 {
        let runs = rows_local * (width_local / axis);
        let axis_local = if per_head { axis } else { width_local };
        return ctx.emit(&mut |cx| {
            let xb = cx.read(x)?;
            let wb = cx.read(weight)?;
            let yb = cx.write(y)?;
            if pes > 1 {
                cx.over(pes);
                let tile = Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 1,
                };
                cx.shard(&xb, tile);
                cx.shard(&yb, tile);
            }
            cx.library("k_rmsnorm");
            cx.call(
                "k_rmsnorm",
                vec![
                    Arg::Ptr(xb),
                    Arg::Ptr(wb),
                    Arg::Ptr(yb),
                    Arg::Int(i64::from(runs)),
                    Arg::Int(i64::from(axis_local)),
                    Arg::Float(eps),
                    Arg::Bool(plus_one),
                ],
            );
            Ok(())
        });
    }
    // Two phases: each block's sums of squares, then the scale; the sums
    // buffer is declared by name in both (the second declare finds it).
    let ss_name = format!("ss_{}_{}", x.buf, y.buf);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let ss = cx
            .program()
            .declare(&ss_name, Dtype::F32, y.rows, cg, true)?;
        cx.over(pes);
        let tile = Shard::Tile {
            rows: rg,
            cols: cg,
            segments: 1,
        };
        cx.shard(&xb, tile);
        cx.shard(&ss, tile);
        cx.library("k_rowsq");
        cx.call(
            "k_rowsq",
            vec![Arg::Ptr(xb), Arg::Ptr(ss), Arg::Int(i64::from(rows_local)), Arg::Int(i64::from(width_local))],
        );
        Ok(())
    })?;
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let wb = cx.read(weight)?;
        let yb = cx.write(y)?;
        let ss = cx.program().declare(&ss_name, Dtype::F32, y.rows, cg, true)?;
        cx.over(pes);
        let tile = Shard::Tile {
            rows: rg,
            cols: cg,
            segments: 1,
        };
        cx.shard(&xb, tile);
        cx.shard(&yb, tile);
        // The gain's block goes by its shape: a column `[width][1]` by rows,
        // a row `[1][width]` by columns; every row group holds it.
        let gain = if wb.rows == 1 {
            Shard::ColsBy { parts: cg, period: 1 }
        } else {
            Shard::RowsBy { parts: cg, period: 1 }
        };
        cx.shard(&wb, gain);
        cx.shard(&ss, Shard::RowsBy { parts: rg, period: cg });
        cx.library("k_rmsnorm_scaled");
        cx.call(
            "k_rmsnorm_scaled",
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(wb),
                Arg::Ptr(ss),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
                Arg::Int(i64::from(cg)),
                Arg::Int(i64::from(width)),
                Arg::Float(eps),
                Arg::Bool(plus_one),
            ],
        );
        Ok(())
    })
}

pub fn rmsnorm_gated(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    sigmoid_gate: bool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated";
    expect(OP, x, &[Dtype::F32, Dtype::Bf16])?;
    expect(OP, gate, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, weight, &[Dtype::F32, Dtype::Bf16])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if head_dim == 0
        || !x.width.is_multiple_of(head_dim)
        || weight.elements() != u64::from(head_dim)
    {
        return Err(refuse(
            OP,
            format!(
                "x is {} wide, weight holds {}, head_dim {head_dim}",
                x.width,
                weight.elements()
            ),
        ));
    }
    if gate.width != x.width || gate.rows < y.rows || y.width != x.width || y.rows > x.rows {
        return Err(refuse(
            OP,
            format!(
                "x {}x{}, gate {}x{}, y {}x{}",
                x.rows, x.width, gate.rows, gate.width, y.rows, y.width
            ),
        ));
    }
    // Rows past a PE spread over row groups, a row past a PE over its
    // heads; the gain (one head) rides whole.
    let (rg, cg) = if x.rows == y.rows && gate.rows == y.rows {
        tile_split(
            OP,
            y.rows,
            x.width,
            3 * u64::from(x.width),
            0,
            weight.elements(),
            head_dim,
        )?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    let runs = (y.rows / rg) * (x.width / cg / head_dim);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let gb = cx.read(gate)?;
        let wb = cx.read(weight)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            for b in [&xb, &gb, &yb] {
                cx.shard(
                    b,
                    Shard::Tile {
                        rows: rg,
                        cols: cg,
                        segments: 1,
                    },
                );
            }
        }
        cx.library("k_rmsnorm_gated");
        cx.call(
            "k_rmsnorm_gated",
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(gb),
                Arg::Ptr(wb),
                Arg::Ptr(yb),
                Arg::Int(i64::from(runs)),
                Arg::Int(i64::from(head_dim)),
                Arg::Float(eps),
                Arg::Bool(sigmoid_gate),
            ],
        );
        Ok(())
    })
}

pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if x.width != y.width || x.rows > y.rows {
        return Err(refuse(
            OP,
            format!("x is {}x{}, y is {}x{}", x.rows, x.width, y.rows, y.width),
        ));
    }
    let (rg, cg) = if x.rows == y.rows {
        tile_split(OP, x.rows, x.width, 2 * u64::from(x.width), 0, 0, 1)?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    let n = x.elements() / u64::from(pes);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            let tile = Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            };
            cx.shard(&xb, tile);
            cx.shard(&yb, tile);
        }
        cx.library("k_residual_add");
        cx.call("k_residual_add", vec![Arg::Ptr(xb), Arg::Ptr(yb), Arg::Int(n as i64)]);
        Ok(())
    })
}

/// `out += bias` per column.
pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add_bias";
    expect(OP, bias, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, out, &[Dtype::Bf16, Dtype::F32])?;
    if bias.elements() != u64::from(out.width) {
        return Err(refuse(
            OP,
            format!(
                "a {}-element bias over {} columns",
                bias.elements(),
                out.width
            ),
        ));
    }
    let (rows, width) = (out.rows, out.width);
    let (rg, cg) = tile_split(OP, rows, width, u64::from(width), 1, 0, 1)?;
    let pes = rg * cg;
    ctx.emit(&mut |cx| {
        let bb = cx.read(bias)?;
        cx.read(out)?;
        let ob = cx.write(out)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(
                &ob,
                Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 1,
                },
            );
            cx.shard(&bb, column_split(&bb, cg));
        }
        cx.library("k_add_bias");
        cx.call(
            "k_add_bias",
            vec![Arg::Ptr(ob), Arg::Ptr(bb), Arg::Int(i64::from(rows / rg)), Arg::Int(i64::from(width / cg))],
        );
        Ok(())
    })
}

/// How a per-column vector (`[width][1]` or `[1][width]`) splits with `cg`
/// column groups, every row group holding it.
fn column_split(b: &crate::program::Buf, cg: u32) -> Shard {
    if b.rows == 1 {
        Shard::ColsBy {
            parts: cg,
            period: 1,
        }
    } else {
        Shard::RowsBy {
            parts: cg,
            period: 1,
        }
    }
}

/// `y = (x - mean) / sqrt(var + eps) · weight + bias` per row.
pub fn layernorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, weight, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, bias, &[Dtype::Bf16, Dtype::F32])?;
    if weight.elements() != u64::from(x.width) || bias.elements() != u64::from(x.width) {
        return Err(refuse(
            OP,
            format!(
                "weight holds {} and bias {} for rows of {}",
                weight.elements(),
                bias.elements(),
                x.width
            ),
        ));
    }
    if y.width != x.width || y.rows > x.rows {
        return Err(refuse(
            OP,
            format!("y is {}x{}, x is {}x{}", y.rows, y.width, x.rows, x.width),
        ));
    }
    let width = x.width;
    // Rows over row groups; a row past a PE over column groups too, the
    // row's sum and sum of squares gathered from every block first.
    let (rg, cg) = if x.rows == y.rows {
        tile_split(OP, y.rows, width, 2 * u64::from(width), 2, 0, 1)?
    } else {
        (1, 1)
    };
    let (rows_local, width_local) = (y.rows / rg, width / cg);
    let pes = rg * cg;
    if cg == 1 {
        return ctx.emit(&mut |cx| {
            let xb = cx.read(x)?;
            let wb = cx.read(weight)?;
            let bb = cx.read(bias)?;
            let yb = cx.write(y)?;
            if pes > 1 {
                cx.over(pes);
                let tile = Shard::Tile {
                    rows: rg,
                    cols: 1,
                    segments: 1,
                };
                cx.shard(&xb, tile);
                cx.shard(&yb, tile);
            }
            cx.library("k_layernorm");
            cx.call(
                "k_layernorm",
                vec![
                    Arg::Ptr(xb),
                    Arg::Ptr(wb),
                    Arg::Ptr(bb),
                    Arg::Ptr(yb),
                    Arg::Int(i64::from(rows_local)),
                    Arg::Int(i64::from(width)),
                    Arg::Float(eps),
                ],
            );
            Ok(())
        });
    }
    let st_name = format!("st_{}_{}", x.buf, y.buf);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let st = cx
            .program()
            .declare(&st_name, Dtype::F32, y.rows, 2 * cg, true)?;
        cx.over(pes);
        cx.shard(
            &xb,
            Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            },
        );
        cx.shard(
            &st,
            Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 2,
            },
        );
        cx.library("k_rowstats");
        cx.call(
            "k_rowstats",
            vec![Arg::Ptr(xb), Arg::Ptr(st), Arg::Int(i64::from(rows_local)), Arg::Int(i64::from(width_local))],
        );
        Ok(())
    })?;
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let wb = cx.read(weight)?;
        let bb = cx.read(bias)?;
        let yb = cx.write(y)?;
        let st = cx.program().declare(&st_name, Dtype::F32, y.rows, 2 * cg, true)?;
        cx.over(pes);
        let tile = Shard::Tile {
            rows: rg,
            cols: cg,
            segments: 1,
        };
        cx.shard(&xb, tile);
        cx.shard(&yb, tile);
        cx.shard(&wb, column_split(&wb, cg));
        cx.shard(&bb, column_split(&bb, cg));
        cx.shard(&st, Shard::RowsBy { parts: rg, period: cg });
        cx.library("k_layernorm_scaled");
        cx.call(
            "k_layernorm_scaled",
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(wb),
                Arg::Ptr(bb),
                Arg::Ptr(st),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
                Arg::Int(i64::from(cg)),
                Arg::Int(i64::from(width)),
                Arg::Float(eps),
            ],
        );
        Ok(())
    })
}
