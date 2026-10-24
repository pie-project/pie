//! Gated feed-forward activations.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::Shard;
use crate::plan::Footprint;
use crate::tensor::Tensor;

pub fn swiglu(ctx: &Ctx<'_>, packed: Tensor, intermediate: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.glu";
    expect(OP, packed, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if packed.width != 2 * intermediate || y.width != intermediate || y.rows > packed.rows {
        return Err(refuse(
            OP,
            format!(
                "packed is {}x{}, y is {}x{}, intermediate is {intermediate}",
                packed.rows, packed.width, y.rows, y.width
            ),
        ));
    }
    // Rows past a PE spread over row groups, a row past a PE over column
    // groups: the same block of `g` and of `u` lands on one PE.
    let (rg, cg) = if packed.rows == y.rows {
        ctx.plan().tile(
            OP,
            Footprint {
                rows: y.rows,
                width: intermediate,
                per_row: u64::from(packed.width + y.width),
                per_col: 0,
                whole: 0,
                merge: 0,
                unit: 1,
            },
        )?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    let rows = y.rows / rg;
    let inter = intermediate / cg;
    ctx.emit(&mut |cx| {
        let pb = cx.read(packed)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(
                &pb,
                Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 2,
                },
            );
            cx.shard(
                &yb,
                Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 1,
                },
            );
        }
        cx.call(
            Kernel::Swiglu,
            vec![
                Arg::Ptr(pb),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(inter)),
            ],
        );
        Ok(())
    })
}

/// [`swiglu`] over a gate and an up plane held apart, element for element.
pub fn swiglu_split(ctx: &Ctx<'_>, gate: Tensor, up: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.glu";
    expect(OP, gate, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, up, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if (up.rows, up.width) != (gate.rows, gate.width) || y.width != gate.width || y.rows > gate.rows
    {
        return Err(refuse(
            OP,
            format!(
                "gate is {}x{}, up is {}x{}, y is {}x{}",
                gate.rows, gate.width, up.rows, up.width, y.rows, y.width
            ),
        ));
    }
    let (rg, cg) = if gate.rows == y.rows {
        ctx.plan().tile(
            OP,
            Footprint {
                rows: y.rows,
                width: gate.width,
                per_row: 3 * u64::from(gate.width),
                per_col: 0,
                whole: 0,
                merge: 0,
                unit: 1,
            },
        )?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    let n = y.elements() / u64::from(pes);
    ctx.emit(&mut |cx| {
        let gb = cx.read(gate)?;
        let ub = cx.read(up)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            let tile = Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            };
            cx.shard(&gb, tile);
            cx.shard(&ub, tile);
            cx.shard(&yb, tile);
        }
        cx.call(
            Kernel::SwigluSplit,
            vec![Arg::Ptr(gb), Arg::Ptr(ub), Arg::Ptr(yb), Arg::Int(n as i64)],
        );
        Ok(())
    })
}

/// `y = ½x(1 + tanh(√(2/π)(x + 0.044715x³)))`, element for element.
pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.gelu_tanh";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if y.width != x.width || y.rows > x.rows {
        return Err(refuse(
            OP,
            format!("y is {}x{}, x is {}x{}", y.rows, y.width, x.rows, x.width),
        ));
    }
    let (rg, cg) = if x.rows == y.rows {
        ctx.plan().tile(
            OP,
            Footprint {
                rows: y.rows,
                width: x.width,
                per_row: 2 * u64::from(x.width),
                per_col: 0,
                whole: 0,
                merge: 0,
                unit: 1,
            },
        )?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    let n = y.elements() / u64::from(pes);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
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
        cx.call(
            Kernel::GeluTanh,
            vec![Arg::Ptr(xb), Arg::Ptr(yb), Arg::Int(n as i64)],
        );
        Ok(())
    })
}
