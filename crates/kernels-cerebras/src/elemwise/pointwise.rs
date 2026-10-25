//! Plain elementwise arithmetic: `elemwise.binary` and the square root.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::Shard;
use crate::plan::Footprint;
use crate::tensor::Tensor;

/// Discriminants are `k_binary`'s op codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add = 0,
    Mul = 1,
    Div = 2,
    /// Floored, as Python's `%`.
    Mod = 3,
}

pub fn binary(ctx: &Ctx<'_>, op: BinaryOp, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.binary";
    for t in [x, y, z] {
        expect(OP, t, &[Dtype::Bf16, Dtype::F32])?;
    }
    if [x, y].iter().any(|t| t.width != z.width || t.rows < z.rows) {
        return Err(refuse(
            OP,
            format!(
                "x is {}x{}, y is {}x{}, z is {}x{}; this op does not broadcast",
                x.rows, x.width, y.rows, y.width, z.rows, z.width
            ),
        ));
    }
    let (rg, cg) = if x.rows == z.rows && y.rows == z.rows {
        ctx.plan().tile(
            OP,
            Footprint {
                rows: z.rows,
                width: z.width,
                per_row: 3 * u64::from(z.width),
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
    let n = z.elements() / u64::from(pes);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let yb = cx.read(y)?;
        let zb = cx.write(z)?;
        if pes > 1 {
            cx.over(pes);
            let tile = Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            };
            cx.shard(&xb, tile);
            cx.shard(&yb, tile);
            cx.shard(&zb, tile);
        }
        cx.call(
            Kernel::Binary,
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(yb),
                Arg::Ptr(zb),
                Arg::Int(n as i64),
                Arg::Int(op as i64),
            ],
        );
        Ok(())
    })
}

pub fn sqrt(ctx: &Ctx<'_>, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    let (rg, cg) = ctx.plan().tile(
        OP,
        Footprint {
            rows: x.rows,
            width: x.width,
            per_row: u64::from(x.width),
            per_col: 0,
            whole: 0,
            merge: 0,
            unit: 1,
        },
    )?;
    let pes = rg * cg;
    let n = x.elements() / u64::from(pes);
    ctx.emit(&mut |cx| {
        cx.read(x)?;
        let xb = cx.write(x)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(
                &xb,
                Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 1,
                },
            );
        }
        cx.call(Kernel::Sqrt, vec![Arg::Ptr(xb), Arg::Int(n as i64)]);
        Ok(())
    })
}
