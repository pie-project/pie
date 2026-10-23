//! Sigmoid gates and the scaled silu.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect, tile_split};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::Shard;
use crate::tensor::Tensor;

/// `x = silu(s · x)`, in place.
pub fn silu_scaled(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.silu_scaled";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    let (rg, cg) = tile_split(OP, x.rows, x.width, u64::from(x.width), 0, 0, 1)?;
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
        cx.call(
            Kernel::SiluScaled,
            vec![Arg::Ptr(xb), Arg::Int(n as i64), Arg::Float(s)],
        );
        Ok(())
    })
}

pub fn sigmoid_mul(ctx: &Ctx<'_>, gate: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.gate_sigmoid_mul";
    expect(OP, gate, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    if gate.width != x.width || gate.rows < x.rows {
        return Err(refuse(
            OP,
            format!(
                "gate is {}x{}, x is {}x{}",
                gate.rows, gate.width, x.rows, x.width
            ),
        ));
    }
    let (rg, cg) = if gate.rows == x.rows {
        tile_split(OP, x.rows, x.width, 2 * u64::from(x.width), 0, 0, 1)?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    let n = x.elements() / u64::from(pes);
    ctx.emit(&mut |cx| {
        let gb = cx.read(gate)?;
        cx.read(x)?;
        let xb = cx.write(x)?;
        if pes > 1 {
            cx.over(pes);
            let tile = Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            };
            cx.shard(&gb, tile);
            cx.shard(&xb, tile);
        }
        cx.call(
            Kernel::SigmoidMul,
            vec![Arg::Ptr(xb), Arg::Ptr(gb), Arg::Int(n as i64)],
        );
        Ok(())
    })
}
