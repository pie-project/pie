//! Gated feed-forward activations.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect, tile_split};
use crate::error::{Error, refuse};
use crate::program::Shard;
use crate::tensor::Tensor;

pub fn swiglu(ctx: &Ctx<'_>, packed: Tensor, intermediate: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu";
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
        tile_split(
            OP,
            y.rows,
            intermediate,
            u64::from(packed.width + y.width),
            0,
            0,
            1,
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
        cx.library("k_swiglu");
        cx.call(
            "k_swiglu",
            vec![Arg::Ptr(pb), Arg::Ptr(yb), Arg::Int(i64::from(rows)), Arg::Int(i64::from(inter))],
        );
        Ok(())
    })
}

/// `y = ½x(1 + tanh(√(2/π)(x + 0.044715x³)))`, element for element.
pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_gelu_tanh";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if y.width != x.width || y.rows > x.rows {
        return Err(refuse(
            OP,
            format!("y is {}x{}, x is {}x{}", y.rows, y.width, x.rows, x.width),
        ));
    }
    let (rg, cg) = if x.rows == y.rows {
        tile_split(OP, y.rows, x.width, 2 * u64::from(x.width), 0, 0, 1)?
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
        cx.library("k_gelu_tanh");
        cx.call("k_gelu_tanh", vec![Arg::Ptr(xb), Arg::Ptr(yb), Arg::Int(n as i64)]);
        Ok(())
    })
}
