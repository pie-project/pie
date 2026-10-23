//! Choosing from rows: top-k and argmax, and the block draft's selector walk.

pub mod selector_walk;

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::{FabricOp, FabricStep, Footprint, Guarded, HostOp, Shard};
use crate::tensor::Tensor;

/// The `k` largest of each row of `x`, largest first: ties go to the lower
/// column, NaN is never picked, and a slot no value fills answers value 0
/// at column 0. Values land f32, indices i32.
pub fn topk(
    ctx: &Ctx<'_>,
    x: Tensor,
    k: u32,
    values: Tensor,
    indices: Tensor,
) -> Result<(), Error> {
    const OP: &str = "sampling.top_k";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, values, &[Dtype::F32])?;
    expect(OP, indices, &[Dtype::I32])?;
    let (rows, width) = (x.rows, x.width);
    if rows == 0 || width == 0 || k == 0 || k > width {
        return Err(refuse(OP, format!("x is {rows}x{width}, k = {k}")));
    }
    if values.rows != rows || values.width != k || indices.rows != rows || indices.width != k {
        return Err(refuse(
            OP,
            format!(
                "the values are {}x{} and the indices {}x{} for {rows} rows of {k}",
                values.rows, values.width, indices.rows, indices.width
            ),
        ));
    }
    // Rows over PEs (a row whole on its PE); a row past a PE ranks on the
    // host.
    let split = ctx
        .tile(
            OP,
            Footprint {
                rows,
                width,
                per_row: u64::from(width) + 2 * u64::from(k),
                per_col: 0,
                whole: 0,
                unit: width,
            },
        )
        .ok();
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let vb = cx.write(values)?;
        let ib = cx.write(indices)?;
        let Some((rg, _)) = split else {
            cx.host(HostOp::TopK {
                x: xb.name,
                values: vb.name,
                indices: ib.name,
                rows,
                width,
                k,
            });
            return Ok(());
        };
        if rg > 1 {
            cx.over(rg);
            for b in [&xb, &vb, &ib] {
                cx.shard(b, Shard::Rows(rg));
            }
        }
        cx.call(
            Kernel::Topk,
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(vb),
                Arg::Ptr(ib),
                Arg::Int(i64::from(rows / rg)),
                Arg::Int(i64::from(width)),
                Arg::Int(i64::from(k)),
            ],
        );
        Ok(())
    })
}

/// Writes the index of each row's largest value into column `column` of the
/// i32 `y`; ties go to the lowest index, NaN is never picked, an all-NaN row
/// answers 0. The rows ranked are the vocabulary-sized logits the host
/// already holds, so the host ranks them.
pub fn argmax(ctx: &Ctx<'_>, x: Tensor, column: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "sampling.argmax";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::I32])?;
    if x.rows == 0 || x.width == 0 {
        return Err(refuse(OP, format!("x is {}x{}", x.rows, x.width)));
    }
    if column >= y.width {
        return Err(refuse(
            OP,
            format!(
                "column {column} is outside the {}-wide plane it writes",
                y.width
            ),
        ));
    }
    if x.rows != y.rows {
        return Err(refuse(
            OP,
            format!("{} rows ranked into {} rows", x.rows, y.rows),
        ));
    }
    // Each column block's PE finds its greatest lane and index (a pair a
    // row); the blocks of a row group (one rectangle row) gather those on
    // the row's first PE, which picks among them into its rows of y.
    let fabric = ctx.fabric().fabric_sum;
    let whole = if fabric {
        crate::linear::gemm::FABRIC_SUM_WORDS
    } else {
        0
    };
    let (rg, cg) = ctx.tile(
        OP,
        Footprint {
            rows: x.rows,
            width: x.width,
            per_row: u64::from(x.width) + 4,
            per_col: 0,
            whole,
            unit: 1,
        },
    )?;
    let (rows_local, width_local) = (x.rows / rg, x.width / cg);
    let pes = rg * cg;
    if cg > 1 && !fabric {
        return argmax_host_merge(ctx, x, column, y, rg, cg);
    }
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        let parts = cx.scratch("am_parts", 2 * u64::from(rows_local));
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
            cx.shard(
                &yb,
                Shard::Roots {
                    rows: rg,
                    cols: 1,
                    depth: cg,
                },
            );
        }
        cx.call(
            Kernel::ArgmaxPart,
            vec![
                Arg::Ptr(xb),
                Arg::Scratch(parts.clone(), "f32"),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
            ],
        );
        if cg == 1 {
            cx.call(
                Kernel::ArgmaxMerge,
                vec![
                    Arg::Scratch(parts.clone(), "f32"),
                    Arg::Ptr(yb),
                    Arg::Int(1),
                    Arg::Int(i64::from(rows_local)),
                    Arg::Int(i64::from(width_local)),
                    Arg::Int(i64::from(column)),
                    Arg::Int(i64::from(y.width)),
                ],
            );
            return Ok(());
        }
        let all = cx.scratch("am_all", 2 * u64::from(rows_local) * u64::from(cg));
        let steps = vec![FabricStep {
            calls: Vec::new(),
            op: FabricOp::Gather {
                send: parts.clone(),
                recv: all.clone(),
                count: 2 * u64::from(rows_local),
            },
        }];
        let finish = vec![Guarded::root(
            Kernel::ArgmaxMerge,
            vec![
                Arg::Scratch(all.clone(), "f32"),
                Arg::Ptr(yb.clone()),
                Arg::Int(i64::from(cg)),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
                Arg::Int(i64::from(column)),
                Arg::Int(i64::from(y.width)),
            ],
        )];
        cx.collectives((cg, rg), steps, finish);
        Ok(())
    })
}

/// The argmax with the blocks' pairs merged on the host (`Fabric::fabric_sum`
/// off).
fn argmax_host_merge(
    ctx: &Ctx<'_>,
    x: Tensor,
    column: u32,
    y: Tensor,
    rg: u32,
    cg: u32,
) -> Result<(), Error> {
    let (rows_local, width_local) = (x.rows / rg, x.width / cg);
    let pes = rg * cg;
    let parts_name = format!("am_{}_{}_{column}", x.buf, y.buf);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let parts = cx.declare(&parts_name, Dtype::F32, x.rows, 2 * cg, true)?;
        if pes > 1 {
            cx.over(pes);
            let tile = Shard::Tile {
                rows: rg,
                cols: cg,
                segments: 1,
            };
            cx.shard(&xb, tile);
            cx.shard(&parts, tile);
        }
        cx.call(
            Kernel::ArgmaxPart,
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(parts),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width_local)),
            ],
        );
        Ok(())
    })?;
    ctx.emit(&mut |cx| {
        let parts = cx.declare(&parts_name, Dtype::F32, x.rows, 2 * cg, true)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        cx.host(HostOp::ArgmaxMerge {
            parts: parts.name,
            y: yb.name,
            rows: x.rows,
            cg,
            block: width_local,
            column,
            y_width: y.width,
        });
        Ok(())
    })
}
