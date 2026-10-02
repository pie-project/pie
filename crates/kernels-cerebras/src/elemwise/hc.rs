//! Hyper-connections: `M` residual streams of width `H` ride one
//! `[rows, M·H]` row. Reference: kernels-xla `elemwise/hc.rs` (every
//! backend computes the same).
//!
//! A phase tiles its rows over row groups and, when a row is past a PE,
//! its `H` columns over column groups: the wide planes as `M`-segment tiles
//! (`Shard::Tile { segments: M }`), so one PE holds the same columns of
//! every stream and reduces over the streams alone.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect, tile_split};
use crate::error::{Error, refuse};
use crate::program::Shard;
use crate::tensor::Tensor;

const MAX_HC_MULT: u32 = 8;

/// The stream count a `wide`-wide row fans into `hidden`-wide streams.
fn stream_fan(op: &'static str, wide: u32, hidden: u32, streams: u32) -> Result<u32, Error> {
    if hidden == 0 || wide == 0 || !wide.is_multiple_of(hidden) {
        return Err(refuse(
            op,
            format!(
                "the {wide}-wide row is not a whole number of {hidden}-wide hyper-connection streams"
            ),
        ));
    }
    let fan = wide / hidden;
    if fan > MAX_HC_MULT {
        return Err(refuse(
            op,
            format!("the stream count is {fan}, above the {MAX_HC_MULT} the mixers take"),
        ));
    }
    if fan != streams {
        return Err(refuse(
            op,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    Ok(fan)
}

fn tile(rows: u32, cols: u32, segments: u32) -> Shard {
    Shard::Tile {
        rows,
        cols,
        segments,
    }
}

/// `y[n, s·H + h] = x[n, h]` for each of `streams` streams.
pub fn expand(ctx: &Ctx<'_>, x: Tensor, streams: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_expand";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    let m = stream_fan(OP, y.width, x.width, streams)?;
    if y.rows != x.rows {
        return Err(refuse(OP, "the expansion lands one wide row per row"));
    }
    let h = x.width;
    let (rg, cg) = tile_split(OP, x.rows, h, u64::from(h + y.width), 0, 0, 1)?;
    let pes = rg * cg;
    let (rows, h_local) = (x.rows / rg, h / cg);
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(&xb, tile(rg, cg, 1));
            cx.shard(&yb, tile(rg, cg, m));
        }
        cx.library("k_hc_expand");
        cx.call(
            "k_hc_expand",
            vec![Arg::Ptr(xb), Arg::Ptr(yb), Arg::Int(i64::from(rows)), Arg::Int(i64::from(h_local)), Arg::Int(i64::from(m))],
        );
        Ok(())
    })
}

/// `y[n, h] = (1/M) Σ_s normed[n, s, h] · σ(gates[n, s, h])`.
pub fn mix(
    ctx: &Ctx<'_>,
    gates: Tensor,
    normed: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_mix";
    expect(OP, gates, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, normed, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    let m = stream_fan(OP, normed.width, y.width, streams)?;
    if gates.width != normed.width || gates.rows != normed.rows || y.rows != normed.rows {
        return Err(refuse(
            OP,
            "the gates ride the normed rectangle element for element",
        ));
    }
    let h = y.width;
    let (rg, cg) = tile_split(OP, y.rows, h, u64::from(2 * normed.width + h), 0, 0, 1)?;
    let pes = rg * cg;
    let (rows, h_local) = (y.rows / rg, h / cg);
    ctx.emit(&mut |cx| {
        let gb = cx.read(gates)?;
        let vb = cx.read(normed)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(&gb, tile(rg, cg, m));
            cx.shard(&vb, tile(rg, cg, m));
            cx.shard(&yb, tile(rg, cg, 1));
        }
        cx.library("k_hc_mix");
        cx.call(
            "k_hc_mix",
            vec![
                Arg::Ptr(gb),
                Arg::Ptr(vb),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(h_local)),
                Arg::Int(i64::from(m)),
            ],
        );
        Ok(())
    })
}

/// `hyper[n, s, h] += 2σ(gates[n, s] / M) · o[n, h]`, in place.
pub fn inject(
    ctx: &Ctx<'_>,
    o: Tensor,
    gates: Tensor,
    streams: u32,
    hyper: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_inject";
    expect(OP, o, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, gates, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, hyper, &[Dtype::Bf16, Dtype::F32])?;
    let m = stream_fan(OP, hyper.width, o.width, streams)?;
    if gates.width != m || o.rows != hyper.rows || gates.rows < hyper.rows {
        return Err(refuse(
            OP,
            "one gate logit per stream, one output row per wide row",
        ));
    }
    let h = o.width;
    // The gates split by rows alone, every column group holding its row
    // group's; a gate plane taller than the rows rides whole on one PE.
    let (rg, cg) = if gates.rows == hyper.rows {
        tile_split(OP, hyper.rows, h, u64::from(hyper.width + h), 0, 0, 1)?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    let (rows, h_local) = (hyper.rows / rg, h / cg);
    ctx.emit(&mut |cx| {
        let ob = cx.read(o)?;
        let gb = cx.read(gates)?;
        cx.read(hyper)?;
        let hb = cx.write(hyper)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(&ob, tile(rg, cg, 1));
            cx.shard(&hb, tile(rg, cg, m));
            cx.shard(
                &gb,
                Shard::RowsBy {
                    parts: rg,
                    period: cg,
                },
            );
        }
        cx.library("k_hc_inject");
        cx.call(
            "k_hc_inject",
            vec![
                Arg::Ptr(ob),
                Arg::Ptr(gb),
                Arg::Ptr(hb),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(h_local)),
                Arg::Int(i64::from(m)),
            ],
        );
        Ok(())
    })
}

/// Per stream, `gate = σ(sign(d)·√max(|d|, 1e-6))` for `d = key·query / √H`
/// (`d = 0` gates by σ(0)); `y[n, s, h] = gate · value[n, h]`.
pub fn ple_gate(
    ctx: &Ctx<'_>,
    key: Tensor,
    query: Tensor,
    value: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.ple_gate";
    for t in [key, query, value, y] {
        expect(OP, t, &[Dtype::Bf16, Dtype::F32])?;
    }
    let m = stream_fan(OP, y.width, value.width, streams)?;
    if key.width != y.width
        || query.width != y.width
        || key.rows != y.rows
        || query.rows != y.rows
        || value.rows != y.rows
    {
        return Err(refuse(
            OP,
            "the key and query ride the stream row they gate, the value one row per row",
        ));
    }
    let h = value.width;
    // Rows past a PE split over row groups; a row past a PE over column
    // groups of every stream, the dot then gathered from the blocks'
    // partials in a second phase (like the wide rmsnorm).
    let (rg, cg) = tile_split(OP, y.rows, h, u64::from(3 * y.width + h), 0, 0, 1)?;
    let pes = rg * cg;
    let (rows, h_local) = (y.rows / rg, h / cg);
    if cg == 1 {
        return ctx.emit(&mut |cx| {
            let kb = cx.read(key)?;
            let qb = cx.read(query)?;
            let vb = cx.read(value)?;
            let yb = cx.write(y)?;
            if rg > 1 {
                cx.over(rg);
                for b in [&kb, &qb, &vb, &yb] {
                    cx.shard(b, Shard::Rows(rg));
                }
            }
            cx.library("k_ple_gate");
            cx.call(
                "k_ple_gate",
                vec![
                    Arg::Ptr(kb),
                    Arg::Ptr(qb),
                    Arg::Ptr(vb),
                    Arg::Ptr(yb),
                    Arg::Int(i64::from(rows)),
                    Arg::Int(i64::from(h)),
                    Arg::Int(i64::from(m)),
                ],
            );
            Ok(())
        });
    }
    // The partial dots: `[rows][M · cg]`, block `c` of stream `s` at `s · cg
    // + c`, declared by name in both phases (the second declare finds it).
    let pd_name = format!("pd_{}_{}", key.buf, y.buf);
    ctx.emit(&mut |cx| {
        let kb = cx.read(key)?;
        let qb = cx.read(query)?;
        let pd = cx
            .program()
            .declare(&pd_name, Dtype::F32, y.rows, m * cg, true)?;
        cx.over(pes);
        cx.shard(&kb, tile(rg, cg, m));
        cx.shard(&qb, tile(rg, cg, m));
        cx.shard(&pd, tile(rg, cg, m));
        cx.library("k_ple_dot");
        cx.call(
            "k_ple_dot",
            vec![
                Arg::Ptr(kb),
                Arg::Ptr(qb),
                Arg::Ptr(pd),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(h_local)),
                Arg::Int(i64::from(m)),
            ],
        );
        Ok(())
    })?;
    ctx.emit(&mut |cx| {
        let vb = cx.read(value)?;
        let yb = cx.write(y)?;
        let pd = cx
            .program()
            .declare(&pd_name, Dtype::F32, y.rows, m * cg, true)?;
        cx.over(pes);
        cx.shard(&vb, tile(rg, cg, 1));
        cx.shard(&yb, tile(rg, cg, m));
        cx.shard(
            &pd,
            Shard::RowsBy {
                parts: rg,
                period: cg,
            },
        );
        cx.library("k_ple_gate_scaled");
        cx.call(
            "k_ple_gate_scaled",
            vec![
                Arg::Ptr(vb),
                Arg::Ptr(pd),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(h_local)),
                Arg::Int(i64::from(m)),
                Arg::Int(i64::from(cg)),
                Arg::Int(i64::from(h)),
            ],
        );
        Ok(())
    })
}
