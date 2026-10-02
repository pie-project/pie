//! Rotary position embeddings.
//!
//! Reference: kernels-cuda `rope_partial` (NeoX pairing `(i, i + r/2)` over the
//! first `rotary_dim` lanes of every head, `inv_freq_i = theta^(-2i/r)`).

use dtype::Dtype;

use crate::csl::Arg;
use crate::csl::f32_lit;
use crate::cx::{Ctx, expect, tile_split};
use crate::error::{Error, refuse};
use crate::program::Shard;
use crate::tensor::Tensor;

/// Rotates the first `rotary_dim` lanes of every `head_dim` head of `q` and
/// `k` in place by the row's position.
pub fn partial(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    rotate(
        ctx,
        "elementwise.rope_partial",
        &[q, k],
        positions,
        rotary_dim,
        head_dim,
        theta,
    )
}

/// [`partial`] over `q` alone.
pub fn partial_q(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    rotate(
        ctx,
        "elementwise.rope_partial_q",
        &[q],
        positions,
        rotary_dim,
        head_dim,
        theta,
    )
}

fn rotate(
    ctx: &Ctx<'_>,
    op: &'static str,
    ts: &[Tensor],
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    expect(op, positions, &[Dtype::I32])?;
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
        return Err(refuse(
            op,
            format!("rotary_dim {rotary_dim} does not fit head_dim {head_dim}"),
        ));
    }
    for t in ts {
        expect(op, *t, &[Dtype::Bf16, Dtype::F32])?;
        if !t.width.is_multiple_of(head_dim) {
            return Err(refuse(
                op,
                format!("width {} is not a multiple of head_dim {head_dim}", t.width),
            ));
        }
        if positions.elements() < u64::from(t.rows) {
            return Err(refuse(
                op,
                format!("{} positions for {} rows", positions.elements(), t.rows),
            ));
        }
    }
    let half = rotary_dim / 2;
    let inv_freq: Vec<String> = (0..half)
        .map(|i| f32_lit(theta.powf(-(2.0 * i as f32) / rotary_dim as f32)))
        .collect();
    let Some(first) = ts.first().copied() else {
        return Ok(());
    };
    // Rows past a PE spread over row groups, a row past a PE over its heads
    // (the positions ride by row); the column groups divide every tensor's
    // heads.
    let same_rows =
        ts.iter().all(|t| t.rows == first.rows) && positions.elements() == u64::from(first.rows);
    let (rg, cg) = if same_rows {
        let heads_gcd = ts.iter().map(|t| t.width / head_dim).fold(0, gcd).max(1);
        let unit = head_dim * (first.width / head_dim).max(1) / heads_gcd;
        let per_row: u64 = ts.iter().map(|t| u64::from(t.width)).sum::<u64>() + 1;
        tile_split(
            op,
            first.rows,
            first.width,
            per_row,
            0,
            u64::from(half),
            unit,
        )?
    } else {
        (1, 1)
    };
    let pes = rg * cg;
    ctx.emit(&mut |cx| {
        let pb = cx.read(positions)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(
                &pb,
                Shard::RowsBy {
                    parts: rg,
                    period: cg,
                },
            );
        }
        debug_assert_eq!(inv_freq.len(), half as usize);
        let table = cx.table("inv_freq", "f32", &inv_freq);
        cx.library("k_rope");
        for t in ts {
            cx.read(*t)?;
            let b = cx.write(*t)?;
            if pes > 1 {
                cx.shard(
                    &b,
                    Shard::Tile {
                        rows: rg,
                        cols: cg,
                        segments: 1,
                    },
                );
            }
            cx.call(
                "k_rope",
                vec![
                    Arg::Ptr(b),
                    Arg::Ptr(pb.clone()),
                    Arg::Scratch(table.clone(), "f32"),
                    Arg::Int(i64::from(t.rows / rg)),
                    Arg::Int(i64::from(t.width / cg)),
                    Arg::Int(i64::from(head_dim)),
                    Arg::Int(i64::from(half)),
                ],
            );
        }
        Ok(())
    })
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}
