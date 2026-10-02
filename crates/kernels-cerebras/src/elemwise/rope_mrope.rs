//! Multimodal rotary embeddings: each rotated pair reads one of three
//! position axes `(t, h, w)` from an i32 `[rows, 3]` stream. The pairs of
//! a head (their two lanes, their axis, their frequency) are a table built
//! here and carried into the program as constants; the kernel walks it per
//! row and head. Reference: kernels-xla `elemwise/rope_mrope.rs`.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::csl::f32_lit;
use crate::cx::{Ctx, expect, tile_split};
use crate::error::{Error, refuse};
use crate::program::Shard;
use crate::tensor::Tensor;

pub const AXES: u32 = 3;

const OP: &str = "elementwise.rope_mrope";

/// One rotated pair of a head: lanes `lo` and `hi`, position axis `axis`,
/// frequency `freq` (the angle is `position[axis] · freq`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pair {
    pub lo: u32,
    pub hi: u32,
    pub axis: u32,
    pub freq: f32,
}

fn validate(
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
) -> Result<(), Error> {
    expect(OP, positions, &[Dtype::I32])?;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the rotated prefix {rotary_dim} is not a whole number of pairs within the {head_dim}-wide head"
            ),
        ));
    }
    for t in [q, k] {
        expect(OP, t, &[Dtype::Bf16, Dtype::F32])?;
        if t.width == 0 || !t.width.is_multiple_of(head_dim) {
            return Err(refuse(
                OP,
                format!("width {} is not a multiple of head_dim {head_dim}", t.width),
            ));
        }
        if positions.elements() < u64::from(t.rows) * u64::from(AXES) {
            return Err(refuse(
                OP,
                format!(
                    "{} position words for {} rows of {AXES} axes",
                    positions.elements(),
                    t.rows
                ),
            ));
        }
    }
    let stated: u32 = sections.iter().sum();
    if stated > head_dim / 2 {
        return Err(refuse(
            OP,
            format!(
                "the sections {sections:?} name {stated} pairs and a {head_dim}-wide head has {}",
                head_dim / 2
            ),
        ));
    }
    Ok(())
}

fn blocked_pairs(sections: [u32; AXES as usize], rotary_dim: u32) -> Result<u32, Error> {
    let total: u32 = sections.iter().sum();
    if total == 0 {
        return Err(refuse(
            OP,
            format!("the sections {sections:?} name no frequency pair"),
        ));
    }
    Ok((rotary_dim / 2).min(total))
}

fn inv_freq(theta: f32, i: u32, rotary_dim: u32) -> f32 {
    theta.powf(-(2.0 * i as f32) / rotary_dim as f32)
}

/// Pair `i` of the first `r/2` reads axis `h` when `i % 3 == 1 && i < 3·s1`,
/// `w` when `i % 3 == 2 && i < 3·s2`, else `t`; pairs `(i, i + r/2)` at
/// `theta^(-2i/r)`.
pub fn interleaved_pairs(sections: [u32; AXES as usize], rotary_dim: u32, theta: f32) -> Vec<Pair> {
    let half = rotary_dim / 2;
    (0..half)
        .map(|i| {
            let axis = match i % 3 {
                1 if i < 3 * sections[1] => 1,
                2 if i < 3 * sections[2] => 2,
                _ => 0,
            };
            Pair {
                lo: i,
                hi: i + half,
                axis,
                freq: inv_freq(theta, i, rotary_dim),
            }
        })
        .collect()
}

/// Pairs `(i, i + d/2)` for `i < min(r/2, Σs)`, the sections taking pairs
/// in order `t, h, w`, each at `theta^(-2·within/Σs)`.
pub fn blocked_table(
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<Vec<Pair>, Error> {
    let pairs = blocked_pairs(sections, rotary_dim)?;
    let total: u32 = sections.iter().sum();
    let half = head_dim / 2;
    let (s0, s1) = (sections[0], sections[1]);
    Ok((0..pairs)
        .map(|i| {
            let (axis, within) = if i < s0 {
                (0, i)
            } else if i < s0 + s1 {
                (1, i - s0)
            } else {
                (2, i - s0 - s1)
            };
            Pair {
                lo: i,
                hi: i + half,
                axis,
                freq: inv_freq(theta, within, total),
            }
        })
        .collect())
}

/// Each section owns `2·s` contiguous channels, rotated as its own neox half
/// pair `(before·2 + j, before·2 + j + s)` at `theta^(-j/s)`.
pub fn split_table(
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    theta: f32,
) -> Result<Vec<Pair>, Error> {
    let pairs = blocked_pairs(sections, rotary_dim)?;
    let (s0, s1) = (sections[0], sections[1]);
    let mut out = Vec::with_capacity(pairs as usize);
    for i in 0..pairs {
        let (axis, within, before) = if i < s0 {
            (0, i, 0)
        } else if i < s0 + s1 {
            (1, i - s0, s0)
        } else {
            (2, i - s0 - s1, s0 + s1)
        };
        let width = sections[axis as usize];
        if width == 0 {
            continue;
        }
        let lo = 2 * before + within;
        out.push(Pair {
            lo,
            hi: lo + width,
            axis,
            freq: theta.powf(-(within as f32) / width as f32),
        });
    }
    Ok(out)
}

pub fn interleaved(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    validate(q, k, positions, sections, rotary_dim, head_dim)?;
    rotate(
        ctx,
        &[q, k],
        positions,
        head_dim,
        &interleaved_pairs(sections, rotary_dim, theta),
    )
}

pub fn blocked(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    validate(q, k, positions, sections, rotary_dim, head_dim)?;
    let table = blocked_table(sections, rotary_dim, head_dim, theta)?;
    rotate(ctx, &[q, k], positions, head_dim, &table)
}

pub fn split(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    validate(q, k, positions, sections, rotary_dim, head_dim)?;
    let table = split_table(sections, rotary_dim, theta)?;
    rotate(ctx, &[q, k], positions, head_dim, &table)
}

/// Rotates every head of `ts` in place by the table, each row reading its
/// three positions.
fn rotate(
    ctx: &Ctx<'_>,
    ts: &[Tensor],
    positions: Tensor,
    head_dim: u32,
    table: &[Pair],
) -> Result<(), Error> {
    let Some(first) = ts.first().copied() else {
        return Ok(());
    };
    let n = table.len() as u32;
    // Rows past a PE spread over row groups, a row past a PE over its heads
    // (the positions ride by row, three a row).
    let same_rows = ts.iter().all(|t| t.rows == first.rows)
        && positions.elements() == u64::from(first.rows) * u64::from(AXES);
    let (rg, cg) = if same_rows {
        let heads_gcd = ts.iter().map(|t| t.width / head_dim).fold(0, gcd).max(1);
        let unit = head_dim * (first.width / head_dim).max(1) / heads_gcd;
        let per_row: u64 = ts.iter().map(|t| u64::from(t.width)).sum::<u64>() + u64::from(AXES);
        tile_split(
            OP,
            first.rows,
            first.width,
            per_row,
            0,
            4 * u64::from(n),
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
            cx.shard(&pb, Shard::RowsBy { parts: rg, period: cg });
        }
        let list = |f: &dyn Fn(&Pair) -> String| table.iter().map(f).collect::<Vec<String>>();
        let lo = cx.table("mrope_lo", "i32", &list(&|p| p.lo.to_string()));
        let hi = cx.table("mrope_hi", "i32", &list(&|p| p.hi.to_string()));
        let axis = cx.table("mrope_axis", "i32", &list(&|p| p.axis.to_string()));
        let freq = cx.table("mrope_freq", "f32", &list(&|p| f32_lit(p.freq)));
        cx.library("k_rope_table");
        for t in ts {
            cx.read(*t)?;
            let b = cx.write(*t)?;
            if pes > 1 {
                cx.shard(&b, Shard::Tile { rows: rg, cols: cg, segments: 1 });
            }
            cx.call(
                "k_rope_table",
                vec![
                    Arg::Ptr(b),
                    Arg::Ptr(pb.clone()),
                    Arg::Scratch(lo.clone(), "i32"),
                    Arg::Scratch(hi.clone(), "i32"),
                    Arg::Scratch(axis.clone(), "i32"),
                    Arg::Scratch(freq.clone(), "f32"),
                    Arg::Int(n as i64),
                    Arg::Int(i64::from(t.rows / rg)),
                    Arg::Int(i64::from(t.width / cg)),
                    Arg::Int(i64::from(head_dim)),
                ],
            );
        }
        Ok(())
    })
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}
