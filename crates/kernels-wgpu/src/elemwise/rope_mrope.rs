#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "rope/mrope.wgsl";

const GROUP: u32 = 64;

pub const AXES: u32 = 3;

fn heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

fn rotate(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    x: Tensor,
    positions: Tensor,
    base: f32,
    head_dim: u32,
    pairs: u32,
    heads: u32,
    sections: [u32; AXES as usize],
) -> Result<(), Error> {
    if heads == 0 || pairs == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([pairs, heads, x.rows], [GROUP, 1, 1])),
        &[
            x.arg_mut(),
            positions.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, sections[0])?.arg(),
            stated(op, sections[1])?.arg(),
            stated(op, sections[2])?.arg(),
            stated(op, heads)?.arg(),
            stated(op, pairs)?.arg(),
        ],
    )
}

struct Geometry {
    num_q_heads: u32,
    num_kv_heads: u32,
}

fn validate(
    op: &'static str,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
) -> Result<Geometry, Error> {
    debug_assert_eq!(k.dtype, q.dtype, "`{op}` rotates q and k in one element");
    nonzero(op, "the head width this rotation states", head_dim)?;
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(refuse(
            op,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front of is {head_dim}"
            ),
        ));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the rotated prefix {rotary_dim} is not a whole number of pairs"),
        ));
    }
    let num_q_heads = heads(op, "query", q.width, head_dim)?;
    let num_kv_heads = heads(op, "key", k.width, head_dim)?;
    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the position stream is {:?}, and this rotation reads i32 (t, h, w) triples",
                positions.dtype
            ),
        ));
    }
    if positions.width != AXES || positions.rows != q.rows {
        return Err(refuse(
            op,
            format!(
                "the position stream is {} x {}, and this rotation reads one (t, h, w) triple per \
                 one of {} rotated rows",
                positions.rows, positions.width, q.rows
            ),
        ));
    }
    let half = head_dim / 2;
    let stated_pairs: u32 = sections.iter().copied().sum();
    if stated_pairs > half {
        return Err(refuse(
            op,
            format!(
                "the sections {sections:?} name {stated_pairs} frequency pairs and a {head_dim}-wide head has {half}"
            ),
        ));
    }
    nonzero(op, "rows", q.rows)?;
    Ok(Geometry {
        num_q_heads,
        num_kv_heads,
    })
}

fn blocked_pairs(
    op: &'static str,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
) -> Result<u32, Error> {
    let total: u32 = sections.iter().copied().sum();
    if total == 0 {
        return Err(refuse(
            op,
            format!(
                "the sections {sections:?} name no frequency pair, and a blocked rotation divides its ladder by the pairs the sections tile"
            ),
        ));
    }
    Ok((rotary_dim / 2).min(total))
}

fn both(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    pairs: u32,
    head_dim: u32,
    theta: f32,
    geom: Geometry,
) -> Result<(), Error> {
    let base = theta.log2();
    rotate(
        ctx,
        op,
        entry,
        q,
        positions,
        base,
        head_dim,
        pairs,
        geom.num_q_heads,
        sections,
    )?;
    rotate(
        ctx,
        op,
        entry,
        k,
        positions,
        base,
        head_dim,
        pairs,
        geom.num_kv_heads,
        sections,
    )
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
    const OP: &str = "elementwise.rope_mrope";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "rope_mrope_interleaved_bf16" });
    let geom = validate(OP, q, k, positions, sections, rotary_dim, head_dim)?;
    both(
        ctx,
        OP,
        entry,
        q,
        k,
        positions,
        sections,
        rotary_dim / 2,
        head_dim,
        theta,
        geom,
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
    const OP: &str = "elementwise.rope_mrope";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "rope_mrope_blocked_bf16" });
    let geom = validate(OP, q, k, positions, sections, rotary_dim, head_dim)?;
    let pairs = blocked_pairs(OP, sections, rotary_dim)?;
    both(
        ctx, OP, entry, q, k, positions, sections, pairs, head_dim, theta, geom,
    )
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
    const OP: &str = "elementwise.rope_mrope";
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "rope_mrope_split_bf16" });
    let geom = validate(OP, q, k, positions, sections, rotary_dim, head_dim)?;
    let pairs = blocked_pairs(OP, sections, rotary_dim)?;
    both(
        ctx, OP, entry, q, k, positions, sections, pairs, head_dim, theta, geom,
    )
}
