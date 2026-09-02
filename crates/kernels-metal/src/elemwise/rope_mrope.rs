//! The multimodal rotary: [`rope`](crate::elemwise::rope)'s partial arm over
//! a position that is a `(t, h, w)` triple rather than a scalar, mirroring
//! `kernels_cuda::elemwise::rope_mrope`. A file of its own beside `rope.rs`
//! because this reads a different position stream (`[rows, 3]`) under a
//! different section-split statute. Unverified on device: written against
//! `elemwise/rope_neox.metal` on a box with no Metal compiler.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, head_group, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/rope_mrope.metal";

/// The axes a multimodal position carries: time, and the patch's row and
/// column in its grid.
pub const AXES: u32 = 3;

/// The head count a row's width spells at a stated head width. A zero-wide
/// row is the `k`-shaped absence `rope::partial_q` already uses — zero heads,
/// and this entry fires no launch for it.
fn heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

/// One rotation, over one tensor. The family's shape: `rope_neox.metal`'s
/// arms each take a single tensor, so the entry fires twice rather than
/// handing one grid both rows — which keeps `num_kv_heads` out of the shader
/// entirely.
#[allow(clippy::too_many_arguments)]
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
    if heads == 0 {
        return Ok(());
    }
    let lanes = [pairs, heads, x.rows];
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            x.arg_mut(),
            positions.arg(),
            base.arg(),
            stated(op, head_dim)?.arg(),
            stated(op, sections[0])?.arg(),
            stated(op, sections[1])?.arg(),
            stated(op, sections[2])?.arg(),
        ],
    )
}

/// What both forms check before either fires: the two head counts a
/// `(q, k)` pair spells. One validation for two forms, deliberately — the
/// two arms differ only in shader-side arithmetic, not in what a caller can
/// get wrong about the geometry.
struct Geometry {
    num_q_heads: u32,

    num_kv_heads: u32,
}

#[allow(clippy::too_many_arguments)]
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
    if head_dim % 2 != 0 {
        return Err(refuse(
            op,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(refuse(
            op,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front \
                 of is {head_dim}"
            ),
        ));
    }
    if rotary_dim % 2 != 0 {
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
                "the position stream is {} x {}, and this rotation reads one (t, h, w) triple \
                 per one of {} rotated rows",
                positions.rows, positions.width, q.rows
            ),
        ));
    }

    // Sections may not run past the head's own frequency pairs; a
    // checkpoint whose sections don't fit is refused, not truncated.
    let half = head_dim / 2;
    let stated_pairs: u32 = sections.iter().copied().sum();
    if stated_pairs > half {
        return Err(refuse(
            op,
            format!(
                "the sections {sections:?} name {stated_pairs} frequency pairs and a \
                 {head_dim}-wide head has {half}"
            ),
        ));
    }

    nonzero(op, "rows", q.rows)?;
    Ok(Geometry {
        num_q_heads,
        num_kv_heads,
    })
}

/// The 3D rotary, section-split and interleaved (`MropeForm::Interleaved`).
///
/// `q` and `k` are rotated in place at their stated head geometry.
/// `positions` is `i32`, one `(t, h, w)` triple per rotated row — a
/// `[rows, 3]` rectangle. `sections` is the checkpoint's own `mrope_section`,
/// a trace constant so it arrives stated rather than read from device
/// memory. `rotary_dim` is the rotated prefix of each head, as in
/// [`rope::partial`](crate::elemwise::rope::partial) — state it equal to
/// `head_dim` for the full rotation.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16; a refusal for a row
/// width that is not a whole number of heads, a head with no whole number of
/// rotation pairs, a rotated prefix wider than the head, a position stream
/// that is not `[rows, 3]` `i32`, or sections whose interleaved prefix does
/// not fit the head's frequency pairs.
#[allow(clippy::too_many_arguments)]
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
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "rope_mrope_interleaved_bfloat16" });
    let geom = validate(OP, q, k, positions, sections, rotary_dim, head_dim)?;

    let pairs = rotary_dim / 2;
    let base = theta.log2();
    rotate(
        ctx,
        OP,
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
        OP,
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

/// The tower's rotation: contiguous sections, and each restarts the
/// frequency ladder (`MropeForm::Blocked`). [`interleaved`]'s signature and
/// validation, a different shader: sections are contiguous blocks (pairs
/// `[0, s0)` turn by `t`, `[s0, s0+s1)` by `h`, `[s0+s1, s0+s1+s2)` by `w`;
/// `s0 == 0` spells a two-axis rotation), and each block restarts the
/// frequency ladder at denominator `Σ sections` rather than `head_dim`. The
/// grid is `min(rotary_dim / 2, Σ sections)`.
///
/// # Errors
///
/// As [`interleaved`], plus a refusal for sections that name no pair at all —
/// a rotation of nothing, which the interleaved form spells as a launch over
/// the whole prefix and this one cannot spell at all.
#[allow(clippy::too_many_arguments)]
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
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "rope_mrope_blocked_bfloat16" });
    let geom = validate(OP, q, k, positions, sections, rotary_dim, head_dim)?;

    // A zero total is refused rather than launched at zero extent: the
    // shader divides by it.
    let total: u32 = sections.iter().copied().sum();
    if total == 0 {
        return Err(refuse(
            OP,
            format!(
                "the sections {sections:?} name no frequency pair, and a blocked rotation \
                 divides its ladder by the pairs the sections tile"
            ),
        ));
    }
    let pairs = (rotary_dim / 2).min(total);
    let base = theta.log2();
    rotate(
        ctx,
        OP,
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
        OP,
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

