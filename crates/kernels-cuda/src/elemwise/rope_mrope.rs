//! `RopeMrope`: the multimodal rotary — [`rope`](crate::elemwise::rope)'s
//! partial arm over a position that is a triple. A file of its own beside
//! `rope.rs` since it reads a different stream (`[rows, 3]`) under a
//! different section split, not just a differently-shaped position.
//!
//! Deliberately not here: the fused `qk_rmsnorm_rotate_mrope` the unit next
//! door carries, for trunks that norm their heads. This op is the plain
//! rotation the trace names; both agree on the section formula
//! (transcribed, not shared).

use crate::error::Error;
use dtype::Dtype;

use crate::elemwise::rope::ROTATE_BLOCK;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/rope_mrope.cuh";

/// The axes a multimodal position carries: time, and the patch's row and
/// column in its grid.
pub const AXES: u32 = 3;

/// The 3D rotary, section-split and interleaved. `q` and `k` are rotated in
/// place at their stated head geometry. `positions` is `i32`, one `(t, h, w)`
/// triple per rotated row — a `[rows, 3]` rectangle. `sections` is the
/// checkpoint's own `mrope_section`; a trace constant, so it arrives stated
/// rather than read from device memory.
///
/// `rotary_dim` is the rotated prefix of each head, as in
/// [`rope::partial`](crate::elemwise::rope::partial) — state it equal to
/// `head_dim` for the full rotation.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16; a refusal for a row
/// width that is not a whole number of heads, a rotated prefix wider than the
/// head, a position stream that is not `[rows, 3]` `i32`, or sections whose
/// interleaved prefix does not fit the head's frequency pairs.
pub fn interleaved(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    fire(
        ctx,
        "::pie::elemwise::rope_mrope<::pie::bf16>",
        q,
        k,
        positions,
        sections,
        rotary_dim,
        head_dim,
        theta,
    )
}

/// The tower's rotation: contiguous sections, each restarting the ladder.
/// Same operands and refusals as [`interleaved`], differing only in section
/// layout: pairs `[0, s0)` turn by `t`, `[s0, s0+s1)` by `h`,
/// `[s0+s1, s0+s1+s2)` by `w`, and the `i`-th pair of its block turns at
/// `theta^(-2i / Σsections)`. Both arms pair `(d, d + head_dim/2)`
/// (`rotate_half`); `mrope_interleaved` selects how the sections are handed
/// out, not the pairing.
///
/// # Errors
///
/// As [`interleaved`].
pub fn blocked(
    ctx: &Ctx,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    fire(
        ctx,
        "::pie::elemwise::rope_mrope_blocked<::pie::bf16>",
        q,
        k,
        positions,
        sections,
        rotary_dim,
        head_dim,
        theta,
    )
}

/// The two arms' one body: every refusal, and the launch, with the entry name
/// as the only thing that varies.
#[allow(clippy::too_many_arguments)]
fn fire(
    ctx: &Ctx,
    entry: &'static str,
    q: &mut Tensor,
    k: &mut Tensor,
    positions: Tensor,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_mrope";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` rotates q and k in one element");

    nonzero(OP, "the head width this rotation states", head_dim)?;
    if head_dim % 2 != 0 {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(refuse(
            OP,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front \
                 of is {head_dim}"
            ),
        ));
    }
    let num_q_heads = heads(OP, "query", q.width, head_dim)?;
    let num_kv_heads = heads(OP, "key", k.width, head_dim)?;

    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {:?}, and this rotation reads i32 (t, h, w) triples",
                positions.dtype
            ),
        ));
    }
    if positions.width != AXES || positions.rows != q.rows {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {} x {}, and this rotation reads one (t, h, w) triple \
                 per one of {} rotated rows",
                positions.rows, positions.width, q.rows
            ),
        ));
    }

    // The interleaved prefix is three pairs wide per section step, and it may
    // not run past the head's own frequency pairs — a checkpoint whose
    // sections do not fit is a text to fix, not a rotation to truncate.
    let half = head_dim / 2;
    let stated_pairs: u32 = sections.iter().copied().sum();
    if stated_pairs > half {
        return Err(refuse(
            OP,
            format!(
                "the sections {sections:?} name {stated_pairs} frequency pairs and a \
                 {head_dim}-wide head has {half}"
            ),
        ));
    }

    let rows = nonzero(OP, "rows", q.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, entry).apply(Launch::per_row(rows, ROTATE_BLOCK)),
        &[
            q.arg(),
            k.arg(),
            positions.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, rotary_dim)?.arg(),
            theta.arg(),
            stated(OP, sections[0])?.arg(),
            stated(OP, sections[1])?.arg(),
            stated(OP, sections[2])?.arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}

/// The head count a row's width spells at a stated head width. A zero-wide
/// row is the `k`-shaped absence `rope::partial_q` already uses — zero
/// heads, and the unit reads none.
fn heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}
