//! `RopeMrope`: the multimodal rotary — [`rope`](crate::elemwise::rope)'s
//! partial arm over a position that is a triple (`.wiki/alto/multimodal.md`
//! §2's second op), and this plane's mirror of
//! `kernels_cuda::elemwise::rope_mrope`.
//!
//! A file of its own beside `rope.rs` rather than an arm inside it: the
//! scalar rotations are one op family with one position stream, and this one
//! reads a different stream (`[rows, 3]`) under a different statute (the
//! section split). Sharing the file would have meant one entry with a
//! sometimes-absent section triple, which is the shape the design's "a fourth
//! axis with a fourth fact, not a flag" ruling says not to build.
//!
//! **THE ROTATION IS `neox_prop_mb`'S, AND THE FREQUENCY IS SPELLED THE WAY
//! THIS PLANE SPELLS IT.** The CUDA twin computes `powf(theta, -2 * i /
//! head_dim)` from the raw theta; every rope arm here takes `base =
//! log2(theta)` from the entry and computes `exp2(-d * base)` in the shader,
//! which is the same number by the same identity. Transcribing the twin's
//! `powf` instead would have put one arm of this family in a different idiom
//! than its four neighbours for no gain.
//!
//! **THE TAIL IS NOT DISPATCHED RATHER THAN DISPATCHED AND DROPPED.** The
//! twin walks every pair of the head and `continue`s past `rotary_dim / 2`;
//! this plane sizes the grid to the rotated pairs, exactly as
//! [`rope::partial`](crate::elemwise::rope::partial) does. Same rows
//! rotated, same rows left alone.
//!
//! **UNVERIFIED ON DEVICE**, as the whole M2 mirror is: written against
//! `elemwise/rope_neox.metal` on a box with no Metal compiler. What the tests
//! below pin is the host half.

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
    rotary_dim: u32,
    heads: u32,
    sections: [u32; AXES as usize],
) -> Result<(), Error> {
    if heads == 0 {
        return Ok(());
    }
    let lanes = [rotary_dim / 2, heads, x.rows];
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

/// **THE 3D ROTARY, SECTION-SPLIT AND INTERLEAVED.**
///
/// `q` and `k` are rotated in place at their stated head geometry.
/// `positions` is `i32`, one `(t, h, w)` triple per rotated row — a
/// `[rows, 3]` rectangle, which is the mrope shape of the scalar entries'
/// `[rows, 1]` position stream. `sections` is the checkpoint's own
/// `mrope_section` (qwen36 states `[11, 11, 10]`); it is a trace constant, so
/// it arrives stated rather than read from device memory.
///
/// `rotary_dim` is the rotated prefix of each head, as in
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
    if rotary_dim % 2 != 0 {
        return Err(refuse(
            OP,
            format!("the rotated prefix {rotary_dim} is not a whole number of pairs"),
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

    nonzero(OP, "rows", q.rows)?;
    let base = theta.log2();
    rotate(
        ctx,
        OP,
        entry,
        q,
        positions,
        base,
        head_dim,
        rotary_dim,
        num_q_heads,
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
        rotary_dim,
        num_kv_heads,
        sections,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::ArgValue;
    use crate::probe::Probe;

    /// qwen36's stated sections.
    const QWEN36: [u32; 3] = [11, 11, 10];

    fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::Bf16)
    }

    fn triples(rows: u32) -> Tensor {
        Tensor::new(9, rows, AXES, Dtype::I32)
    }

    #[test]
    fn the_rotation_fires_once_per_tensor_over_the_rotated_pairs() {
        let probe = Probe::default();
        let (rows, head_dim) = (40u32, 64u32);
        interleaved(
            &probe,
            bf16(1, rows, 16 * head_dim),
            bf16(2, rows, 4 * head_dim),
            triples(rows),
            QWEN36,
            head_dim,
            head_dim,
            1_000_000.0,
        )
        .expect("the rotation enqueues");
        let fires = probe.fires();
        assert_eq!(fires.len(), 2, "q and k are two launches on this plane");
        assert_eq!(fires[0].0.entrypoint, "rope_mrope_interleaved_bfloat16");
        assert_eq!(fires[0].0.file, FILE);
        // One thread per rotated pair, per head, per row.
        assert_eq!(fires[0].0.lanes, [head_dim / 2, 16, rows]);
        assert_eq!(fires[0].0.group, [head_dim / 2, 1, 1]);
        assert_eq!(fires[1].0.lanes, [head_dim / 2, 4, rows]);
        // The sections arrive stated, all three of them.
        assert_eq!(fires[0].1[4], ArgValue::I32(11));
        assert_eq!(fires[0].1[5], ArgValue::I32(11));
        assert_eq!(fires[0].1[6], ArgValue::I32(10));
        // The shader takes log2(theta), as every arm of this family does.
        assert_eq!(fires[0].1[2], ArgValue::F32(1_000_000.0f32.log2()));
    }

    #[test]
    fn a_partial_prefix_shrinks_the_grid_rather_than_branching_in_the_shader() {
        let probe = Probe::default();
        interleaved(
            &probe,
            bf16(1, 8, 4 * 128),
            bf16(2, 8, 128),
            triples(8),
            [10, 10, 10],
            64,
            128,
            10_000.0,
        )
        .expect("a 64-wide prefix of a 128-wide head enqueues");
        let fires = probe.fires();
        assert_eq!(fires[0].0.lanes, [32, 4, 8]);
        // The head width the shader pairs across is still the whole head.
        assert_eq!(fires[0].1[3], ArgValue::I32(128));
    }

    /// The `k`-shaped absence: zero heads is a launch not made, not a
    /// zero-extent grid the driver has to mean something by.
    #[test]
    fn a_zero_wide_key_row_fires_once() {
        let probe = Probe::default();
        interleaved(
            &probe,
            bf16(1, 8, 2 * 64),
            bf16(2, 8, 0),
            triples(8),
            QWEN36,
            64,
            64,
            10_000.0,
        )
        .expect("a query-only rotation enqueues");
        assert_eq!(probe.fires().len(), 1);
    }

    #[test]
    fn a_scalar_position_stream_is_refused_by_name() {
        let probe = Probe::default();
        let why = interleaved(
            &probe,
            bf16(1, 8, 64),
            bf16(2, 8, 64),
            Tensor::new(9, 8, 1, Dtype::I32),
            QWEN36,
            64,
            64,
            10_000.0,
        )
        .expect_err("this rotation reads triples, not scalars");
        assert!(format!("{why}").contains("(t, h, w)"), "{why}");

        let wrong_dtype = interleaved(
            &probe,
            bf16(1, 8, 64),
            bf16(2, 8, 64),
            Tensor::new(9, 8, AXES, Dtype::U32),
            QWEN36,
            64,
            64,
            10_000.0,
        )
        .expect_err("the position stream is i32");
        assert!(format!("{wrong_dtype}").contains("i32"), "{wrong_dtype}");
        assert!(probe.fires().is_empty(), "a refused rotation launched anyway");
    }

    /// [11, 11, 10] is 32 pairs, which a 64-wide head has and a 32-wide one
    /// does not.
    #[test]
    fn sections_wider_than_the_heads_pairs_are_refused_by_name() {
        let probe = Probe::default();
        let why = interleaved(
            &probe,
            bf16(1, 8, 32),
            bf16(2, 8, 32),
            triples(8),
            QWEN36,
            32,
            32,
            10_000.0,
        )
        .expect_err("32 sectioned pairs do not fit 16");
        assert!(format!("{why}").contains("frequency pairs"), "{why}");
    }

    #[test]
    fn a_prefix_wider_than_the_head_is_refused_by_name() {
        let probe = Probe::default();
        let why = interleaved(
            &probe,
            bf16(1, 8, 64),
            bf16(2, 8, 64),
            triples(8),
            QWEN36,
            96,
            64,
            10_000.0,
        )
        .expect_err("a 96-wide prefix of a 64-wide head is not a prefix");
        assert!(format!("{why}").contains("rotated prefix"), "{why}");
    }
}
