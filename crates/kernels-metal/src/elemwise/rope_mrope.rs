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

/// What both forms check before either fires, and what they both need after:
/// the two head counts a `(q, k)` pair spells, and the head's own frequency
/// pair count.
///
/// **ONE VALIDATION FOR TWO FORMS**, deliberately. The two arms differ in
/// which pair takes which axis and at what frequency — which is the shader's
/// business — and in nothing a caller can get wrong about the geometry. A
/// second copy of these seven refusals is a second place for them to drift.
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

    // The sections may not run past the head's own frequency pairs — a
    // checkpoint whose sections do not fit is a text to fix, not a rotation
    // to truncate. It is the interleaved prefix's ceiling AND the blocked
    // tiling's: three pairs per section step reaches `Σ sections` either way.
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

/// **THE 3D ROTARY, SECTION-SPLIT AND INTERLEAVED** — the trunk's form
/// (`MropeForm::Interleaved`, one crate up).
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
    let geom = validate(OP, q, k, positions, sections, rotary_dim, head_dim)?;

    // Every pair of the rotated prefix turns — which of the three axes it
    // turns by is the shader's interleaved test, and the pairs above the
    // prefix are the tail this plane does not dispatch.
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

/// **THE TOWER'S ROTATION: CONTIGUOUS SECTIONS, AND EACH RESTARTS THE
/// LADDER** — `MropeForm::Blocked`
/// (`.wiki/alto/multimodal.md` §6.3), and this plane's mirror of
/// `kernels_cuda::elemwise::rope_mrope::blocked`.
///
/// [`interleaved`]'s signature, [`interleaved`]'s validation, and a different
/// shader. Two things differ inside it, and the second is the one nobody
/// would guess:
///
/// * the sections are CONTIGUOUS BLOCKS — pairs `[0, s0)` turn by `t`,
///   `[s0, s0+s1)` by `h`, `[s0+s1, s0+s1+s2)` by `w`. Both towers state
///   `[0, head_dim/4, head_dim/4]`, so they turn by `(h, w)` and read no `t`
///   at all; `s0 == 0` is how a two-axis rotation is spelled here, rather
///   than by a second position shape;
/// * and each block RESTARTS the frequency ladder, at a denominator that is
///   `Σ sections` rather than `head_dim`.
///
/// **WHY THIS IS A SECOND ENTRY AND NOT A FLAG.** A rotation that handed the
/// sections out the other way would answer plausible numbers for the wrong
/// checkpoint — the refusal this retires said exactly that — and the two
/// forms differ in the frequency ladder as well as the split, so one shader
/// with a mode word would be two kernels sharing a register file.
///
/// **THE GRID IS `min(rotary_dim / 2, Σ sections)`.** The twin walks every
/// pair of the head and `continue`s past both bounds; this plane sizes the
/// launch, so a pair past the sections is never a thread. That is exact
/// rather than approximate because the blocks start at pair zero and tile
/// upward: the pairs that turn are a PREFIX, which is the only shape a grid
/// can express.
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

    // **THE SECTIONS ARE THE LADDER'S WIDTH AND THE GRID'S**, which is the
    // one arithmetic this entry does that its twin does not. A zero total is
    // refused rather than launched at zero extent: the shader divides by it.
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

    /// **BOTH TOWERS' SECTIONS**: `[0, head_dim/4, head_dim/4]`, which turns
    /// by `(h, w)` and reads no `t` at all.
    const TOWER: [u32; 3] = [0, 18, 18];

    /// **THE BLOCKED FORM IS THE INTERLEAVED ONE'S LAUNCH AT ANOTHER
    /// POINT**, over a grid the sections size rather than the head.
    #[test]
    fn the_blocked_rotation_launches_over_the_pairs_the_sections_tile() {
        let probe = Probe::default();
        // gemma's wide tower: 16 heads at head_dim 72, so 36 pairs a head and
        // `[0, 18, 18]` tiles all of them.
        let (rows, head_dim) = (256u32, 72u32);
        blocked(
            &probe,
            bf16(1, rows, 16 * head_dim),
            bf16(2, rows, 16 * head_dim),
            triples(rows),
            TOWER,
            head_dim,
            head_dim,
            100.0,
        )
        .expect("the tower's rotation enqueues");
        let fires = probe.fires();
        assert_eq!(fires.len(), 2, "q and k are two launches on this plane");
        assert_eq!(fires[0].0.entrypoint, "rope_mrope_blocked_bfloat16");
        assert_eq!(fires[0].0.file, FILE);
        assert_eq!(fires[0].0.lanes, [36, 16, rows]);
        assert_eq!(fires[0].0.group, [36, 1, 1]);
        // The sections arrive stated, and `s0` is a real zero rather than an
        // absence: the shader reads it to place the first block's boundary.
        assert_eq!(fires[0].1[4], ArgValue::I32(0));
        assert_eq!(fires[0].1[5], ArgValue::I32(18));
        assert_eq!(fires[0].1[6], ArgValue::I32(18));
        assert_eq!(fires[0].1[3], ArgValue::I32(72));
        // `rope_theta = 100.0` on both towers, and this family states log2.
        assert_eq!(fires[0].1[2], ArgValue::F32(100.0f32.log2()));
    }

    /// **THE TWO FORMS ARE ONE VALIDATION AND TWO SHADERS.** Same handles,
    /// same sections, same grid — only the point changes, which is the whole
    /// content of the refusal this retires.
    #[test]
    fn the_two_forms_differ_in_the_point_and_not_the_geometry() {
        let (rows, head_dim) = (8u32, 64u32);
        type Arm = fn(&Ctx<'_>, Tensor, Tensor, Tensor, [u32; 3], u32, u32, f32) -> Result<(), Error>;
        let call = |f: Arm| {
            let probe = Probe::default();
            f(
                &probe,
                bf16(1, rows, 4 * head_dim),
                bf16(2, rows, head_dim),
                triples(rows),
                [0, 16, 16],
                head_dim,
                head_dim,
                10_000.0,
            )
            .expect("the rotation enqueues");
            probe.fires()
        };
        let inter = call(interleaved);
        let block = call(blocked);
        assert_eq!(inter.len(), block.len());
        // `[0, 16, 16]` tiles exactly the 32 pairs a 64-wide head holds, so
        // the two grids coincide here — which is what makes the POINT the
        // only difference the test can be reading.
        assert_eq!(inter[0].0.lanes, block[0].0.lanes);
        assert_eq!(inter[0].0.group, block[0].0.group);
        assert_eq!(inter[0].1, block[0].1);
        assert_ne!(inter[0].0.entrypoint, block[0].0.entrypoint);
    }

    /// **SECTIONS NARROWER THAN THE PREFIX SHRINK THE GRID.** The twin walks
    /// every pair and `continue`s past `total`; this plane never dispatches
    /// them. The blocks tile upward from pair zero, so the pairs that turn
    /// are a PREFIX and a grid can say so exactly.
    #[test]
    fn sections_narrower_than_the_prefix_shrink_the_grid() {
        let probe = Probe::default();
        blocked(
            &probe,
            bf16(1, 8, 128),
            bf16(2, 8, 128),
            triples(8),
            [0, 10, 10],
            128,
            128,
            10_000.0,
        )
        .expect("a 20-pair rotation of a 64-pair head enqueues");
        let fires = probe.fires();
        assert_eq!(fires[0].0.lanes, [20, 1, 8]);
        // The head the shader pairs across is still the whole head — the
        // partner of pair `i` is at `i + head_dim / 2` and not `i + total`.
        assert_eq!(fires[0].1[3], ArgValue::I32(128));
    }

    /// A blocked rotation divides its ladder by the pairs the sections tile,
    /// so sections that tile nothing are refused rather than launched.
    #[test]
    fn sections_that_name_no_pair_are_refused_by_name() {
        let probe = Probe::default();
        let why = blocked(&probe, bf16(1, 8, 64), bf16(2, 8, 64), triples(8), [0, 0, 0], 64, 64, 10_000.0)
            .expect_err("a ladder of no pairs has no denominator");
        assert!(format!("{why}").contains("no frequency pair"), "{why}");
        assert!(probe.fires().is_empty());
    }

    /// The blocked form inherits every one of the interleaved form's
    /// refusals, which is what `validate` being one function means.
    #[test]
    fn the_blocked_form_refuses_what_the_interleaved_one_refuses() {
        let probe = Probe::default();

        let scalar = blocked(
            &probe,
            bf16(1, 8, 64),
            bf16(2, 8, 64),
            Tensor::new(9, 8, 1, Dtype::I32),
            TOWER,
            64,
            64,
            100.0,
        )
        .expect_err("this rotation reads triples, not scalars");
        assert!(format!("{scalar}").contains("(t, h, w)"), "{scalar}");

        let wide = blocked(&probe, bf16(1, 8, 32), bf16(2, 8, 32), triples(8), QWEN36, 32, 32, 100.0)
            .expect_err("32 sectioned pairs do not fit 16");
        assert!(format!("{wide}").contains("frequency pairs"), "{wide}");

        let ragged = blocked(&probe, bf16(1, 8, 70), bf16(2, 8, 72), triples(8), TOWER, 72, 72, 100.0)
            .expect_err("a 70-wide row is not a whole number of 72-wide heads");
        assert!(format!("{ragged}").contains("whole number of"), "{ragged}");

        assert!(probe.fires().is_empty(), "a refused rotation launched anyway");
    }
}
