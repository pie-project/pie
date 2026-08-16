//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.
#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{keys, Ask, Bind, Buf, BufMut, Ctx, Fire, I32s, Param, ParamF32, ParamOr, Routine};
use crate::routine::OutSlot;

/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "neox_decode_bfloat16",
    "neox_freqs_decode_bfloat16",
    "neox_freqs_mb_bfloat16",
    "neox_mb_bfloat16",
    "neox_prop_decode_bfloat16",
    "neox_prop_mb_bfloat16",
    "neox_strided_bfloat16",
];

/// The grid every rotation in this file launches: `[rotary/2, heads, rows]`.
///
/// `neox.slang` is `[numthreads(1, 1, 1)]` and reads its own shape back out of
/// `gl_NumWorkGroups` -- `.x` is the pair count and `.y` is the head count --
/// because the launch already knows both and no scalar carries them. So this
/// grid is not a division of work over a rectangle. It IS the shape, and every
/// extent below is a fact the shader will read rather than a tuning choice.
///
/// `rotary` is the STATEMENT's, not the fire's. gemma-4 rotates a quarter of
/// each full-attention head and all of each sliding one, over the same tensor
/// width, so it cannot come off the rectangle -- under `kernel!` it was
/// `grid_param = Some(3)`, a param index the kernel itself never reads and
/// only the grid does. An `Env` argument is what that sentence becomes.
///
/// # Errors
///
/// [`Refusal::Empty`] for an extent with nothing in it, and
/// [`Refusal::Narrow`] for two shapes this kernel cannot express:
///
/// An ODD `rotary` would halve to a pair count that leaves the last element
/// unrotated -- the shader pairs `i` with `i + pair_half` and a stray element
/// belongs to neither half -- and the result is a rotation that is correct in
/// every dimension but one, at every position but zero.
///
/// A `width` the head does not divide means the head count is not a whole
/// number, and the shader multiplies `n_head * head_dim` to find each row's
/// base. `driver-vulkan`'s `geometry::rope_heads` refuses the same shape with
/// `Ungeometric::Unheaded`; this is that refusal moved to where the fact is.
fn rope_grid(rotary: i32, width: i32, head_dim: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if rotary <= 0 {
        return Err(Refusal::Empty { what: "rotary" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if rotary % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "rotary is not a whole number of pairs",
            at: i64::from(rotary),
        });
    }
    if width <= 0 || width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "width is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok([
        rotary.unsigned_abs() / 2,
        width.unsigned_abs() / head_dim.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}

/// The geometric ladder, one row, in place.
///
/// In place on ONE tensor: buffer 0 is both the input and the result, which is
/// why `x` is the only buffer and why it is `BufMut`.
///
/// A `rows` of one is not an argument here but a fact of the SYMBOL:
/// `PIE_DECODE` makes the shader assign `row = 0`, so a taller grid would
/// rotate row zero `rows` times and leave the rest alone. That is exactly the
/// failure `neox_freqs_mb` was hiding -- see its own line -- and it is why the
/// decode and multi-batch forms are separate symbols rather than one grid.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "neox_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, 1)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// The batched form of [`neox_decode`]: one row per token, each at its own
/// position.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_mb(
    ctx: &Ctx<'_>,
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "neox_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// The rotation a deployment that RESCALES its ladder takes.
///
/// Same body as [`neox_decode`] with the frequencies read from a buffer
/// instead of raised from a base, which is the only form that can express
/// llama-3's piecewise rescaling or YaRN's -- neither is a base.
///
/// `mscale` is YaRN's attention-temperature correction, and it is `1.0` for a
/// deployment that has none, which is every llama-3 one: its rescaling lives
/// in the frequencies and not in a gain.
///
/// `base` is absent rather than ignored. `PIE_FREQS` changes the push block's
/// LAYOUT -- `{ float scale; int head_dim; float mscale; }` against
/// `{ float scale; float base; int head_dim; }` -- so passing one here would
/// not be a wasted word, it would put `head_dim` where the shader reads a
/// float.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    inv_freq: Ask<keys::RopeFrequencies, Buf>,
    head_dim: ParamOr<1, keys::HeadDim, i32>,
    mscale: ParamF32<2>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "neox_freqs_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, 1)?,
        },
        &[
            x.v(),
            position.v(),
            scale.v(),
            inv_freq.v(),
            head_dim.v(),
            mscale.v(),
        ],
    )
}

/// The batched form of the rescaled ladder, and the row a PREFILL on
/// llama-3.1, llama-3.2 or any YaRN deployment needs.
///
/// This kernel is why the crossing is worth doing. Its row was BARE, so a
/// statement had nothing to name and named the decode symbol instead -- a
/// single-row kernel over a multi-row grid, which rotates row zero and leaves
/// every row after it untouched. Rope is the identity at position zero, so row
/// zero agreed with the reference either way and the failure was silent.
/// `tests/routines.rs` now checks that every entrypoint a body names is one
/// the shader tree declares, and this function is the only thing that names
/// this symbol.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    inv_freq: Ask<keys::RopeFrequencies, Buf>,
    head_dim: ParamOr<1, keys::HeadDim, i32>,
    mscale: ParamF32<2>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "neox_freqs_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
        },
        &[
            x.v(),
            position.v(),
            scale.v(),
            inv_freq.v(),
            head_dim.v(),
            mscale.v(),
        ],
    )
}

/// gemma's rotation: the same geometric ladder over a PROPORTIONAL slice of
/// each head rather than all of it.
///
/// The arithmetic differs from [`neox_decode`] in two places -- the exponent
/// is `2i/head_dim` rather than `i/pair_half`, and the pair is `i` with
/// `i + head_dim/2` rather than `i + pair_half` -- which are the same thing
/// when the rotation covers the whole head and different when it does not.
/// That is the entire reason this is a separate symbol.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "neox_prop_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, 1)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// The batched form of [`neox_prop_decode`], and gemma's prefill rotation.
///
/// Its row was one of the 285 that state no operands at all, so this signature
/// is the FIRST statement of its binding order that has ever existed. It was
/// read off `rope/neox.slang`: buffer 0 is `x`, buffer 1 is `position`, and
/// the push block under `PIE_PROP` is `{ float scale; float base;
/// int head_dim; }`.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "neox_prop_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// The geometric ladder over rows that are not contiguous.
///
/// `row_pitch` is the only thing that separates it from [`neox_mb`]: the
/// shader takes `row_base = row * row_pitch` instead of
/// `row * n_head * head_dim`. A packed QKV projection is where that arises --
/// q and k live in one buffer, so rotating q means striding over k.
///
/// Also an unstated row, so this signature is the first statement of its
/// bindings too. `PIE_STRIDED`'s push block is `{ float scale; float base;
/// int head_dim; int row_pitch; }`, and `row_pitch` is LAST.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    row_pitch: Param<4, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    if *row_pitch < *width {
        return Err(Refusal::Narrow {
            what: "row_pitch is narrower than the row it strides over",
            at: i64::from(*row_pitch),
        });
    }
    ctx.dispatch(
        Fire {
            entrypoint: "neox_strided_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
        },
        &[
            x.v(),
            position.v(),
            scale.v(),
            base.v(),
            head_dim.v(),
            row_pitch.v(),
        ],
    )
}

/// This family's routines.
///
/// EVERY ONE STATES `in_place = &[(0, 0)]`, and an earlier draft that removed
/// it had the right gate and the wrong word. The claim then was that a
/// rotation's single operand is an `Out` with no input to alias -- but
/// `dsl::metal::rope_one`, the only place any plane's neox is stated, places
/// that operand as an INPUT and declares no result on purpose. A statement
/// that declared a separate result had `Out(0)` bind the RESULT's slot, which
/// no kernel had written, and the rotated value everything downstream wanted
/// was never produced; position zero makes rope the identity, so the first
/// reference gate agreed anyway. Stating no result is what makes
/// `dispatch::reorder` bind `Out(0)` to the one widthed operand -- the input,
/// the buffer the kernel mutates.
///
/// So `(0, 0)` is not a second spelling of `BufMut`. It is the only thing that
/// says the write and the placement are ONE buffer, which §6.2's arity rule
/// needs on both sides: without it a rotation reads NOTHING against a
/// statement placing one operand, and writes one pointer against a statement
/// declaring none.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(neox_decode, in_place = &[(0, 0)]),
    crate::routine!(neox_freqs_decode, in_place = &[(0, 0)]),
    crate::routine!(neox_freqs_mb, in_place = &[(0, 0)]),
    crate::routine!(neox_mb, in_place = &[(0, 0)]),
    crate::routine!(neox_prop_decode, in_place = &[(0, 0)]),
    crate::routine!(neox_prop_mb, in_place = &[(0, 0)]),
    crate::routine!(neox_strided, in_place = &[(0, 0)]),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    type Call = (String, [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    /// A decode rotation asks for ONE row and a batched one asks for all of
    /// them.
    ///
    /// This is the failure this family is named for. `neox_freqs_mb`'s row was
    /// bare, so a statement had nothing to name and named
    /// `neox_freqs_decode` -- and `PIE_DECODE` makes the shader assign
    /// `row = 0`, so a three-row grid rotated row zero three times and left
    /// rows one and two exactly as they arrived. Rope is the identity at
    /// position zero, so the first row of a prefill agreed with the reference
    /// and the rest were silently unrotated.
    ///
    /// A decode form taking no `rows` argument at all is the shape that makes
    /// that unspellable: there is no grid to get wrong.
    #[test]
    fn a_decode_rotation_is_one_row_and_a_batched_one_is_all_of_them() {
        let seen = Seen::default();
        neox_freqs_decode(
            &seen,
            OutSlot::new(BufMut(0)),
            Ask::new(I32s(1)),
            ParamF32::new(1.0),
            Ask::new(Buf(2)),
            ParamOr::new(128),
            ParamF32::new(1.0),
            ParamOr::new(128),
            Ask::new(4096),
        )
        .expect("a launch");
        neox_freqs_mb(
            &seen,
            OutSlot::new(BufMut(0)),
            Ask::new(I32s(1)),
            ParamF32::new(1.0),
            Ask::new(Buf(2)),
            ParamOr::new(128),
            ParamF32::new(1.0),
            ParamOr::new(128),
            Ask::new(4096),
            Ask::new(3),
        )
        .expect("three tokens is a launch");

        let calls = seen.0.borrow();
        let fired: Vec<(&str, [u32; 3])> = calls
            .iter()
            .map(|(e, lanes, _)| (e.as_str(), *lanes))
            .collect();
        assert_eq!(
            fired,
            vec![
                ("neox_freqs_decode_bfloat16", [64, 32, 1]),
                ("neox_freqs_mb_bfloat16", [64, 32, 3]),
            ],
            "64 pairs of a 128-wide rotation, 32 heads of a 4096-wide tensor, \
             and the row count is the ONLY difference"
        );
    }

    /// The grid is `[rotary/2, heads, rows]`, and the rotary width is the
    /// STATEMENT's, not the tensor's.
    ///
    /// gemma-4 rotates a quarter of each full-attention head and all of each
    /// sliding one, over the same tensor width. Under `kernel!` that was
    /// `grid_param = Some(3)` -- a param index the kernel never reads and only
    /// the grid does, which is a fact stated in a place designed for something
    /// else. Here it is an argument.
    #[test]
    fn the_rotary_width_is_the_statements_and_the_head_count_is_the_tensors() {
        let seen = Seen::default();
        neox_decode(
            &seen,
            OutSlot::new(BufMut(0)),
            Ask::new(I32s(1)),
            ParamF32::new(1.0),
            ParamF32::new(10_000.0),
            ParamOr::new(256),
            ParamOr::new(64),
            Ask::new(2048),
        )
        .expect("a quarter-rotated 256-wide head is a launch");

        assert_eq!(
            seen.0.borrow()[0].1,
            [32, 8, 1],
            "32 pairs of a 64-wide ROTATION over 8 heads of a 2048-wide tensor \
             whose heads are 256 wide -- the rotary width and the head width \
             are different numbers and both are read"
        );
    }

    /// A shape the shader cannot express is refused, and the refusal says
    /// which.
    ///
    /// Neither is a hypothetical. An odd rotary width halves to a pair count
    /// that leaves the last element unrotated -- the shader pairs `i` with
    /// `i + pair_half` and a stray element belongs to neither half -- so the
    /// result is correct in every dimension but one, at every position but
    /// zero. And a width the head does not divide gives a fractional head
    /// count, which the shader multiplies back out to find each row's base.
    #[test]
    fn a_shape_the_rotation_cannot_express_is_refused() {
        assert!(
            matches!(
                rope_grid(127, 4096, 128, 1),
                Err(Refusal::Narrow {
                    what: "rotary is not a whole number of pairs",
                    at: 127
                })
            ),
            "an odd rotary width would leave one element of each head unturned"
        );
        assert!(
            matches!(
                rope_grid(128, 4095, 128, 1),
                Err(Refusal::Narrow {
                    what: "width is not a whole number of heads",
                    at: 4095
                })
            ),
            "`geometry::rope_heads` refuses the same shape as `Unheaded`"
        );
        assert!(
            matches!(rope_grid(128, 4096, 128, 0), Err(Refusal::Empty { .. })),
            "and no tokens is nothing to rotate, not a zero grid"
        );
    }

    /// `neox_strided` refuses a pitch narrower than the row it strides over.
    ///
    /// The pitch exists because q and k share one buffer in a packed QKV
    /// projection, so rotating q means striding over k -- which means the
    /// pitch is always at least the width. A smaller one makes consecutive
    /// rows OVERLAP: row 1 starts inside row 0, and the second half of every
    /// row is rotated twice at two different positions. The output is finite,
    /// plausible and wrong.
    #[test]
    fn a_stride_narrower_than_the_row_is_refused() {
        let seen = Seen::default();
        assert!(
            matches!(
                neox_strided(
                    &seen,
                    OutSlot::new(BufMut(0)),
                    Ask::new(I32s(1)),
                    ParamF32::new(1.0),
                    ParamF32::new(10_000.0),
                    ParamOr::new(128),
                    Param::new(2048),
                    ParamOr::new(128),
                    Ask::new(4096),
                    Ask::new(3),
                ),
                Err(Refusal::Narrow {
                    what: "row_pitch is narrower than the row it strides over",
                    at: 2048
                })
            ),
            "a 2048 pitch over a 4096 row overlaps every pair of rows"
        );
        assert!(
            seen.0.borrow().is_empty(),
            "and it is refused before anything reaches the driver"
        );
    }

    /// The four spellings of the schedule are four SYMBOLS, each with its own
    /// push block.
    ///
    /// `PIE_FREQS` and `PIE_STRIDED` change the push block's LAYOUT, not just
    /// its contents: `{ scale, head_dim, mscale }` against
    /// `{ scale, base, head_dim }` against `{ scale, base, head_dim,
    /// row_pitch }`. So a `base` passed to the frequency form would not be a
    /// wasted word -- it would put `head_dim` where the shader reads a float.
    /// The signatures differ because the blocks do.
    #[test]
    fn each_schedule_pushes_the_block_its_own_symbol_declares() {
        let seen = Seen::default();
        neox_mb(
            &seen,
            OutSlot::new(BufMut(0)),
            Ask::new(I32s(1)),
            ParamF32::new(0.5),
            ParamF32::new(10_000.0),
            ParamOr::new(128),
            ParamOr::new(128),
            Ask::new(4096),
            Ask::new(3),
        )
        .expect("a launch");
        neox_freqs_mb(
            &seen,
            OutSlot::new(BufMut(0)),
            Ask::new(I32s(1)),
            ParamF32::new(0.5),
            Ask::new(Buf(9)),
            ParamOr::new(128),
            ParamF32::new(0.75),
            ParamOr::new(128),
            Ask::new(4096),
            Ask::new(3),
        )
        .expect("a launch");
        neox_strided(
            &seen,
            OutSlot::new(BufMut(0)),
            Ask::new(I32s(1)),
            ParamF32::new(0.5),
            ParamF32::new(10_000.0),
            ParamOr::new(128),
            Param::new(8192),
            ParamOr::new(128),
            Ask::new(4096),
            Ask::new(3),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        let scalars: Vec<Vec<&ArgValue>> = calls
            .iter()
            .map(|(_, _, args)| {
                args.iter()
                    .filter(|a| !matches!(a, ArgValue::Buffer { .. }))
                    .collect()
            })
            .collect();
        assert_eq!(
            scalars,
            vec![
                vec![
                    &ArgValue::F32(0.5),
                    &ArgValue::F32(10_000.0),
                    &ArgValue::I32(128)
                ],
                vec![
                    &ArgValue::F32(0.5),
                    &ArgValue::I32(128),
                    &ArgValue::F32(0.75)
                ],
                vec![
                    &ArgValue::F32(0.5),
                    &ArgValue::F32(10_000.0),
                    &ArgValue::I32(128),
                    &ArgValue::I32(8192)
                ],
            ],
            "scale then base then head_dim; scale then head_dim then mscale; \
             and the strided form's row_pitch LAST"
        );
        assert_eq!(
            calls[1].2[3],
            ArgValue::Buffer {
                handle: 9,
                writes: false
            },
            "and only the frequency form binds a second buffer, at 2, read-only"
        );
    }
}
