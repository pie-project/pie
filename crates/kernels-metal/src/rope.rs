//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, I32s, Routine};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[
    ("rope/neox.metal", "neox_decode_bfloat16"),
    ("rope/neox.metal", "neox_freqs_decode_bfloat16"),
    ("rope/neox.metal", "neox_freqs_mb_bfloat16"),
    ("rope/neox.metal", "neox_mb_bfloat16"),
    ("rope/neox.metal", "neox_prop_decode_bfloat16"),
    ("rope/neox.metal", "neox_prop_mb_bfloat16"),
    ("rope/neox.metal", "neox_strided_bfloat16"),
];

/// The rotation's grid: one thread per PAIR, per head, per row.
///
/// `[rotary/2, width/head_dim, rows]`, which is what `driver-metal`'s
/// `launch::rope_rows` has always computed. The halving is the whole geometry:
/// a neox rotation turns `(i, i + rotary/2)` together, so a thread owns a pair
/// and not a channel, and the shader reads the pair count back off `grid.x`
/// rather than being told it.
///
/// That last fact is why `rotary` is checked for being EVEN. An odd extent
/// would floor here, and the shader would then divide its frequency exponent
/// by a number one pair short of the rotation it was asked for -- a wrong
/// angle on every channel, not a missing one, which no bounds check catches.
///
/// # Errors
///
/// [`Refusal::Empty`] for an extent of zero, and [`Refusal::Narrow`] for a
/// rotary that is not a whole number of pairs or a width that is not a whole
/// number of heads. Both refuse rather than round: the head count is the
/// grid's `y` and a width that is not a multiple of `head_dim` would silently
/// rotate a fraction of the last head.
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

/// The threadgroup a rotation of this many pairs takes.
///
/// One group per (head, row), as wide as the pair count -- which is
/// `launch::rope_rows`'s `tg: [half, 1, 1]`. It is derived from the lanes
/// rather than stated beside them because the two cannot disagree: the shader
/// reads its pair count off `threads_per_grid.x` and its lane off
/// `thread_position_in_grid.x`, so a group narrower than `grid.x` would leave
/// the top pairs of every head unrotated and a wider one would dispatch lanes
/// past the end of the pair list.
const fn rope_group(lanes: [u32; 3]) -> [u32; 3] {
    [lanes[0], 1, 1]
}

/// The shader all seven rotations are compiled from.
const NEOX_FILE: &str = "rope/neox.metal";

/// The geometric ladder, one row, in place.
///
/// In place on ONE tensor: buffer 0 is both the input and the result, which is
/// why `x` is the only buffer and why it is `BufMut`.
///
/// A `rows` of one is not an argument here but a fact of the SYMBOL: the
/// decode body assigns `m = 0`, so a taller grid would rotate row zero `rows`
/// times and leave the rest alone. That is exactly the failure `neox_freqs_mb`
/// was hiding -- see its own line -- and it is why the decode and multi-batch
/// forms are separate symbols rather than one grid.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = rope_grid(*rotary, *width, head_dim, 1)?;
    ctx.dispatch(
        Fire {
            entrypoint: "neox_decode_bfloat16",
            file: NEOX_FILE,
            lanes,
            group: rope_group(lanes),
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
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = rope_grid(*rotary, *width, head_dim, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "neox_mb_bfloat16",
            file: NEOX_FILE,
            lanes,
            group: rope_group(lanes),
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
/// `base` is absent rather than ignored: this entrypoint declares
/// `(x, position, scale, inv_freq, head_dim, mscale)` and there is no slot for
/// one. Passing it would put `head_dim` where the shader reads a float.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    inv_freq: Buf,
    head_dim: i32,
    mscale: f32,
    rotary: Env<i32>,
    width: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = rope_grid(*rotary, *width, head_dim, 1)?;
    ctx.dispatch(
        Fire {
            entrypoint: "neox_freqs_decode_bfloat16",
            file: NEOX_FILE,
            lanes,
            group: rope_group(lanes),
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
/// zero agreed with the reference either way and the failure was silent. This
/// function is the only thing that names this symbol.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    inv_freq: Buf,
    head_dim: i32,
    mscale: f32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = rope_grid(*rotary, *width, head_dim, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "neox_freqs_mb_bfloat16",
            file: NEOX_FILE,
            lanes,
            group: rope_group(lanes),
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
/// That is the entire reason this is a separate symbol. Verified against
/// mlx_lm at head_dim=512, rotary=128: the channels that move are [0,63] and
/// [256,319], not [0,127].
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = rope_grid(*rotary, *width, head_dim, 1)?;
    ctx.dispatch(
        Fire {
            entrypoint: "neox_prop_decode_bfloat16",
            file: NEOX_FILE,
            lanes,
            group: rope_group(lanes),
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// The batched form of [`neox_prop_decode`], and gemma's prefill rotation.
///
/// Its row was one of the ones that state no operands at all, so this
/// signature is the FIRST statement of its binding order that has ever
/// existed. It was read off `rope/neox.metal`: `rope_neox_prop_mb` declares
/// `x` at buffer 0, `position` at 1, `scale` at 2, `base` at 3 and `head_dim`
/// at 4.
///
/// # Errors
///
/// See [`rope_grid`].
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = rope_grid(*rotary, *width, head_dim, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "neox_prop_mb_bfloat16",
            file: NEOX_FILE,
            lanes,
            group: rope_group(lanes),
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// The geometric ladder over rows that are not contiguous.
///
/// `row_pitch` is the only thing that separates it from [`neox_mb`]: the
/// shader takes `row_base = m * row_pitch` instead of `m * n_head * head_dim`.
/// A packed QKV projection is where that arises -- q and k live in one buffer,
/// so rotating q means striding over k.
///
/// Also an unstated row, so this signature is the first statement of its
/// bindings too. `rope_neox_strided` declares `row_pitch` at buffer 5, LAST.
///
/// # Errors
///
/// See [`rope_grid`], plus [`Refusal::Narrow`] for a pitch narrower than the
/// row it strides over -- which would make consecutive rows overlap rather
/// than tile, and is the one thing about this kernel that cannot be seen from
/// the grid.
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    row_pitch: i32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    if row_pitch < *width {
        return Err(Refusal::Narrow {
            what: "row_pitch is narrower than the row it strides over",
            at: i64::from(row_pitch),
        });
    }
    let lanes = rope_grid(*rotary, *width, head_dim, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "neox_strided_bfloat16",
            file: NEOX_FILE,
            lanes,
            group: rope_group(lanes),
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
/// None of them states `in_place`, and that is worth a line because it looks
/// like it should. These rotations ARE in place: buffer 0 is read and written.
/// But `in_place` names `(input, output)` pairs of TRACE OPERANDS that must be
/// given the same address, and a rotation has only one operand -- an `Out`.
/// There is no input to alias it to. What makes it in place is that the single
/// buffer is `BufMut`, which the signature already says.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(neox_decode),
    crate::routine!(neox_freqs_decode),
    crate::routine!(neox_freqs_mb),
    crate::routine!(neox_mb),
    crate::routine!(neox_prop_decode),
    crate::routine!(neox_prop_mb),
    crate::routine!(neox_strided),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// The entrypoint, the lanes, the group and the arguments.
    type Call = (&'static str, [u32; 3], [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint, fire.lanes, fire.group, args.to_vec()));
            Ok(())
        }
    }

    /// A decode rotation takes ONE row however many the fire has, and its
    /// multi-batch twin takes them all.
    ///
    /// This is the assertion `neox_freqs_mb` existed to fail. Its row was bare,
    /// so the statement named `neox_freqs_decode` instead -- a body that
    /// assigns `m = 0` -- and every prefill on llama-3.1, llama-3.2 or any
    /// YaRN deployment rotated row zero and left the rest of the prompt
    /// unrotated. Rope is the identity at position zero, so row zero agreed
    /// with the reference and nothing reported anything.
    #[test]
    fn a_decode_rotation_takes_one_row_and_its_twin_takes_all_of_them() {
        let seen = Seen::default();
        neox_freqs_decode(
            &seen,
            BufMut(1),
            I32s(2),
            1.0,
            Buf(3),
            128,
            1.0,
            Env(128),
            Env(2048),
        )
        .expect("a launch");
        neox_freqs_mb(
            &seen,
            BufMut(1),
            I32s(2),
            1.0,
            Buf(3),
            128,
            1.0,
            Env(128),
            Env(2048),
            Env(37),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(calls[0].0, "neox_freqs_decode_bfloat16");
        assert_eq!(
            calls[0].1,
            [64, 16, 1],
            "64 pairs, 16 heads, ONE row -- the body assigns m = 0, so a taller \
             grid rotates row zero that many times"
        );
        assert_eq!(calls[1].0, "neox_freqs_mb_bfloat16");
        assert_eq!(
            calls[1].1,
            [64, 16, 37],
            "and the batched symbol is the one that gets the rows"
        );
    }

    /// The threadgroup is as wide as the pair count, always.
    ///
    /// The shader reads its pair count off `threads_per_grid.x` and its lane
    /// off `thread_position_in_grid.x`. A group narrower than `grid.x` leaves
    /// the top pairs of every head unrotated; a wider one dispatches lanes
    /// past the end of the pair list. Deriving it removes the chance of the
    /// two disagreeing.
    #[test]
    fn the_threadgroup_is_exactly_the_pair_count() {
        let seen = Seen::default();
        for rotary in [64, 128, 256] {
            neox_mb(
                &seen,
                BufMut(1),
                I32s(2),
                1.0,
                10_000.0,
                128,
                Env(rotary),
                Env(2048),
                Env(4),
            )
            .expect("a launch");
        }
        for (_, lanes, group, _) in seen.0.borrow().iter() {
            assert_eq!(*group, [lanes[0], 1, 1]);
        }
    }

    /// An odd rotary is refused rather than floored.
    ///
    /// The shader divides its frequency exponent by `grid.x`. A floored pair
    /// count is a WRONG ANGLE on every channel of every head -- not a missing
    /// one -- which no bounds check anywhere can catch.
    #[test]
    fn an_odd_rotary_is_a_refusal_because_flooring_it_rotates_by_the_wrong_angle() {
        let seen = Seen::default();
        let refused = neox_decode(
            &seen,
            BufMut(1),
            I32s(2),
            1.0,
            10_000.0,
            128,
            Env(65),
            Env(2048),
        )
        .expect_err("65 is not a whole number of pairs");
        assert!(
            matches!(
                refused,
                Refusal::Narrow {
                    what: "rotary is not a whole number of pairs",
                    at: 65
                }
            ),
            "got {refused:?}"
        );

        let refused = neox_decode(
            &seen,
            BufMut(1),
            I32s(2),
            1.0,
            10_000.0,
            128,
            Env(64),
            Env(2000),
        )
        .expect_err("2000 is not a whole number of 128-wide heads");
        assert!(
            matches!(
                refused,
                Refusal::Narrow {
                    what: "width is not a whole number of heads",
                    at: 2000
                }
            ),
            "got {refused:?}"
        );
        assert!(seen.0.borrow().is_empty(), "and neither was encoded");
    }

    /// The two forms that had NO operand statement before this file get one,
    /// and it is the shader's.
    ///
    /// `neox_prop_mb` and `neox_strided` were bare rows -- nothing anywhere
    /// said what they bind. These orders were read off `rope/neox.metal`, where
    /// `rope_neox_strided` declares `row_pitch` at buffer 5, after `head_dim`.
    #[test]
    fn the_two_unstated_rotations_bind_what_the_shader_declares() {
        let seen = Seen::default();
        neox_prop_mb(
            &seen,
            BufMut(1),
            I32s(2),
            0.5,
            10_000.0,
            256,
            Env(128),
            Env(2048),
            Env(5),
        )
        .expect("a launch");
        neox_strided(
            &seen,
            BufMut(1),
            I32s(2),
            0.5,
            10_000.0,
            256,
            4096,
            Env(128),
            Env(2048),
            Env(5),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            calls[0].3,
            vec![
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::F32(0.5),
                ArgValue::F32(10_000.0),
                ArgValue::I32(256),
            ]
        );
        assert_eq!(
            calls[1].3,
            vec![
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::F32(0.5),
                ArgValue::F32(10_000.0),
                ArgValue::I32(256),
                ArgValue::I32(4096),
            ],
            "the pitch is LAST, at buffer 5"
        );
    }

    /// A pitch narrower than the row it strides over is refused.
    ///
    /// It would make consecutive rows OVERLAP rather than tile, so row `m+1`
    /// would be rotated at row `m`'s angle over part of its width. Nothing
    /// about the grid can see it, which is why the body checks.
    #[test]
    fn a_pitch_narrower_than_its_row_is_a_refusal() {
        let seen = Seen::default();
        let refused = neox_strided(
            &seen,
            BufMut(1),
            I32s(2),
            0.5,
            10_000.0,
            256,
            1024,
            Env(128),
            Env(2048),
            Env(5),
        )
        .expect_err("a 1024 pitch cannot tile 2048-wide rows");
        assert!(
            matches!(
                refused,
                Refusal::Narrow {
                    what: "row_pitch is narrower than the row it strides over",
                    at: 1024
                }
            ),
            "got {refused:?}"
        );
        assert!(seen.0.borrow().is_empty());
    }
}
