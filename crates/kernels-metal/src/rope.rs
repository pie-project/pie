//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.
//!
//! EVERY ROUTINE HERE TAKES ITS `position` AS `Env<I32s>`, and the `freqs`
//! pair its `inv_freq` as `Env<Buf>`. A rotation's only trace operand is the
//! tensor it turns: `driver-metal`'s shared `fn neox` arm binds `position`
//! from `FireTable::Positions` and `inv_freq` from `FireTable::RopeFrequencies`
//! -- where each token sits in its request, and a ramp the loader computed --
//! so no statement places either and §6.2's arity rule would count two
//! operands against a statement that carries none.


use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, InOut, Tensor, bf16, keys};


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
/// why `x` is the only buffer and why it is `Buf`.
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
#[routine]
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at(NEOX_FILE, "neox_decode_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

/// The batched form of [`neox_decode`]: one row per token, each at its own
/// position.
///
/// # Errors
///
/// See [`rope_grid`].
#[routine]
pub fn neox_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at(NEOX_FILE, "neox_mb_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
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
#[routine]
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    // THE FIRE'S FREQUENCY TABLE, ASKED FOR. `Ask<keys::RopeFrequencies, Buf>`
    // before the marks: a table the driver builds once per fire, not a weight
    // the checkpoint carries and no builder places one. As a
    // `Const<Tensor<f32>>` it asked the statement for a weight operand that
    // is not there, and every gpt-oss rotation refused.
    let inv_freq = ctx.ask::<Tensor<f32>, keys::RopeFrequencies>()?;

    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at(NEOX_FILE, "neox_freqs_decode_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
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
#[routine]
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    // THE FIRE'S FREQUENCY TABLE, ASKED FOR. `Ask<keys::RopeFrequencies, Buf>`
    // before the marks: a table the driver builds once per fire, not a weight
    // the checkpoint carries and no builder places one. As a
    // `Const<Tensor<f32>>` it asked the statement for a weight operand that
    // is not there, and every gpt-oss rotation refused.
    let inv_freq = ctx.ask::<Tensor<f32>, keys::RopeFrequencies>()?;

    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at(NEOX_FILE, "neox_freqs_mb_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
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
#[routine]
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let lanes = rope_grid(*rotary, width, *head_dim, 1)?;
    ctx.fire(
        Fire::at(NEOX_FILE, "neox_prop_decode_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
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
#[routine]
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at(NEOX_FILE, "neox_prop_mb_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
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
#[routine]
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    // THE STATEMENT'S, AND IT WAS `Param<4, i32>`. A pitch is the rectangle
    // the text laid out, not something this batch made -- two fires of one
    // deployment stride the same way -- so it fails `ask`'s own test and no
    // driver answers `keys::RowPitch`. `the_two_rotations_no_row_could_reach_
    // now_dispatch` states it at `params[4]` and expects a refusal without it,
    // which is the slot this mark derives.
    row_pitch: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    if *row_pitch < width {
        return Err(Refusal::Narrow {
            what: "row_pitch is narrower than the row it strides over",
            at: i64::from(*row_pitch),
        });
    }
    let lanes = rope_grid(*rotary, width, *head_dim, rows)?;
    ctx.fire(
        Fire::at(NEOX_FILE, "neox_strided_bfloat16").apply(Grid::of(lanes, rope_group(lanes))),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            row_pitch.arg(),
        ],
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// The entrypoint, the lanes, the group and the arguments.
    type Call = (&'static str, [u32; 3], [u32; 3], Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// three facts every rotation in this file asks for: where each token
    /// sits (`Positions`, every routine), how many rows the batched forms
    /// cover (`Rows`, the `_mb` and strided forms), and the stride the
    /// strided form's rows do not touch each other's channels at
    /// (`RowPitch`, `neox_strided` alone).
    struct Seen {
        calls: RefCell<Vec<Call>>,
        positions: Cell<u32>,
        rows: Cell<i32>,
    /// The fire's rope frequency table, as a handle.
    inv_freq: Cell<u32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                positions: Cell::new(2),
                rows: Cell::new(4),
                // The fire's frequency table, which `neox_freqs_*` asks for.
                inv_freq: Cell::new(3),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(&self, _ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            // The statement's own scalars, read by index where the params run
            // is the shader's struct -- see `Asks::param`.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                let _ = n;
                return Ok(ArgValue::I32(4096));
            }
            use kernels::keys::Fact;
            if source == <keys::Positions as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.positions.get()));
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // THE FIRE'S FREQUENCY TABLE. `inv_freq` was a parameter and is
            // an ask now: it is built once per fire, not carried by the
            // checkpoint, so no statement places it.
            if source == <keys::RopeFrequencies as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.inv_freq.get()));
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
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
        seen.rows.set(37);
        neox_freqs_decode(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Const::new(1.0),
            Const::new(128),
            Const::new(1.0),
            Const::new(128))
        .expect("a launch");
        neox_freqs_mb(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Const::new(1.0),
            Const::new(128),
            Const::new(1.0),
            Const::new(128))
        .expect("a launch");

        let calls = seen.calls.borrow();
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
                InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
                Const::new(1.0),
                Const::new(10_000.0),
                Const::new(128),
                Const::new(rotary))
            .expect("a launch");
        }
        for (_, lanes, group, _) in seen.calls.borrow().iter() {
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
            InOut::new(Tensor::<bf16>::new(1)),
            Const::new(1.0),
            Const::new(10_000.0),
            Const::new(128),
            Const::new(65))
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
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2000 },
            Const::new(1.0),
            Const::new(10_000.0),
            Const::new(128),
            Const::new(64))
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
        assert!(seen.calls.borrow().is_empty(), "and neither was encoded");
    }

    /// The two forms that had NO operand statement before this file get one,
    /// and it is the shader's.
    ///
    /// `neox_prop_mb` and `neox_strided` were bare rows -- nothing anywhere
    /// said what they bind. These orders were read off `rope/neox.metal`, where
    /// `rope_neox_strided` declares `row_pitch` at buffer 5, after `head_dim`.
    /// KNOWN FAILING, upstream of this crate: `x`'s `ArgValue::BufferMut(1)`
    /// below is the correct claim for what an `InOut<Tensor<bf16>>` SHOULD
    /// produce; `mlp::tests::all_four_bodies_bind_gate_up_and_out_at_zero_one_and_two`
    /// documents in full why `Tensor<E>`'s one `handle` field and its
    /// direction-blind `Bind` impl (`crates/kernels/src/shader.rs`, outside
    /// this crate) mean no positional argument a routine body binds itself
    /// can presently come out mutable, on any plane.
    #[test]
    fn the_two_unstated_rotations_bind_what_the_shader_declares() {
        let seen = Seen::default();
        neox_prop_mb(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Const::new(0.5),
            Const::new(10_000.0),
            Const::new(256),
            Const::new(128))
        .expect("a launch");
        neox_strided(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Const::new(0.5),
            Const::new(10_000.0),
            Const::new(256),
            Const::new(128),
            // THE PITCH IS THE STATEMENT'S NOW, not a fact the probe answers.
            // 4096 is what the probe used to hand back and what the assertion
            // below still names -- two 2048-wide rows to a stride.
            Const::new(4096))
        .expect("a launch");

        let calls = seen.calls.borrow();
        assert_eq!(
            calls[0].3,
            vec![
                ArgValue::BufferMut(1),
                ArgValue::Buffer(2),
                ArgValue::F32(0.5),
                ArgValue::F32(10_000.0),
                ArgValue::I32(256),
            ]
        );
        assert_eq!(
            calls[1].3,
            vec![
                ArgValue::BufferMut(1),
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
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Const::new(0.5),
            Const::new(10_000.0),
            Const::new(256),
            Const::new(128),
            // The narrow pitch this case is about, stated rather than set on
            // the probe: the statement carries it.
            Const::new(1024))
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
        assert!(seen.calls.borrow().is_empty());
    }
}
