//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Const, Ctx, Fire, InOut, Tensor, bf16, keys};
use kernels::routine::Refusal;

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
/// why `x` is the only buffer and why it is `bf16`.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("neox_decode_bfloat16", ctx.best()), "neox_decode_bfloat16").apply(rope_grid(*rotary, width, *head_dim, 1)?),
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
    ctx.fire(
        Fire::at(crate::routine::module_path("neox_mb_bfloat16", ctx.best()), "neox_mb_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
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
/// `base` is absent rather than ignored. `PIE_FREQS` changes the push block's
/// LAYOUT -- `{ float scale; int head_dim; float mscale; }` against
/// `{ float scale; float base; int head_dim; }` -- so passing one here would
/// not be a wasted word, it would put `head_dim` where the shader reads a
/// float.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("neox_freqs_decode_bfloat16", ctx.best()), "neox_freqs_decode_bfloat16").apply(rope_grid(*rotary, width, *head_dim, 1)?),
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
/// zero agreed with the reference either way and the failure was silent.
/// `tests/routines.rs` now checks that every entrypoint a body names is one
/// the shader tree declares, and this function is the only thing that names
/// this symbol.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("neox_freqs_mb_bfloat16", ctx.best()), "neox_freqs_mb_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
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
/// That is the entire reason this is a separate symbol.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("neox_prop_decode_bfloat16", ctx.best()), "neox_prop_decode_bfloat16").apply(rope_grid(*rotary, width, *head_dim, 1)?),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
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
    ctx.fire(
        Fire::at(crate::routine::module_path("neox_prop_mb_bfloat16", ctx.best()), "neox_prop_mb_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
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
    // driver answers `keys::RowPitch`. Metal's twin declares it identically;
    // the three planes must ask the binder the same questions.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("neox_strided_bfloat16", ctx.best()), "neox_strided_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
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
    use crate::routine::{ArgValue, Const, Encode, InOut, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that remembers, and answers the facts this family's bodies
    /// ask for.
    ///
    /// `positions` backs every `ctx.ask::<Tensor<i32>, keys::Positions>()` in
    /// this file, so no rotation can even reach `rope_grid` unless the probe
    /// binds one. `rows` answers the batched forms' own row ask, and
    /// `row_pitch` answers the strided form's extra ask. `width` never reaches
    /// `resolve`: every body reads it off the `InOut` mark itself, so tests
    /// that care about it state it on `x` directly.
    ///
    /// The defaults are representative nonzero values. A test that asserts a
    /// specific dispatch or drives a refusal path overrides the fact it needs
    /// before the call; a test that only cares that a call gets far enough to
    /// fire should not accidentally hit an empty-grid refusal because the
    /// probe invented zero.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        positions: Cell<u32>,
        rows: Cell<i32>,
        row_pitch: Cell<i32>,
    /// The fire's rope frequency table, as a handle.
    inv_freq: Cell<u32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                positions: Cell::new(700),
                rows: Cell::new(3),
                row_pitch: Cell::new(8192),
                // The fire's frequency table, which `neox_freqs_*` asks for.
                // 9 is what `each_schedule_pushes_the_block_its_own_symbol_
                // declares` pins by name; it predates the ask -- `inv_freq`
                // was a parameter that test supplied directly (`Buf(9)`)
                // before the table moved into the body.
                inv_freq: Cell::new(9),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(
            &self,
            ty: kernels::Ty,
            source: kernels::Source,
        ) -> Result<ArgValue, Refusal> {
            // The statement's own scalars, read by index where the params run
            // is the shader's struct -- see `Asks::param`.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                let _ = n;
                return Ok(ArgValue::I32(4096));
            }
            use kernels::keys::Fact;
            if source == <keys::Positions as Fact>::SOURCE {
                return Ok(ArgValue::Buffer {
                    handle: self.positions.get(),
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // THE FIRE'S FREQUENCY TABLE. `inv_freq` was a parameter and is
            // an ask now: it is built once per fire, not carried by the
            // checkpoint, so no statement places it.
            if source == <keys::RopeFrequencies as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.inv_freq.get(), writes: false, rows: 0, width: 0 });
            }
            if matches!(ty, kernels::Ty::Buf) {
                return Ok(ArgValue::Buffer { handle: 900, writes: false, rows: 0, width: 0 });
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
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
            InOut { ptr: Tensor::<bf16>::new(0), rows: 0, width: 4096 },
            Const::new(1.0),
            Const::new(128),
            Const::new(1.0),
            Const::new(128),
        )
        .expect("a launch");
        neox_freqs_mb(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(0), rows: 0, width: 4096 },
            Const::new(1.0),
            Const::new(128),
            Const::new(1.0),
            Const::new(128),
        )
        .expect("three tokens is a launch");

        let calls = seen.calls.borrow();
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
            "64 pairs of a 128-wide rotation, 32 heads of a 4096-wide tensor, and the row count is the ONLY difference"
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
            InOut { ptr: Tensor::<bf16>::new(0), rows: 0, width: 2048 },
            Const::new(1.0),
            Const::new(10_000.0),
            Const::new(256),
            Const::new(64),
        )
        .expect("a quarter-rotated 256-wide head is a launch");

        assert_eq!(
            seen.calls.borrow()[0].1,
            [32, 8, 1],
            "32 pairs of a 64-wide ROTATION over 8 heads of a 2048-wide tensor whose heads are 256 wide -- the rotary width and the head width are different numbers and both are read"
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
                    InOut { ptr: Tensor::<bf16>::new(0), rows: 0, width: 4096 },
                    Const::new(1.0),
                    Const::new(10_000.0),
                    Const::new(128),
                    Const::new(128),
                    // The narrow pitch this case is about, stated rather than
                    // set on the probe: the statement carries it now.
                    Const::new(2048),
                ),
                Err(Refusal::Narrow {
                    what: "row_pitch is narrower than the row it strides over",
                    at: 2048
                })
            ),
            "a 2048 pitch over a 4096 row overlaps every pair of rows"
        );
        assert!(
            seen.calls.borrow().is_empty(),
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
            InOut { ptr: Tensor::<bf16>::new(0), rows: 0, width: 4096 },
            Const::new(0.5),
            Const::new(10_000.0),
            Const::new(128),
            Const::new(128),
        )
        .expect("a launch");
        neox_freqs_mb(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(0), rows: 0, width: 4096 },
            Const::new(0.5),
            Const::new(128),
            Const::new(0.75),
            Const::new(128),
        )
        .expect("a launch");
        neox_strided(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(0), rows: 0, width: 4096 },
            Const::new(0.5),
            Const::new(10_000.0),
            Const::new(128),
            Const::new(128),
            // What the probe used to hand back for `keys::RowPitch`.
            Const::new(8192),
        )
        .expect("a launch");

        let calls = seen.calls.borrow();
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
            "scale then base then head_dim; scale then head_dim then mscale; and the strided form's row_pitch LAST"
        );
        assert_eq!(
            calls[1].2[3],
            ArgValue::Buffer {
                handle: 9,
                writes: false,
                    rows: 0,
                    width: 0
            },
            "and only the frequency form binds a second buffer, at 2, read-only"
        );
    }
}

