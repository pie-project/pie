//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, keys};

/// Threads per threadgroup for every elementwise body in this file.
///
/// Metal declares no group size in the source -- `[[kernel]]` takes no
/// `[numthreads]` -- so the number lives here and reaches the encoder as the
/// second half of `dispatchThreads:threadsPerThreadgroup:`. 256 is what
/// `driver-metal`'s `geometry` has always used for `LaunchRule::Elementwise`,
/// and this is the same statement moved, not a new one.
const GROUP_X: u32 = 256;

/// gemma's gated activation: `gelu_tanh(gate) * up`, the tanh approximation
/// and not the erf one.
///
/// `mlp/gated.metal`. Buffers 0..=2 are gate, up and out, in that order, which
/// is the order this signature states and not the order a trace states them.
/// There is no buffer 3.
///
/// **`GegluParams` is gone from the kernel, not merely unread by it.** The
/// struct had one member, named `unused` and genuinely so: it was a per-row
/// element count that once bounded a whole-tensor dispatch, and the body has
/// bounded itself with the grid ever since -- `dispatchThreads:` launches
/// exactly the element count, so there is no tail to guard.
///
/// What kept the argument alive after the field died was the ROW. It stated
/// `params: Buf`, and an argument table with a hole in it is not something the
/// encoder can be asked for, so this body forwarded `ctx.params()` into a slot
/// the kernel then `(void)`'d. The row is retired; the slot is not the last
/// one (`silu_mul_strided` still takes its pitch at buffer 4, which is why
/// THAT kernel is dark), so nothing is shifted by removing this argument, and
/// the kernel's own parameter list is now three buffers and a thread position.
/// `hold::params_block` is not called for this dispatch at all, which means
/// `driver-metal` stages no packed run for it either.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
#[routine]
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.metal", "geglu_tanh_bfloat16").apply(Grid::of(elementwise(width, rows)?, [GROUP_X, 1, 1])),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

/// The same activation over rows that are not contiguous.
///
/// gemma's PLE reads a narrow gate out of a wide buffer, so each of the three
/// operands has its own pitch and all three ride the `GegluStridedParams`
/// block along with the width.
///
/// The grid is FLAT -- `width * rows` on x -- and the body recovers the row
/// and the column by dividing by the width. That is not a simplification:
/// this kernel used to read `gid.y` as its row while its own row said
/// `Elementwise`, so `gid.y` was structurally zero and every row of gemma's
/// PLE tail but the first came back holding whatever the arena was born with,
/// on every layer of every prefill longer than one token, and the dispatch
/// succeeded.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
#[routine]
pub fn geglu_tanh_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE FIVE THAT WERE `GegluStridedParams`, in the order the row states
    // them. `stated_width`/`stated_rows` are the launch's own rectangle, which
    // this body also derives -- not a duplication introduced here: the struct
    // already held both, the shader read the staged words and the grid came
    // from the body, and both still do.
    stated_width: Const<u32>,
    stated_rows: Const<u32>,
    gate_pitch: Const<u32>,
    up_pitch: Const<u32>,
    out_pitch: Const<u32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.metal", "geglu_tanh_strided_bfloat16").apply(Grid::of(elementwise(width, rows)?, [GROUP_X, 1, 1])),
        &[
            gate.arg(),
            up.arg(),
            out.arg(),
            stated_width.arg(),
            stated_rows.arg(),
            gate_pitch.arg(),
            up_pitch.arg(),
            out_pitch.arg(),
        ],
    )
}

/// gpt-oss's activation, which is not anyone else's.
///
/// The gate is clamped ABOVE only, the linear branch is clamped both ways and
/// carries a `+1`, and the sigmoid takes an `alpha`. `silu_mul` cannot serve
/// it: dropping either the clamp or the `+1` produces a model that runs and is
/// wrong, which is why this is a symbol a text names rather than a flag.
///
/// `limit` and `alpha` are `Const` marks now: they were `GptOssSwiGluParams`'
/// second and third fields, behind a dead per-row element count held only to
/// keep them at their offsets. `Const` slots are the statement's run counted in
/// declaration order, so reaching `limit` at word 1 means naming word 0 --
/// `_stated_elements` is that name, and it is declared and NOT passed, so the
/// dead word stops at the host instead of riding to the device inside a struct.
/// The holder goes when the DSL stops stating the word.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
#[routine]
pub fn gptoss_swiglu(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    _stated_elements: Const<u32>,
    limit: Const<f32>,
    alpha: Const<f32>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.metal", "gptoss_swiglu_bfloat16").apply(Grid::of(elementwise(width, rows)?, [GROUP_X, 1, 1])),
        &[gate.arg(), up.arg(), out.arg(), limit.arg(), alpha.arg()],
    )
}

/// The plain gated FFN activation: `silu(gate) * up`.
///
/// Three buffers and no parameter block, because it needs no scalar the grid
/// does not give it. It indexes `tid` raw and guards nothing, which is the
/// shape the other three bodies in this file have been brought to -- and, in
/// [`geglu_tanh`]'s case, brought to all the way: that one no longer carries
/// an unread block either, so this is no longer the only kernel here without
/// one.
///
/// The silu is computed through bf16 twice on purpose: `sigmoid` and the
/// product are each rounded to bf16 before the multiply, which is what MLX
/// does, and matching it is what lets this backend's output be compared
/// against the reference bit for bit rather than within a tolerance.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
#[routine]
pub fn silu_mul(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("mlp/gated.metal", "silu_mul_bfloat16").apply(Grid::of(elementwise(width, rows)?, [GROUP_X, 1, 1])),
        &[gate.arg(), up.arg(), out.arg()],
    )
}


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
    ("mlp/gated.metal", "geglu_tanh_bfloat16"),
    ("mlp/gated.metal", "geglu_tanh_strided_bfloat16"),
    ("mlp/gated.metal", "gptoss_swiglu_bfloat16"),
    ("mlp/gated.metal", "silu_mul_bfloat16"),
    ("mlp/gated.metal", "silu_mul_strided_bfloat16"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// facts this family's bodies ask for.
    ///
    /// `rows` backs every `ctx.ask::<i32, keys::Rows>()` in this file --
    /// under `kernel!` it rode its own `Ask::new(_)` argument, and now the
    /// body asks the fire for it. `params_handle` answers `ctx.params()`,
    /// standing in for the `Block::new(Buf(4))` every body but `silu_mul`
    /// used to take -- TWO bodies still ask, `geglu_tanh_strided` for its
    /// pitches and `gptoss_swiglu` for its limit and alpha. `geglu_tanh` was
    /// the third until `mlp/gated.metal` lost `GegluParams` outright, and
    /// `all_four_bodies_bind_gate_up_and_out_at_zero_one_and_two` is where
    /// that absence is asserted rather than assumed.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        params_handle: Cell<u32>,
        /// THE STATEMENT\'S SCALAR RUN, for a body that reads a word by
        /// index. Empty means "4096 at every slot", which is a plausible
        /// stride for the rows these tests build; a case that means a
        /// particular tiling or split count sets its own.
        words: RefCell<Vec<i32>>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(2),
                params_handle: Cell::new(4),
                words: RefCell::default(),
            }
        }
    }

    impl Encode for Seen {
        // A PROBE HAS NO FIRE BEHIND IT, so it answers only the facts this
        // file's bodies ask for and refuses everything else honestly --
        // answering zero for an unasked fact would let a body under test pass
        // while the fact it asked for went unanswered on a real driver.
        fn resolve(&self, _ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // THE STATEMENT'S OWN SCALARS, which a body reads by index when its
            // params run is a struct and no `Const` mark can name a word inside
            // it -- see `Asks::param`. The probe answers a number that is
            // plausible for every reader: a stride wide enough for the rows
            // these tests build, and a positive tiling.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                return Ok(ArgValue::I32(
                    self.words.borrow().get(usize::from(n)).copied().unwrap_or(4096),
                ));
            }
            if source == kernels::Source::Slot(kernels::Kind::Params, 0) {
                return Ok(ArgValue::Buffer(self.params_handle.get()));
            }
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// Every body in this family binds `gate, up, out` at 0, 1, 2 -- which is
    /// the one thing the file's first line says they have in common.
    ///
    /// A positional bind in the TRACE's order rather than the kernel's is how
    /// `rms_single_row` came to read its own output as the weight, so the
    /// order is worth a test even where it looks obvious.
    ///
    /// KNOWN FAILING, upstream of this crate: `out` is an `Out<Tensor<bf16>>`
    /// and `ArgValue::BufferMut(3)` below is the correct claim for what its
    /// `.arg()` SHOULD produce. `kernels::shader::Tensor<E>` has one field,
    /// `handle: u32`, with nowhere to keep a direction, and its `Bind` impl
    /// (`crates/kernels/src/shader.rs`, outside this crate) reads `V::buffer
    /// (self.handle)` unconditionally -- `Out`/`InOut` delegate to it exactly
    /// as `In` does, so no positional argument a routine body binds itself
    /// can presently come out `BufferMut` on any plane. `kernels::routine::
    /// Elem for Tensor<E>` sets `Read = Write = Self` and its own comment
    /// says why: "a binding index has one [value]... so `Read` and `Write`
    /// are both `Self`, and the `Ty` still splits, because the TABLE must say
    /// which way the launch drives the operand even where the value cannot."
    /// That table -- the declared `Ty` a `KernelSig` carries -- is read-only
    /// from here too, and nothing between a routine body's `ctx.fire()` and
    /// `Encode::fire` re-derives a value from it: `Ctx<'_>` is `dyn Encode`
    /// with no wrapping. The assertion states the correct claim rather than
    /// one weakened to match the gap.
    #[test]
    fn all_four_bodies_bind_gate_up_and_out_at_zero_one_and_two() {
        let seen = Seen::default();
        seen.rows.set(2);
        geglu_tanh(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            In { ptr: Tensor::<bf16>::new(2), rows: 0, width: 8 },
            Out::new(Tensor::<bf16>::new(3)),
        )
        .expect("a launch");
        geglu_tanh_strided(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            In { ptr: Tensor::<bf16>::new(2), rows: 0, width: 8 },
            Out::new(Tensor::<bf16>::new(3)),
            // The five that were `GegluStridedParams`. The pitches differ from
            // the width on purpose: gemma's PLE reads a narrow gate out of a
            // wide table, which is the whole reason this form exists.
            Const::new(8),
            Const::new(2),
            Const::new(8),
            Const::new(64),
            Const::new(8),
        )
        .expect("a launch");
        gptoss_swiglu(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            In { ptr: Tensor::<bf16>::new(2), rows: 0, width: 8 },
            Out::new(Tensor::<bf16>::new(3)),
            // Word 0 is the dead element count the struct opened with; it is
            // declared so `limit` lands on word 1 and is deliberately not
            // passed, so it reaches no argument list.
            Const::new(0),
            Const::new(7.0),
            Const::new(1.702),
        )
        .expect("a launch");
        silu_mul(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            In { ptr: Tensor::<bf16>::new(2), rows: 0, width: 8 },
            Out::new(Tensor::<bf16>::new(3)),
        )
        .expect("a launch");

        let calls = seen.calls.borrow();
        assert_eq!(calls.len(), 4, "one activation is one dispatch");
        for (fire, args) in calls.iter() {
            assert_eq!(
                args[..3],
                [
                    ArgValue::Buffer(1),
                    ArgValue::Buffer(2),
                    ArgValue::BufferMut(3)
                ],
                "`{}` binds gate, up, out at 0, 1, 2 -- and `out` says so in \
                 the VALUE, which is where the encoder reads the direction",
                fire.entrypoint
            );
            assert_eq!(
                fire.file, "mlp/gated.metal",
                "`{}`: all four are one file because they are one binding \
                 contract, and Metal compiles from (path, entry name)",
                fire.entrypoint
            );
        }
        assert_eq!(
            calls[0].1.len(),
            3,
            "geglu_tanh takes NO params buffer: `GegluParams` was one field \
             named `unused`, the grid is the extent, and the struct and its \
             `[[buffer(3)]]` argument are both gone from `mlp/gated.metal`. \
             A fourth argument here would be bound into a slot the kernel \
             does not declare"
        );
        assert_eq!(
            calls[1].1.len(),
            8,
            "the strided one takes `GegluStridedParams`' five words as five \
             `setBytes` scalars at buffers 3..=7, where they used to be one \
             `constant GegluStridedParams&` at buffer 3"
        );
        assert_eq!(
            calls[2].1.len(),
            5,
            "gpt-oss takes `limit` at buffer 3 and `alpha` at 4. FIVE and not \
             six: `GptOssSwiGluParams` opened with a dead element count, and \
             the mark that holds its slot is declared and never passed, so the \
             dead word reaches no argument list"
        );
        assert_eq!(
            calls[3].1.len(),
            3,
            "silu_mul needs no scalar the grid does not give it -- and since \
             `GegluParams` went, geglu_tanh above says the same"
        );
    }

    /// The grid is the whole rectangle, on one axis.
    ///
    /// This is the assertion the strided kernel failed for its entire life. It
    /// read `gid.y` as its row under a rule that puts every thread on x, so
    /// `gid.y` was structurally zero and rows 1.. of gemma's PLE tail were
    /// never written. The number below is `width * rows` and not `width`, and
    /// on Metal it is exact -- `dispatchThreads:` launches precisely this many
    /// threads, which is why these bodies carry no bounds guard.
    #[test]
    fn the_grid_is_every_element_of_the_rectangle_and_not_one_row_of_it() {
        let seen = Seen::default();
        seen.rows.set(8);
        geglu_tanh_strided(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 256 },
            In { ptr: Tensor::<bf16>::new(2), rows: 0, width: 256 },
            Out::new(Tensor::<bf16>::new(3)),
            Const::new(256),
            Const::new(8),
            Const::new(256),
            Const::new(256),
            Const::new(256),
        )
        .expect("eight rows is a launch");

        let calls = seen.calls.borrow();
        let (fire, _) = &calls[0];
        assert_eq!(
            fire.lanes,
            [256 * 8, 1, 1],
            "256 wide by 8 rows is 2048 elements and so 2048 threads; a grid of \
             [256, 1, 1] would write the first row and leave seven holding \
             whatever the arena was born with"
        );
        assert_eq!(fire.group, [GROUP_X, 1, 1]);
    }

    /// An empty rectangle is refused, not dispatched.
    ///
    /// A routed expert that won no tokens has zero rows, so this arrives
    /// honestly. `dispatchThreads:` with a zero extent runs nothing and
    /// reports success, which is the silence this plane exists to end.
    #[test]
    fn a_rectangle_with_no_elements_is_a_refusal_and_not_a_launch() {
        let seen = Seen::default();
        seen.rows.set(0);
        let refused = silu_mul(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 8 },
            In { ptr: Tensor::<bf16>::new(2), rows: 0, width: 8 },
            Out::new(Tensor::<bf16>::new(3)),
        )
        .expect_err("zero rows is not a launch");
        assert!(
            matches!(refused, Refusal::Empty { what: "rows" }),
            "got {refused:?}"
        );
        assert!(
            seen.calls.borrow().is_empty(),
            "and nothing was encoded on the way to refusing"
        );
    }
}
