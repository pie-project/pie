//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, keys,
};
use kernels::routine::Refusal;
use kernels_macros::routine;

/// gemma's gated activation: `gelu_tanh(gate) * up`, the tanh approximation
/// and not the erf one.
///
/// `mlp/gated.slang` under `PIE_GEGLU`. Bindings 0..=2 are gate, up and out,
/// in that order, which is the order this signature states and not the order a
/// trace states them. There is no binding 3.
///
/// **The `GegluParams` block is gone, source and all.** `PIE_PARAMS(3,
/// GegluParams, p)` declared a `std430 readonly buffer` whose one member was
/// named `unused` and was -- a per-row element count that once bounded a
/// whole-tensor dispatch, retired because `past_the_end` bounds the store with
/// `GetDimensions` on the output instead. The block outlived its field only
/// because the row stated `params: Buf` and every backend built a bind layout
/// out of the row.
///
/// This body used to open `let _params = ctx.params()?;` and then drop the
/// handle without forwarding it, which was correct and looked like a mistake:
/// `slangc` emits no binding decoration for a global the entrypoint never
/// reads, so the module has no descriptor at 3 and forwarding one would be an
/// arity refusal at the first real dispatch. What the call still did was mint
/// `hold::params_block`'s sentinel and stage the statement's scalar run for a
/// slot that does not exist. With the declaration deleted there is nothing
/// left to ask for, and the source, the compiled module and this signature all
/// say three.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
#[routine]
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("geglu_tanh_bfloat16", ctx.best()),
            "geglu_tanh_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

/// The same activation over rows that are not contiguous.
///
/// gemma's PLE reads a narrow gate out of a wide buffer, so each of the three
/// operands has its own pitch and all three ride the `GegluStridedParams`
/// block along with the width and the row count.
///
/// The grid is FLAT -- `width * rows` on x -- and the shader recovers the row
/// and the column by division. That is not a simplification: this kernel used
/// to be dispatched on two axes while its row said `Elementwise`, so
/// everything past the first sixteen rows came back holding the sentinel it
/// was born with, and the dispatch succeeded. The body now computes what the
/// rule it states actually means.
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
    out_pitch: Const<u32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("geglu_tanh_strided_bfloat16", ctx.best()),
            "geglu_tanh_strided_bfloat16",
        )
        .apply(elementwise(width, rows)?),
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
    alpha: Const<f32>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("gptoss_swiglu_bfloat16", ctx.best()),
            "gptoss_swiglu_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), limit.arg(), alpha.arg()],
    )
}

/// The plain gated FFN activation: `silu(gate) * up`.
///
/// Three bindings and no parameter block, which it needs none of: the output's
/// own length is the bound and the grid is the extent. It was the only kernel
/// in this file that could say so; [`geglu_tanh`] now says it too, having lost
/// a block whose single field held nothing.
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
    out: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("silu_mul_bfloat16", ctx.best()),
            "silu_mul_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[gate.arg(), up.arg(), out.arg()],
    )
}

/// [`silu_mul`] over rows that are not contiguous.
///
/// The three tensors share one pitch rather than three: `silu_mul_strided`'s
/// body walks `tid.y * row_pitch + tid.x` for all of gate, up and out, which
/// is the packed-projection case where the row a lane owns is the same
/// distance apart in each. That is why this takes one `row_pitch` where
/// [`geglu_tanh_strided`] takes a params block holding three.
///
/// The pitch is a PUSH CONSTANT, not a params buffer. `mlp/gated.slang`
/// declares `Push { int row_pitch }` for this instantiation alone, and the
/// grid is `ElementwiseRows` -- `x` over the row's width, `y` over the rows --
/// because the body reads two axes rather than recovering them from a flat
/// index.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
#[routine]
pub fn silu_mul_strided(
    ctx: &Ctx<'_>,
    gate: In<Tensor<bf16>>,
    up: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<1>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowPitch`, which no driver answers.
    let row_pitch = ctx.param(1)?;
    let width = gate.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("silu_mul_strided_bfloat16", ctx.best()),
            "silu_mul_strided_bfloat16",
        )
        .apply(elementwise_rows(width, rows)?),
        &[gate.arg(), up.arg(), out.arg(), row_pitch.arg()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that remembers, and answers the facts this family's bodies
    /// ask for.
    ///
    /// `width` came off the mark -- every routine here reads `gate.width`
    /// directly, so a test states it at construction and this probe never
    /// sees it. `rows` did not: it is `ctx.ask::<i32, keys::Rows>()` in every
    /// one of these five bodies, unconditionally and before the rectangle is
    /// even checked, so `resolve` has to answer it for ANY call to run past
    /// its first line. The default is a representative nonzero count so a
    /// call that does not care about the exact number still runs whole; a
    /// test asserting a specific dispatch, or driving the empty-rows
    /// refusal, sets it first. `row_pitch` answers `silu_mul_strided`'s own
    /// ask the same way. `ctx.params()`, which TWO of these five still call --
    /// `geglu_tanh` was the third until its block was deleted from
    /// `mlp/gated.slang` outright -- is answered generically by `Ty` alone
    /// rather than by a field of its own, since no test here inspects what a
    /// params block holds.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        row_pitch: Cell<i32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(3),
                row_pitch: Cell::new(4096),
            }
        }
    }

    impl Encode for Seen {
        // Answers the two NAMED facts this file's bodies ask by name, and
        // `ctx.params()` -- `Ty::Buf` at `Source::Slot(Params, 0)` -- by type
        // alone, since no test here inspects what the params block holds.
        // Anything else is refused: a probe that invented an answer to a
        // fact it does not know would let a body pass under test while the
        // same fact went unanswered on a real driver.
        fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            // The statement's own scalars, read by index where the params run
            // is the shader's struct -- see `Asks::param`.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                let _ = n;
                return Ok(ArgValue::I32(4096));
            }
            use kernels::Source as Src;
            if source == Src::Named("rows") {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == Src::Named("row_pitch") {
                return Ok(ArgValue::I32(self.row_pitch.get()));
            }
            if matches!(ty, kernels::Ty::Buf) {
                return Ok(ArgValue::Buffer {
                    handle: 900,
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            Err(Refusal::Unstated {
                what: "a fact this probe does not answer",
            })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
                .borrow_mut()
                .push((fire.entrypoint.to_owned(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    /// All four activations are one lane per element, on x alone.
    ///
    /// `tests/routines.rs` asks the questions that hold for every routine in
    /// the crate; this asks the one that is about what these four kernels
    /// MEAN. `gated.slang` is `[numthreads(256, 1, 1)]` and indexes with a
    /// flat `dispatchThreadID.x`, so the grid is the element count and the
    /// row structure is the shader's business, not the launch's.
    ///
    /// `geglu_tanh_strided` is the one worth stating out loud. Its operands
    /// each have their own pitch, which reads like a two-dimensional launch
    /// and was dispatched as one -- `[width, rows, 1]` against a shader that
    /// divides a flat index -- so every row past the sixteenth was never
    /// visited and the dispatch reported success. It is `width * rows` here
    /// for the same reason as the other three.
    #[test]
    fn every_activation_launches_one_lane_per_element_on_x_alone() {
        let seen = Seen::default();
        // `seen.rows` defaults to 3, which is what every assertion below
        // treats "3 rows" as meaning; `width: 64` rides each mark directly.
        geglu_tanh(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 3,
                width: 64,
            },
            In {
                ptr: Tensor::<bf16>::new(1),
                rows: 3,
                width: 64,
            },
            Out {
                ptr: Tensor::<bf16>::new(2),
                rows: 3,
                width: 64,
            },
        )
        .expect("64 wide by 3 rows is a launch");
        geglu_tanh_strided(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 3,
                width: 64,
            },
            In {
                ptr: Tensor::<bf16>::new(1),
                rows: 3,
                width: 64,
            },
            Out {
                ptr: Tensor::<bf16>::new(2),
                rows: 3,
                width: 64,
            },
            // The five that were `GegluStridedParams`. The `up` pitch is wider
            // than the width on purpose: gemma's PLE reads a narrow gate out of
            // a wide table, which is why this form exists at all.
            Const::new(64),
            Const::new(3),
            Const::new(64),
            Const::new(512),
            Const::new(64),
        )
        .expect("a pitch does not make it two-dimensional");
        gptoss_swiglu(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 3,
                width: 64,
            },
            In {
                ptr: Tensor::<bf16>::new(1),
                rows: 3,
                width: 64,
            },
            Out {
                ptr: Tensor::<bf16>::new(2),
                rows: 3,
                width: 64,
            },
            // Word 0 is the dead element count `GptOssSwiGluParams` opened
            // with; the mark holds its slot so `limit` lands on word 1, and it
            // is deliberately not passed.
            Const::new(0),
            Const::new(7.0),
            Const::new(1.702),
        )
        .expect("64 wide by 3 rows is a launch");
        silu_mul(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 3,
                width: 64,
            },
            In {
                ptr: Tensor::<bf16>::new(1),
                rows: 3,
                width: 64,
            },
            Out {
                ptr: Tensor::<bf16>::new(2),
                rows: 3,
                width: 64,
            },
        )
        .expect("64 wide by 3 rows is a launch");

        let calls = seen.calls.borrow();
        let fired: Vec<(&str, [u32; 3])> = calls
            .iter()
            .map(|(e, lanes, _)| (e.as_str(), *lanes))
            .collect();
        assert_eq!(
            fired,
            vec![
                ("geglu_tanh_bfloat16", [192, 1, 1]),
                ("geglu_tanh_strided_bfloat16", [192, 1, 1]),
                ("gptoss_swiglu_bfloat16", [192, 1, 1]),
                ("silu_mul_bfloat16", [192, 1, 1]),
            ],
            "one dispatch each, named whole, over `width * rows` lanes on x"
        );

        // AND THE DENSE GEGLU BINDS THREE, not four. Its `GegluParams` block
        // was a `PIE_PARAMS(3, ...)` declaration for a struct with one field
        // named `unused` -- and `slangc` had already compiled the binding out
        // for being unread, so the body took the handle as `_params` and threw
        // it away. The declaration is gone from `mlp/gated.slang` now.
        //
        // The two arms that DO read their numbers no longer take a block
        // either: five pitches and gpt-oss's limit and alpha are numbers no
        // grid supplies, so unlike `GegluParams` they became `Const` marks
        // rather than nothing, and they ride a push range packed from what the
        // body passes. That is what makes these four counts differ from one
        // another, and an arity that drifted would be a refusal at the first
        // real dispatch -- worth pinning here rather than discovering on a
        // device.
        let bound: Vec<usize> = calls.iter().map(|(_, _, args)| args.len()).collect();
        assert_eq!(
            bound,
            vec![3, 8, 5, 3],
            "geglu_tanh and silu_mul take gate, up and out and stop. The \
             strided GeGLU takes those three plus `GegluStridedParams`' five \
             words. gpt-oss takes FIVE and not six: its struct opened with a \
             dead element count, and the mark holding that slot is declared \
             and never passed, so the dead word reaches no argument list"
        );
    }

    /// A rectangle with no elements is refused, not dispatched.
    ///
    /// `vkCmdDispatch(0, 1, 1)` is valid Vulkan: it runs nothing, returns
    /// success, and leaves the output holding whatever it held. An activation
    /// is the last thing before a down-projection, so what that costs is a
    /// hidden state made of uninitialised memory, silently.
    ///
    /// Zero arrives here honestly. `width` is an expert's intermediate size
    /// and `rows` is how many tokens routed to it, which is zero on any step
    /// where an expert wins nothing.
    #[test]
    fn an_empty_rectangle_is_refused() {
        let seen = Seen::default();
        seen.rows.set(0);
        assert!(
            matches!(
                silu_mul(
                    &seen,
                    In {
                        ptr: Tensor::<bf16>::new(0),
                        rows: 0,
                        width: 64
                    },
                    In {
                        ptr: Tensor::<bf16>::new(1),
                        rows: 0,
                        width: 64
                    },
                    Out {
                        ptr: Tensor::<bf16>::new(2),
                        rows: 0,
                        width: 64
                    },
                ),
                Err(Refusal::Empty { what: "rows" })
            ),
            "no rows routed to this expert is not a launch of nothing"
        );
        // `rows` is asked unconditionally, before the rectangle is checked,
        // so it still has to resolve to something here even though the
        // width failure below fires first; its value plays no other part.
        seen.rows.set(3);
        assert!(
            matches!(
                silu_mul(
                    &seen,
                    In {
                        ptr: Tensor::<bf16>::new(0),
                        rows: 3,
                        width: 0
                    },
                    In {
                        ptr: Tensor::<bf16>::new(1),
                        rows: 3,
                        width: 0
                    },
                    Out {
                        ptr: Tensor::<bf16>::new(2),
                        rows: 3,
                        width: 0
                    },
                ),
                Err(Refusal::Empty { what: "width" })
            ),
            "a zero-wide row is refused too, and the refusal NAMES which \
             extent was empty -- the two arrive from different places and a \
             caller that wants to fall back needs to know which"
        );
        assert!(
            seen.calls.borrow().is_empty(),
            "a refusal reaches the driver as nothing at all"
        );
    }

    /// A rectangle too big to count in 32 bits is refused rather than wrapped.
    ///
    /// The lane count is a `u32` because `vkCmdDispatch` takes one. A product
    /// that does not fit would wrap to a SMALL number, which dispatches a
    /// grid that covers a fraction of the buffer and succeeds -- the same
    /// failure shape as the empty grid, arrived at from the other end.
    #[test]
    fn a_rectangle_too_large_to_count_is_refused() {
        let seen = Seen::default();
        seen.rows.set(1 << 13);
        let e = silu_mul(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 1 << 13,
                width: 1 << 20,
            },
            In {
                ptr: Tensor::<bf16>::new(1),
                rows: 1 << 13,
                width: 1 << 20,
            },
            Out {
                ptr: Tensor::<bf16>::new(2),
                rows: 1 << 13,
                width: 1 << 20,
            },
        );
        assert!(
            matches!(
                e,
                Err(Refusal::Grid {
                    what: "width * rows",
                    at: 8_589_934_592
                })
            ),
            "the refusal carries the true product, computed in 64 bits, so the \
             number in the message is the one that did not fit: {e:?}"
        );
    }
}
