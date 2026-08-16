//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine, elementwise};

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
/// `mlp/gated.metal`. Buffers 0..=3 are gate, up, out and the `GegluParams`
/// block, in that order, which is the order this signature states and not the
/// order a trace states them.
///
/// `params`' one member is named `unused` and it is: the body bounds itself
/// with the grid. The buffer is still bound, because the entrypoint declares
/// the argument and an argument table with a hole in it is not something the
/// encoder can be asked for.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "geglu_tanh_bfloat16",
            file: "mlp/gated.metal",
            lanes: elementwise(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[gate.v(), up.v(), out.v(), params.v()],
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
pub fn geglu_tanh_strided(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "geglu_tanh_strided_bfloat16",
            file: "mlp/gated.metal",
            lanes: elementwise(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[gate.v(), up.v(), out.v(), params.v()],
    )
}

/// gpt-oss's activation, which is not anyone else's.
///
/// The gate is clamped ABOVE only, the linear branch is clamped both ways and
/// carries a `+1`, and the sigmoid takes an `alpha`. `silu_mul` cannot serve
/// it: dropping either the clamp or the `+1` produces a model that runs and is
/// wrong, which is why this is a symbol a text names rather than a flag.
///
/// `limit` and `alpha` ride the `GptOssSwiGluParams` block, so they reach the
/// kernel through `params` rather than as `f32` arguments here.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
pub fn gptoss_swiglu(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gptoss_swiglu_bfloat16",
            file: "mlp/gated.metal",
            lanes: elementwise(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[gate.v(), up.v(), out.v(), params.v()],
    )
}

/// The plain gated FFN activation: `silu(gate) * up`.
///
/// Three buffers and no parameter block -- the only kernel in this file
/// without one, because it needs no scalar the grid does not give it. It
/// indexes `tid` raw and guards nothing, which is the shape the other three
/// bodies in this file have now been brought to.
///
/// The silu is computed through bf16 twice on purpose: `sigmoid` and the
/// product are each rounded to bf16 before the multiply, which is what MLX
/// does, and matching it is what lets this backend's output be compared
/// against the reference bit for bit rather than within a tolerance.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
pub fn silu_mul(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "silu_mul_bfloat16",
            file: "mlp/gated.metal",
            lanes: elementwise(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[gate.v(), up.v(), out.v()],
    )
}

/// This family's routines.
///
/// Four of the five. `silu_mul_strided` does NOT cross, and the reason is a
/// property of Metal's argument table rather than a judgment: its entrypoint
/// declares `row_pitch` at **buffer(4)** with buffer(3) left empty, and a
/// routine's argument list is positional -- the index in the list IS the slot.
/// Neither this plane nor the table plane it replaces can express a hole, so
/// crossing it would mean either binding a null at 3 or renumbering the
/// entrypoint. It is DARK -- no text names it, and no statement produces the
/// row pitch it wants -- so neither is worth doing to a symbol nothing calls.
/// `kernels-vulkan` left it uncrossed too.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(geglu_tanh),
    crate::routine!(geglu_tanh_strided),
    crate::routine!(gptoss_swiglu),
    crate::routine!(silu_mul),
];

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
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// Every body in this family binds `gate, up, out` at 0, 1, 2 -- which is
    /// the one thing the file's first line says they have in common.
    ///
    /// A positional bind in the TRACE's order rather than the kernel's is how
    /// `rms_single_row` came to read its own output as the weight, so the
    /// order is worth a test even where it looks obvious.
    #[test]
    fn all_four_bodies_bind_gate_up_and_out_at_zero_one_and_two() {
        let seen = Seen::default();
        geglu_tanh(&seen, Buf(1), Buf(2), BufMut(3), Buf(4), Env(8), Env(2)).expect("a launch");
        geglu_tanh_strided(&seen, Buf(1), Buf(2), BufMut(3), Buf(4), Env(8), Env(2))
            .expect("a launch");
        gptoss_swiglu(&seen, Buf(1), Buf(2), BufMut(3), Buf(4), Env(8), Env(2)).expect("a launch");
        silu_mul(&seen, Buf(1), Buf(2), BufMut(3), Env(8), Env(2)).expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(calls.len(), 4, "one activation is one dispatch");
        for (fire, args) in calls.iter() {
            assert_eq!(
                args[..3],
                [
                    ArgValue::Buffer(1),
                    ArgValue::Buffer(2),
                    ArgValue::Buffer(3)
                ],
                "`{}` binds gate, up, out at 0, 1, 2",
                fire.entrypoint
            );
            assert_eq!(
                fire.file, "mlp/gated.metal",
                "`{}`: all four are one file because they are one binding \
                 contract, and Metal compiles from (path, entry name)",
                fire.entrypoint
            );
        }
        assert_eq!(calls[0].1.len(), 4, "geglu_tanh takes GegluParams at 3");
        assert_eq!(
            calls[1].1.len(),
            4,
            "the strided one takes its pitches at 3"
        );
        assert_eq!(
            calls[2].1.len(),
            4,
            "gpt-oss takes its limit and alpha at 3"
        );
        assert_eq!(
            calls[3].1.len(),
            3,
            "silu_mul is the only one without a parameter block -- it needs no \
             scalar the grid does not give it"
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
        geglu_tanh_strided(&seen, Buf(1), Buf(2), BufMut(3), Buf(4), Env(256), Env(8))
            .expect("eight rows is a launch");

        let calls = seen.0.borrow();
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
        let refused = silu_mul(&seen, Buf(1), Buf(2), BufMut(3), Env(8), Env(0))
            .expect_err("zero rows is not a launch");
        assert!(
            matches!(refused, Refusal::Empty { what: "rows" }),
            "got {refused:?}"
        );
        assert!(
            seen.0.borrow().is_empty(),
            "and nothing was encoded on the way to refusing"
        );
    }
}
