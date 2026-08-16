//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use kernels::routine::Refusal;
use kernels::{KernelSig, kernel};

use crate::axes::*;
use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine, elementwise};

/// gemma's gated activation: `gelu_tanh(gate) * up`, the tanh approximation
/// and not the erf one.
///
/// `mlp/gated.slang` under `PIE_GEGLU`. Bindings 0..=3 are gate, up, out and
/// the `GegluParams` block, in that order, which is the order this signature
/// states and not the order a trace states them.
///
/// `params` is a buffer and not a scalar run: `PIE_PARAMS` expands to a
/// `std430 readonly buffer`. Its one member is named `unused`, and it is --
/// the shader bounds itself with `GetDimensions` on the output rather than
/// with a count the caller passes. The binding still has to be filled, because
/// a descriptor the layout declares and nothing writes is a fault inside
/// `vkCreateComputePipelines`, not an error.
///
/// # Errors
///
/// See [`crate::routine::elementwise`].
pub fn geglu_tanh(
    ctx: &Ctx<'_>,
    gate: Buf,
    up: Buf,
    out: BufMut,
    _params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "geglu_tanh_bfloat16",
            lanes: elementwise(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v()],
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
            lanes: elementwise(*width, *rows)?,
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
/// shader through `params` rather than as `f32` arguments here.
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
            lanes: elementwise(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v(), params.v()],
    )
}

/// The plain gated FFN activation: `silu(gate) * up`.
///
/// Three bindings and no parameter block -- the only kernel in this file
/// without one, because it needs no scalar the output's own length does not
/// give it.
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
            lanes: elementwise(*width, *rows)?,
        },
        &[gate.v(), up.v(), out.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(geglu_tanh),
    crate::routine!(geglu_tanh_strided),
    crate::routine!(gptoss_swiglu),
    crate::routine!(silu_mul),
];

/// The entrypoints this family's crossed routines spell, now that their rows
/// are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "geglu_tanh_bfloat16",
    "geglu_tanh_strided_bfloat16",
    "gptoss_swiglu_bfloat16",
    "silu_mul_bfloat16",
];

/// What is left of the table: the one kernel in `mlp/gated.slang` that has no
/// routine. `silu_mul_strided` was never written, here or on Metal, and until
/// it is the row is how it is dispatched -- which is why
/// `driver-vulkan/src/arm.rs` RESERVES its stem rather than letting
/// `silu_mul` claim it.
pub static KERNELS: &[KernelSig] = &[
    // 1 in mlp/gated.slang
    kernel!(silu_mul_strided "silu_mul_strided", file = Some("mlp/gated.slang"), axes = &[BF16]),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    type Call = (String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that remembers. See `sample.rs` for why a body is worth
    /// checking without a device.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
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
        geglu_tanh(&seen, Buf(0), Buf(1), BufMut(2), Buf(3), Env(64), Env(3))
            .expect("64 wide by 3 rows is a launch");
        geglu_tanh_strided(&seen, Buf(0), Buf(1), BufMut(2), Buf(3), Env(64), Env(3))
            .expect("a pitch does not make it two-dimensional");
        gptoss_swiglu(&seen, Buf(0), Buf(1), BufMut(2), Buf(3), Env(64), Env(3))
            .expect("64 wide by 3 rows is a launch");
        silu_mul(&seen, Buf(0), Buf(1), BufMut(2), Env(64), Env(3))
            .expect("64 wide by 3 rows is a launch");

        let calls = seen.0.borrow();
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
        assert!(
            matches!(
                silu_mul(&seen, Buf(0), Buf(1), BufMut(2), Env(64), Env(0)),
                Err(Refusal::Empty { what: "rows" })
            ),
            "no rows routed to this expert is not a launch of nothing"
        );
        assert!(
            matches!(
                silu_mul(&seen, Buf(0), Buf(1), BufMut(2), Env(0), Env(3)),
                Err(Refusal::Empty { what: "width" })
            ),
            "a zero-wide row is refused too, and the refusal NAMES which \
             extent was empty -- the two arrive from different places and a \
             caller that wants to fall back needs to know which"
        );
        assert!(
            seen.0.borrow().is_empty(),
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
        let e = silu_mul(&seen, Buf(0), Buf(1), BufMut(2), Env(1 << 20), Env(1 << 13));
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
