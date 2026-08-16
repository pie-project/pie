//! Sampling. One kernel: the device argmax with its EOS compare.
//!
//! The first family to cross to the routine shape
//! (`.wiki/kernel-x/refactor-bigplan.md` §7 Stage 3), chosen because it is one
//! kernel and so is the smallest thing that can prove the whole surface —
//! and because no text names its symbol, so the crossing cannot change what
//! any model computes. `kernels-vulkan` crossed the same family first for the
//! same two reasons.

use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};

/// `sample/argmax.metal:24` — "Grid{1024, n_rows, 1}, Threadgroup{1024,1,1}".
///
/// One threadgroup, 32 simdgroups, owns one row. The kernel reads its own
/// width back off `threads_per_threadgroup` and strides the vocabulary by it,
/// so the number is load-bearing in both places at once: it is the lane count
/// on x AND the reduction's stride, and a grid stating one without the other
/// would have threads reading past the row or leaving its tail unscanned.
const GROUP_X: u32 = 1024;

/// Device argmax over each row's logits, with the EOS compare fused.
///
/// `sample/argmax.metal`. One threadgroup owns one row and reduces through
/// simdgroup shuffles and then threadgroup memory, keeping the LOWEST index on
/// every tie — which is what makes it bit-identical to the host scan it
/// replaced, and the reason that property is stated in the shader's own header
/// rather than left to be rediscovered.
///
/// The four buffers are the shader's `buffer(0..=3)` IN THAT ORDER, which is
/// the order this signature states and not the order a trace states them.
/// `params` is a buffer and not a scalar run: it is `constant ArgmaxParams&`,
/// and the struct holds `eos_ids[8]` — an array no scalar parameter slot can
/// carry.
///
/// # Errors
///
/// [`Refusal::Empty`] when there is no row to sample. A refusal and not a
/// zero-lane dispatch on purpose: `dispatchThreads:` over an empty grid is
/// legal Metal that runs nothing and reports success, so `next_token` would
/// keep whatever it held and the loop would sample a stale token.
pub fn argmax_logits(
    ctx: &Ctx<'_>,
    logits: Buf,
    next_token: BufMut,
    params: Buf,
    eos_flag: BufMut,
    rows: Env<u32>,
) -> Result<(), Refusal> {
    if *rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `tg_pos.y` is the row, and x carries the one threadgroup that reduces it.
    ctx.dispatch(
        Fire {
            entrypoint: "argmax_logits_bfloat16",
            file: "sample/argmax.metal",
            lanes: [GROUP_X, *rows, 1],
            group: [GROUP_X, 1, 1],
        },
        &[logits.v(), next_token.v(), params.v(), eos_flag.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[crate::routine!(argmax_logits)];

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
pub static ENTRYPOINTS: &[(&str, &str)] = &[("sample/argmax.metal", "argmax_logits_bfloat16")];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do.
    ///
    /// The whole point of the crossing is that a body is reachable without a
    /// device: everything `argmax_logits` decides — the entrypoint spelling,
    /// the file, the lane counts, the threadgroup, the argument ORDER — is
    /// host arithmetic over host values, and none of it needs a GPU to check.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// The body names the entrypoint the shader tree declares, in the file it
    /// declares it in, and lays its arguments out in the SHADER's binding
    /// order.
    ///
    /// Every half of this has been wrong somewhere in this tree, in ways
    /// nothing reported. An entrypoint assembled by pasting an axis suffix
    /// onto a row name is how `neox_freqs_mb` came to name the DECODE symbol —
    /// a single-row kernel over a multi-row grid, which rotated row zero and
    /// left every row after it untouched. And a positional bind in the trace's
    /// order rather than the kernel's is how `rms_single_row` read its own
    /// output as the weight: the shader binds `0=x, 1=w, 2=out` and the trace
    /// hands over `In(0), Out(0), Weight(0)`.
    #[test]
    fn the_body_asks_for_the_entrypoint_and_the_order_the_shader_declares() {
        let seen = Seen::default();
        argmax_logits(&seen, Buf(10), BufMut(11), Buf(12), BufMut(13), Env(7))
            .expect("seven rows is a launch");

        let calls = seen.0.borrow();
        assert_eq!(calls.len(), 1, "one batch of logits is one dispatch");
        let (fire, args) = &calls[0];
        assert_eq!(
            fire.entrypoint, "argmax_logits_bfloat16",
            "spelled whole, as the census has it -- the shader's \
             `instantiate_argmax(bfloat16, bfloat)` is what produces that name"
        );
        assert_eq!(
            fire.file, "sample/argmax.metal",
            "Metal compiles at run time from (path, entry name), so a body that \
             named the entrypoint without the file could not be dispatched"
        );
        assert_eq!(
            fire.lanes,
            [GROUP_X, 7, 1],
            "THREADS: one threadgroup wide on x, one row per group on y -- the \
             shader reads the row off `tg_pos.y`. Writing 1 here instead of \
             1024, as a threadGROUP count, would launch one thread per row and \
             the simd reductions would read lanes that were never dispatched."
        );
        assert_eq!(
            fire.group,
            [GROUP_X, 1, 1],
            "and the kernel reads this same number back off \
             `threads_per_threadgroup` to stride the vocabulary, so a smaller \
             group would leave the tail of every row unscanned"
        );
        assert_eq!(
            args,
            &[
                ArgValue::Buffer(10),
                ArgValue::Buffer(11),
                ArgValue::Buffer(12),
                ArgValue::Buffer(13),
            ],
            "buffer(0..=3) in the shader's order -- logits, next_token, params, \
             eos_flag"
        );
    }

    /// No rows is a refusal, not a dispatch of nothing.
    #[test]
    fn an_empty_batch_is_refused_rather_than_dispatched() {
        let seen = Seen::default();
        assert_eq!(
            argmax_logits(&seen, Buf(0), BufMut(1), Buf(2), BufMut(3), Env(0)),
            Err(Refusal::Empty { what: "rows" })
        );
        assert!(
            seen.0.borrow().is_empty(),
            "a refusal must not have reached the device first"
        );
    }

    /// The derived row says what the signature says.
    ///
    /// This is the property the whole crossing is for: the table cannot drift
    /// from the code, because there is only one statement of it. Four buffers
    /// the trace supplies and one extent the environment does — a `rows` the
    /// program being run never names.
    #[test]
    fn the_derived_row_is_the_signature() {
        let row = &ROUTINES[0];
        assert_eq!(row.name, "argmax_logits");
        assert_eq!(
            row.args,
            &[
                (kernels::Ty::Buf, crate::routine::Supplier::Trace),
                (kernels::Ty::BufMut, crate::routine::Supplier::Trace),
                (kernels::Ty::Buf, crate::routine::Supplier::Trace),
                (kernels::Ty::BufMut, crate::routine::Supplier::Trace),
                (kernels::Ty::U32, crate::routine::Supplier::Env),
            ]
        );
    }
}
