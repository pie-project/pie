//! Sampling. One kernel: the device argmax with its EOS compare.
//!
//! The first family to cross to the routine shape
//! (`.wiki/kernel-x/vulkan-refactor.md` §6 step 1), chosen because it is one
//! kernel and so is the smallest thing that can prove the whole surface.

use kernels::routine::Refusal;

use crate::routine::{keys, Ask, Bind, Buf, BufMut, Ctx, Fire, Routine};
use crate::routine::{InSlot, OutSlot};

/// `sample/argmax.slang:11` — `#define PIE_GROUP_X 1024`.
///
/// Stated here only to be handed back as a LANE count on x. The division into
/// workgroups is the driver's, from the SPIR-V's own `LocalSize`; this is the
/// width of the one workgroup a row is given, which is a different fact that
/// happens to be the same number.
const GROUP_X: u32 = 1024;

/// Device argmax over each row's logits, with the EOS compare fused.
///
/// `sample/argmax.slang`. One workgroup owns one row and reduces through
/// `groupshared` rather than a subgroup, so the kernel does not assume a
/// 32-lane subgroup — which is what lets it run unchanged on a device whose
/// subgroup is 64 (AMD) or 16 (some Intel).
///
/// The four buffers are the shader's bindings 0..=3 IN THAT ORDER, which is
/// the order this signature states and not the order a trace states them.
/// `params` is a buffer and not a scalar run: `PIE_PARAMS` expands to a
/// `std430 readonly buffer`, and the block holds `eos_ids[8]`, an array no
/// push-constant scalar list can carry.
///
/// # Errors
///
/// [`Refusal::Empty`] when there is no row to sample. That is a refusal and
/// not a zero-lane dispatch on purpose: `vkCmdDispatch(0, 1, 1)` is legal
/// Vulkan that runs nothing and reports success.
pub fn argmax_logits(
    ctx: &Ctx<'_>,
    logits: InSlot<0, Buf>,
    next_token: OutSlot<0, BufMut>,
    params: InSlot<1, Buf>,
    eos_flag: OutSlot<1, BufMut>,
    rows: Ask<keys::Rows, u32>,
) -> Result<(), Refusal> {
    if *rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `group.y` is the row, and x carries the one workgroup that reduces it.
    ctx.dispatch(
        Fire {
            entrypoint: "argmax_logits_bfloat16",
            lanes: [GROUP_X, *rows, 1],
        },
        &[logits.v(), next_token.v(), params.v(), eos_flag.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[crate::routine!(argmax_logits)];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// A row's `axes` used to generate these, and `entrypoints()` collected them
/// off the table. A retired row still has its shader and its module -- the
/// crossing moved WHO NAMES IT, not what exists -- so the name has to be
/// stated somewhere or the census would read a successful crossing as a
/// shader that had disappeared. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &["argmax_logits_bfloat16"];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// An `Encode` that remembers what it was asked to do.
    ///
    /// The whole point of the crossing is that a body is reachable without a
    /// device: everything `argmax_logits` decides — the entrypoint spelling,
    /// the lane counts, the argument ORDER — is host arithmetic over host
    /// values, and none of it needs a GPU to check. `tests/gpu.rs` proves the
    /// kernel computes an argmax; this proves the body asks for the right one.
    /// One recorded dispatch: the entrypoint, the lanes, the argument list.
    type Call = (String, [u32; 3], Vec<ArgValue>);

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

    /// The body names the entrypoint the shader tree declares, and lays its
    /// arguments out in the SHADER's binding order.
    ///
    /// Both halves have been wrong here before, in ways nothing reported. An
    /// entrypoint assembled by pasting an axis suffix onto a row name is how
    /// `neox_freqs_mb` came to name the DECODE symbol — a single-row kernel
    /// over a multi-row grid, which rotated row zero and left every row after
    /// it untouched. And a positional bind in the trace's order rather than
    /// the kernel's is how `rms_single_row` read its own output as the weight:
    /// the shader binds `0=x, 1=w, 2=out` and the trace hands over
    /// `In(0), Out(0), Weight(0)`.
    #[test]
    fn the_body_asks_for_the_entrypoint_and_the_order_the_shader_declares() {
        let seen = Seen::default();
        argmax_logits(&seen, InSlot::new(Buf(10)), OutSlot::new(BufMut(11)), InSlot::new(Buf(12)), OutSlot::new(BufMut(13)), Ask::new(7))
            .expect("seven rows is a launch");

        let calls = seen.0.borrow();
        assert_eq!(calls.len(), 1, "one batch of logits is one dispatch");
        let (entrypoint, lanes, args) = &calls[0];
        assert_eq!(
            entrypoint, "argmax_logits_bfloat16",
            "the entrypoint is spelled whole, as its instantiate line has it"
        );
        assert_eq!(
            *lanes,
            [GROUP_X, 7, 1],
            "one workgroup wide on x, one row per group on y -- the shader reads \
             the row off `group.y`"
        );
        assert_eq!(
            args,
            &[
                ArgValue::Buffer {
                    handle: 10,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 11,
                    writes: true
                },
                ArgValue::Buffer {
                    handle: 12,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 13,
                    writes: true
                },
            ],
            "bindings 0..=3 in the shader's order -- logits, next_token, params, \
             eos_flag -- and the two the kernel WRITES carry that, off the \
             `BufMut` in the signature and nothing else"
        );
    }

    /// No rows is a refusal, not a dispatch of nothing.
    ///
    /// `vkCmdDispatch` with a zero group count is legal, runs nothing and
    /// reports success. A caller that got `Ok` here would go on to read a
    /// `next_token` buffer that still holds whatever it held.
    #[test]
    fn an_empty_batch_is_refused_rather_than_dispatched() {
        let seen = Seen::default();
        assert_eq!(
            argmax_logits(&seen, InSlot::new(Buf(0)), OutSlot::new(BufMut(1)), InSlot::new(Buf(2)), OutSlot::new(BufMut(3)), Ask::new(0)),
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
