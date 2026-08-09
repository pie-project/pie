//! The 64-byte record one op hands its kernel.
//!
//! An emitted region kernel is generic over the op it runs: the same MSL
//! function body serves every `add`, and what distinguishes one dispatch from
//! the next is a [`OpParams`] record the host writes into a parameter buffer,
//! one per region, indexed by the region's ordinal. It is a flat sixteen `u32`
//! words because that is what a kernel can read without a struct layout
//! agreement beyond "sixteen words in this order".
//!
//! Most of the record is a copy of the plan's op. Four things are not, and they
//! are the reason this is code rather than a `memcpy`:
//!
//! * **`a1` carries `pivot_threshold`'s predicate.** That op takes one argument
//!   and a predicate *value id*, and the id rides in the second argument slot
//!   because the kernel already has a binding for it. Anything that decides
//!   which slots to bind has to know this, which is why
//!   [`OpParams::binds_second_argument`] exists rather than each call site
//!   re-deriving `args.len() > 1 || tag == PIVOT_THRESHOLD`.
//! * **`o0` falls back to `a0` for an op with no results.** A kernel binds its
//!   output slot unconditionally, so a result-less op needs the slot to point
//!   at something real; pointing it at the op's own first input is the choice
//!   that needs no extra allocation.
//! * **`imm` is the vocabulary for an intrinsic.** The plan cannot know it -- it
//!   is a property of the model, not of the program -- so the runtime
//!   substitutes it.
//! * **`imm2` is the MTP draft row for the two MTP intrinsics**, for the same
//!   reason, and only when the fire actually has one.
//!
//! # What this module deliberately does not do
//!
//! Nothing here touches a buffer, a channel cell or the GPU. The C++ filled
//! this record inside a six-hundred-line loop that also resolved slot handles,
//! validated channel bindings, checked logits row ranges and jumped to
//! `cleanup_failure` with a `goto`. Those are separate concerns that happen to
//! share a loop, and the record is the one part of it that is a pure function of
//! the plan plus three runtime numbers -- so it is a pure function here, and it
//! is tested without a device.

use driver_api::plan::LaunchOp;
use tensor_ir::op::IntrinsicId;
use tensor_ir::op::tags;

/// The runtime numbers an op's record needs that its plan cannot carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Runtime {
    /// The model's vocabulary size, which every logits intrinsic is shaped on.
    pub vocab: u32,
    /// Which draft row this fire is running, for the MTP intrinsics.
    ///
    /// `None` where the C++ used a negative `int`: the sentinel and the value
    /// shared one field there, so every read had to remember to test it, and
    /// the two places that did test it wrote the condition out longhand.
    pub mtp_draft_row: Option<u32>,
}

/// One op's parameters, in the layout the kernels read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct OpParams {
    /// `PTIR_OP_*`.
    pub tag: u32,
    /// First argument's value slot.
    pub a0: u32,
    /// Second argument's value slot, or `pivot_threshold`'s predicate.
    pub a1: u32,
    /// Third argument's value slot.
    pub a2: u32,
    /// First result's value slot, or `a0` for an op with no results.
    pub o0: u32,
    /// Second result's value slot, or `o0` for an op with fewer than two.
    pub o1: u32,
    /// The op's immediate, or the vocabulary for an intrinsic.
    pub imm: u32,
    /// The op's second immediate, or the MTP draft row.
    pub imm2: u32,
    /// The op's third immediate.
    pub imm3: u32,
    /// RNG kind: 0 uniform, 1 gumbel.
    pub kind: u32,
    /// `pivot_threshold`'s predicate tag.
    pub pred_tag: u32,
    /// A const literal's dtype.
    pub lit_dtype: u32,
    /// A const literal's raw bits.
    pub lit_bits: u32,
    /// The channel slot a channel op targets.
    pub channel_slot: u32,
    /// `PTIR_INTR_*`, for `intrinsic_val`.
    pub intr: u32,
    /// The fixed cell size a `chan_put` writes into; filled when the sink is
    /// bound, which is not this module's job.
    pub sink_bytes: u32,
}

/// Sixteen words is the agreement with the emitted kernels. The C++ asserted
/// the same number; keeping the assertion means a field added here fails to
/// compile instead of shifting every subsequent op's parameters by four bytes.
const _: () = assert!(size_of::<OpParams>() == 64);

/// The `u32` a channel op's slot is written as when the op targets no channel.
///
/// The wire form spells "no channel" as `u32::MAX`, and the device record has
/// no such spelling -- it would be indistinguishable from a real slot to a
/// kernel that read it. Zero is what the C++ wrote and is kept, because no
/// kernel reads `channel_slot` for an op that is not a channel op. The
/// substitution is named so that the value is not mistaken for slot zero by
/// anyone reading a dumped record.
const NO_CHANNEL: u32 = 0;

impl OpParams {
    /// Build one op's record.
    ///
    /// `result_base` is the op's first result slot within its stage -- the
    /// running prefix sum of result counts in plan order, which the caller
    /// tracks because it is a property of the walk, not of the op.
    #[must_use]
    pub fn of(op: &LaunchOp, result_base: u32, runtime: Runtime) -> OpParams {
        let tag = op.code as u8;
        let intrinsic = tag == tags::INTRINSIC_VAL;
        let a0 = op.args.first().copied().unwrap_or(0);
        // `pivot_threshold` has one real argument and a predicate value id;
        // the id rides in the second slot because the kernel binds it there.
        let a1 = op
            .args
            .get(1)
            .copied()
            .unwrap_or(if tag == tags::PIVOT_THRESHOLD {
                op.pred_payload
            } else {
                0
            });
        // A kernel binds its output slot unconditionally, so a result-less op
        // points it at its own first input rather than at nothing.
        let o0 = if op.result_count > 0 { result_base } else { a0 };
        OpParams {
            tag: u32::from(op.code),
            a0,
            a1,
            a2: op.args.get(2).copied().unwrap_or(0),
            o0,
            // Saturating rather than wrapping: a second result slot at
            // `u32::MAX` cannot exist, and aliasing it onto slot zero would
            // hand the kernel a live value to overwrite.
            o1: if op.result_count > 1 {
                o0.saturating_add(1)
            } else {
                o0
            },
            imm: if intrinsic { runtime.vocab } else { op.imm },
            imm2: match runtime.mtp_draft_row {
                Some(row) if intrinsic && is_mtp(op.intrinsic) => row,
                _ => op.imm2,
            },
            imm3: op.imm3,
            kind: u32::from(op.rng_kind),
            pred_tag: u32::from(op.pred_tag),
            lit_dtype: u32::from(op.lit_dtype),
            lit_bits: op.lit_bits,
            channel_slot: if op.channel == u32::MAX {
                NO_CHANNEL
            } else {
                op.channel
            },
            intr: u32::from(op.intrinsic),
            sink_bytes: 0,
        }
    }

    /// Whether the kernel's second argument slot must be bound to a value.
    ///
    /// Not `a1 != 0`: slot zero is a real value, and `pivot_threshold` binds
    /// the slot with no second argument at all. The C++ wrote this condition
    /// inline at each of the three places it bound arguments, which is three
    /// chances for one of them to drift.
    #[must_use]
    pub fn binds_second_argument(op: &LaunchOp) -> bool {
        op.args.len() > 1 || op.code as u8 == tags::PIVOT_THRESHOLD
    }
}

/// Whether an intrinsic id is one of the two the MTP draft row applies to.
fn is_mtp(intrinsic: u16) -> bool {
    intrinsic == IntrinsicId::MtpLogits as u16 || intrinsic == IntrinsicId::MtpDrafts as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(code: u8) -> LaunchOp {
        LaunchOp {
            code: u16::from(code),
            result_count: 1,
            channel: u32::MAX,
            ..LaunchOp::default()
        }
    }

    /// The straightforward case, so the interesting ones below are visibly
    /// departures from it.
    #[test]
    fn an_ordinary_op_copies_its_arguments_and_takes_its_result_base() {
        let mut add = op(tags::ADD);
        add.args = vec![4, 5];
        add.imm = 11;
        let params = OpParams::of(&add, 9, Runtime::default());
        assert_eq!((params.a0, params.a1, params.a2), (4, 5, 0));
        assert_eq!((params.o0, params.o1), (9, 9));
        assert_eq!(params.imm, 11);
        assert_eq!(params.tag, u32::from(tags::ADD));
    }

    /// Two results occupy consecutive slots, because `result_base` is a prefix
    /// sum and an op's results are contiguous within it.
    #[test]
    fn a_two_result_op_takes_consecutive_slots() {
        let mut sort = op(tags::SORT_DESC);
        sort.result_count = 2;
        let params = OpParams::of(&sort, 9, Runtime::default());
        assert_eq!((params.o0, params.o1), (9, 10));
    }

    /// A kernel binds its output slot whether or not the op has one, so the
    /// slot must name a real value. `chan_put` is the op this exists for.
    #[test]
    fn a_result_less_op_points_its_output_slot_at_its_own_first_input() {
        let mut put = op(tags::CHAN_PUT);
        put.result_count = 0;
        put.args = vec![7];
        put.channel = 3;
        let params = OpParams::of(&put, 9, Runtime::default());
        assert_eq!(
            (params.o0, params.o1),
            (7, 7),
            "an output slot pointing at nothing is a binding the kernel cannot make"
        );
        assert_eq!(params.channel_slot, 3);
    }

    /// `pivot_threshold` smuggles a value id through the second argument slot.
    /// Both the record and the binding predicate have to agree about it.
    #[test]
    fn pivot_threshold_puts_its_predicate_in_the_second_argument_slot() {
        let mut pivot = op(tags::PIVOT_THRESHOLD);
        pivot.args = vec![4];
        pivot.pred_payload = 6;
        pivot.pred_tag = 2;
        let params = OpParams::of(&pivot, 9, Runtime::default());
        assert_eq!((params.a0, params.a1), (4, 6));
        assert_eq!(params.pred_tag, 2);
        assert!(
            OpParams::binds_second_argument(&pivot),
            "the predicate is in the slot, so the slot must be bound"
        );
    }

    /// The same predicate, without the special case, would leave the slot
    /// unbound and the kernel reading whatever was there.
    #[test]
    fn an_ordinary_one_argument_op_does_not_bind_the_second_slot() {
        let mut neg = op(tags::NEG);
        neg.args = vec![4];
        assert!(!OpParams::binds_second_argument(&neg));
    }

    /// The plan cannot carry the vocabulary; it is a property of the model.
    #[test]
    fn an_intrinsic_takes_the_runtime_vocabulary_over_its_own_immediate() {
        let mut logits = op(tags::INTRINSIC_VAL);
        logits.intrinsic = IntrinsicId::Logits as u16;
        logits.imm = 1;
        let params = OpParams::of(
            &logits,
            0,
            Runtime {
                vocab: 128_256,
                mtp_draft_row: None,
            },
        );
        assert_eq!(params.imm, 128_256);
    }

    /// The draft row applies to the two MTP intrinsics and to nothing else --
    /// substituting it into a plain `logits` would silently reshape it.
    #[test]
    fn only_the_mtp_intrinsics_take_the_draft_row() {
        let runtime = Runtime {
            vocab: 32,
            mtp_draft_row: Some(3),
        };
        for (intrinsic, expected) in [
            (IntrinsicId::MtpLogits, 3),
            (IntrinsicId::MtpDrafts, 3),
            (IntrinsicId::Logits, 7),
            (IntrinsicId::Hidden, 7),
        ] {
            let mut intr = op(tags::INTRINSIC_VAL);
            intr.intrinsic = intrinsic as u16;
            intr.imm2 = 7;
            assert_eq!(
                OpParams::of(&intr, 0, runtime).imm2,
                expected,
                "{}",
                intrinsic.name()
            );
        }
    }

    /// A fire with no draft row leaves the op's own immediate alone. The C++
    /// spelled the absent row as a negative `int` in the same field as the
    /// value, so every read had to remember to test it.
    #[test]
    fn a_fire_without_a_draft_row_leaves_the_immediate_alone() {
        let mut mtp = op(tags::INTRINSIC_VAL);
        mtp.intrinsic = IntrinsicId::MtpLogits as u16;
        mtp.imm2 = 7;
        let params = OpParams::of(
            &mtp,
            0,
            Runtime {
                vocab: 32,
                mtp_draft_row: None,
            },
        );
        assert_eq!(params.imm2, 7);
    }

    /// The wire spells "no channel" as `u32::MAX` and the device record has no
    /// such spelling. Zero is what goes out, and no kernel reads the field for
    /// an op that is not a channel op.
    #[test]
    fn an_op_that_touches_no_channel_writes_slot_zero() {
        let params = OpParams::of(&op(tags::ADD), 0, Runtime::default());
        assert_eq!(params.channel_slot, NO_CHANNEL);
    }

    /// A second result slot cannot exist past the end of the address space, so
    /// the increment saturates rather than wrapping onto slot zero -- which is
    /// a live value the kernel would then overwrite.
    #[test]
    fn a_second_result_slot_saturates_instead_of_wrapping_onto_slot_zero() {
        let mut two = op(tags::SORT_DESC);
        two.result_count = 2;
        let params = OpParams::of(&two, u32::MAX, Runtime::default());
        assert_eq!(params.o1, u32::MAX);
        assert_ne!(params.o1, 0);
    }
}
