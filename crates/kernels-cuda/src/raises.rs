//! What a CUDA fire raises, declared.
//!
//! The vocabulary is [`kernels::raises`] — [`kernels::raises::Raise`] and
//! `raise!`, beside `Fact` and `fact!`. The DECLARATIONS are here because a
//! raise's `Value` is this plane's own type and `kernels` has no dependencies:
//! it can hold a fact whose value is `f32` or `*const i32`, and it cannot name
//! a `PrefillPlanCache`. The same split puts [`kernels::routine::Elem`] there
//! and each plane's `Tensor` with the plane.
//!
//! # What is here and what is not
//!
//! One entry per object that is RAISED — made once before the launches, by a
//! prep the text states, and read by every statement that names it. The two
//! attention schedules are the first because they are the largest: 45 of
//! `keys.rs`'s 182 keys are these two objects unfolded into scalars, read back
//! at 130 `ask` sites across six launchers.
//!
//! A per-fire scalar is NOT a raise. `keys::Rows` is the token count, and the
//! test is unchanged from `keys.rs`'s: *two fires of the same model, on the
//! same deployment, can see different answers here*. What separates a raise
//! from a fact is not variability but SHAPE — one object with one lifetime,
//! against one number.

// The plan caches, as raises.
//
// Both name the CACHE and not the dispatch-ready `PrefillPlan`/`DecodePlan`
// beside it. The cache is what the fire raised — `fire/launch.rs`'s
// `raise_attn_plans` writes into it and `bind::attn_plan` hands it back — and
// the dispatch aggregate is derived from it per launch by
// `dispatch::prefill_plan_of`. Naming the derived one would raise the
// question of which fire's derivation, which is the disagreement this design
// exists to close.
/// Stamp the crossing impls for one raise's aggregate.
///
/// `raise!` declares the KEY and the value type; this declares how a pointer at
/// that value crosses. Both are needed and neither implies the other, so they
/// sit adjacent and this macro exists rather than a second hand-written pair.
///
/// The orphan rule is why it is a macro at all -- `impl<T: Raise> Abi for
/// *const T::Value` puts an uncovered parameter ahead of the first local type,
/// which is the same wall `jit::arg`'s own doc describes for
/// `impl<T: Abi> Arg<Cuda> for T`.
///
/// **`CPP` is empty and that is the claim.** The spelling exists so a Rust
/// mirror and a `__global__`'s parameter can be checked against each other; a
/// raised object reaches no `__global__`, so there is no declaration to check
/// and a string here would be a spelling nothing spells.
macro_rules! raise_abi {
    ($($value:ty),* $(,)?) => {
        $(
            impl $crate::jit::Abi for *const $value {
                const CPP: &'static str = "";
                const TY: kernels::Ty = kernels::Ty::Raised;

                fn arg(&self) -> $crate::jit::ArgValue {
                    $crate::jit::ArgValue::Ptr((*self).cast::<core::ffi::c_void>().cast_mut())
                }

                fn unpack(
                    value: &$crate::jit::ArgValue,
                    at: usize,
                ) -> Result<Self, kernels::Refusal> {
                    match value {
                        $crate::jit::ArgValue::Ptr(p) => Ok(p.cast::<$value>().cast_const()),
                        _ => Err(kernels::Refusal::Kind { at, want: kernels::Ty::Raised }),
                    }
                }
            }

            $crate::arg_via_abi!(*const $value);
        )*
    };
}

raise_abi!(
    crate::attn::fa2::plan::PrefillPlanCache,
    crate::attn::fa2::plan::DecodePlanCache,
);

kernels::raise!(
    /// The paged-PREFILL schedule this fire raised.
    ///
    /// `PrepKind::PrefillAttention` produces it; the three prefill launchers
    /// consume it. Its 26 leaves are `keys::Fa2Prefill*` today, and each
    /// records *the value the plan was built at* — including four that look
    /// like model geometry and are not, because a fire that restated those
    /// from the checkpoint would be free to disagree with the plan it is
    /// executing.
    Fa2Prefill = "fa2.prefill" => crate::attn::fa2::plan::PrefillPlanCache
);

kernels::raise!(
    /// The paged-DECODE schedule this fire raised. [`Fa2Prefill`]'s twin, with
    /// 19 leaves rather than 26: the seven it lacks are what a prefill
    /// schedule has and a decode one does not — a QO tile assignment, a
    /// per-row merge indptr, a CTA tile width.
    ///
    /// Two of these may stand at once. `raise_attn_plans` raises a windowed
    /// schedule and a full-attention one where a stack states both widths, and
    /// which of them a statement executes is recovered at bind time from the
    /// window on its `LaunchSpec` — `bind::attn_plan`'s guess, which naming
    /// the plan as an operand is what deletes.
    Fa2Decode = "fa2.decode" => crate::attn::fa2::plan::DecodePlanCache
);

#[cfg(test)]
mod tests {
    use super::{Fa2Decode, Fa2Prefill};
    use crate::attn::fa2::plan::{DecodePlanCache, PrefillPlanCache};
    use kernels::Ty;
    use kernels::raises::{Raise, Struct};
    use kernels::routine::{Elem, In};

    /// The carrier resolves to a pointer at the plane's own aggregate.
    ///
    /// Written as a binding and not an `assert`: there is no runtime value to
    /// compare, and a `let` of the spelled type is the only thing that fails
    /// if `Elem::Read` ever resolves elsewhere.
    #[test]
    fn a_raise_carries_a_pointer_at_the_planes_own_type() {
        let cache = PrefillPlanCache::default();
        let read: <Struct<Fa2Prefill> as Elem>::Read = &raw const cache;
        assert!(!read.is_null());

        let decode = DecodePlanCache::default();
        let read: <Struct<Fa2Decode> as Elem>::Read = &raw const decode;
        assert!(!read.is_null());
    }

    /// THE POINT OF THE WHOLE DESIGN: it is a mark, so it is positional.
    ///
    /// `In<Struct<Fa2Prefill>>` is the same wrapper an activation takes. What
    /// that buys is stated in `.wiki/designs/design-struct.md` §4.3 -- a mark
    /// is enumerated by `arity_problem` and `check_plan` where an `ask` is a
    /// call the derived column cannot see.
    #[test]
    fn a_raise_is_an_operand_and_not_a_question() {
        let cache = PrefillPlanCache::default();
        let plan: In<Struct<Fa2Prefill>> =
            In { ptr: &raw const cache, rows: 0, width: 0 };
        assert!(core::ptr::eq(plan.ptr, &raw const cache));
    }

    /// A raise binds as [`Ty::Raised`] in BOTH directions.
    ///
    /// One kind and not two: `Elem`'s pair exists because a read and a write
    /// are different C++ spellings, and a raised object has no C++ spelling to
    /// differ in. See `Ty::Raised`'s own doc.
    #[test]
    fn both_directions_are_the_one_kind() {
        assert_eq!(<Struct<Fa2Prefill> as Elem>::TY_CONST, Ty::Raised);
        assert_eq!(<Struct<Fa2Prefill> as Elem>::TY_MUT, Ty::Raised);
        assert_eq!(<Struct<Fa2Prefill> as Elem>::CPP_CONST, "");
        assert_eq!(<Struct<Fa2Prefill> as Elem>::CPP_MUT, "");
    }

    /// The word is written once, here, and the two do not collide.
    /// THE CARRIER IS AN ARGUMENT, which is what makes it a mark.
    ///
    /// `In<E>` binds only where `E::Read: Arg<B>`, so this compiles exactly
    /// when a raise can occupy an operand slot. It is the one claim stage 1
    /// could not make and stage 4 needs.
    #[test]
    fn a_raise_can_occupy_an_operand_slot() {
        use kernels::routine::Arg;

        let cache = PrefillPlanCache::default();
        let value = crate::jit::ArgValue::Ptr((&raw const cache).cast::<core::ffi::c_void>().cast_mut());
        let back = <In<Struct<Fa2Prefill>> as Arg<crate::jit::Cuda>>::unpack(&value, 0)
            .expect("a raise unpacks from the pointer the binder minted");
        assert!(core::ptr::eq(back.ptr, &raw const cache));
        assert_eq!(<In<Struct<Fa2Prefill>> as Arg<crate::jit::Cuda>>::TY, Ty::Raised);
    }

    /// And a value of the wrong kind is refused, not reinterpreted.
    #[test]
    fn a_raise_refuses_a_value_that_is_not_a_pointer() {
        use kernels::routine::Arg;

        let err = <In<Struct<Fa2Prefill>> as Arg<crate::jit::Cuda>>::unpack(
            &crate::jit::ArgValue::I32(7),
            3,
        )
        .expect_err("a scalar is not a raise");
        assert!(matches!(err, kernels::Refusal::Kind { at: 3, want: Ty::Raised }));
    }

    #[test]
    fn each_raise_says_its_own_word() {
        assert_eq!(Fa2Prefill::KEY, "fa2.prefill");
        assert_eq!(Fa2Decode::KEY, "fa2.decode");
        assert_ne!(Fa2Prefill::KEY, Fa2Decode::KEY);
    }
}
