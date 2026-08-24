//! The eager resolve pass: every step's `Call` checked at LOAD, so a
//! refusal lands with the op named instead of mid-fire.
//!
//! # Check-then-bind
//!
//! `.wiki/baker.md` states the rule: *"Load runs an eager resolve pass over
//! the whole lowered plan — every `(op × axes)` JIT-compiled / looked up
//! before serving — so refusals land at load with the op named, never
//! mid-fire."* A driver that discovers at token 300 that it cannot answer
//! `mlp.swiglu` at f32 has already told a caller the model was loaded.
//!
//! # What this pass CAN answer, and what it cannot
//!
//! Two of the three questions are answerable with nothing but the plan and
//! the plane's tables, and both are answered here; the third kind of call
//! refuses outright, which is also an answer:
//!
//! * a `Call::Point` resolves iff the plane claims the point AT THE ELEMENT
//!   ITS WITNESS SLOT RIDES, and that element is known at load because it is
//!   in `Program::slots` — settled by the width walk, not by the fire's data.
//!   The witness and the element set both come off
//!   `kernels_cuda::points_dispatch::CLAIMED`, which the same generator
//!   writes as the arms;
//! * a `Call::Tier2` resolves the same way against `TIER2`, the census of the
//!   plane's inherent surface, which the same generator writes beside it.
//!
//! The third — **is the CUDA source for this instantiation compilable** —
//! is not answered HERE, and the reason it cannot be is worth keeping. A
//! claim body builds its own JIT symbol from the operands it was handed
//! (`kernels-cuda`'s `Norm::rmsnorm_residual_add` spells
//! `::pie::norm::rmsnorm_residual_add<{T::CPP}, 256>` and chooses the block
//! width from the row it is about to write), and `jit::nvrtc::compile_text`
//! takes a fully-built `Job`. So a warm out here would mean replicating
//! each body's symbol construction — a second spelling of the thing most
//! likely to drift — and the census in `kernels_cuda::jit::warm` says the
//! copy could not even be exact: seven of qwen decode's nineteen
//! instantiations are picked from facts no table out here holds, and they
//! carry 79 % of the compile time.
//!
//! **The seam is closed, from the other side.** `serve::load::warm_lane`
//! walks the lane with `kernels_cuda::jit::warm::pass` open, so every
//! `Ctx::fire` resolves its instantiation and launches nothing: the warm
//! arm IS the fire arm, there is no second spelling, and an instantiation
//! that will not compile refuses `load_model` by name instead of arriving
//! at token 1. What this pass still answers, and answers first, is the
//! POINT: a lane naming a point the plane does not claim has no body to run
//! and would warm nothing.

use std::collections::BTreeSet;

use kernels::bound::{Axis, Site};
use kernels_cuda::points_dispatch::{CLAIMED, TIER2};
use model_compiler::program::{Call, Dt, Program, Slot};
use model_ir::plan::Plan;

/// One step the resolve pass could not bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unresolved {
    /// Index into `plan.ops` — the first statement that asked.
    pub op: u32,
    /// The statement's point, as the plan spells it.
    pub kernel: String,
    /// What the driver was asked to reach, and could not.
    pub call: String,
    pub why: String,
}

impl std::fmt::Display for Unresolved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "op {} `{}` -> {}: {}",
            self.op, self.kernel, self.call, self.why
        )
    }
}

/// Check every step of `program` against the plane this driver fires.
///
/// Answers the whole measurement and not the first failure: a lane missing
/// four points should report four, so one load says what the whole backlog
/// is. Deduplicated by `(call, dtype)` so a 24-layer stack reports a
/// missing point once rather than twenty-four times, with the FIRST op that
/// asked — which is the one a bisect wants.
pub(crate) fn check(plan: &Plan, program: &Program) -> Vec<Unresolved> {
    let mut out = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for step in &program.steps {
        let op = &plan.ops[step.op as usize];
        let (call, why) = match &step.call {
            // A CANON SYMBOL REFUSES HERE, AND IT DID NOT USED TO.
            //
            // This asked `kernels_cuda::routine(sym)` whether a row of the
            // registry answered the symbol and CONTINUED when one did — which
            // was true of both symbols that still reach this arm, and told
            // the caller the model was loaded. Then the fire reached
            // `baker::staging::symbol`, which had been a bare refusal since
            // R4b retired the last of its five arms — and is now deleted, its
            // sentence inlined into `Fire::step`'s own `Call::Symbol` arm. A
            // pass whose whole reason is that "a driver that discovers at
            // token 300 that it cannot answer `mlp.swiglu` has already told a
            // caller the model was loaded" cannot be the thing that does it.
            //
            // The registry row is not the question. Whether a row exists says
            // the plane HAS a launcher; what a fire needs is the STAGING
            // between the statement and that launcher — the operands it does
            // not carry, the results it does not state — and this driver
            // states none. Two symbols reach here across all sixteen catalog
            // rows, `hc.collapse` and `norm.res_blend`, both argued in
            // `kernels/src/points.rs`, both waiting on something the floor
            // does not have. They refuse, and now they refuse where a refusal
            // is a sentence a load can print.
            Call::Symbol(sym) => (
                (*sym).to_string(),
                "a canon symbol, and this driver's staging shim answers none: the \
                 statement's operands are not the routine's"
                    .to_string(),
            ),
            // THE SAME QUESTION TWICE, AGAINST THE TABLE THE CALL CAN REACH.
            // A `Call::Point` is answered by a tier-1 claim and a
            // `Call::Tier2` by an inherent method, and neither census can
            // answer for the other — a row in the wrong one would be a row no
            // call arrives at. What the two share is the question, so they
            // share the reading of it.
            Call::Point(point) | Call::Tier2(point) => {
                let census = match &step.call {
                    Call::Tier2(_) => TIER2,
                    _ => CLAIMED,
                };
                let dt = witness_dt(census, program, op, point);
                match claimed(census, point, dt) {
                    Claim::Yes => continue,
                    Claim::NoPoint => (
                        point.clone(),
                        "this plane answers no point of that name; see the family's \
                         `*_CLAIMS`, or `TIER2_POINTS` for an inherent one"
                            .to_string(),
                    ),
                    Claim::NoDtype => (
                        point.clone(),
                        format!(
                            "this plane answers it, but not at {}",
                            dt.map_or_else(
                                || "an unstated element".to_string(),
                                |d| format!("{d:?}")
                            ),
                        ),
                    ),
                }
            }
        };
        let key = format!("{call}@{why}");
        if seen.insert(key) {
            out.push(Unresolved {
                op: step.op,
                kernel: op.kernel.clone(),
                call,
                why,
            });
        }
    }
    out
}

/// Whether the plane claims `point`, and at the element its witness rides.
enum Claim {
    Yes,
    NoPoint,
    NoDtype,
}

/// The element a point's arm is SELECTED on, read where the generated
/// dispatch reads it.
///
/// NOT "the first result", which is what this used to assume and what the
/// hand-written claim table carried a paragraph about: thirteen of the
/// plane's arms witness their axis at `Site::In(0)` instead — `ssm.gdn_prep`
/// and `ssm.gated_delta` state an f32 result over a bf16 operand, and
/// `mla.kv_append` states no result at all. The generator picks one witness
/// per point and now writes it down, so this reads the same slot the fire
/// will.
///
/// `None` is a point that quantifies over nothing, or a witness column this
/// statement does not carry; both resolve by name and refuse at the fire if
/// the statement is malformed.
fn witness_dt(
    census: &[(&str, Option<Site>, &[Axis])],
    program: &Program,
    op: &model_ir::plan::Op,
    point: &str,
) -> Option<Dt> {
    let (_, witness, _) = census.iter().find(|(p, _, _)| *p == point)?;
    let value = match (*witness)? {
        Site::In(i) => *op.inputs.get(i)?,
        Site::Out(i) => *op.outputs.get(i)?,
        // A bank's element is the CHECKPOINT's and lives in the parameter
        // table, not in a slot the walk sized. No point declares one as a
        // witness (the generator asserts it), so this is unreachable rather
        // than unimplemented.
        Site::Const(_) => return None,
    };
    dtype_of(program, value)
}

fn dtype_of(program: &Program, value: model_ir::plan::ValueId) -> Option<Dt> {
    match program.slots.get(value as usize)? {
        Slot::Arena { dtype, .. } => Some(*dtype),
        Slot::Alias(to) => dtype_of(program, *to),
        _ => None,
    }
}

/// What a rectangle the walk sized rides, as the floor names it — the same
/// mapping `baker::bound::axis` makes at the fire.
fn axis(dt: Dt) -> Axis {
    match dt {
        Dt::Bf16 => Axis::Bf16,
        Dt::F32 => Axis::F32,
        Dt::I32 => Axis::I32,
        Dt::U32 => Axis::U32,
        Dt::U8 => Axis::U8,
    }
}

fn claimed(census: &[(&str, Option<Site>, &[Axis])], point: &str, dt: Option<Dt>) -> Claim {
    let Some((_, _, elements)) = census.iter().find(|(p, _, _)| *p == point) else {
        return Claim::NoPoint;
    };
    match dt {
        // A point that quantifies over nothing, or whose witness column this
        // statement does not carry, is reached by name: the arm has no match
        // to select and the fire is where a malformed statement is caught.
        None => Claim::Yes,
        Some(d) if elements.contains(&axis(d)) => Claim::Yes,
        Some(_) => Claim::NoDtype,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every point the generated census claims is one the compiler can
    /// actually route to a point call on this plane.
    ///
    /// The failure this catches is a family whose claim table names something
    /// `model_compiler` does not route as a point — a spelling that resolves
    /// nothing, whose only symptom is a load-time refusal for a point that IS
    /// implemented.
    #[test]
    fn every_claimed_point_is_a_point_the_compiler_routes() {
        for (point, _, _) in CLAIMED {
            let call = model_compiler::program::call_of(model_ir::kernels::Backend::Cuda, point);
            assert_eq!(
                call,
                Some(Call::Point((*point).to_string())),
                "`{point}` is claimed by this plane but `call_of` answers {call:?}; \
                 either the spelling is wrong or the point is not a claim",
            );
        }
    }

    /// The same, for the TIER-2 census — and the routing it checks is the
    /// other variant, which is the whole reason the two tables are two.
    ///
    /// A tier-2 row is spelled by its METHOD and reached by a statement
    /// spelled `cuda::<method>`, so what has to hold is that the lowering
    /// strips the plane gate and answers `Call::Tier2` with exactly the name
    /// this plane's table carries. A row whose statement `call_of` answered
    /// `Call::Point` for — or answered nothing for — would be an arm no lane
    /// arrives at.
    #[test]
    fn every_tier2_point_is_a_statement_the_compiler_gates_to_this_plane() {
        for (point, _, _) in TIER2 {
            let statement = format!("cuda::{point}");
            let call =
                model_compiler::program::call_of(model_ir::kernels::Backend::Cuda, &statement);
            assert_eq!(
                call,
                Some(Call::Tier2((*point).to_string())),
                "`{statement}` is this plane's tier-2 surface but `call_of` answers \
                 {call:?}",
            );
            // AND IT IS ONE PLANE'S. The gate is the point of the prefix: the
            // same statement on another plane is a lowering violation, not a
            // backlog row.
            assert_eq!(
                model_compiler::program::call_of(model_ir::kernels::Backend::Metal, &statement,),
                None,
                "`{statement}` resolves on a plane that does not declare it",
            );
        }
    }

    /// The two censuses are DISJOINT, which is what lets one match hold both.
    ///
    /// A tier-1 point is `family.method` and carries a dot; an inherent
    /// method cannot. If a name ever stood in both, the generated match's
    /// first arm would answer for the second and one of the two calls would
    /// fire the wrong kernel.
    #[test]
    fn no_name_stands_in_both_censuses() {
        for (point, _, _) in TIER2 {
            assert!(
                !point.contains('.'),
                "`{point}` is a tier-2 name wearing a family prefix",
            );
            assert!(
                !CLAIMED.iter().any(|(p, _, _)| p == point),
                "`{point}` stands in both censuses",
            );
        }
    }

    /// The census is a set: a duplicated point would make the second row
    /// unreachable and its elements silently unclaimed.
    #[test]
    fn the_claim_table_names_each_point_once() {
        let mut seen = BTreeSet::new();
        for (point, _, _) in CLAIMED {
            assert!(seen.insert(*point), "`{point}` is claimed twice");
        }
    }

    /// The three answers, on a census row that exists.
    #[test]
    fn a_claim_is_checked_at_the_dtype_and_not_only_by_name() {
        assert!(matches!(
            claimed(CLAIMED, "norm.rmsnorm", Some(Dt::Bf16)),
            Claim::Yes
        ));
        assert!(matches!(
            claimed(CLAIMED, "norm.rmsnorm", Some(Dt::F32)),
            Claim::Yes
        ));
        assert!(matches!(
            claimed(CLAIMED, "norm.rmsnorm", Some(Dt::I32)),
            Claim::NoDtype
        ));
        assert!(matches!(
            claimed(CLAIMED, "norm.rmsnorm.nosuch", Some(Dt::Bf16)),
            Claim::NoPoint
        ));
        assert!(
            matches!(claimed(CLAIMED, "norm.rmsnorm", None), Claim::Yes),
            "a statement whose witness column it does not carry is reached by name",
        );
    }

    /// THE WITNESS IS THE GENERATOR'S, and this is the row that proves the
    /// census carries it rather than assuming the result.
    ///
    /// `ssm.gdn_prep` states an f32 result over a bf16 operand and its arm
    /// selects on `Site::In(0)`. A pass that read `outputs.first()` — which
    /// this one did — would have asked whether the plane instantiates the
    /// point at F32 and got the right answer for the wrong reason, and would
    /// have got the WRONG answer for `mla.kv_append`, which states no result
    /// at all.
    #[test]
    fn the_census_reads_the_witness_the_arm_reads() {
        let of = |point: &str| {
            CLAIMED
                .iter()
                .find(|(p, _, _)| *p == point)
                .map(|(_, w, _)| *w)
        };
        assert_eq!(of("ssm.gdn_prep"), Some(Some(Site::In(0))));
        assert_eq!(of("mla.kv_append"), Some(Some(Site::In(0))));
        assert_eq!(of("norm.rmsnorm"), Some(Some(Site::Out(0))));
    }
}
