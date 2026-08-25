//! The eager resolve pass: every step's `Call` checked at LOAD, so a refusal
//! lands with the statement named instead of mid-fire.
//!
//! `.wiki/baker.md` states the rule: *"Load runs an eager resolve pass over the
//! whole lowered plan -- every `(op × axes)` JIT-compiled / looked up before
//! serving -- so refusals land at load with the op named, never mid-fire."* A
//! driver that discovers at token 300 that it cannot answer `mlp.swiglu` at f32
//! has already told a caller the model was loaded.
//!
//! ON A YOUNG PLANE THE PASS IS NOT A FORMALITY. When `driver-wgpu` landed, no
//! catalog row's lane bound on it at all -- every SKU refused at `gemm.matmul`
//! before it refused anywhere else -- and this pass is what turned that from a
//! mid-fire surprise into a load-time sentence naming the point and the first
//! statement that asked. `driver-metal` landed with the same measurement.
//!
//! # What this pass CAN answer, and what it cannot
//!
//! **What is answered here** is whether the plane CLAIMS a point AT THE ELEMENT
//! ITS WITNESS SLOT RIDES, and that element is known at load because it is in
//! `Program::slots` -- settled by the width walk, not by the fire's data. The
//! witness and the element set both come off [`Plane::CLAIMED`], which the same
//! generator writes as the arms; `Call::Tier2` asks [`Plane::TIER2`], which on
//! both shader planes is EMPTY and therefore refuses every tier-2 statement by
//! name. That is the honest answer: a tier-2 point is declared by an inherent
//! `impl Ctx` block, and `Ctx` is a trait object on a shader plane, which Rust
//! has no inherent impl for.
//!
//! **What is NOT answered here** -- and the gap is worth stating rather than
//! leaving silent -- is whether the ENTRY POINT a claim body will name is one
//! the shader tree actually stamps. On cuda the equivalent question is "is this
//! instantiation compilable", and the answer is the same: the name is built
//! INSIDE the body, out of the operands it was handed (each plane's `quant`
//! composes one from four axes at the fire), so warming would mean either
//! replicating each body's construction out here or firing the kernels for real
//! at load.
//!
//! **The seam:** each plane already enumerates every entry point its tree can
//! reach -- `kernels_metal::census()`, `kernels_wgpu::source::declared()` --
//! and the generated dispatch is the file that knows which body answers which
//! point. When the generator learns to emit a `warm(point, axes)` beside each
//! arm, built from the same `Fire::at` expression the body uses, the two lists
//! can be joined and this pass grows a third question. Until then a name the
//! tree does not stamp is reported by the pipeline compiler at first fire, and
//! it is reported by name.
//!
//! [`Plane::CLAIMED`]: crate::Plane::CLAIMED
//! [`Plane::TIER2`]: crate::Plane::TIER2

use std::collections::BTreeSet;

use kernels::bound::Site;
use kernels::points::ScalarKind;
use model_compiler::program::{Call, Dt, Program, Slot};
use model_ir::plan::{Op, Plan, ValueId};

use crate::walk::{Census, Plane};

/// One step the resolve pass could not bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unresolved {
    /// Index into `plan.ops` -- the first statement that asked.
    pub op: u32,
    /// The statement's point, as the plan spells it.
    pub kernel: String,
    /// What the driver was asked to reach, and could not.
    pub call: String,
    /// Why it could not: the plane answers no point of that name, or answers it
    /// at other elements than the one this lane's rectangles ride.
    pub why: String,
}

impl core::fmt::Display for Unresolved {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "op {} `{}` -> {}: {}",
            self.op, self.kernel, self.call, self.why
        )
    }
}

/// Check every step of `program` against the plane `P` fires.
///
/// Answers the whole measurement and not the first failure: a lane missing four
/// points should report four, so one load says what the whole backlog is.
/// Deduplicated by `(call, why)` so a 24-layer stack reports a missing point
/// once rather than twenty-four times, with the FIRST statement that asked --
/// which is the one a bisect wants.
#[must_use]
pub fn check<P: Plane>(plan: &Plan, program: &Program) -> Vec<Unresolved> {
    let mut out = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for step in &program.steps {
        let Some(op) = plan.ops.get(step.op as usize) else {
            continue;
        };
        let (call, why) = match &step.call {
            // A CANON SYMBOL REFUSES HERE, and the SENTENCE is the plane's:
            // metal's `CANON` has two rows whose STAGING no statement carries
            // (the embed's affine bank is three operands where the declaration
            // states one; the sorted combine reads a permutation no point can
            // name), and wgpu states no `CANON` table at all. A pass whose
            // whole reason is that a refusal should land at load cannot be the
            // thing that lets one through, whichever of the two it is.
            Call::Symbol(sym) => ((*sym).to_string(), P::NO_SYMBOL_AT_LOAD.to_string()),
            // THE SAME QUESTION TWICE, AGAINST THE TABLE THE CALL CAN REACH. A
            // `Call::Point` is answered by a tier-1 claim and a `Call::Tier2`
            // by an inherent method, and neither census can answer for the
            // other -- a row in the wrong one would be a row no call arrives
            // at.
            Call::Point(point) | Call::Tier2(point) => {
                let census = match &step.call {
                    Call::Tier2(_) => P::TIER2,
                    _ => P::CLAIMED,
                };
                let dt = witness_dt(census, program, op, point);
                match claimed(census, point, dt) {
                    Claim::Yes => continue,
                    Claim::NoPoint => (point.clone(), P::NO_POINT.to_string()),
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

/// The element a point's arm is SELECTED on, read where the generated dispatch
/// reads it.
///
/// NOT "the first result". `ssm.gdn_prep` states an f32 result over a bf16
/// operand and `attention.kv_append` states no result at all; the generator
/// picks one witness per point and writes it into the census, so this reads the
/// same slot the fire will.
fn witness_dt(census: Census, program: &Program, op: &Op, point: &str) -> Option<Dt> {
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

fn dtype_of(program: &Program, value: ValueId) -> Option<Dt> {
    match program.slots.get(value as usize)? {
        Slot::Arena { dtype, .. } => Some(*dtype),
        Slot::Alias(to) => dtype_of(program, *to),
        _ => None,
    }
}

/// What a rectangle the walk sized rides -- the same mapping
/// `crate::bound::axis` makes at the fire.
fn axis(dt: Dt) -> ScalarKind {
    match dt {
        Dt::Bf16 => ScalarKind::Bf16,
        Dt::F32 => ScalarKind::F32,
        Dt::I32 => ScalarKind::I32,
        Dt::U32 => ScalarKind::U32,
        Dt::U8 => ScalarKind::U8,
    }
}

fn claimed(census: Census, point: &str, dt: Option<Dt>) -> Claim {
    let Some((_, _, elements)) = census.iter().find(|(p, _, _)| *p == point) else {
        return Claim::NoPoint;
    };
    match dt {
        // A point that quantifies over nothing, or whose witness column this
        // statement does not carry, is reached by name: the arm has no match to
        // select and the fire is where a malformed statement is caught.
        None => Claim::Yes,
        Some(d) if elements.contains(&axis(d)) => Claim::Yes,
        Some(_) => Claim::NoDtype,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A census of one point, shaped exactly as a generator writes one.
    ///
    /// A FIXTURE AND NOT A PLANE'S TABLE, which is the one change this test
    /// took on the way here. Both drivers asked the same three questions of
    /// `norm.rmsnorm` in their own `CLAIMED`, so each was measuring two things
    /// at once -- that [`claimed`] has three answers, and that its own plane
    /// happens to instantiate that point at Bf16 and not at I32. The second
    /// half stayed with the plane (see each driver's `baker::resolve`); this is
    /// the first half, asked of the function rather than of a table.
    const ONE: Census = &[(
        "norm.rmsnorm",
        Some(Site::Out(0)),
        &[ScalarKind::Bf16, ScalarKind::F32],
    )];

    /// The three answers, on a census row that exists.
    #[test]
    fn a_claim_is_checked_at_the_dtype_and_not_only_by_name() {
        assert!(matches!(
            claimed(ONE, "norm.rmsnorm", Some(Dt::Bf16)),
            Claim::Yes
        ));
        assert!(matches!(
            claimed(ONE, "norm.rmsnorm", Some(Dt::I32)),
            Claim::NoDtype
        ));
        assert!(matches!(
            claimed(ONE, "norm.rmsnorm.nosuch", Some(Dt::Bf16)),
            Claim::NoPoint
        ));
        assert!(
            matches!(claimed(ONE, "norm.rmsnorm", None), Claim::Yes),
            "a statement whose witness column it does not carry is reached by name",
        );
    }
}
