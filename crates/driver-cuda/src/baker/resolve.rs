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
//! the plane's tables, and both are answered here:
//!
//! * a `Call::Symbol` resolves iff `kernels_cuda::routine(sym)` finds a row
//!   — that is the same lookup the fire makes, run early;
//! * a `Call::Point` resolves iff the shim claims the point AT THE RESULT
//!   DTYPE, and the dtype is known at load because it is in
//!   `Program::slots` — settled by the width walk, not by the fire's data.
//!
//! The third — **is the CUDA source for this instantiation compilable** —
//! is NOT answered, and the reason is worth stating rather than leaving as
//! a silent gap. A claim body builds its own JIT symbol from the operands
//! it was handed (`kernels-cuda`'s `Norm::rmsnorm_residual_add` spells
//! `::pie::norm::rmsnorm_residual_add<{T::CPP}, 256>` and chooses the block
//! width from the row it is about to write), and `jit::nvrtc::compile_text`
//! takes a fully-built `Job`. So warming would mean either replicating each
//! body's symbol construction out here — a second spelling of the thing
//! most likely to drift — or firing the kernels for real at load. Neither
//! is "where cheap".
//!
//! **The seam:** when `#[claims]` generates the dispatch (W5), it can also
//! emit a `warm(point, axes)` beside each arm, built from the same symbol
//! expression the body uses. That is the honest place for it, and it is one
//! generator away. Until then the JIT's own first-fire compile is what
//! reports an uncompilable instantiation, and it reports it by name.

use std::collections::BTreeSet;

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
        // The result dtype, which is what a point's arm is selected on. A
        // statement with no result is an effect (`attn::write_kv_to_pages`
        // states none), and effects are name-resolved only.
        let dt = op
            .outputs
            .first()
            .and_then(|v| match &program.slots[*v as usize] {
                Slot::Arena { dtype, .. } => Some(*dtype),
                Slot::Alias(to) => match &program.slots[*to as usize] {
                    Slot::Arena { dtype, .. } => Some(*dtype),
                    _ => None,
                },
                _ => None,
            });
        let (call, why) = match &step.call {
            Call::Symbol(sym) => {
                if kernels_cuda::routine(sym).is_some() {
                    continue;
                }
                (
                    (*sym).to_string(),
                    "no row of this plane's signature table answers the symbol".to_string(),
                )
            }
            Call::Point(point) => {
                match claimed(point, dt) {
                    Claim::Yes => continue,
                    Claim::NoPoint => (
                        point.clone(),
                        "this driver's point shim states no arm for the point"
                            .to_string(),
                    ),
                    Claim::NoDtype => (
                        point.clone(),
                        format!(
                            "the point shim claims it, but not at {}",
                            dt.map_or_else(|| "an unstated dtype".to_string(), |d| format!("{d:?}")),
                        ),
                    ),
                }
            }
            Call::Tier2(statement) => (
                statement.clone(),
                "a tier-2 statement, and this driver states no tier-2 shim".to_string(),
            ),
        };
        let key = format!("{call}@{dt:?}");
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

/// Whether the point shim answers `point`, and at `dt`.
enum Claim {
    Yes,
    NoPoint,
    NoDtype,
}

/// The points [`super::points_shim`] answers, and the result dtypes each is
/// instantiated at.
///
/// # Why a table beside a `match`, and why the drift is safe
///
/// The firing side is a `match` on the point name with a nested `match` on
/// the dtype — the shape a generated dispatch has, and the shape the smoke
/// proved. A `match` cannot be enumerated, so the CHECK needs its own
/// spelling of the same fact, and two spellings of one fact is exactly the
/// drift this tree keeps writing comments about.
///
/// It is tolerable here for one reason, and only this one: **both
/// directions of drift surface as a named refusal, never as a wrong
/// answer.**
///
/// * a row here the `match` does not answer ⇒ load passes, the first fire
///   of that op refuses with the point named. Late, loud, correct.
/// * an arm in the `match` this table does not list ⇒ load refuses with
///   the point named. Early, loud, over-strict.
///
/// Neither can produce a number. The pair dies together when `#[claims]`
/// emits both halves from the claim table (W5) — at which point the table
/// IS the match and the question stops existing.
const CLAIMED: &[(&str, &[Dt])] = &[
    ("norm.rmsnorm", &[Dt::Bf16, Dt::F32]),
    ("norm.rmsnorm_plus_one", &[Dt::Bf16, Dt::F32]),
    ("norm.rmsnorm_per_head", &[Dt::Bf16]),
    ("norm.rmsnorm_per_head_plus_one", &[Dt::Bf16]),
    ("norm.rmsnorm_gated", &[Dt::Bf16]),
    ("norm.residual_add", &[Dt::Bf16, Dt::F32]),
    ("gemm.matmul", &[Dt::Bf16]),
    ("gemm.lm_head", &[Dt::Bf16]),
    ("gemm.attention_landing", &[Dt::Bf16]),
    ("mlp.swiglu", &[Dt::Bf16]),
    ("gate.sigmoid_mul", &[Dt::Bf16]),
    ("layout.embed", &[Dt::Bf16]),
    ("layout.split_q_gate", &[Dt::Bf16]),
    ("layout.split_rows", &[Dt::Bf16]),
    ("rope.partial", &[Dt::Bf16]),
    ("ssm.causal_conv1d", &[Dt::Bf16]),
    // THE DTYPE HERE IS THE RESULT'S, and both gdn points state an f32
    // result over a bf16 operand — the decay row and the recurrence's
    // output are accumulated, not activated. `check` reads
    // `outputs.first()`'s slot, so F32 is the honest row even though the
    // shim's own `match` selects on the packed operand's element (which is
    // what the generated dispatch does too: `Site::In(0)`).
    ("ssm.gdn_prep", &[Dt::F32]),
    ("ssm.gated_delta", &[Dt::F32]),
];

fn claimed(point: &str, dt: Option<Dt>) -> Claim {
    let Some((_, dtypes)) = CLAIMED.iter().find(|(p, _)| *p == point) else {
        return Claim::NoPoint;
    };
    match dt {
        // A point that resolves and states no result is an effect the shim
        // reaches by name; the dtype arm is not what selects it.
        None => Claim::Yes,
        Some(d) if dtypes.contains(&d) => Claim::Yes,
        Some(_) => Claim::NoDtype,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every point the table claims is one the compiler can actually route
    /// to a point call on this plane.
    ///
    /// The failure this catches is a typo: a row spelled `norm.rmsnorm_gate`
    /// claims nothing, resolves nothing, and its only symptom is a load-time
    /// refusal for a point that IS implemented.
    #[test]
    fn every_claimed_point_is_a_point_the_compiler_routes() {
        for (point, _) in CLAIMED {
            let call = model_compiler::program::call_of(
                model_ir::kernels::Backend::Cuda,
                point,
            );
            assert_eq!(
                call,
                Some(Call::Point((*point).to_string())),
                "`{point}` is claimed by the shim but `call_of` answers {call:?}; \
                 either the spelling is wrong or the point is not a claim",
            );
        }
    }

    /// The table is a set: a duplicated point would make the second row
    /// unreachable and its dtypes silently unclaimed.
    #[test]
    fn the_claim_table_names_each_point_once() {
        let mut seen = BTreeSet::new();
        for (point, _) in CLAIMED {
            assert!(seen.insert(*point), "`{point}` is claimed twice");
        }
    }

    /// The three answers, on a table row that exists.
    #[test]
    fn a_claim_is_checked_at_the_dtype_and_not_only_by_name() {
        assert!(matches!(claimed("norm.rmsnorm", Some(Dt::Bf16)), Claim::Yes));
        assert!(matches!(claimed("norm.rmsnorm", Some(Dt::F32)), Claim::Yes));
        assert!(matches!(
            claimed("norm.rmsnorm", Some(Dt::I32)),
            Claim::NoDtype
        ));
        assert!(matches!(claimed("mlp.swiglu", Some(Dt::F32)), Claim::NoDtype));
        assert!(matches!(claimed("moe.matmul_select", None), Claim::NoPoint));
        assert!(
            matches!(claimed("norm.rmsnorm", None), Claim::Yes),
            "an effect with no result is reached by name",
        );
    }
}
