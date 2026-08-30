//! What `validate_singleton_plan` is *for*, stated as damage it has to catch.
//!
//! The validator's job is to reject plans a well-formed compiler never
//! produces, which is why almost every one of its error paths is unreachable
//! from the corpus alone. The goldens therefore pin only the accept path: they
//! would still pass if the whole function were `Ok(...)`.
//!
//! So this damages each corpus stage in the 23 ways `common/msl_mutations.rs`
//! describes — the list the deleted C++ validator's `kMutations` used — and
//! requires the validator, or the emitter behind it, to notice. That turns the
//! accept path from "these bytes came out" into "and nothing else would have".
//!
//! Scope: these are *native* mutations of a [`eta_compiler::plan::CompiledStage`], not
//! the oracle's wire-level ones. `metal_msl_golden.rs` explains why the wire
//! mutations are not compared across the two implementations; the point here is
//! not cross-checking C++ but pinning what the Rust validator guarantees.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;
#[path = "common/msl_mutations.rs"]
mod msl_mutations;

use eta_compiler::codegen::metal::validate_singleton_plan;
use msl_corpus::{corpus_stages, extended_stages};
use msl_mutations::{MUTATIONS, mutate};

/// Mutations that a valid plan survives, with the reason. Everything else has
/// to be rejected wherever it applies.
///
/// Kept as a list rather than a per-case skip so that a mutation moving between
/// the two groups is a visible diff. `singleton_plan_rejection_is_total` below
/// is what stops an entry from being added to silence a real hole: an entry
/// only holds if the mutation is *never* rejected, so a partially caught
/// mutation belongs in neither group and fails.
const TOLERATED: &[(&str, &str)] = &[(
    "none",
    "the unmutated plan; here to prove the harness accepts something",
)];

/// Every mutation, and how the validator answered on every stage it applied to.
///
/// Only stages the validator accepts unmutated take part. The others — the
/// corpus has three, plans the singleton path cannot express at all — are
/// already rejected, so damaging them would prove nothing about the damage.
fn verdicts() -> Vec<(&'static str, usize, usize)> {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .chain(extended_stages())
        .filter(|stage| validate_singleton_plan(&stage.plan).is_ok())
        .collect();
    MUTATIONS
        .iter()
        .map(|mutation| {
            let (mut applied, mut rejected) = (0, 0);
            for stage in &stages {
                let mut damaged = stage.plan.clone();
                if !mutate(&mut damaged, mutation) {
                    continue;
                }
                applied += 1;
                if validate_singleton_plan(&damaged).is_err() {
                    rejected += 1;
                }
            }
            (*mutation, applied, rejected)
        })
        .collect()
}

/// Every mutation has to reach at least one stage, or it is describing damage
/// the corpus cannot express and is checking nothing.
#[test]
fn every_mutation_reaches_the_corpus() {
    let dead: Vec<_> = verdicts()
        .into_iter()
        .filter(|(_, applied, _)| *applied == 0)
        .map(|(mutation, _, _)| mutation)
        .collect();
    assert!(
        dead.is_empty(),
        "these mutations applied to no accepted corpus stage, so they check \
         nothing: {dead:?}. Either the corpus lost the shape they need or they \
         describe damage the plan types no longer allow."
    );
}

/// The accept path is not vacuous: the corpus really does contain stages the
/// validator says yes to.
#[test]
fn the_unmutated_corpus_is_accepted() {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .chain(extended_stages())
        .collect();
    assert!(!stages.is_empty(), "the corpus is empty");
    let accepted = stages
        .iter()
        .filter(|stage| validate_singleton_plan(&stage.plan).is_ok())
        .count();
    assert!(
        accepted > 0,
        "the validator rejected all {} corpus stages, so every rejection below \
         proves nothing",
        stages.len()
    );
}

/// A mutation is either always caught or never caught. A mutation caught on
/// some stages and not others is the interesting case — it means the check
/// exists but depends on plan shape, and the stages it misses are plans that
/// reach the emitter damaged.
#[test]
fn singleton_plan_rejection_is_total() {
    let mut holes = Vec::new();
    for (mutation, applied, rejected) in verdicts() {
        let tolerated = TOLERATED.iter().any(|(name, _)| *name == mutation);
        if tolerated {
            assert_eq!(
                rejected, 0,
                "{mutation:?} is listed in TOLERATED but is rejected on \
                 {rejected}/{applied} stages — it is a real check now, so take it \
                 out of the list"
            );
            continue;
        }
        if rejected != applied {
            holes.push(format!(
                "{mutation}: caught on {rejected} of {applied} stages"
            ));
        }
    }
    assert!(
        holes.is_empty(),
        "validate_singleton_plan lets damaged plans through:\n  {}\n\
         Either the check is missing, or the mutation is harmless and belongs \
         in TOLERATED with the reason why.",
        holes.join("\n  ")
    );
}
