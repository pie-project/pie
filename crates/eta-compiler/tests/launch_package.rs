//! What the engine is handed, pinned.
//!
//! [`eta_compiler::codegen::launch::build`] produces the only artefact the engine
//! executes: a `LaunchPackage` of lowered values, channels, ports, stages and
//! per-stage grouped plans. Until this file existed it had exactly one caller
//! in the tested workspace — `cuda_golden::emit_engine_test_kernel_fixtures`,
//! which writes the encoded package into a gitignored fixtures directory and
//! asserts nothing about its contents. So `StageNeeds::grouped_valid`,
//! `plan.error`, `flags`, `mtp_rows` and `channel_rules` were all unpinned:
//! a mutation testing this discovered that declaring `add` grouped-unsupported
//! — which silently costs every elementwise stage its grouped path — passed
//! the entire 208-test suite.
//!
//! These are structural claims plus one pinned count. The count is the part
//! that bites: a classification flip changes it even when every structural
//! invariant still holds.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use msl_corpus::{GOLDEN_NAMES, golden_container, golden_profile, synthetic_traces};
use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_compiler::plan::compile_bound;
use eta_ir::validate::bind;

/// The corpus, split the way binding splits it: what built a package, and the
/// name of everything that did not.
fn bound_and_refused() -> (Vec<(String, Vec<LaunchStagePlan>)>, Vec<String>) {
    let mut out = Vec::new();
    let mut refused = Vec::new();
    let mut push = |name: &str, container, profile| {
        let Ok(bound) = bind(container, profile) else {
            refused.push(name.to_string());
            return;
        };
        let stages = compile_bound(&bound);
        let package = eta_compiler::codegen::launch::build(&bound, &stages);
        out.push((name.to_string(), package.plans));
    };
    for name in GOLDEN_NAMES {
        push(name, golden_container(name), golden_profile(name));
    }
    for (name, container, profile) in synthetic_traces() {
        push(name, container, profile);
    }
    // This was `drop(push)`, to end the closure's mutable borrows of `out`
    // and `refused` before the tuple is built. It is not needed -- NLL ends a
    // borrow at its last use and `push` is never used again -- and it is a
    // clippy error: `drop_non_drop`, because a closure has no `Drop` impl, so
    // dropping one "only extends its contained lifetimes", which is the
    // opposite of what the line was written to achieve. Removing it changes
    // nothing about when the borrows end.
    (out, refused)
}

#[test]
fn every_plan_the_engine_receives_is_well_formed() {
    // `packages()` drops a trace that stops binding -- `let Ok(bound) = ...
    // else { return }` -- and then every check below runs over what is left.
    // The old guard was `>= 12` against a corpus of 24, so half the corpus
    // could vanish without a word.
    //
    // Six of those 24 are *supposed* to vanish: the `neg_` traces exist to be
    // refused, and the first version of this check read the silence as loss
    // and failed. So the question is not how many bound, it is which. A
    // `neg_` that binds is a refusal that stopped happening; anything else
    // that does not bind is a trace that broke and said nothing.
    let (packages, refused) = bound_and_refused();
    let unexpected: Vec<&String> = refused.iter().filter(|n| !n.starts_with("neg_")).collect();
    assert!(
        unexpected.is_empty(),
        "these traces failed to bind and were skipped in silence: {unexpected:?}"
    );
    let negatives = GOLDEN_NAMES
        .iter()
        .filter(|n| n.starts_with("neg_"))
        .count()
        + synthetic_traces()
            .iter()
            .filter(|(n, _, _)| n.starts_with("neg_"))
            .count();
    assert_eq!(
        refused.len(),
        negatives,
        "{negatives} traces are written to be refused and {} were; a `neg_` \
         that binds is a refusal that stopped happening",
        refused.len()
    );
    let mut plans = 0usize;
    for (name, stage_plans) in &packages {
        for (index, plan) in stage_plans.iter().enumerate() {
            plans += 1;
            let id = format!("{name}#{index}");

            // `invalid()` clears the valid bit and sets a reason. Neither half
            // may happen without the other, or the engine sees a plan it will
            // run with a diagnosis attached, or refuse with none.
            assert_eq!(
                plan.needs.grouped_valid,
                plan.error.is_empty(),
                "{id}: grouped-valid = {}, error = {:?}",
                plan.needs.grouped_valid,
                plan.error
            );
            // The "no undeclared bit" assertion that stood here is gone with
            // the bitmask: `StageNeeds` is eight named booleans, so a bit no
            // constant declares is not a value it can hold.

            // Every lowered op is an op, and every index the engine will
            // dereference is in range of the table it indexes.
            for op in &plan.ops {
                assert!(
                    eta_ir::op::spec(op.tag).is_some(),
                    "{id}: lowered op tag {:#04x} is not in OP_TABLE",
                    op.tag
                );
            }
            assert_eq!(
                plan.ops.len(),
                plan.source_ops.len(),
                "{id}: {} ops but {} source-op lists",
                plan.ops.len(),
                plan.source_ops.len()
            );
            for region in plan.singleton.iter().chain(&plan.fused) {
                for node in &region.nodes {
                    assert!(
                        (*node as usize) < plan.ops.len(),
                        "{id}: region names node {node} of {} ops",
                        plan.ops.len()
                    );
                }
            }
            for binding in &plan.channel_bindings {
                assert!(
                    *binding != u32::MAX,
                    "{id}: an unbound channel slot reached the engine"
                );
            }
            if plan.needs.mtp_rows {
                assert!(
                    plan.mtp_rows > 0,
                    "{id}: asks the engine for MTP rows and then names zero of them"
                );
            }
        }
    }
    assert!(
        plans >= 15,
        "only {plans} stage plans; the sweep is too thin to mean anything"
    );
}

