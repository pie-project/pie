//! What the driver is handed, pinned.
//!
//! [`tensor_compiler::codegen::launch::build`] produces the only artefact the driver
//! executes: a `LaunchPackage` of lowered values, channels, ports, stages and
//! per-stage grouped plans. Until this file existed it had exactly one caller
//! in the tested workspace — `cuda_golden::emit_driver_test_kernel_fixtures`,
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

use driver_api::program::LaunchStagePlan;
use msl_corpus::{GOLDEN_NAMES, golden_container, golden_profile, synthetic_traces};
use tensor_compiler::plan::compile_bound;
use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::{ModelProfile, Stage};
use tensor_ir::types::{DType, Shape};
use tensor_ir::validate::bind;

/// Every corpus trace's launch package, tagged with the trace it came from.
fn packages() -> Vec<(String, Vec<LaunchStagePlan>)> {
    bound_and_refused().0
}

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
        let package = tensor_compiler::codegen::launch::build(&bound, &stages);
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
fn every_plan_the_driver_receives_is_well_formed() {
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
            // may happen without the other, or the driver sees a plan it will
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

            // Every lowered op is an op, and every index the driver will
            // dereference is in range of the table it indexes.
            for op in &plan.ops {
                assert!(
                    tensor_ir::op::spec(op.tag).is_some(),
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
                    "{id}: an unbound channel slot reached the driver"
                );
            }
            if plan.needs.mtp_rows {
                assert!(
                    plan.mtp_rows > 0,
                    "{id}: asks the driver for MTP rows and then names zero of them"
                );
            }
        }
    }
    assert!(
        plans >= 15,
        "only {plans} stage plans; the sweep is too thin to mean anything"
    );
}

/// How many stage plans the grouped path accepts, pinned.
///
/// The structural claims above hold just as well when a stage is wrongly
/// refused — an invalid plan with a reason attached is well-formed. This is the
/// assertion that notices, and it is why `grouped_supported_tag` narrowing by
/// one op is now a test failure rather than a silent loss of the grouped path
/// for every stage containing that op.
#[test]
fn the_grouped_path_accepts_the_same_stages_it_always_has() {
    let packages = packages();
    let mut valid = 0usize;
    let mut refused: Vec<String> = Vec::new();
    for (name, stage_plans) in &packages {
        for (index, plan) in stage_plans.iter().enumerate() {
            if plan.needs.grouped_valid {
                valid += 1;
            } else {
                refused.push(format!("{name}#{index}: {}", plan.error));
            }
        }
    }
    refused.sort();
    assert_eq!(
        (valid, refused.as_slice()),
        (
            EXPECTED_GROUPED_VALID,
            EXPECTED_REFUSALS
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .as_slice()
        ),
        "the grouped classification moved; if that is intended, say why here"
    );
}

// 17 → 18: the corpus grew the `lora_prologue` golden (stage-4 lora sink),
// whose single prologue stage — channel peeks feeding a pass-wide sink — is
// grouped-valid like any other sink-carrying stage. No existing stage's
// classification moved.
const EXPECTED_GROUPED_VALID: usize = 18;

/// A prologue `lora` sink raises its own stage flag, and only its own.
///
/// The sink-flag derivation used to set `StageNeeds::page_mask` for
/// every `sink_call`; now that two first-party sinks exist it dispatches on
/// the resolved name, and this is what notices if that dispatch regresses in
/// either direction — a lora stage flagged as a page mask would have the
/// driver look for a page selection that never comes, and vice versa.
#[test]
fn the_lora_sink_raises_its_own_stage_flag() {
    let chan = |shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(DType::F32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: true,
    };
    let container = TraceContainer {
        names: vec!["lora".to_string()],
        channels: vec![
            chan(Shape::new(&[2, 2, 4]).unwrap()), // A [num_layers, R, d]
            chan(Shape::new(&[2, 4, 2]).unwrap()), // B [num_layers, d_out, R]
            chan(Shape::vector(4)),                // SITES
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::ChanRead(0),
                Op::ChanRead(1),
                Op::ChanRead(2),
                Op::SinkCall {
                    name: 0,
                    args: vec![0, 1, 2],
                },
            ],
        }],
        externs: Vec::new(),
    };
    let bound = bind(container, ModelProfile::dummy()).expect("the lora prologue binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let plan = &package.plans[0];
    assert!(
        plan.needs.lora,
        "a stage writing the lora sink must tell the driver so"
    );
    assert!(
        !plan.needs.page_mask,
        "a lora stage is not a page-mask stage"
    );

    // Over the corpus: whichever sink flag a plan raises, its name table
    // holds the sink that earns it — and both flags still occur (the quest
    // traces' page mask, `synthetic_sink_call`'s lora).
    let (mut page_mask_plans, mut lora_plans) = (0usize, 0usize);
    for (name, stage_plans) in &packages() {
        for plan in stage_plans {
            if plan.needs.page_mask {
                page_mask_plans += 1;
                assert!(
                    plan.names.iter().any(|n| n == "attn_page_mask"),
                    "{name}: a page-mask flag with no attn_page_mask in the name table"
                );
            }
            if plan.needs.lora {
                lora_plans += 1;
                assert!(
                    plan.names.iter().any(|n| n == "lora"),
                    "{name}: a lora flag with no lora in the name table"
                );
            }
        }
    }
    assert!(
        page_mask_plans > 0,
        "the corpus has quest traces; their page-mask flag went missing"
    );
    assert!(
        lora_plans > 0,
        "the corpus's synthetic lora sink lost its flag"
    );
}

/// Both refusals are the same rule: the grouped runtime binds only the
/// intrinsics its lane table describes, so a stage reading one it does not know
/// is refused rather than mis-bound. `metal_msl_golden` records these same two
/// cases as places the C++ oracle bound the request to the logits buffer and
/// ran anyway.
const EXPECTED_REFUSALS: &[&str] = &[
    "pentathlon_iter#1: stage uses an unsupported intrinsic",
    "synthetic_mtp_drafts#0: stage uses an unsupported intrinsic",
];
