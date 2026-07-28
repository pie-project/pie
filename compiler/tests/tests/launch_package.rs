//! What the driver is handed, pinned.
//!
//! [`pie_codegen::launch::build`] produces the only artefact the driver
//! executes: a `LaunchPackage` of lowered values, channels, ports, stages and
//! per-stage grouped plans. Until this file existed it had exactly one caller
//! in the tested workspace — `cuda_golden::emit_driver_test_kernel_fixtures`,
//! which writes the encoded package into a gitignored fixtures directory and
//! asserts nothing about its contents. So `PIE_STAGE_GROUPED_VALID`,
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
use pie_driver_abi::plan::LaunchStagePlan;
use pie_driver_abi::{
    PIE_STAGE_GROUPED_VALID, PIE_STAGE_REQUIRES_ATTN_SCORE, PIE_STAGE_REQUIRES_KERNEL_CALL,
    PIE_STAGE_REQUIRES_LAYER, PIE_STAGE_REQUIRES_MTP_ROWS, PIE_STAGE_REQUIRES_PAGE_MASK,
    PIE_STAGE_REQUIRES_QUERY,
};
use pie_ir::validate::bind;
use pie_plan::compile_bound;

/// Every corpus trace's launch package, tagged with the trace it came from.
fn packages() -> Vec<(String, Vec<LaunchStagePlan>)> {
    let mut out = Vec::new();
    let mut push = |name: &str, container, profile| {
        let Ok(bound) = bind(container, profile) else {
            return;
        };
        let stages = compile_bound(&bound);
        let package = pie_codegen::launch::build(&bound, &stages);
        out.push((name.to_string(), package.plans));
    };
    for name in GOLDEN_NAMES {
        push(name, golden_container(name), golden_profile(name));
    }
    for (name, container, profile) in synthetic_traces() {
        push(name, container, profile);
    }
    out
}

/// Every bit `lower_stage_plan` is allowed to set.
const DECLARED_FLAGS: u32 = PIE_STAGE_GROUPED_VALID
    | PIE_STAGE_REQUIRES_QUERY
    | PIE_STAGE_REQUIRES_LAYER
    | PIE_STAGE_REQUIRES_ATTN_SCORE
    | PIE_STAGE_REQUIRES_MTP_ROWS
    | PIE_STAGE_REQUIRES_KERNEL_CALL
    | PIE_STAGE_REQUIRES_PAGE_MASK;

#[test]
fn every_plan_the_driver_receives_is_well_formed() {
    let packages = packages();
    assert!(
        packages.len() >= 12,
        "only {} traces built a launch package; the corpus did not load",
        packages.len()
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
                plan.flags & PIE_STAGE_GROUPED_VALID != 0,
                plan.error.is_empty(),
                "{id}: grouped-valid = {}, error = {:?}",
                plan.flags & PIE_STAGE_GROUPED_VALID != 0,
                plan.error
            );
            assert_eq!(
                plan.flags & !DECLARED_FLAGS,
                0,
                "{id}: flags {:#010x} carry a bit no PIE_STAGE_* constant declares",
                plan.flags
            );

            // Every lowered op is an op, and every index the driver will
            // dereference is in range of the table it indexes.
            for op in &plan.ops {
                let tag = u8::try_from(op.code).expect("an op code is a wire tag");
                assert!(
                    pie_ir::op::spec(tag).is_some(),
                    "{id}: lowered op code {:#06x} is not in OP_TABLE",
                    op.code
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
            if plan.flags & PIE_STAGE_REQUIRES_MTP_ROWS != 0 {
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
            if plan.flags & PIE_STAGE_GROUPED_VALID != 0 {
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

const EXPECTED_GROUPED_VALID: usize = 17;

/// Both refusals are the same rule: the grouped runtime binds only the
/// intrinsics its lane table describes, so a stage reading one it does not know
/// is refused rather than mis-bound. `metal_msl_golden` records these same two
/// cases as places the C++ oracle bound the request to the logits buffer and
/// ran anyway.
const EXPECTED_REFUSALS: &[&str] = &[
    "pentathlon_iter#1: stage uses an unsupported intrinsic",
    "synthetic_mtp_drafts#0: stage uses an unsupported intrinsic",
];
