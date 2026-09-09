#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_compiler::plan::compile_bound;
use eta_ir::validate::bind;
use msl_corpus::{GOLDEN_NAMES, golden_container, golden_profile, synthetic_traces};

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
    (out, refused)
}

#[test]
fn every_plan_the_engine_receives_is_well_formed() {
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

            assert_eq!(
                plan.needs.grouped_valid,
                plan.error.is_empty(),
                "{id}: grouped-valid = {}, error = {:?}",
                plan.needs.grouped_valid,
                plan.error
            );

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
