use engine_cuda::experts::{Attachments, Budgets, Plan};
use model_dsl::{Dtype, Platform};
use model_ir::Trace;

fn micro() -> (models::qwen_3::model::Model, Trace) {
    let m = models::qwen_3::model::Model::a3b_micro(Dtype::Bf16, Dtype::Bf16, 1);
    let trace = model_dsl::trace_hybrid("qwen35-a3b-micro", &m, Platform::Cuda);
    (m, trace)
}

fn full_demand(trace: &Trace) -> u64 {
    Plan::of(trace, &Attachments::new(), Budgets::uncapped())
        .expect("a bf16 routed text plans")
        .device_demand()
}

#[test]
fn routed_experts_stream_every_case() {
    a_budget_under_the_planes_that_cannot_move_is_refused_by_name();
    a_host_budget_under_the_pinned_tier_is_refused_by_name();
    an_uncapped_budget_opens_no_tier_at_all();
}

fn a_budget_under_the_planes_that_cannot_move_is_refused_by_name() {
    let (_, trace) = micro();
    let why = Plan::of(&trace, &Attachments::new(), Budgets::device(1 << 16))
        .expect_err("64 KiB holds no model");
    let said = why.to_string();
    assert!(
        said.contains("REGISTERED") && said.contains("cannot be moved to another tier"),
        "the refusal names the planes that cannot hold less: {said}"
    );
}

fn a_host_budget_under_the_pinned_tier_is_refused_by_name() {
    let (_, trace) = micro();
    let full = full_demand(&trace);
    let plan = Plan::of(&trace, &Attachments::new(), Budgets::device(full * 3 / 4))
        .expect("three quarters streams");
    assert!(plan.streams());
    let residency = engine::load::Residency {
        device_weight_budget: Some(full * 3 / 4),
        host_weight_budget: Some(plan.host_demand() - 1),
        ..engine::load::Residency::uncapped()
    };
    let why = residency
        .admit(plan.device_demand(), plan.host_demand())
        .expect_err("a pinned tier one byte short does not admit");
    let said = why.to_string();
    assert!(
        said.contains("host_weight_budget") && said.contains("pinned host"),
        "the refusal names the tier and the field: {said}"
    );
}

fn an_uncapped_budget_opens_no_tier_at_all() {
    let (_, trace) = micro();
    let plan = Plan::of(&trace, &Attachments::new(), Budgets::uncapped()).expect("uncapped plans");
    assert!(!plan.streams());
    assert_eq!(
        plan.host_demand(),
        0,
        "a fully-resident load pins nothing — dev's `place_all` allocates no host tier"
    );
}
