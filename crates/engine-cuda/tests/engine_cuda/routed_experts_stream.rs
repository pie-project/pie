use engine_cuda::experts::{Attachments, Budgets, Plan, Ranking};
use poem::{Dtype, Platform};
use poem_ir::Trace;

fn micro() -> Trace {
    poem_compiler::catalog::trace_of("qwen3-a3b-micro", Dtype::Bf16, Dtype::Bf16, Platform::Cuda)
}

fn full_demand(trace: &Trace) -> u64 {
    Plan::of(trace, &Attachments::new(), Budgets::uncapped())
        .expect("a bf16 routed text plans")
        .device_demand()
}

#[test]
fn routed_experts_stream_every_case() {
    a_host_budget_under_the_pinned_tier_is_refused_by_name();
    an_uncapped_budget_opens_no_tier_at_all();
    a_plan_whose_cache_takes_the_banks_seats_no_expert();
    a_bank_the_cache_does_not_take_stays_whole_on_the_device();
}

/// The dense routed banks of the micro text, and their bytes.
fn banks(trace: &Trace) -> (Vec<usize>, u64) {
    let cached = engine_cuda::store::layout::streamed_experts(trace);
    assert!(
        !cached.is_empty(),
        "the micro text's banks are plain dense selects' the weight cache streams"
    );
    let bytes = cached
        .iter()
        .map(|&param| {
            let plane = &trace.params[param];
            (plane.shape.iter().product::<u64>() * 2).next_multiple_of(4096)
        })
        .sum();
    (cached, bytes)
}

/// The weight cache streams the banks: under a budget the plan seats no
/// expert itself, the budget decides only what else leaves the device, and
/// the banks' bytes come off the device's demand when the load moves them.
fn a_plan_whose_cache_takes_the_banks_seats_no_expert() {
    let trace = micro();
    let full = full_demand(&trace);
    let (cached, bank_bytes) = banks(&trace);
    let ranking = || {
        Ranking::of(&trace, &Attachments::new())
            .expect("ranks")
            .caching(&cached)
    };
    // A budget that holds everything but the banks cuts nothing.
    let mut plan =
        Plan::cut(&ranking(), Budgets::device(full - bank_bytes)).expect("the dense planes fit");
    assert!(plan.banks().is_empty() && !plan.streams());
    // The banks are the pool's from the cut on: the weight tier is never
    // charged for them, before the load moves them or after.
    assert_eq!(plan.device_demand(), full - bank_bytes);
    plan.host_experts(&trace, &cached).expect("the banks pin");
    assert_eq!(plan.device_demand(), full - bank_bytes);
    assert_eq!(plan.cached_experts(), cached.as_slice());
    // So is a load with no budget at all.
    let plan = Plan::cut(&ranking(), Budgets::uncapped()).expect("uncapped plans");
    assert_eq!(plan.device_demand(), full - bank_bytes);
    // A tighter one spills dense planes, and still seats no expert.
    let tight = (full - bank_bytes) * 3 / 4;
    let mut plan = Plan::cut(&ranking(), Budgets::device(tight)).expect("dense planes spill");
    assert!(plan.banks().is_empty(), "no expert is the plan's to seat");
    assert!(plan.streams(), "a dense plane left the device");
    assert!(
        plan.device_demand() <= tight,
        "{} on the device under a budget of {tight}, before the banks move",
        plan.device_demand()
    );
    plan.host_experts(&trace, &cached).expect("the banks pin");
    assert!(
        plan.device_demand() <= tight,
        "{} on the device under a budget of {tight}",
        plan.device_demand()
    );
}

/// A dense routed bank the cache does not take is not streamed by the plan
/// either: it stays whole on the device, so a budget under it is refused.
fn a_bank_the_cache_does_not_take_stays_whole_on_the_device() {
    let trace = micro();
    let full = full_demand(&trace);
    let (_, bank_bytes) = banks(&trace);
    let ranking = Ranking::of(&trace, &Attachments::new())
        .expect("ranks")
        .caching(&[]);
    let plan = Plan::cut(&ranking, Budgets::device(full - 1)).expect("one plane short spills");
    assert!(plan.banks().is_empty());
    let why = Plan::cut(&ranking, Budgets::device(bank_bytes / 2))
        .expect_err("half the banks' bytes does not hold them");
    assert!(why.to_string().contains("device_weight_budget"), "{why}");
}

fn a_host_budget_under_the_pinned_tier_is_refused_by_name() {
    let trace = micro();
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
    let trace = micro();
    let plan = Plan::of(&trace, &Attachments::new(), Budgets::uncapped()).expect("uncapped plans");
    assert!(!plan.streams());
    assert_eq!(
        plan.host_demand(),
        0,
        "a fully-resident load pins nothing — dev's `place_all` allocates no host tier"
    );
}
