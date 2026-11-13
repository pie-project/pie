use engine_cuda::experts::{Attachments, Plan, Ranking};
use poem::{Dtype, Platform};
use poem_ir::Trace;

fn micro() -> Trace {
    poem_compiler::catalog::trace_of("qwen3-a3b-micro", Dtype::Bf16, Dtype::Bf16, Platform::Cuda)
}

fn full_demand(trace: &Trace) -> u64 {
    Plan::of(trace, &Attachments::new())
        .expect("a bf16 routed text plans")
        .device_demand()
}

#[test]
fn routed_experts_stream_every_case() {
    a_plan_opens_no_tier_at_all();
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

/// The weight cache streams the banks: the plan seats no expert itself,
/// and the banks' bytes come off the device's demand from the cut on.
fn a_plan_whose_cache_takes_the_banks_seats_no_expert() {
    let trace = micro();
    let full = full_demand(&trace);
    let (cached, bank_bytes) = banks(&trace);
    let ranking = Ranking::of(&trace, &Attachments::new())
        .expect("ranks")
        .caching(&cached);
    let mut plan = Plan::cut(&ranking);
    assert!(!plan.streams());
    // The banks are the pool's from the cut on: the weight tier is never
    // charged for them, before the load moves them or after.
    assert_eq!(plan.device_demand(), full - bank_bytes);
    plan.host_experts(&trace, &cached).expect("the banks pin");
    assert_eq!(plan.device_demand(), full - bank_bytes);
    assert_eq!(plan.cached_experts(), cached.as_slice());
}

/// A dense routed bank the cache does not take stays whole on the device,
/// with every other dense plane.
fn a_bank_the_cache_does_not_take_stays_whole_on_the_device() {
    let trace = micro();
    let full = full_demand(&trace);
    let ranking = Ranking::of(&trace, &Attachments::new())
        .expect("ranks")
        .caching(&[]);
    let plan = Plan::cut(&ranking);
    assert!(!plan.streams());
    assert_eq!(plan.device_demand(), full);
}

fn a_plan_opens_no_tier_at_all() {
    let trace = micro();
    let plan = Plan::of(&trace, &Attachments::new()).expect("plans");
    assert!(!plan.streams());
    assert_eq!(
        plan.host_demand(),
        0,
        "a load whose weights all live on the device pins nothing"
    );
}
