use poem_ir::{Fused, Linear, Operation, Platform, RouteScore, Trace};

/// The routes a trace picks off its logits by score (softmax, sigmoid,
/// sqrt-softplus), its hashed ones, and its fused router products.
fn routes(trace: &Trace) -> (usize, usize, usize) {
    let (mut scored, mut hashed, mut fused) = (0, 0, 0);
    for node in &trace.nodes {
        match &node.op {
            Operation::Linear(Linear::MoeRoute {
                score: RouteScore::Hash { .. },
                ..
            }) => hashed += 1,
            Operation::Linear(Linear::MoeRoute {
                score:
                    RouteScore::Softmax | RouteScore::Sigmoid { .. } | RouteScore::SqrtSoftplus { .. },
                ..
            }) => scored += 1,
            Operation::Fused(Fused::MatmulMoeRoute { .. }) => fused += 1,
            _ => {}
        }
    }
    (scored, hashed, fused)
}

/// Every scored route of DeepSeek-V4 (its bias-routed layers), Qwen3-MoE
/// and Kimi-K3 (896 experts, past what the fused kernel stages: it is
/// formed all the same, and the backend runs the two) folds with the
/// router's product into one op; a hashed route stays apart. The fused
/// traces still check.
#[test]
fn a_router_product_and_its_pick_fuse() {
    let rules = poem_compiler::fuse::rules::forming(&["fused.matmul_moe_route"]);
    let mut hashed_kept = 0;
    let mut widest = 0;
    for package in ["deepseek_v4", "qwen3_moe", "kimi_k3"] {
        let mut rows = 0;
        for d in poem_compiler::catalog::deployments().filter(|d| d.package.name() == package) {
            let Ok(trace) = d.try_trace(Platform::Cuda) else {
                continue;
            };
            let (scored, hashed, _) = routes(&trace);
            let fused = poem_compiler::fuse::fuse(trace, &rules);
            assert_eq!(
                routes(&fused),
                (0, hashed, scored),
                "{}: every scored route folds, no hashed one",
                d.name
            );
            if let Err(faults) = poem_ir::check(&fused) {
                panic!("{}: {faults:?}", d.name);
            }
            hashed_kept += hashed;
            widest = fused
                .nodes
                .iter()
                .filter_map(|node| match &node.op {
                    Operation::Fused(Fused::MatmulMoeRoute { experts, .. }) => Some(*experts),
                    _ => None,
                })
                .fold(widest, u32::max);
            rows += usize::from(scored > 0);
        }
        assert!(rows > 0, "{package} states a routed row");
    }
    assert!(hashed_kept > 0, "DeepSeek-V4's hashed layers stay apart");
    assert_eq!(
        widest, 896,
        "Kimi-K3's router is formed past the kernel's 512"
    );
}
