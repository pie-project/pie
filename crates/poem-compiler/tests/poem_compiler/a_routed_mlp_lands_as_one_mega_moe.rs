use poem_ir::{Fused, Linear, Operation, Platform};

/// How a mega-MoE folds its routes: the weights first (the GLU takes them,
/// the shared row added) or the weighted sum.
#[derive(Debug, PartialEq)]
enum Fold {
    First { shared: bool },
    Weighted,
}

fn folds(trace: &poem_ir::Trace) -> Vec<Fold> {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Fused(Fused::MoeMega { glu, sum, .. }) => Some(match (&**glu, &**sum) {
                (
                    Linear::Glu {
                        weights: Some(_), ..
                    },
                    Linear::MoeCombine {
                        weights: None,
                        shared,
                        ..
                    },
                ) => Fold::First {
                    shared: shared.is_some(),
                },
                (
                    Linear::Glu { weights: None, .. },
                    Linear::MoeCombine {
                        weights: Some(_),
                        shared: None,
                        ..
                    },
                ) => Fold::Weighted,
                other => panic!("a mega-MoE folds as no kernel does: {other:?}"),
            }),
            _ => None,
        })
        .collect()
}

/// Every routed MLP of a row whose banks the mega-MoE takes lands as one,
/// ahead of the grouped experts that would take it otherwise: DeepSeek-V4's
/// W4A8 experts their weights first with the shared row added, a bf16
/// row's (DeepSeek-V4's, GLM's) under the weighted sum; and the fused trace
/// checks.
#[test]
fn a_routed_mlp_lands_as_one_mega_moe() {
    let rules = poem_compiler::fuse::rules::forming(&[
        "fused.moe_mega",
        "fused.moe_experts",
        "fused.moe_select_glu",
    ]);
    for (deployment, fold) in [
        (
            "dsv4-flash-fp8b128a8-mxfp4a8-kv-bf16",
            Fold::First { shared: true },
        ),
        (
            "dsv4-flash-mini-fp8b128a8-mxfp4a8-kv-e4m3",
            Fold::First { shared: true },
        ),
        ("dsv4-flash-bf16-kv-bf16", Fold::Weighted),
        ("glm47-bf16-kv-bf16", Fold::Weighted),
    ] {
        let row = poem_compiler::catalog::deployment(deployment)
            .unwrap_or_else(|| panic!("the catalog states {deployment}"));
        let trace = row.trace(Platform::Cuda);
        let layers = trace
            .nodes
            .iter()
            .filter(|node| matches!(node.op, Operation::Linear(Linear::MoeCombine { .. })))
            .count();
        let fused = poem_compiler::fuse::fuse(trace, &rules);
        assert!(layers > 0, "{deployment}: a routed MLP");
        let landed = folds(&fused);
        assert_eq!(landed.len(), layers, "{deployment}: every routed MLP");
        assert!(
            landed.iter().all(|f| *f == fold),
            "{deployment}: {landed:?}"
        );
        if let Err(faults) = poem_ir::check(&fused) {
            panic!("{deployment}: {faults:?}");
        }
    }
}

/// A bank the mega-MoE does not take (DeepSeek-V4's MXFP4 experts against
/// bf16 rows, GLM's NVFP4 against NVFP4) is left to the grouped experts.
#[test]
fn a_bank_the_mega_moe_does_not_take_stays_with_the_grouped_experts() {
    let rules = poem_compiler::fuse::rules::forming(&["fused.moe_mega", "fused.moe_experts"]);
    for deployment in [
        "dsv4-flash-bf16-mxfp4-kv-bf16",
        "glm47-bf16-nvfp4a4-nvfp4-kv-bf16",
    ] {
        let row = poem_compiler::catalog::deployment(deployment)
            .unwrap_or_else(|| panic!("the catalog states {deployment}"));
        let fused = poem_compiler::fuse::fuse(row.trace(Platform::Cuda), &rules);
        assert!(folds(&fused).is_empty(), "{deployment}");
        assert!(
            fused
                .nodes
                .iter()
                .any(|node| matches!(node.op, Operation::Fused(Fused::MoeExperts { .. }))),
            "{deployment}"
        );
    }
}
