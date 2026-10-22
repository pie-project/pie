use poem_ir::{Fault, Platform};

/// Every rule run over every catalog trace leaves a trace whose values are
/// each written by the op that claims them, and read only after it.
#[test]
fn every_fused_trace_keeps_its_order() {
    let kernels: Vec<&str> = poem_compiler::fuse::kernels().collect();
    let mut broken = Vec::new();
    for deployment in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits())
    {
        for platform in [Platform::Cuda, Platform::Metal] {
            let fused = poem_compiler::fuse::fuse(deployment.trace(platform), &kernels);
            let Err(faults) = poem_ir::check(&fused) else {
                continue;
            };
            broken.extend(
                faults
                    .into_iter()
                    .filter(|fault| {
                        matches!(
                            fault,
                            Fault::UseBeforeDef { .. }
                                | Fault::PhantomDef { .. }
                                | Fault::DoubleOutput { .. }
                                | Fault::ForeignOutput { .. }
                                | Fault::DefNodeOutOfRange { .. }
                                | Fault::OutOfRange { .. }
                        )
                    })
                    .map(|fault| format!("{} on {platform:?}: {fault}", deployment.name)),
            );
        }
    }
    assert!(broken.is_empty(), "{}", broken.join("\n"));
}

/// A catalog model's pattern folds on every layer it appears in: the kernel
/// replaces each `unfused` op one for one.
#[test]
fn catalog_models_fold_their_patterns_on_every_layer() {
    let count = |t: &poem_ir::Trace, op: &str| {
        t.nodes
            .iter()
            .filter(|node| poem_ir::Operands::name(&node.op) == op)
            .count()
    };
    for (deployment, unfused, kernels, landed) in [
        (
            "gemma4-31b-bf16-kv-bf16",
            "gather.split_qkv",
            &["fused.qkv_fused_qknorm_rope_vnorm_write"][..],
            "fused.qkv_fused_qknorm_rope_vnorm_write",
        ),
        (
            "gptoss-120b-bf16-mxfp4-kv-bf16",
            "gather.split_qkv",
            &["fused.split_rope_yarn_append", "fused.rope_yarn_append"][..],
            "fused.split_rope_yarn_append",
        ),
        (
            "kimik3-l4-bf16-mxfp4a8-kv-bf16",
            "attn.mla_prefill",
            &["fused.mla_prefill_expanded"][..],
            "fused.mla_prefill_expanded",
        ),
        (
            "kimik3-l4-bf16-mxfp4a8-kv-bf16",
            "elemwise.res_blend",
            &["fused.res_blend_rmsnorm"][..],
            "fused.res_blend_rmsnorm",
        ),
    ] {
        let trace = poem_compiler::catalog::deployment(deployment)
            .unwrap_or_else(|| panic!("the catalog states {deployment}"))
            .trace(Platform::Cuda);
        let layers = count(&trace, unfused);
        let fused = poem_compiler::fuse::fuse(trace, kernels);
        assert!(layers > 0, "{deployment}: {unfused}");
        assert_eq!(
            (count(&fused, unfused), count(&fused, landed)),
            (0, layers),
            "{deployment}: {landed}"
        );
    }
}
