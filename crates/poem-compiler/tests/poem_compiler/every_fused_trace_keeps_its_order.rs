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

/// Gemma 4's attention prologue is a pattern the compiler folds, on every
/// layer whose q, k and v share one projection.
#[test]
fn gemma_4_writes_its_kv_through_the_fused_kernel() {
    let deployment = poem_compiler::catalog::deployment("gemma4-31b-bf16-kv-bf16")
        .expect("the catalog states gemma4-31b");
    let trace = deployment.trace(Platform::Cuda);
    let count = |t: &poem_ir::Trace, op: &str| {
        t.nodes
            .iter()
            .filter(|node| poem_ir::Operands::name(&node.op) == op)
            .count()
    };
    let layers = count(&trace, "layout.split_qkv");
    let fused =
        poem_compiler::fuse::fuse(trace, &["custom_cuda.qkv_fused_qknorm_rope_vnorm_write"]);
    assert!(layers > 0);
    assert_eq!(count(&fused, "layout.split_qkv"), 0);
    assert_eq!(
        count(&fused, "custom_cuda.qkv_fused_qknorm_rope_vnorm_write"),
        layers
    );
}

#[test]
fn gpt_oss_rotates_and_writes_its_kv_from_the_packed_row() {
    let deployment = poem_compiler::catalog::deployment("gptoss-120b-bf16-mxfp4-kv-bf16")
        .expect("the catalog states gptoss-120b");
    let trace = deployment.trace(Platform::Cuda);
    let count = |t: &poem_ir::Trace, op: &str| {
        t.nodes
            .iter()
            .filter(|node| poem_ir::Operands::name(&node.op) == op)
            .count()
    };
    let layers = count(&trace, "layout.split_qkv");
    let fused = poem_compiler::fuse::fuse(
        trace,
        &[
            "elementwise.split_rope_yarn_append",
            "elementwise.rope_yarn_append",
        ],
    );
    assert!(layers > 0);
    assert_eq!(
        (
            count(&fused, "layout.split_qkv"),
            count(&fused, "elementwise.split_rope_yarn_append")
        ),
        (0, layers)
    );
}

/// Kimi-K3's latent attention prefill (its query cut, absorbs and scoring)
/// is a pattern the compiler folds, on every MLA layer.
#[test]
fn kimi_k3_folds_its_latent_prefill() {
    let deployment = poem_compiler::catalog::deployment("kimik3-l4-bf16-mxfp4a8-kv-bf16")
        .expect("the catalog states kimik3-l4");
    let trace = deployment.trace(Platform::Cuda);
    let count = |t: &poem_ir::Trace, op: &str| {
        t.nodes
            .iter()
            .filter(|node| poem_ir::Operands::name(&node.op) == op)
            .count()
    };
    let prefills = count(&trace, "attention.mla_prefill");
    let fused = poem_compiler::fuse::fuse(trace, &["attention.mla_prefill_expanded"]);
    assert!(prefills > 0);
    assert_eq!(count(&fused, "attention.mla_prefill"), 0);
    assert_eq!(count(&fused, "attention.mla_prefill_expanded"), prefills);
}

/// Kimi-K3's AttnRes blends whose rows only a norm reads fold with it.
#[test]
fn kimi_k3_folds_its_blends_into_their_norms() {
    let deployment = poem_compiler::catalog::deployment("kimik3-l4-bf16-mxfp4a8-kv-bf16")
        .expect("the catalog states kimik3-l4");
    let trace = deployment.trace(Platform::Cuda);
    let count = |t: &poem_ir::Trace, op: &str| {
        t.nodes
            .iter()
            .filter(|node| poem_ir::Operands::name(&node.op) == op)
            .count()
    };
    let blends = count(&trace, "elementwise.res_blend");
    let fused = poem_compiler::fuse::fuse(trace, &["elementwise.res_blend_rmsnorm"]);
    assert!(blends > 0);
    assert_eq!(count(&fused, "elementwise.res_blend_rmsnorm"), blends);
    assert_eq!(count(&fused, "elementwise.res_blend"), 0);
}
