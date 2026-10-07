use poem_ir::{Fault, Platform};

/// Every rule run over every catalog trace leaves a trace whose values are
/// each written by the op that claims them, and read only after it.
#[test]
fn every_fused_trace_keeps_its_order() {
    let kernels: Vec<&str> = poem_compiler::fuse::kernels().collect();
    let mut broken = Vec::new();
    for sku in models::skus() {
        for platform in [Platform::Cuda, Platform::Metal] {
            let fused = poem_compiler::fuse::fuse((sku.trace)(platform), &kernels);
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
                    .map(|fault| format!("{} on {platform:?}: {fault}", sku.name)),
            );
        }
    }
    assert!(broken.is_empty(), "{}", broken.join("\n"));
}

/// Gemma 4's attention prologue is a pattern the compiler folds, on every
/// layer whose q, k and v share one projection.
#[test]
fn gemma_4_writes_its_kv_through_the_fused_kernel() {
    let sku = models::sku("gemma4-31b-bf16-kv-bf16").expect("the catalog states gemma4-31b");
    let trace = (sku.trace)(Platform::Cuda);
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
