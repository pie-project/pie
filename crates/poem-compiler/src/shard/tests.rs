use poem_ir::Platform;

/// A row-parallel projection leaves each rank a partial sum, which is reduced
/// once, right after it, and nothing else is added.
#[test]
fn a_split_row_reduces_each_projection_once() {
    let deployment = crate::catalog::deployment("qwen35-d0.8b-bf16-kv-bf16-tp2")
        .expect("the catalog ships the row");
    let trace = deployment.trace(Platform::Cuda);
    let count = |op: &str| {
        trace
            .nodes
            .iter()
            .filter(|node| poem_ir::Operands::name(&node.op) == op)
            .count()
    };
    assert_eq!(
        count("collective.all_reduce"),
        48,
        "one per mixer and one per mlp"
    );
    assert_eq!(
        count("collective.all_gather"),
        0,
        "the 0.8b row ties no vocab-split head"
    );
    let whole = crate::catalog::deployment("qwen35-d0.8b-bf16-kv-bf16")
        .expect("the one-rank row")
        .trace(Platform::Cuda);
    assert_eq!(trace.nodes.len(), whole.nodes.len() + 48);
}

/// A package that leaves its cuts to the compiler splits as Megatron does:
/// each layer's attention and MLP reduced once, after their row-cut
/// projections, and the vocab-split head gathered once.
#[test]
fn an_inferred_package_reduces_each_projection_once() {
    let deployment = crate::catalog::deployment("llama31-8b-bf16-kv-bf16-tp2")
        .expect("the catalog ships the row");
    assert!(
        deployment
            .package
            .infers_cuts()
            .expect("a package that says"),
        "llama 3 leaves its cuts to the compiler"
    );
    let trace = deployment.trace(Platform::Cuda);
    let count = |op: &str| {
        trace
            .nodes
            .iter()
            .filter(|node| poem_ir::Operands::name(&node.op) == op)
            .count()
    };
    let whole = crate::catalog::deployment("llama31-8b-bf16-kv-bf16")
        .expect("the one-rank row")
        .trace(Platform::Cuda);
    assert_eq!(
        count("collective.all_reduce"),
        2 * LLAMA_8B_LAYERS,
        "one per attention and one per mlp"
    );
    assert_eq!(count("collective.all_gather"), 1, "the logits, once");
    assert_eq!(
        trace.nodes.len(),
        whole.nodes.len() + 2 * LLAMA_8B_LAYERS + 1
    );
}

/// The layers of Llama 3.1 8B.
const LLAMA_8B_LAYERS: usize = 32;
