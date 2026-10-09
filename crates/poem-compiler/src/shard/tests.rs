use poem_ir::Platform;

/// A row-parallel projection leaves each rank a partial sum, which is reduced
/// once, right after it, and nothing else is added.
#[test]
fn a_split_row_reduces_each_projection_once() {
    let deployment =
        models::deployment("qwen35-d0.8b-bf16-kv-bf16-tp2").expect("the catalog ships the row");
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
    let whole = models::deployment("qwen35-d0.8b-bf16-kv-bf16")
        .expect("the one-rank row")
        .trace(Platform::Cuda);
    assert_eq!(trace.nodes.len(), whole.nodes.len() + 48);
}
