use poem_ir::{Fault, Platform};

/// Every catalog row that runs on more than one rank splits, and the split
/// trace reads each value only after the op, or the collective, that makes it.
#[test]
fn every_split_row_reads_what_it_makes_in_order() {
    let mut broken = Vec::new();
    for sku in models::skus().filter(|sku| sku.recipe.tp > 1) {
        for platform in [Platform::Cuda, Platform::Metal] {
            let Err(faults) = poem_ir::check(&(sku.trace)(platform)) else {
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

/// A row-parallel projection leaves each rank a partial sum, which is reduced
/// once, right after it, and nothing else is added.
#[test]
fn a_split_row_reduces_each_projection_once() {
    let sku = models::sku("qwen35-d0.8b-bf16-kv-bf16-tp2").expect("the catalog ships the row");
    let trace = (sku.trace)(Platform::Cuda);
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
    let whole = (models::sku("qwen35-d0.8b-bf16-kv-bf16")
        .expect("the one-rank row")
        .trace)(Platform::Cuda);
    assert_eq!(trace.nodes.len(), whole.nodes.len() + 48);
}
