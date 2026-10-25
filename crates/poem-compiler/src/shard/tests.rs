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

/// A projection a package states whole reads the share before it gathered:
/// each block of GLM-5.3-Flash's vision tower cuts its gate and up by their
/// columns and gathers what its whole down projection reads, and its merger
/// does the same, one gather each and none in the text model.
#[test]
fn a_whole_projection_reads_its_share_gathered() {
    let deployment = crate::catalog::deployment("glm53-flash-vision-u8g64-u2g64-kv-bf16-tp2")
        .expect("the catalog ships the row");
    let trace = deployment.trace(Platform::Cuda);
    let shard = |name: &str| {
        &trace
            .params
            .iter()
            .find(|p| p.name == name)
            .expect("the tower states it")
            .shard
    };
    assert!(matches!(
        shard("tower.blocks.0.gate_up"),
        poem_ir::Shard::Cut { .. }
    ));
    assert_eq!(*shard("tower.blocks.0.down"), poem_ir::Shard::Replicated);
    let gathers = trace
        .nodes
        .iter()
        .filter(|node| poem_ir::Operands::name(&node.op) == "collective.all_gather")
        .count();
    assert_eq!(
        gathers,
        GLM_TOWER_BLOCKS + 1,
        "one a block and one for the merger"
    );
}

/// The blocks of GLM-5.3-Flash's vision tower.
const GLM_TOWER_BLOCKS: usize = 24;

/// A padded q, k and v split shards by its heads: MiMo's FP8 rows, whose qkv
/// is stored unpadded, split eight ways run every one-rank op, each padded
/// split taking apart its rank's share (a query head in eight, a kv head a
/// group of ranks shares), and add nothing but the collectives.
#[test]
fn a_padded_qkv_split_takes_apart_each_ranks_heads() {
    const WORLD: u64 = 8;
    let split = crate::catalog::deployment("mimo26-flash-mini-e4m3g128a8-mxfp4a8-kv-bf16-tp8")
        .expect("the catalog ships the row")
        .trace(Platform::Cuda);
    let whole = crate::catalog::deployment("mimo26-flash-mini-e4m3g128a8-mxfp4a8-kv-bf16")
        .expect("the one-rank row")
        .trace(Platform::Cuda);
    let ops = |trace: &poem_ir::Trace| {
        let mut names: Vec<&str> = trace
            .nodes
            .iter()
            .map(|node| poem_ir::Operands::name(&node.op))
            .filter(|name| !name.starts_with("collective."))
            .collect();
        names.sort_unstable();
        names
    };
    assert_eq!(ops(&split), ops(&whole), "every op but the collectives");
    let padded = |trace: &poem_ir::Trace| -> Vec<([u32; 2], [u32; 2], u32)> {
        trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                poem_ir::Operation::Gather(poem_ir::Gather::SplitQkvPadded {
                    heads,
                    widths,
                    head_dim,
                    ..
                }) => Some((*heads, *widths, *head_dim)),
                _ => None,
            })
            .collect()
    };
    let (split, whole) = (padded(&split), padded(&whole));
    assert!(!whole.is_empty(), "the FP8 rows split q, k and v padded");
    assert_eq!(split.len(), whole.len());
    for ((heads, widths, head_dim), (all, all_widths, all_dim)) in split.iter().zip(&whole) {
        let kv_parts =
            poem_ir::Shard::parts(Some(u64::from(all[1])), WORLD).expect("kv heads split");
        assert_eq!(
            u64::from(heads[0]),
            u64::from(all[0]) / WORLD,
            "a rank's query heads"
        );
        assert_eq!(
            u64::from(heads[1]),
            u64::from(all[1]) / kv_parts,
            "a rank's kv heads"
        );
        assert_eq!((widths, head_dim), (all_widths, all_dim));
    }
}
