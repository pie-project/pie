const SKU: &str = "qwen38-flash-next-full-u4g64-u2g128-kv-bf16";

#[test]
fn the_full_row_emits_the_gather_the_planner_keys_on() {
    use model_dsl::{Attention, Def, Layout, Operation, Platform};

    let trace = (models::sku(SKU)
        .expect("this build ships the full 2-bit row")
        .trace)(Platform::Metal);

    let mut heads: Vec<usize> = Vec::new();
    let mut tables: Vec<(String, Vec<u64>)> = Vec::new();
    for node in &trace.nodes {
        match &node.op {
            Operation::Attention(
                Attention::PleNgramIds { primes, .. }
                | Attention::PleNgramIdsChunked { primes, .. },
            ) => heads.push(primes.len()),
            Operation::Layout(Layout::EmbedConcat { table, .. }) => {
                let Some(Def::Weight(w)) = trace.values.get(table.0 as usize).map(|d| &d.def)
                else {
                    panic!("the gather's table operand is not a weight");
                };
                let param = &trace.params[*w as usize];
                tables.push((param.name.clone(), param.shape.clone()));
            }
            _ => {}
        }
    }

    assert_eq!(
        heads,
        vec![16, 16],
        "the two hasher arms of one fire, sixteen hashed heads each — \
         `Plan::of` reads the head count off `primes.len()` and seats that many \
         rows per fired token"
    );
    assert_eq!(
        tables.len(),
        1,
        "exactly one concatenating gather in the plan"
    );
    let (name, shape) = &tables[0];
    assert_eq!(name, "ple.table");
    assert_eq!(
        shape.first().copied(),
        Some(320_001_536),
        "`Plan::of` takes the slab's row count from `params[table].shape[0]`, \
         and the full row's table is the shipped 320 001 536 rows — three \
         hundred and twenty million, which is why this class exists"
    );
    assert_eq!(
        shape.get(1).copied(),
        Some(160),
        "hidden 2560 over sixteen heads: the row a single head's gather lands"
    );
}
