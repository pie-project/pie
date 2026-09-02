//! Pins the qwen4 text's names and widths against the shipped 2-bit
//! artifact `qwen38-flash-next-full-u4g64-u2g128-kv-bf16` (the parent
//! `mini-l4-e16-p8` was carved from), at the shipped `Dims`. Reads facts
//! from a local snapshot or `$PIE_HEADER_MANIFEST` (a JSON header census
//! that avoids downloading the 68 GiB artifact); neither present is a
//! skip, not a red run. The one test that needs real bytes,
//! [`the_shipped_artifact_identifies_as_the_full_two_bit_row`], skips on
//! the manifest alone.


/// The catalog row the shipped artifact is declared by — not the plain
/// `qwen38-flash-next-u4g64-u2g128-kv-bf16`, which is the miniature's.
const SKU: &str = "qwen38-flash-next-full-u4g64-u2g128-kv-bf16";

// ── where a census gets its facts ───────────────────────────────────────────

// ── the censuses ────────────────────────────────────────────────────────────

/// The gathered class is inherited, not added: `engine-metal::gather::
/// Plan::of` finds its table structurally (the table param of a
/// `Layout::EmbedConcat`, head count off `Attention::PleNgramIds`), so this
/// asserts the two facts that planner reads off this row's own trace and
/// no device — one concatenating gather with a 320,001,536-row table
/// beside one hasher with sixteen primes.
#[test]
fn the_full_row_emits_the_gather_the_planner_keys_on() {
    use model_dsl::{Attention, Def, Layout, Operation, Platform};

    let trace = (models::sku(SKU).expect("this build ships the full 2-bit row").trace)(Platform::Metal);

    let mut heads: Vec<usize> = Vec::new();
    let mut tables: Vec<(String, Vec<u64>)> = Vec::new();
    for node in &trace.nodes {
        match &node.op {
            Operation::Attention(
                Attention::PleNgramIds { primes, .. }
                | Attention::PleNgramIdsChunked { primes, .. },
            ) => heads.push(primes.len()),
            Operation::Layout(Layout::EmbedConcat { table, .. }) => {
                // `Plan::of`'s own resolution: the gather's table operand
                // is a `Def::Weight` row, and nothing else resolves there.
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

    // Two hasher nodes and one gather: the forward splits on `qo == 1`
    // (prefill/decode arms). `Plan::of` folds them with a `max`, so what
    // it reads is sixteen either way.
    assert_eq!(
        heads,
        vec![16, 16],
        "the two hasher arms of one fire, sixteen hashed heads each — \
         `Plan::of` reads the head count off `primes.len()` and seats that many \
         rows per fired token"
    );
    assert_eq!(tables.len(), 1, "exactly one concatenating gather in the plan");
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

