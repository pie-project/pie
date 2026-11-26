//! Every attention over a kv row states how it reads that row, masked or
//! not: a lane's mask is data, so a masked arm is shaped as the plain arm
//! is, and the plan values the arms share are read at one reading. MiniMax
//! M3 shares its plans between dense layers and sparse ones, whose arms
//! read the mask beside the indexer's selection: a sparse layer's masked
//! prefill once stated no reading, so its kv row stated no heads, and the
//! plans its decodes share with a dense layer were read at two.

use std::collections::BTreeMap;

use poem::Platform;
use poem_exec::store::kv::{reads, row_of, width_of};
use poem_ir::{Attn, Fused, Operation};

#[test]
fn every_attention_reads_its_plan_at_one_reading_minimax() {
    let row = poem_compiler::catalog::deployment("minimax-m3-mini-bf16-kv-bf16")
        .expect("the catalog lists MiniMax M3 mini");
    let trace = row.trace(Platform::Metal);

    let mut heads: BTreeMap<usize, u32> = BTreeMap::new();
    for node in &trace.nodes {
        let attn = match &node.op {
            Operation::Attn(attn) => attn,
            Operation::Fused(Fused::LseSink { attn, .. }) => attn,
            _ => continue,
        };
        let cache = match attn {
            Attn::Prefill { cache, .. } | Attn::Decode { cache, .. } => *cache,
            _ => continue,
        };
        if row_of(&trace, cache).is_none() {
            continue;
        }
        let read = reads(&node.op)
            .unwrap_or_else(|| panic!("an attention over a kv row states its reading: {attn:?}"));
        if let (Some(row), Some(kv)) = (row_of(&trace, read.cache), read.kv_heads) {
            heads.insert(row, kv);
        }
    }

    let mut plans = BTreeMap::new();
    for node in &trace.nodes {
        let Some(read) = reads(&node.op) else {
            continue;
        };
        let Some(row) = row_of(&trace, read.cache) else {
            continue;
        };
        let kv_heads = heads.get(&row).copied();
        assert!(kv_heads.is_some(), "kv row {row} states its heads");
        let width = width_of(&trace, read.q).expect("a query is a rectangle");
        let reading = (read.head_dim, kv_heads, width, read.window);
        let known = plans.entry(read.plan).or_insert(reading);
        assert_eq!(
            *known, reading,
            "plan value {} is read at one reading",
            read.plan.0
        );
    }
    assert!(!plans.is_empty(), "MiniMax M3 attends its kv rows");
}
