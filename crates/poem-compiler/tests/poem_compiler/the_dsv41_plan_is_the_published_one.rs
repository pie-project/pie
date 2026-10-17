//! DeepSeek-V4.1-Flash's package lays out the release's plan: an Engram
//! table as tall as the checkpoint states (`engram_num_embeddings:
//! [384006168, 384016682]`), and CSA2's cross-layer reuse — a reuse layer
//! reads its source's pool and index keys rather than writing its own —
//! and its Causal Encoder-Decoder's bounded replay, where a prompt's
//! leading rows run the encoder alone.

use poem_ir::{Attention, CacheRow, Fact, Operands, Operation, Platform};

#[test]
fn the_dsv41_plan_is_the_published_one() {
    let row = poem_compiler::catalog::deployment("dsv41-flash-bf16-mxfp4-kv-bf16")
        .expect("the V4.1 row ships");
    let trace = row.trace(Platform::Cuda);
    let rows_of = |name: &str| {
        trace
            .params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("no `{name}`"))
            .shape[0]
    };
    assert_eq!(rows_of("layer.1.engram.embed"), 384_006_168);
    assert_eq!(rows_of("layer.14.engram.embed"), 384_016_682);
    assert!(
        trace
            .params
            .iter()
            .all(|p| !p.name.starts_with("layer.0.engram"))
    );

    let kv_rows: Vec<&str> = trace
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    // Layers 2, 8, 14 and 20 are the KV sources; every other pooling layer
    // reads theirs.
    for source in [2, 8, 14, 20] {
        assert!(
            kv_rows.contains(&format!("pool.{source}").as_str()),
            "{kv_rows:?}"
        );
        assert!(
            kv_rows.contains(&format!("index.{source}").as_str()),
            "{kv_rows:?}"
        );
    }
    for reuse in [3, 21, 24] {
        assert!(
            !kv_rows.contains(&format!("pool.{reuse}").as_str()),
            "{kv_rows:?}"
        );
        assert!(
            !kv_rows.contains(&format!("index.{reuse}").as_str()),
            "{kv_rows:?}"
        );
    }
    // The ratio-2 sources gate their compressor; the ratio-1 one projects.
    assert!(
        trace
            .params
            .iter()
            .any(|p| p.name == "layer.2.compressor.wgate")
    );
    assert!(
        trace
            .params
            .iter()
            .all(|p| p.name != "layer.20.compressor.wgate")
    );
    assert!(trace.params.iter().any(|p| p.name == "layer.20.indexer.wk"));
    assert!(trace.params.iter().all(|p| p.name != "layer.24.indexer.wk"));

    // Bounded replay: a prompt's encoder-only rows run the encoder and the
    // decoder's sources (layer 20's pool and index keys, from the encoder's
    // last hidden states), and nothing past them; every decoder window
    // reads above its rows' replay floor.
    let encode = trace
        .facts
        .table
        .iter()
        .position(|fact| *fact == Fact::Flag("encode".into()))
        .expect("the trace reads the encode flag");
    let skipped = 1u64 << encode;
    let mut floored = 0;
    for node in &trace.nodes {
        let Some(layer) = node.layer else { continue };
        let runs_skipped = node.guard.holds(skipped);
        let sources = matches!(
            node.op,
            Operation::Attention(
                Attention::PoolKvAppend { .. } | Attention::PoolBoundaryPrefill { .. }
            )
        );
        if layer > 20 || (layer == 20 && !sources && !runs_skipped) {
            assert!(
                !runs_skipped,
                "layer {layer}'s `{}` runs on encoder-only rows",
                node.op.name()
            );
        }
        if layer == 20 && sources {
            assert!(
                runs_skipped,
                "layer 20's `{}` skips encoder-only rows",
                node.op.name()
            );
        }
        if let Operation::Attention(Attention::PrefillLse { floor, .. }) = &node.op {
            assert_eq!(floor.is_some(), layer >= 20, "layer {layer}'s window floor");
            floored += usize::from(floor.is_some());
        }
    }
    assert_eq!(floored, 20, "every decoder layer's window is floored");
}

/// V4.1's DSpark head beside its release format: three stages, each over a
/// window of its own one block wider than the trunk's, keyed from the
/// target layers' inputs, and the Markov head walking the block's readouts.
#[test]
fn the_dsv41_dspark_head_drafts_through_its_stages() {
    let row = poem_compiler::catalog::deployments()
        .find(|d| d.name.starts_with("dsv41-flash-mini") && d.name.contains("dspark"))
        .unwrap_or_else(|| {
            panic!(
                "no V4.1 DSpark row in {:?}",
                poem_compiler::catalog::deployments()
                    .map(|d| d.name.as_str())
                    .filter(|n| n.starts_with("dsv41"))
                    .collect::<Vec<_>>()
            )
        });
    let trace = row.trace(Platform::Cuda);
    let kv_rows: Vec<&str> = trace
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    for stage in 0..3 {
        assert!(
            kv_rows.contains(&format!("dspark.{stage}").as_str()),
            "{kv_rows:?}"
        );
    }
    let count = |want: fn(&Operation) -> bool| trace.nodes.iter().filter(|n| want(&n.op)).count();
    assert_eq!(
        count(|op| matches!(
            op,
            Operation::Attention(Attention::WindowSink { window: 133, .. })
        )),
        3,
        "a stage's window reads the trunk's 128 and the block's 5"
    );
    assert_eq!(
        count(|op| matches!(
            op,
            Operation::Attention(Attention::MarkovWalk { block: 5, .. })
        )),
        1
    );
    assert!(trace.params.iter().any(|p| p.name == "dspark.main_proj"));
    assert!(
        trace
            .params
            .iter()
            .any(|p| p.name == "dspark.2.experts_down")
    );
}
