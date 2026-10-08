//! DeepSeek-V4.1-Flash's package lays out the release's plan: an Engram
//! table as tall as the checkpoint states (`engram_num_embeddings:
//! [384006168, 384016682]`), and CSA2's cross-layer reuse — a reuse layer
//! reads its source's pool and index keys rather than writing its own.

use poem_ir::{CacheRow, Platform};

#[test]
fn the_dsv41_plan_is_the_published_one() {
    let row = models::deployment("dsv41-flash-bf16-mxfp4-kv-bf16").expect("the V4.1 row ships");
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
}
