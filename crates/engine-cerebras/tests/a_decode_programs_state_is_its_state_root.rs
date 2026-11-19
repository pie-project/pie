//! A decode program of stream units states its state once (host.json `state`: each K / V or
//! recurrent column a region), and the engine reads it from there alone; a program compiled
//! before the root is refused.

use engine_cerebras::reading::Reading;

fn decode(model: &str, lanes: u32) -> (tempfile::TempDir, Reading) {
    let trace = poem_compiler::catalog::repository()
        .parse(model)
        .unwrap_or_else(|| panic!("{model} parses"))
        .trace(poem_ir::Platform::Cerebras);
    let dir = tempfile::tempdir().expect("a scratch directory");
    kernels_cerebras::compile::stream::compile_decode(
        &trace,
        256,
        lanes,
        true,
        None,
        &Default::default(),
        dir.path(),
    )
    .unwrap_or_else(|e| panic!("{model}: {e}"));
    let reading = Reading::open(dir.path()).expect("host.json states the program");
    (dir, reading)
}

#[test]
fn a_paged_programs_lanes_pages_and_rows_are_its_state_roots() {
    let (dir, r) = decode("qwen3-0.6b-bf16-kv-bf16-l2", 4);
    let g = r.paging.expect("its attention columns page");
    assert_eq!(g.lanes, 4);
    assert!(g.pages > 1 && g.size >= 64, "{g:?}");
    // every attention PE takes the page tables and the rows
    let n = |rs: &[engine_cerebras::sdk::Rect]| rs.iter().map(|r| r.w * r.h).sum::<i32>();
    assert!(n(&r.ptab) > 0 && n(&r.rowp) >= n(&r.ptab));
    // the description is the file's `state` alone: the old keys are refused
    let at = dir.path().join("host.json");
    let mut h: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&at).expect("host.json")).expect("json");
    let state = h["state"].clone();
    assert_eq!(state["lanes"], 4);
    h["rowp"] = serde_json::json!([]);
    std::fs::write(&at, serde_json::to_string(&h).expect("json")).expect("write");
    let e = Reading::open(dir.path()).expect_err("a top-level rowp is refused");
    assert!(e.contains("before the state root"), "{e}");
}

#[test]
fn a_recurrences_rows_go_to_its_state_columns() {
    // (Qwen3.5's first layer a gated delta rule: no pages, its state a lane slot)
    let (_dir, r) = decode("qwen35-d0.8b-bf16-kv-bf16-l1", 1);
    assert!(r.paging.is_none());
    assert!(r.ptab.is_empty());
    assert!(!r.rowp.is_empty(), "the state columns take a step's rows");
}
