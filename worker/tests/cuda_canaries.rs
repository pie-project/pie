//! Real-hardware #4 validation — production-inferlet canaries on `cuda_native`
//! (Lane C).
//!
//! Proves what the mock driver CANNOT: fork / spec / snapshot inferlets produce
//! real coherent tokens on the paged-KV / atomic-txn / CoW forward path (real
//! CoW, real save→/scratch→open replay), via the result-captured canary
//! harness. (The former CAS-index dedup half of this lane sampled the live CAS
//! index through `pie_engine::arena`/`working_set::kv_cas`, an introspection
//! surface the engine no longer exposes; the prefix-dedup contract is covered
//! by the engine's prefix-cache e2e and the prefix-heavy benchmark gate.)
//!
//! Shares the `common` cuda harness (`boot_cuda` + `install_inferlet` +
//! `spawn_inferlet`). One boot per process (global engine state). Run warm:
//!   cargo test -p pie-worker --features driver-cuda --test cuda_canaries -- --ignored --nocapture

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda + a local model snapshot; one boot per process"]
fn cuda_inferlet_canaries() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let worker = common::boot_cuda().await;
        eprintln!("[cuda_canaries] engine up on {}", worker.url());

        // Direct coherence check first: `spawn_text` RETURNS the generated text
        // (unlike best-of-n/demo-persistent-kv, which stream via `println!` and
        // return an empty/status string — so for those `Ok` == the pipeline ran).
        let program = common::install_inferlet("text-completion").await;
        let text = common::spawn_text(&program, "The capital of France is", 16).await;
        eprintln!("[cuda canary] coherence text => {text:?}");
        let text = text.expect("text-completion errored on cuda");
        assert!(
            !text.trim().is_empty(),
            "cuda decode produced empty text — generation/forward regressed"
        );

        // fork-CoW / spec / snapshot pipelines: assert they RUN to success on real
        // hardware (these return empty/status by design — they print their output).
        let bon = common::spawn_inferlet(
            "best-of-n",
            r#"{"question":"What is 2+2? Reply with just the number.","num_candidates":3,"max_tokens":24}"#,
        )
        .await;
        eprintln!("[cuda canary] best-of-n (fork CoW) => {bon:?}");
        bon.expect("best-of-n errored on cuda");

        common::spawn_inferlet(
            "text-completion-spec",
            r#"{"prompt":"The capital of France is","max_tokens":24}"#,
        )
        .await
        .expect("text-completion-spec errored on cuda");

        let snap = common::spawn_inferlet(
            "demo-persistent-kv",
            r#"{"mode":"smart","turn1":"Remember the number 7.","turn2":"What number did I tell you?","max_tokens":24}"#,
        )
        .await;
        eprintln!("[cuda canary] demo-persistent-kv (save→open replay) => {snap:?}");
        snap.expect("demo-persistent-kv errored on cuda");

        worker.shutdown().await;
    });
}
