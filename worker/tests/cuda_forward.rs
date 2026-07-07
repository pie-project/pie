//! Real-hardware #2 validation — coherent dense forward on `cuda_native`.
//!
//! Boots the worker's prod embedded path in-proc and drives a text-gen inferlet
//! through `program::add` → `process::spawn`, bypassing the gateway/client edge
//! entirely. A coherent multi-token completion proves the context→working-set
//! forward rewrite (project_kv physical ids → real flashinfer attention over
//! paged KV → atomic-txn commit → KV CAS) runs on real silicon — not just mock.
//!
//! Shares the `common` cuda harness with the Lane-C CAS-dedup and Lane-D
//! fold-parity tests (`boot_cuda()` + `spawn_text()`). Run explicitly:
//!   cargo test -p pie-worker --features driver-cuda --test cuda_forward -- --ignored --nocapture

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda + a local model snapshot; one boot per process"]
fn cuda_native_text_completion_decodes() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // (1) Boot the embedded cuda engine in-proc (loads the model onto the GPU
        //     + bootstraps the runtime) — the worker's prod path.
        let worker = common::boot_cuda().await;
        eprintln!("[cuda_forward] engine up on {}", worker.url());

        // (2) Install + drive a basic text-gen inferlet in-proc (no client edge).
        let program = common::install_inferlet("text-completion").await;
        let result = common::spawn_text(&program, "The capital of France is", 16).await;

        // (3) Real cuda forward: prefill -> multi-token decode -> coherent text.
        eprintln!("[cuda_forward] RESULT = {result:?}");
        let text = result.expect("inferlet errored on cuda");
        assert!(
            !text.trim().is_empty(),
            "cuda forward must decode non-empty text, got empty"
        );

        worker.shutdown().await;
    });
}
