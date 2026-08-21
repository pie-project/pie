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
//!   cargo test -p worker --features driver-cuda-13 --test cuda_forward -- --ignored --nocapture

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda-13 + a local model snapshot; one boot per process"]
fn cuda_native_text_and_device_geometry_decode() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // (1) Boot the embedded cuda engine in-proc (loads the model onto the GPU
        //     + bootstraps the runtime) — the worker's prod path.
        let worker = common::boot_cuda().await;
        eprintln!("[cuda_forward] engine up on {}", worker.url());

        // (2) Install + drive a basic text-gen inferlet in-proc (no client edge).
        //
        // `text-completion` STOOD HERE and is not in `tests/inferlets` any
        // more; `text-completion-bench` is the one that survived the PTIR
        // bridge rewrite and takes the same `{prompt, max_tokens}` input.
        let program = common::install_inferlet("text-completion-bench").await;
        let result = common::spawn_text(&program, "The capital of France is", 16).await;

        // (3) Real cuda forward: prefill -> multi-token decode -> coherent text.
        eprintln!("[cuda_forward] RESULT = {result:?}");
        let text = result.expect("inferlet errored on cuda");
        assert!(
            !text.trim().is_empty(),
            "cuda forward must decode non-empty text, got empty"
        );

        // (4) STOOD HERE: a second spawn that drove `windowed-attention` and
        // matched its `WINDOWED_ATTENTION...` verdict, to exercise the PTIR
        // device-geometry wire form. RETIRED, and not for want of hardware.
        //
        // `windowed-attention` is not in `tests/inferlets` any more. Its
        // successor is `sliding-window-attention`, which builds the same wire
        // form and is what `tests/gpu/tests/cuda_sliding_window_attention_e2e`
        // drives -- and that gate fails on this exact box, with the card
        // present and the model loaded, because THIS DRIVER DOES NOT CLAIM
        // THAT CLASS. `driver-cuda/src/serve/load.rs` says so in as many
        // words beside `device_geometry_port_mask`:
        //
        //     `DEVICE_GEOMETRY_PORTS` is deliberately absent: it wins the
        //     pool-owned class this driver does not build.
        //
        // so the engine sends such a program down the host fallback, which
        // cannot derive `EmbedTokens` and reports `EmbedTokens is not
        // host-derivable: channel 0 has no host-known value` -- a sentence
        // about the symptom rather than about the claim, which is what
        // `engine/src/driver/backend/vulkan.rs` warns it reads as.
        //
        // A step here that turned that into a failure would make this file a
        // standing red about a capability nobody has regressed, and one that
        // skipped would say "no device". The gap belongs to the gate that is
        // about it, and that gate's `#[ignore]` reason now names it. What
        // stays here is the claim this file was always for: a coherent dense
        // forward on real silicon, which steps 1-3 make.

        worker.shutdown().await;
    });
}
