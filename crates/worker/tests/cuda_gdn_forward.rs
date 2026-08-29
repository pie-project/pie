//! Real-hardware validation — coherent HYBRID forward on `cuda_native`.
//!
//! The sibling of `cuda_forward`, pointed at the model whose layers are not
//! all the same shape: Qwen3.5-0.8B runs a gated-delta-net recurrence in
//! three layers out of every four and full paged attention in the fourth.
//! That path owns machinery nothing else on this engine touches — the
//! recurrent-state slabs, `ssm::causal_conv1d_prefill_batched`, the chunked
//! delta rule, `norm::rmsnorm_gated_fp32_in` — and until this file existed,
//! NOTHING IN THE TREE RAN IT.
//!
//! `worker/tests/common/mod.rs` had carried `DEFAULT_GDN_SNAPSHOT` and
//! `gdn_snapshot()` for that purpose the whole time, and not one caller. The
//! cost of that gap was a model that emitted
//!
//!     "\n\n\nqu.c.\n\n. / } )\n0\n -"
//!
//! on real silicon, in the shipped serve path, with every unit test green:
//! the gated RMSNorm was reducing across all sixteen value heads at once
//! because its per-head width was never stated, so it also read its 128-float
//! weight sixteen times over. A forward gate is the only kind of test that
//! catches that class, and a forward gate that only asks for non-empty text
//! is not one — hence `assert_coherent`.
//!
//! Run explicitly (one boot per process, like every gate in this directory):
//!   cargo test --release -p worker --features engine-cuda-13 \
//!       --test cuda_gdn_forward -- --ignored --nocapture

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features engine-cuda-13 + a local Qwen3.5 GDN snapshot (PIE_CUDA_TEST_GDN_SNAPSHOT); one boot per process"]
fn cuda_native_hybrid_gdn_decode_is_coherent() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let snapshot = common::gdn_snapshot();
        if !std::path::Path::new(&snapshot).is_dir() {
            eprintln!("[cuda_gdn_forward] SKIP: no snapshot at {snapshot}");
            return;
        }

        let worker = common::boot_cuda_model(&snapshot).await;
        eprintln!(
            "[cuda_gdn_forward] engine up on {} ({snapshot})",
            worker.url()
        );

        let program = common::install_inferlet("text-completion-bench").await;
        let result = common::spawn_text(&program, "The capital of France is", 16).await;

        eprintln!("[cuda_gdn_forward] RESULT = {result:?}");
        let text = result.expect("inferlet errored on the cuda hybrid path");
        common::assert_coherent(&text, 3);

        worker.shutdown().await;
    });
}
