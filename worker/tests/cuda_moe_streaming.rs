//! Real-hardware validation of weight streaming: a MoE model decodes the same
//! tokens whether its routed experts are resident or paged in through the slab.
//!
//! Streaming is a residency decision, not a numerical one. The group's plan is
//! the stacking contract's expression with the expert axis removed, so a slot
//! holds bit-for-bit what one stride of the stack would have held, and the
//! GEMMs that read it are the same GEMMs. Any difference in the decoded text is
//! a bug -- in the plan, the rebinding, or the eviction.
//!
//! The harness boots one model per process, so the comparison is across two
//! runs rather than inside one. Run both and diff the `MOE_STREAM_RESULT` line:
//!
//!   S=/path/to/qwen3-moe-snapshot
//!   PIE_CUDA_TEST_SNAPSHOT=$S \
//!     cargo test -p pie-worker --features driver-cuda \
//!     --test cuda_moe_streaming -- --ignored --nocapture
//!   PIE_CUDA_TEST_SNAPSHOT=$S PIE_CUDA_TEST_STREAM_EXPERTS=1 \
//!     PIE_CUDA_TEST_EXPERT_CACHE_GB=0.0004 \
//!     cargo test -p pie-worker --features driver-cuda \
//!     --test cuda_moe_streaming -- --ignored --nocapture
//!
//! A deliberately tiny `expert_cache_gb` is the interesting setting: it forces
//! the slab down to a couple of slots so every layer evicts, which is the only
//! way the eviction path gets exercised at all.
//!
//! Both runs take the per-expert dispatch: streamed experts have no fused slab
//! to stride, so the resident run is pinned to the same path rather than the
//! batched decode one. Otherwise the diff would confound a residency change
//! with a change of kernel -- which is exactly what it did on first run, and
//! the batched and per-expert GEMMs do not agree to the last bit.
//!
//! A random-weight fixture is not automatically a useful oracle here: an
//! untrained model's argmax tends to be the same token whatever the experts
//! hold, so a wrong page-in sails through. Scale the fixture's per-expert
//! `down_proj` by a per-expert factor first; then expert identity is visible in
//! the decoded text, which is the only thing this test can see.

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda + a local MoE snapshot; one boot per process"]
fn cuda_moe_decodes_the_same_tokens_streamed_or_resident() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let streaming = std::env::var("PIE_CUDA_TEST_STREAM_EXPERTS").as_deref() == Ok("1");
        // Pin the resident run to the dispatch streaming is forced onto, so the
        // two runs differ in residency and nothing else.
        unsafe { std::env::set_var("PIE_QWEN35_MOE_FORCE_GENERAL", "1") };
        let worker = common::boot_cuda().await;
        eprintln!(
            "[cuda_moe_streaming] engine up on {} (streaming={streaming})",
            worker.url()
        );

        // Temperature 0 is an in-graph argmax, so the emitted tokens are a
        // pure function of the logits. Two runs that route the same experts
        // through the same GEMMs therefore have to emit the same text.
        let started = std::time::Instant::now();
        let result = common::spawn_inferlet(
            "text-completion-bench",
            &std::env::var("PIE_CUDA_TEST_PROMPT_JSON").unwrap_or_else(|_| {
                r#"{"prompt":"The capital of France is","max_tokens":16,"temperature":0.0}"#
                    .to_string()
            }),
        )
        .await;
        let text = result.expect("inferlet errored on cuda");
        // Wall time of the whole request, which is what streaming is finally
        // judged on: per-miss microseconds only matter through this number.
        eprintln!(
            "[cuda_moe_streaming] MOE_STREAM_WALL streaming={streaming} ms={}",
            started.elapsed().as_millis()
        );

        // The line the two runs are diffed on. Printed rather than asserted
        // because the expected value is whatever the other run produced.
        eprintln!("[cuda_moe_streaming] MOE_STREAM_RESULT streaming={streaming} text={text:?}");
        assert!(
            !text.trim().is_empty(),
            "MoE forward must decode non-empty text, got empty"
        );

        worker.shutdown().await;
    });
}
