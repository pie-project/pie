//! Real-hardware validation of weight streaming: a MoE model decodes the same
//! tokens whether its routed experts are resident or paged in through the slab.
//!
//! Streaming is a residency decision, not a numerical one. The group's plan is
//! the stacking contract with the expert axis removed, so a slot holds
//! bit-for-bit what one stride of the stack would have held and the GEMMs that
//! read it are the same GEMMs. Any difference in the decoded text is a bug -- in
//! the plan, the rebinding, or the eviction.
//!
//! One model boots per process, so the comparison is across two runs. Run both
//! and diff the `MOE_STREAM_RESULT` line:
//!
//!   S=/path/to/an/moe/snapshot
//!   PIE_CUDA_TEST_SNAPSHOT=$S \
//!     cargo test --release -p worker --features driver-cuda-13 \
//!     --test cuda_moe_streaming -- --ignored --nocapture
//!   PIE_CUDA_TEST_SNAPSHOT=$S PIE_CUDA_TEST_STREAM_EXPERTS=1 \
//!     PIE_CUDA_TEST_EXPERT_CACHE_GB=0.0004 \
//!     cargo test --release -p worker --features driver-cuda-13 \
//!     --test cuda_moe_streaming -- --ignored --nocapture
//!
//! A deliberately tiny `expert_cache` forces the slab down to a couple of slots
//! so every layer evicts, which is the only way the eviction path is exercised.
//!
//! # `--release`, and it is not a preference
//!
//! A `cargo test` without it builds the engine and the driver at `-O0`, and on
//! a 20-billion-parameter mixture the HOST side is then the whole cost: the
//! card sits at 0% while a batch takes ~36 s, and the harness's 180-second
//! deadline expires around the third one. The same run built `--release`
//! finishes in 95 s including a 13 GB load. Nothing about that is a device
//! fact, which is why it is written here rather than chased.
//!
//! # THE SECOND RUN CANNOT BE MADE ON THIS DRIVER YET
//!
//! `crates/driver-cuda/src/boot.rs` is, by its own first line, "every boot knob
//! this driver reads", and it holds nine: `runahead`, `supergraph`,
//! `trace_supergraph`, `device_transforms`, `kv_envelopes`,
//! `attn_score_window`, `rs_stash_tokens`, `calibrating`, and the KV page size
//! that is not a knob. `stream_routed_experts` is not among them, and neither
//! `expert_cache` nor `expert_host_cache` appears anywhere under
//! `crates/driver-cuda`. The worker composes all three into the boot JSON --
//! `embedded_driver.rs` does it for `cuda_native` specifically -- and this
//! driver reads none of them. Every expert stays resident, always.
//!
//! Which makes the two-run protocol above a comparison of a run with itself,
//! and it duly agrees with itself: run on `openai/gpt-oss-20b`, both halves
//! answered
//!
//! ```text
//!   token_ids [220,20,13,392,637,290,1577,11,472,306,290,279,40,939,11,472]
//! ```
//!
//! token for token, at 63.5 s and 56.8 s -- the streamed half being the FASTER
//! of the two, which is the tell: paging experts through a 429 KiB slab cannot
//! be faster than not paging them. A green pair here would have meant nothing
//! at all.
//!
//! That pair is corroboration and not the proof, and it is worth saying which
//! is which, because the sequence is not stable across BUILDS. Two consecutive
//! resident runs of one binary answer identically -- verified back to back,
//! `[220,18,13,220,623,4928,25,220,16,13,392,220,17,1,392,220]` twice -- so a
//! single build is reproducible and the diff this test performs is meaningful.
//! But that is a different sequence from the pair above, taken on a different
//! build of the same source tree, and nothing in the resident path's config
//! changed between them. Greedy decode on a 20B MXFP4 MoE sits on near-ties
//! that a recompile can tip. So the load-bearing evidence that streaming is
//! unimplemented is the SEARCH -- zero occurrences of the three knob names
//! under `crates/driver-cuda` -- which no rebuild can move.
//!
//! So `PIE_CUDA_TEST_STREAM_EXPERTS=1` refuses rather than passing. It is a
//! request for a thing this driver does not do, and a test that answers such a
//! request with success is worse than one that fails: it retires the ticket.
//! The refusal names `boot.rs`, so the day a knob lands there this file starts
//! working without being edited.
//!
//! # Which snapshot
//!
//! `PIE_CUDA_TEST_SNAPSHOT` must name a MoE checkpoint the catalog serves.
//! A tiny-random one will not do -- `identify` refuses it by shape, at length
//! and by name -- and neither will a 35B-A3B in bf16 on a 24 GB card.
//! `openai/gpt-oss-20b` is 13 GB of MXFP4 experts and is what the numbers above
//! were taken on.

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda-13 + a local MoE snapshot; one boot per process"]
fn cuda_moe_decodes_the_same_tokens_streamed_or_resident() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let streaming = std::env::var("PIE_CUDA_TEST_STREAM_EXPERTS").as_deref() == Ok("1");
        assert!(
            !streaming,
            "PIE_CUDA_TEST_STREAM_EXPERTS asks for expert paging, and \
             `driver-cuda` does not do it: `crates/driver-cuda/src/boot.rs` is \
             every boot knob this driver reads and `stream_routed_experts` is \
             not one of them, nor are `expert_cache` and `expert_host_cache` \
             read anywhere under `crates/driver-cuda`. The worker composes all \
             three into the boot JSON and this driver ignores all three, so a \
             run with them set is the SAME run. Unset it for the resident \
             half, which is the half this driver can make."
        );
        // Pin the resident run to the dispatch streaming is forced onto, so the
        // two runs differ in residency and nothing else.
        unsafe { std::env::set_var("PIE_QWEN35_MOE_FORCE_GENERAL", "1") };
        let worker = common::boot_cuda().await;
        eprintln!(
            "[cuda_moe_streaming] engine up on {} (streaming={streaming})",
            worker.url()
        );

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
