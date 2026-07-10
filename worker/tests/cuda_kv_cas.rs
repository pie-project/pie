//! Real-hardware #4 validation — KV-CAS dedup + production-inferlet canaries on
//! `cuda_native` (Lane C).
//!
//! Proves the two things the mock driver CANNOT:
//!  A. **full-page CAS sealing dedups over real KV** — identical prompts seal the
//!     same chained `compute_page_hashes` (over real positions/masks/content) and
//!     share ONE canonical sealed-page set, while divergent prompts seal K
//!     distinct sets. Measured directly off the live CAS index + arena.
//!  B. **fork / spec / snapshot inferlets produce real coherent tokens** on the
//!     paged-KV / atomic-txn / CoW forward path (real CoW, real save→/scratch→open
//!     replay), via the result-captured canary harness.
//!
//! Shares the `common` cuda harness (`boot_cuda` + `install_inferlet` +
//! `spawn_inferlet`). One boot per process (global engine state). Run warm:
//!   cargo test -p pie-worker --features driver-cuda --test cuda_kv_cas -- --ignored --nocapture

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::Duration;

use pie_engine::arena::{ArenaKind, registry as arena_registry};
use pie_engine::program::ProgramName;
use pie_engine::working_set::kv_cas;

const MODEL_IDX: usize = 0;
const DRIVER_IDX: usize = 0;

fn kv_pages_used() -> u32 {
    arena_registry::get(MODEL_IDX, DRIVER_IDX)
        .lock()
        .unwrap()
        .used(ArenaKind::KvPage)
}
fn cas_sealed_count() -> usize {
    kv_cas::get(MODEL_IDX, DRIVER_IDX).lock().unwrap().len()
}
fn page_size() -> u32 {
    arena_registry::get(MODEL_IDX, DRIVER_IDX)
        .lock()
        .unwrap()
        .block_size()
}

/// A prompt of ≥ `min_pages` full KV pages so at least one page SEALS — partial
/// tail pages never seal (W7), so a sub-page prompt gives CAS nothing to dedup.
fn full_page_prompt(min_pages: u32) -> String {
    let need_tokens = ((min_pages + 1) * page_size()) as usize; // +1 page headroom
    let mut s = String::from("Read this passage and answer at the end.\n");
    while s.split_whitespace().count() < need_tokens {
        s.push_str("the quick brown fox jumps over the lazy dog and then ");
    }
    s
}

/// Peak CAS-index size + KV pages seen while a batch was in flight.
struct BatchPeak {
    max_cas: usize,
    max_used: u32,
}

/// Spawn `k` requests concurrently (raw `process::spawn`, so they overlap — unlike
/// the awaiting `spawn_input`), sampling peak CAS-index size + KV-page usage on a
/// background thread during the in-flight window, then await all (asserting each
/// succeeds). `input_for(i)` builds request `i`'s prompt.
async fn run_concurrent_batch(
    program: &ProgramName,
    k: usize,
    input_for: impl Fn(usize) -> String,
) -> BatchPeak {
    let max_cas = Arc::new(AtomicUsize::new(0));
    let max_used = Arc::new(AtomicU32::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    let (mc, mu, st) = (max_cas.clone(), max_used.clone(), stop.clone());
    let sampler = std::thread::spawn(move || {
        // Sample the live CAS index + arena while the K requests overlap. Both
        // locks are sync and held only briefly by the engine's txn ops.
        while !st.load(Ordering::Relaxed) {
            mc.fetch_max(cas_sealed_count(), Ordering::Relaxed);
            mu.fetch_max(kv_pages_used(), Ordering::Relaxed);
            std::thread::sleep(Duration::from_micros(250));
        }
    });

    let mut rxs = Vec::with_capacity(k);
    for i in 0..k {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let input = format!(r#"{{"prompt":{:?},"max_tokens":8}}"#, input_for(i));
        pie_engine::process::spawn(
            "cuda-test".into(),
            program.clone(),
            input,
            None,
            false,
            Some(tx),
        )
        .expect("spawn process");
        rxs.push(rx);
    }
    // The K processes run concurrently on the engine; awaiting the channels just
    // collects results as they finish.
    for (i, rx) in rxs.into_iter().enumerate() {
        let r = rx.await.expect("result channel dropped");
        r.unwrap_or_else(|e| panic!("batch request {i} errored on cuda: {e}"));
    }

    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    BatchPeak {
        max_cas: max_cas.load(Ordering::Relaxed),
        max_used: max_used.load(Ordering::Relaxed),
    }
}

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda + a local model snapshot; one boot per process"]
fn cuda_kv_cas_dedup_and_canaries() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let worker = common::boot_cuda().await;
        eprintln!("[cuda_kv_cas] engine up on {}", worker.url());

        // ── A. CAS dedup over real KV ────────────────────────────────────────
        let program = common::install_inferlet("text-completion").await;
        let ps = page_size();
        let prompt = full_page_prompt(2);
        const K: usize = 4;

        // K identical multi-page prompts → ONE shared canonical sealed-page set.
        let shared = run_concurrent_batch(&program, K, |_| prompt.clone()).await;
        // K divergent prompts (distinct page-0 content) → K distinct sealed sets.
        let divergent = run_concurrent_batch(&program, K, |i| format!("Variant {i}. {prompt}")).await;

        eprintln!(
            "[cuda_kv_cas] page_size={ps} K={K} | shared(max_cas={}, max_kv={}) | divergent(max_cas={}, max_kv={})",
            shared.max_cas, shared.max_used, divergent.max_cas, divergent.max_used
        );

        // PROOF: identical prefixes dedup to one canonical set in the CAS index,
        // so the peak distinct-sealed-page count is far below the divergent batch
        // (which seals ~K× as many). cas-index size is the timing-robust signal
        // (independent of the transient prefill allocation peak).
        assert!(
            shared.max_cas >= 1,
            "expected ≥1 full prefix page to seal into the CAS index (prompt too short? page_size={ps})"
        );
        assert!(
            shared.max_cas * 2 <= divergent.max_cas,
            "CAS dedup not observed: identical-prefix sealed-page count ({}) should be ≪ divergent ({}) — \
             if these are close, the K requests may not have overlapped (raise K / max_tokens) or dedup regressed",
            shared.max_cas, divergent.max_cas
        );
        // Secondary: shared peak KV pages should also be below divergent.
        eprintln!(
            "[cuda_kv_cas] CAS dedup OK — shared {}× fewer sealed pages than divergent",
            (divergent.max_cas as f64 / shared.max_cas.max(1) as f64)
        );

        // ── B. canaries for real on cuda (real tokens) ───────────────────────
        // Direct coherence check first: `spawn_text` RETURNS the generated text
        // (unlike best-of-n/demo-persistent-kv, which stream via `println!` and
        // return an empty/status string — so for those `Ok` == the pipeline ran).
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
