//! Does `open-with-report` actually detect a replay?
//!
//! Eviction suspends a context rather than deleting it, so `open` succeeds on
//! an evicted snapshot and silently replays the missing prefix. `open-with-report`
//! exists to make that visible. A test that only ever sees the healthy value
//! proves nothing about the instrument, so this asserts BOTH directions:
//!
//!   resident  ->  replayed_pages == 0   (the reuse claim is real)
//!   evicted   ->  replayed_pages  > 0   (the reuse claim is name-level only)
//!
//! Getting the second case to happen is the whole difficulty. Releasing a
//! snapshot's refcount is not enough: pages are content-addressed and live in a
//! trie, so they survive as long as ANY context still references them, and even
//! unreferenced pages linger until something actually reclaims them. So the
//! test drops every reference and then allocates against the pool until the
//! pages are genuinely gone.

use std::sync::{Arc, OnceLock};

mod common;

use common::mock_device::{Behavior, MockBackend};

/// Minimal backend: echo a token per request, record nothing.
struct QuietBehavior;

impl Behavior for QuietBehavior {
    fn handle_fire_batch(&self, req: &pie_bridge::ForwardRequest) -> pie_bridge::ForwardResponse {
        let n = req.qo_indptr.len().saturating_sub(1) as u32;
        pie_bridge::ForwardResponse {
            num_requests: n,
            tokens_indptr: (0..=n).collect(),
            tokens: vec![7; n as usize],
            dists_req_indptr: vec![0; (n + 1) as usize],
            dists_kv_indptr: vec![0],
            logits_req_indptr: vec![0; (n + 1) as usize],
            logits_byte_indptr: vec![0],
            logprobs_req_indptr: vec![0; (n + 1) as usize],
            logprobs_val_indptr: vec![0],
            entropies_indptr: vec![0; (n + 1) as usize],
            spec_indptr: vec![0; (n + 1) as usize],
            spec_tokens: Vec::new(),
            spec_positions: Vec::new(),
            ..Default::default()
        }
    }
}

struct TestState {
    #[allow(dead_code)]
    backend: MockBackend,
    rt: tokio::runtime::Runtime,
    model: usize,
}

static STATE: OnceLock<TestState> = OnceLock::new();

/// A KV-only model (`rs_slots = 0`) on a deliberately small pool, so the pages
/// backing a snapshot can actually be reclaimed. `restore_pause_at_utilization
/// = -1.0` keeps suspended contexts off-GPU rather than eagerly restoring them,
/// which is what lets the evicted case stay evicted long enough to observe.
const POOL_PAGES: usize = 32;

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let backend = MockBackend::new(1, Arc::new(QuietBehavior));
        let rt = tokio::runtime::Runtime::new().unwrap();
        let model = rt.block_on(async {
            let ctx_idx = pie::context::spawn(
                4,
                vec![POOL_PAGES],
                vec![POOL_PAGES],
                4,
                vec![0],
                vec![false],
                4,
                None,
                32.0,
                -1.0,
            );
            let inf_idx = pie::inference::spawn(&[0], 4, 30, "greedy".to_string(), 0).await;
            assert_eq!(ctx_idx, inf_idx);
            ctx_idx
        });
        TestState {
            backend,
            rt,
            model,
        }
    })
}

async fn fresh_pid() -> uuid::Uuid {
    let pid = uuid::Uuid::new_v4();
    pie::context::register_process(pid, None).await.unwrap();
    pid
}

async fn flush_mailbox(model: usize, id: pie::context::ContextId) {
    let _ = pie::context::debug_context_state(model, id).await;
}

/// A context holding one committed page, whose token content is unique to
/// `seed` so it hashes to its own trie entry rather than sharing with another.
async fn committed_context(model: usize, seed: u32) -> pie::context::ContextId {
    let id = pie::context::create(model, fresh_pid().await)
        .await
        .unwrap();
    pie::context::append_working_page_tokens(
        model,
        id,
        vec![seed, seed + 1, seed + 2, seed + 3],
        vec![0, 1, 2, 3],
        vec![],
        None,
        None,
    );
    flush_mailbox(model, id).await;
    pie::context::reserve_working_pages(model, id, 1)
        .await
        .unwrap();
    pie::context::commit_working_pages(model, id, 1)
        .await
        .unwrap();
    id
}

async fn save_as(model: usize, id: pie::context::ContextId, name: &str) -> pie::context::ContextId {
    pie::context::save(model, id, "u".to_string(), Some(name.to_string()))
        .await
        .unwrap();
    pie::context::lookup(model, "u".to_string(), name.to_string())
        .await
        .unwrap()
}

#[test]
fn open_report_sees_a_resident_snapshot_as_resident() {
    let s = state();
    s.rt.block_on(async {
        let source = committed_context(s.model, 100).await;
        let _snap = save_as(s.model, source, "resident").await;

        let (_child, report) = pie::context::fork_with_report(
            s.model,
            _snap,
            fresh_pid().await,
        )
        .await
        .unwrap();

        // Observed: resident_prefix_pages: 1, replayed_pages: 0 — the exact
        // inverse of the evicted case below. That inversion is the evidence
        // that the report tracks residency rather than merely name lookup.
        assert_eq!(
            report.replayed_pages, 0,
            "a snapshot still on GPU must report no replay: {report:?}"
        );
        assert!(
            report.resident_prefix_pages > 0,
            "and must report its prefix as resident: {report:?}"
        );
        assert!(!report.rs_replayed, "kv-only model must not replay rs state");
    });
}

#[test]
fn open_report_sees_an_evicted_snapshot_as_replayed() {
    let s = state();
    s.rt.block_on(async {
        // Save a snapshot, then drop every reference to its pages: the source
        // context AND the snapshot itself. Releasing refcounts alone does not
        // remove the pages from the content-addressed trie.
        let source = committed_context(s.model, 200).await;
        let snap = save_as(s.model, source, "evicted").await;
        pie::context::suspend(s.model, source).await.unwrap();
        pie::context::suspend(s.model, snap).await.unwrap();

        // Now actually reclaim them. Unreferenced pages linger until an
        // allocation needs the space, so churn the pool with contexts whose
        // content hashes differently.
        let mut ballast = Vec::new();
        for i in 0..POOL_PAGES {
            ballast.push(committed_context(s.model, 1000 + (i as u32) * 10).await);
        }

        let (_child, report) =
            pie::context::fork_with_report(s.model, snap, fresh_pid().await)
                .await
                .unwrap();

        // Observed: resident_prefix_pages: 0, replayed_pages: 1.
        //
        // This is the load-bearing assertion in the file. `open` returns Ok in
        // both this case and the resident one above, and `reused_tokens` would
        // be identical — so without this the instrument could be reporting a
        // constant and every gate built on it would still be green.
        assert!(
            report.replayed_pages > 0,
            "an evicted snapshot must report the pages it had to regenerate. \
             Got {report:?} -- if this is all zeroes the instrument cannot see \
             the failure it exists to detect."
        );
        assert_eq!(
            report.resident_prefix_pages, 0,
            "and nothing should have survived as resident: {report:?}"
        );
    });
}
