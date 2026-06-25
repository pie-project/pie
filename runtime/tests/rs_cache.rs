//! Runtime-level rs_cache regression tests.
//!
//! These stay below the WASM API and exercise the context manager directly:
//! fresh slot flags, slot-pressure deferral, replay after fork/take, and
//! the KV-only model path.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

mod common;

use common::mock_device::{Behavior, MockBackend};

struct RecordingBehavior {
    forwards: Mutex<Vec<pie_driver_abi::ForwardRequest>>,
}

impl RecordingBehavior {
    fn new() -> Self {
        Self {
            forwards: Mutex::new(Vec::new()),
        }
    }

    fn clear(&self) {
        self.forwards.lock().unwrap().clear();
    }

    fn forwards(&self) -> Vec<pie_driver_abi::ForwardRequest> {
        self.forwards.lock().unwrap().clone()
    }
}

impl Behavior for RecordingBehavior {
    fn handle_fire_batch(&self, req: &pie_driver_abi::ForwardRequest) -> pie_driver_abi::ForwardResponse {
        self.forwards.lock().unwrap().push(req.clone());
        let n = req.qo_indptr.len().saturating_sub(1) as u32;
        pie_driver_abi::ForwardResponse {
            num_requests: n,
            tokens_indptr: (0..=n).collect(),
            tokens: vec![7; n as usize],
            dists_req_indptr: vec![0; (n + 1) as usize],
            dists_kv_indptr: vec![0],
            dists_ids: Vec::new(),
            dists_probs: Vec::new(),
            logits_req_indptr: vec![0; (n + 1) as usize],
            logits_byte_indptr: vec![0],
            logits_bytes: Vec::new(),
            logprobs_req_indptr: vec![0; (n + 1) as usize],
            logprobs_val_indptr: vec![0],
            logprobs_values: Vec::new(),
            entropies_indptr: vec![0; (n + 1) as usize],
            entropies: Vec::new(),
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
    recorder: Arc<RecordingBehavior>,
    rt: tokio::runtime::Runtime,
    rs_one_slot_model: usize,
    rs_replay_model: usize,
    kv_only_model: usize,
}

static STATE: OnceLock<TestState> = OnceLock::new();

async fn spawn_pair(rs_slots: usize, restore_pause_at_utilization: f64) -> usize {
    let ctx_idx = pie::context::spawn(
        4,
        vec![32],
        vec![32],
        4,
        vec![rs_slots],
        vec![false],
        4,
        None,
        32.0,
        restore_pause_at_utilization,
    );
    let inf_idx = pie::inference::spawn(&[0], 4, 30, "greedy".to_string(), 0).await;
    assert_eq!(ctx_idx, inf_idx);
    ctx_idx
}

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let recorder = Arc::new(RecordingBehavior::new());
        let backend = MockBackend::new(1, recorder.clone());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (rs_one_slot_model, rs_replay_model, kv_only_model) = rt.block_on(async {
            (
                spawn_pair(1, 0.95).await,
                // Keep suspended replay sources off-GPU so fork/take
                // exercise their direct lineage replay paths.
                spawn_pair(4, -1.0).await,
                spawn_pair(0, 0.95).await,
            )
        });
        TestState {
            backend,
            recorder,
            rt,
            rs_one_slot_model,
            rs_replay_model,
            kv_only_model,
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

async fn make_committed_rs_context(model: usize) -> pie::context::ContextId {
    let id = pie::context::create(model, fresh_pid().await)
        .await
        .unwrap();
    let pinned = pie::context::pin(model, id, 0).await.unwrap();
    assert_eq!(pinned.rs_flags, 1);
    pie::context::unpin(model, id);
    flush_mailbox(model, id).await;

    pie::context::append_working_page_tokens(
        model,
        id,
        vec![10, 11, 12, 13],
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

async fn wait_for_replay(model: usize, id: pie::context::ContextId) {
    for _ in 0..100 {
        let debug = pie::context::debug_context_state(model, id).await;
        if debug.contains("state: Active") && debug.contains("pending_replay: false") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "replay did not complete: {}",
        pie::context::debug_context_state(model, id).await
    );
}

#[test]
fn runtime_rs_cache_regressions() {
    let s = state();
    s.rt.block_on(async {
        // Fresh rs slot assignment must set RESET exactly once.
        let ctx = pie::context::create(s.rs_one_slot_model, fresh_pid().await)
            .await
            .unwrap();
        let first = pie::context::pin(s.rs_one_slot_model, ctx, 0)
            .await
            .unwrap();
        assert_eq!(first.rs_slot, Some(0));
        assert_eq!(first.rs_flags, 1);
        pie::context::unpin(s.rs_one_slot_model, ctx);
        flush_mailbox(s.rs_one_slot_model, ctx).await;

        let second = pie::context::pin(s.rs_one_slot_model, ctx, 0)
            .await
            .unwrap();
        assert_eq!(second.rs_slot, Some(0));
        assert_eq!(second.rs_flags, 0);
        pie::context::unpin(s.rs_one_slot_model, ctx);
        flush_mailbox(s.rs_one_slot_model, ctx).await;
        pie::context::destroy(s.rs_one_slot_model, ctx)
            .await
            .unwrap();

        // Slot pressure is non-preemptive: a second pin waits until the
        // resident holder suspends and releases the slot, then receives the
        // freed slot with RESET.
        let holder = pie::context::create(s.rs_one_slot_model, fresh_pid().await)
            .await
            .unwrap();
        let waiter = pie::context::create(s.rs_one_slot_model, fresh_pid().await)
            .await
            .unwrap();
        let held = pie::context::pin(s.rs_one_slot_model, holder, 0)
            .await
            .unwrap();
        assert_eq!(held.rs_slot, Some(0));
        let pending = tokio::spawn(pie::context::pin(s.rs_one_slot_model, waiter, 0));
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!pending.is_finished(), "rs slot waiter should defer");
        pie::context::unpin(s.rs_one_slot_model, holder);
        flush_mailbox(s.rs_one_slot_model, holder).await;
        pie::context::suspend(s.rs_one_slot_model, holder)
            .await
            .unwrap();
        let waited = tokio::time::timeout(Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(waited.rs_slot, Some(0));
        assert_eq!(waited.rs_flags, 1);
        pie::context::unpin(s.rs_one_slot_model, waiter);
        flush_mailbox(s.rs_one_slot_model, waiter).await;

        // KV-only models must not populate rs_cache fields.
        let kv_ctx = pie::context::create(s.kv_only_model, fresh_pid().await)
            .await
            .unwrap();
        let kv_pin = pie::context::pin(s.kv_only_model, kv_ctx, 0).await.unwrap();
        assert_eq!(kv_pin.rs_slot, None);
        assert_eq!(kv_pin.rs_flags, 0);
        pie::context::unpin(s.kv_only_model, kv_ctx);
        flush_mailbox(s.kv_only_model, kv_ctx).await;

        // Fork from a suspended rs context must replay lineage into a fresh
        // slot, and the replay request must carry RESET.
        s.recorder.clear();
        let parent = make_committed_rs_context(s.rs_replay_model).await;
        pie::context::suspend(s.rs_replay_model, parent)
            .await
            .unwrap();
        let child = pie::context::fork(s.rs_replay_model, parent, fresh_pid().await)
            .await
            .unwrap();
        wait_for_replay(s.rs_replay_model, child).await;
        let fork_replays = s.recorder.forwards();
        assert!(
            fork_replays
                .iter()
                .any(|r| !r.rs_slot_ids.is_empty() && r.rs_slot_flags == vec![1]),
            "fork replay should send an rs_cache RESET request: {fork_replays:?}"
        );

        // Save a resident snapshot, suspend that snapshot to make rs state
        // missing, then take it. Take must replay the snapshot lineage.
        s.recorder.clear();
        let source = make_committed_rs_context(s.rs_replay_model).await;
        pie::context::save(
            s.rs_replay_model,
            source,
            "rs-user".to_string(),
            Some("snap".to_string()),
        )
        .await
        .unwrap();
        let snapshot_id =
            pie::context::lookup(s.rs_replay_model, "rs-user".to_string(), "snap".to_string())
                .await
                .unwrap();
        pie::context::suspend(s.rs_replay_model, snapshot_id)
            .await
            .unwrap();
        let taken = pie::context::take(
            s.rs_replay_model,
            "rs-user".to_string(),
            "snap".to_string(),
            fresh_pid().await,
        )
        .await
        .unwrap();
        wait_for_replay(s.rs_replay_model, taken).await;
        let take_replays = s.recorder.forwards();
        assert!(
            take_replays
                .iter()
                .any(|r| !r.rs_slot_ids.is_empty() && r.rs_slot_flags == vec![1]),
            "take replay should send an rs_cache RESET request: {take_replays:?}"
        );
    });
}
