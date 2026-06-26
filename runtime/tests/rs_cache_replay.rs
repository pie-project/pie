//! rs_cache integration test — fork/take replay from a suspended rs context.
//!
//! Spawns a 4-slot context manager with `restore_pause < 0` so suspended
//! replay sources stay off-GPU. This forces fork/take to replay the source's
//! committed lineage into a fresh rs slot, and the replay forward request must
//! carry the rs_cache RESET flag.

use std::sync::{Arc, OnceLock};

mod common;
use common::mock_device::MockBackend;
use common::rs_cache_fixture::{
    RecordingBehavior, fresh_pid, make_committed_rs_context, spawn_managers, wait_for_replay,
};

struct TestState {
    #[allow(dead_code)]
    backend: MockBackend,
    recorder: Arc<RecordingBehavior>,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let recorder = Arc::new(RecordingBehavior::new());
        let backend = MockBackend::new(1, recorder.clone());
        let rt = tokio::runtime::Runtime::new().unwrap();
        // 4 slots + a tiny restore-pause threshold (0.01): any nonzero GPU
        // utilization keeps the restore loop paused, so suspended replay
        // sources stay off-GPU and fork/take must replay their lineage.
        // (The old harness used -1.0 for "always paused"; that out-of-range
        // value now deadlocks the FCFS reserve path — flagged to host owner.
        // An in-range value achieves the same intent without the deadlock.)
        rt.block_on(spawn_managers(4, 0.01));
        TestState {
            backend,
            recorder,
            rt,
        }
    })
}

#[test]
fn fork_and_take_replay_send_reset() {
    let s = state();
    s.rt.block_on(async {
        // Fork from a suspended rs context must replay lineage into a fresh
        // slot, and the replay request must carry RESET.
        s.recorder.clear();
        let parent = make_committed_rs_context().await;
        pie::context::suspend(parent).await.unwrap();
        let child = pie::context::fork(parent, fresh_pid().await).await.unwrap();
        wait_for_replay(child).await;
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
        let source = make_committed_rs_context().await;
        pie::context::save(source, "rs-user".to_string(), Some("snap".to_string()))
            .await
            .unwrap();
        let snapshot_id = pie::context::lookup("rs-user".to_string(), "snap".to_string())
            .await
            .unwrap();
        pie::context::suspend(snapshot_id).await.unwrap();
        let taken = pie::context::take(
            "rs-user".to_string(),
            "snap".to_string(),
            fresh_pid().await,
        )
        .await
        .unwrap();
        wait_for_replay(taken).await;
        let take_replays = s.recorder.forwards();
        assert!(
            take_replays
                .iter()
                .any(|r| !r.rs_slot_ids.is_empty() && r.rs_slot_flags == vec![1]),
            "take replay should send an rs_cache RESET request: {take_replays:?}"
        );
    });
}
