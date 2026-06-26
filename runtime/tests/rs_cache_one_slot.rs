//! rs_cache integration test — single-slot fresh-RESET + slot-pressure.
//!
//! Spawns one context manager with exactly ONE rs_cache slot so it can
//! exercise (a) fresh-slot RESET assignment and (b) non-preemptive slot
//! pressure: a second pin defers until the resident holder suspends and
//! releases the slot, then receives the freed slot with RESET.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::mock_device::MockBackend;
use common::rs_cache_fixture::{RecordingBehavior, flush_mailbox, fresh_pid, spawn_managers};

struct TestState {
    #[allow(dead_code)]
    backend: MockBackend,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let recorder = Arc::new(RecordingBehavior::new());
        let backend = MockBackend::new(1, recorder.clone());
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(spawn_managers(1, 0.95));
        TestState { backend, rt }
    })
}

#[test]
fn fresh_slot_reset_and_slot_pressure() {
    let s = state();
    s.rt.block_on(async {
        // Fresh rs slot assignment must set RESET exactly once.
        let ctx = pie::context::create(fresh_pid().await).await.unwrap();
        let first = pie::context::pin(ctx, 0).await.unwrap();
        assert_eq!(first.rs_slot, Some(0));
        assert_eq!(first.rs_flags, 1);
        pie::context::unpin(ctx);
        flush_mailbox(ctx).await;

        let second = pie::context::pin(ctx, 0).await.unwrap();
        assert_eq!(second.rs_slot, Some(0));
        assert_eq!(second.rs_flags, 0);
        pie::context::unpin(ctx);
        flush_mailbox(ctx).await;
        pie::context::destroy(ctx).await.unwrap();

        // Slot pressure is non-preemptive: a second pin waits until the
        // resident holder suspends and releases the slot, then receives the
        // freed slot with RESET.
        let holder = pie::context::create(fresh_pid().await).await.unwrap();
        let waiter = pie::context::create(fresh_pid().await).await.unwrap();
        let held = pie::context::pin(holder, 0).await.unwrap();
        assert_eq!(held.rs_slot, Some(0));
        let pending = tokio::spawn(pie::context::pin(waiter, 0));
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!pending.is_finished(), "rs slot waiter should defer");
        pie::context::unpin(holder);
        flush_mailbox(holder).await;
        pie::context::suspend(holder).await.unwrap();
        let waited = tokio::time::timeout(Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(waited.rs_slot, Some(0));
        assert_eq!(waited.rs_flags, 1);
        pie::context::unpin(waiter);
        flush_mailbox(waiter).await;
    });
}
