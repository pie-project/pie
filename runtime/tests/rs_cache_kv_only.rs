//! rs_cache integration test — KV-only model must not populate rs fields.
//!
//! Spawns one context manager with ZERO rs_cache slots; pinning a context
//! must leave the rs_cache fields empty.

use std::sync::{Arc, OnceLock};

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
        rt.block_on(spawn_managers(0, 0.95));
        TestState { backend, rt }
    })
}

#[test]
fn kv_only_has_no_rs_fields() {
    let s = state();
    s.rt.block_on(async {
        let kv_ctx = pie::context::create(fresh_pid().await).await.unwrap();
        let kv_pin = pie::context::pin(kv_ctx, 0).await.unwrap();
        assert_eq!(kv_pin.rs_slot, None);
        assert_eq!(kv_pin.rs_flags, 0);
        pie::context::unpin(kv_ctx);
        flush_mailbox(kv_ctx).await;
    });
}
