//! Adapter lifecycle integration tests.
//!
//! Tests adapter CRUD, cloning, and lock/unlock cycles.

use std::sync::{Arc, OnceLock};
mod common;
use common::{create_mock_env, MockEnv, mock_device::EchoBehavior};

struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let env = create_mock_env("adapter-test", 1, 64, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

const MODEL: usize = 0;

#[test]
fn create_and_get() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::adapter::create(MODEL, "lora-1".into()).await.unwrap();
        let found = pie::adapter::get(MODEL, "lora-1".into()).await;
        assert_eq!(found, Some(id));
    });
}

#[test]
fn destroy_adapter() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::adapter::create(MODEL, "lora-destroy".into()).await.unwrap();
        pie::adapter::destroy(MODEL, id).await.unwrap();
        let found = pie::adapter::get(MODEL, "lora-destroy".into()).await;
        assert_eq!(found, None);
    });
}

#[test]
fn clone_adapter() {
    let s = state();
    s.rt.block_on(async {
        let original = pie::adapter::create(MODEL, "lora-original".into()).await.unwrap();
        let cloned = pie::adapter::clone_adapter(MODEL, original, "lora-clone".into()).await;
        assert!(cloned.is_some());
        let cloned_id = cloned.unwrap();
        assert_ne!(original, cloned_id);

        // Both should exist
        let found_orig = pie::adapter::get(MODEL, "lora-original".into()).await;
        let found_clone = pie::adapter::get(MODEL, "lora-clone".into()).await;
        assert_eq!(found_orig, Some(original));
        assert_eq!(found_clone, Some(cloned_id));
    });
}

#[test]
fn lock_unlock_cycle() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::adapter::create(MODEL, "lora-lock".into()).await.unwrap();
        let lock_id = pie::adapter::lock(MODEL, id).await;
        assert!(lock_id > 0, "Lock ID should be > 0");
        // Unlock is sync
        pie::adapter::unlock(MODEL, id, lock_id);
    });
}
