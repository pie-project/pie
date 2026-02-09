//! Context management integration tests.
//!
//! Tests context CRUD, forking, locking, cursor, and buffered tokens.

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
        let env = create_mock_env("ctx-test", 1, 64, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

const MODEL: usize = 0;
const USER: u32 = 1;

#[test]
fn create_and_lookup() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER, "test-ctx".into(), None)
            .await
            .unwrap();

        let found = pie::context::lookup(MODEL, USER, "test-ctx".into()).await;
        assert_eq!(found, Some(id));
    });
}

#[test]
fn create_with_fill() {
    let s = state();
    s.rt.block_on(async {
        let fill_tokens = vec![10, 20, 30];
        let id = pie::context::create(MODEL, USER, "fill-ctx".into(), Some(fill_tokens.clone()))
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id).await;
        let tokens = pie::context::get_buffered_tokens(MODEL, id, lock).await;
        // Fill tokens become tokens_uncommitted, returned as buffered
        assert!(!tokens.is_empty(), "fill tokens should appear as buffered tokens");
        pie::context::release_lock(MODEL, id, lock).unwrap();
    });
}

#[test]
fn destroy_removes_context() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER, "destroy-ctx".into(), None)
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id).await;
        pie::context::destroy(MODEL, id, lock).await.unwrap();

        let found = pie::context::lookup(MODEL, USER, "destroy-ctx".into()).await;
        assert_eq!(found, None);
    });
}

#[test]
fn cursor_ops() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER, "cursor-ctx".into(), None)
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id).await;

        pie::context::set_cursor(MODEL, id, lock, 5).unwrap();
        let cursor = pie::context::get_cursor(MODEL, id, lock).await;
        assert_eq!(cursor, 5);

        pie::context::release_lock(MODEL, id, lock).unwrap();
    });
}

#[test]
fn buffered_token_lifecycle() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER, "buf-tok-ctx".into(), None)
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id).await;

        // Set initial tokens
        pie::context::set_buffered_tokens(MODEL, id, lock, vec![1, 2, 3]).unwrap();

        // Append more
        pie::context::append_buffered_tokens(MODEL, id, lock, vec![4, 5]).unwrap();

        // Read back
        let tokens = pie::context::get_buffered_tokens(MODEL, id, lock).await;
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);

        pie::context::release_lock(MODEL, id, lock).unwrap();
    });
}

#[test]
fn fork_context() {
    let s = state();
    s.rt.block_on(async {
        let parent_id = pie::context::create(MODEL, USER, "parent-ctx".into(), None)
            .await
            .unwrap();

        let child_id = pie::context::fork(MODEL, parent_id, USER, "child-ctx".into())
            .await
            .unwrap();

        assert_ne!(parent_id, child_id);

        // Both should be findable
        let found_parent = pie::context::lookup(MODEL, USER, "parent-ctx".into()).await;
        let found_child = pie::context::lookup(MODEL, USER, "child-ctx".into()).await;
        assert_eq!(found_parent, Some(parent_id));
        assert_eq!(found_child, Some(child_id));
    });
}
