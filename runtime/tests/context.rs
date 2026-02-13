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
const USER: &str = "test-user";

#[test]
fn create_and_lookup() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER.to_string(), "test-ctx".into(), None)
            .await
            .unwrap();

        let found = pie::context::lookup(MODEL, USER.to_string(), "test-ctx".into()).await;
        assert_eq!(found, Some(id));
    });
}

#[test]
fn create_with_fill() {
    let s = state();
    s.rt.block_on(async {
        let fill_tokens = vec![10, 20, 30];
        let id = pie::context::create(MODEL, USER.to_string(), "fill-ctx".into(), Some(fill_tokens.clone()))
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id);
        let tokens = pie::context::get_buffered_tokens(MODEL, id);
        // Fill tokens go to tokens_buffered (awaiting forward pass)
        assert!(!tokens.is_empty(), "fill tokens should appear as buffered tokens");
        pie::context::release_lock(MODEL, id, lock).unwrap();
    });
}

#[test]
fn destroy_removes_context() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER.to_string(), "destroy-ctx".into(), None)
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id);
        pie::context::destroy(MODEL, id, lock).await.unwrap();

        let found = pie::context::lookup(MODEL, USER.to_string(), "destroy-ctx".into()).await;
        assert_eq!(found, None);
    });
}

#[test]
fn cursor_ops() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER.to_string(), "cursor-ctx".into(), None)
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id);

        // Buffer some tokens and mark them forwarded so cursor advances
        pie::context::set_buffered_tokens(MODEL, id, lock, vec![1, 2, 3, 4, 5]).unwrap();
        pie::context::fill(
            MODEL, id, 5,
            vec![0, 1, 2, 3, 4],
            vec![],
            None,
        ).unwrap();

        // Cursor = tokens_filled.len() = 5
        let cursor = pie::context::get_cursor(MODEL, id);
        assert_eq!(cursor, 5);

        // set_cursor truncates filled tokens
        pie::context::set_cursor(MODEL, id, lock, 3).unwrap();
        assert_eq!(pie::context::get_cursor(MODEL, id), 3);

        // set_cursor out of range should fail
        let err = pie::context::set_cursor(MODEL, id, lock, 10);
        assert!(err.is_err(), "set_cursor beyond filled tokens should error");

        pie::context::release_lock(MODEL, id, lock).unwrap();
    });
}

#[test]
fn buffered_token_lifecycle() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(MODEL, USER.to_string(), "buf-tok-ctx".into(), None)
            .await
            .unwrap();

        let lock = pie::context::acquire_lock(MODEL, id);

        // Set initial tokens
        pie::context::set_buffered_tokens(MODEL, id, lock, vec![1, 2, 3]).unwrap();

        // Append more
        pie::context::append_buffered_tokens(MODEL, id, lock, vec![4, 5]).unwrap();

        // Read back
        let tokens = pie::context::get_buffered_tokens(MODEL, id);
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);

        pie::context::release_lock(MODEL, id, lock).unwrap();
    });
}

#[test]
fn fork_context() {
    let s = state();
    s.rt.block_on(async {
        let parent_id = pie::context::create(MODEL, USER.to_string(), "parent-ctx".into(), None)
            .await
            .unwrap();

        let child_id = pie::context::fork(MODEL, parent_id, USER.to_string(), "child-ctx".into())
            .await
            .unwrap();

        assert_ne!(parent_id, child_id);

        // Both should be findable
        let found_parent = pie::context::lookup(MODEL, USER.to_string(), "parent-ctx".into()).await;
        let found_child = pie::context::lookup(MODEL, USER.to_string(), "child-ctx".into()).await;
        assert_eq!(found_parent, Some(parent_id));
        assert_eq!(found_child, Some(child_id));
    });
}

/// Comprehensive test simulating a realistic multi-turn inference lifecycle.
///
/// Timeline:
///   1. Prompt fill with 32 tokens → buffered
///   2. Mark forwarded → promotes buffered to filled
///   3. Commit first page → verify cursor, position, buffer drainage
///   4. Commit second page → verify fully committed state
///   5. Append generation tokens, mark forwarded
///   6. Cursor truncation
///   7. Fork → verify child inherits state
#[test]
fn full_page_lifecycle() {
    let s = state();
    s.rt.block_on(async {
        const PAGE_SIZE: u32 = 16;

        // ── Phase 1: Create with prompt fill (32 tokens = 2 full pages) ──
        let prompt: Vec<u32> = (1000..1032).collect(); // 32 tokens
        let id = pie::context::create(
            MODEL, USER.to_string(), "lifecycle-ctx".into(), Some(prompt.clone()),
        ).await.unwrap();

        let lock = pie::context::acquire_lock(MODEL, id);
        assert_ne!(lock, 0, "lock acquisition should succeed");

        // Tokens per page should match the model config
        assert_eq!(
            pie::context::tokens_per_page(MODEL, id), PAGE_SIZE,
            "tokens_per_page should be 16"
        );

        // All fill tokens are buffered; nothing committed or filled yet
        let buf = pie::context::get_buffered_tokens(MODEL, id);
        assert_eq!(buf.len(), 32, "all 32 fill tokens should be buffered");
        assert_eq!(buf, prompt);

        assert_eq!(pie::context::committed_page_count(MODEL, id), 0);
        assert_eq!(pie::context::get_cursor(MODEL, id), 0, "cursor starts at 0 (no filled tokens)");

        // last_position = None (no filled or committed tokens)
        assert_eq!(pie::context::last_position(MODEL, id), None);

        // ── Phase 2: Mark all 32 tokens as forwarded ──
        let positions: Vec<u32> = (0..32).collect();
        pie::context::fill(MODEL, id, 32, positions, vec![], None).unwrap();

        // Cursor = tokens_filled.len() = 32
        assert_eq!(pie::context::get_cursor(MODEL, id), 32);
        // Buffered should be empty now
        assert!(pie::context::get_buffered_tokens(MODEL, id).is_empty());
        // last_position = max filled position = 31
        assert_eq!(pie::context::last_position(MODEL, id), Some(31));

        // ── Phase 3: Commit first page (positions 0..15) ──
        pie::context::commit_pages(MODEL, id, lock, vec![0]).await.unwrap();

        assert_eq!(pie::context::committed_page_count(MODEL, id), 1);
        // 16 filled tokens remain (second page's worth)
        assert_eq!(pie::context::get_cursor(MODEL, id), 16, "cursor = remaining filled count");
        // last_position = max(committed=15, filled_max=31) = 31
        assert_eq!(pie::context::last_position(MODEL, id), Some(31));

        // ── Phase 4: Commit second page (positions 16..31) ──
        pie::context::commit_pages(MODEL, id, lock, vec![0]).await.unwrap();

        assert_eq!(pie::context::committed_page_count(MODEL, id), 2);
        assert_eq!(pie::context::get_cursor(MODEL, id), 0, "cursor is 0 after full commit");
        // position = max_committed = 31
        assert_eq!(pie::context::last_position(MODEL, id), Some(31));

        // ── Phase 5: Simulate generation — append new tokens, mark forwarded ──
        pie::context::append_buffered_tokens(MODEL, id, lock, vec![2000, 2001, 2002]).unwrap();

        let buf = pie::context::get_buffered_tokens(MODEL, id);
        assert_eq!(buf, vec![2000, 2001, 2002], "generation tokens buffered");

        // No filled tokens, so last_position = max_committed = 31
        assert_eq!(pie::context::last_position(MODEL, id), Some(31));

        // Mark generation tokens forwarded with positions 32, 33, 34
        pie::context::fill(MODEL, id, 3, vec![32, 33, 34], vec![], None).unwrap();
        assert_eq!(pie::context::get_cursor(MODEL, id), 3);
        assert_eq!(pie::context::last_position(MODEL, id), Some(34), "filled position dominates");

        // ── Phase 6: Prepare state with filled + buffered tokens, then fork ──
        // Clear filled tokens from Phase 5
        pie::context::set_cursor(MODEL, id, lock, 0).unwrap();
        // Buffer tokens and mark first 2 forwarded with positions sequential from max_committed (31)
        pie::context::set_buffered_tokens(MODEL, id, lock, vec![3000, 3001, 3002, 3003]).unwrap();
        pie::context::fill(MODEL, id, 2, vec![32, 33], vec![], None).unwrap();
        assert_eq!(pie::context::get_cursor(MODEL, id), 2);
        // Remaining buffered: [3002, 3003]
        assert_eq!(pie::context::get_buffered_tokens(MODEL, id), vec![3002, 3003]);

        pie::context::release_lock(MODEL, id, lock).unwrap();

        let child_id = pie::context::fork(
            MODEL, id, USER.to_string(), "lifecycle-child".into(),
        ).await.unwrap();

        assert_ne!(id, child_id);

        // Verify child state
        let child_lock = pie::context::acquire_lock(MODEL, child_id);
        assert_ne!(child_lock, 0);

        assert_eq!(
            pie::context::committed_page_count(MODEL, child_id), 2,
            "child inherits committed pages"
        );

        // Fork demotes filled → buffered: child gets [3000, 3001, 3002, 3003]
        let child_buf = pie::context::get_buffered_tokens(MODEL, child_id);
        assert_eq!(
            child_buf, vec![3000, 3001, 3002, 3003],
            "child: filled tokens demoted to buffered + existing buffered"
        );

        // Child cursor = 0 (no filled tokens after fork)
        assert_eq!(
            pie::context::get_cursor(MODEL, child_id), 0,
            "child cursor = 0 (filled demoted to buffered)"
        );

        // Child inherits max_committed_position
        let child_pos = pie::context::last_position(MODEL, child_id);
        assert_eq!(child_pos, Some(31), "child inherits max_committed_position = 31");

        // Mutating child buffer doesn't affect parent
        pie::context::append_buffered_tokens(MODEL, child_id, child_lock, vec![9999]).unwrap();
        let parent_buf = pie::context::get_buffered_tokens(MODEL, id);
        assert_eq!(parent_buf, vec![3002, 3003], "parent buffer unchanged after child mutation");

        let child_buf = pie::context::get_buffered_tokens(MODEL, child_id);
        assert_eq!(child_buf, vec![3000, 3001, 3002, 3003, 9999], "child buffer has the appended token");

        pie::context::release_lock(MODEL, child_id, child_lock).unwrap();
    });
}
