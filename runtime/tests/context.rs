//! Context management integration tests.
//!
//! Tests context CRUD, saving, opening, forking, and the full page lifecycle.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
mod common;
use common::{MockEnv, create_mock_env, mock_device::EchoBehavior};

/// Monotonic launch-sequence source for FCFS process registration.
static LAUNCH_SEQ: AtomicU64 = AtomicU64::new(0);
fn next_launch_seq() -> u64 {
    LAUNCH_SEQ.fetch_add(1, Ordering::Relaxed)
}

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

const USER: &str = "test-user";

/// Register a fresh process and return its pid. The context actor
/// panics on `create` if the process isn't registered first.
async fn fresh_pid() -> uuid::Uuid {
    let pid = uuid::Uuid::new_v4();
    pie::context::register_process(pid, next_launch_seq()).await.unwrap();
    pid
}

/// Wait for any pending fire-and-forget messages on the context
/// actor (e.g., `append_working_page_tokens`) to drain. The mailbox
/// is FIFO, so any actor-routed roundtrip after a fire-and-forget
/// send is guaranteed to observe the latter's effects.
async fn flush_mailbox(id: pie::context::ContextId) {
    let _ = pie::context::debug_context_state(id).await;
}

#[test]
fn create_and_save_and_open() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(fresh_pid().await)
            .await
            .unwrap();

        // Anonymous context is not findable by name
        let found = pie::context::lookup(USER.to_string(), "test-ctx".into()).await;
        assert!(found.is_err());

        // Save it with a name
        pie::context::save(id, USER.to_string(), Some("test-ctx".into()))
            .await
            .unwrap();

        // Now it should be findable (lookup + fork returns a different id)
        let snapshot_id = pie::context::lookup(USER.to_string(), "test-ctx".into())
            .await
            .unwrap();
        let found = pie::context::fork(snapshot_id, fresh_pid().await).await;
        assert!(found.is_ok());
    });
}

#[test]
fn destroy_removes_context() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(fresh_pid().await)
            .await
            .unwrap();

        pie::context::destroy(id).await.unwrap();

        // Fork from destroyed context should fail
        let fork_result = pie::context::fork(id, fresh_pid().await).await;
        assert!(
            fork_result.is_err(),
            "fork from destroyed context should fail"
        );
    });
}

#[test]
fn force_destroy() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(fresh_pid().await)
            .await
            .unwrap();

        // Destroy should succeed on a fresh context
        pie::context::destroy(id).await.unwrap();
    });
}

#[test]
fn working_page_token_ops() {
    let s = state();
    s.rt.block_on(async {
        let id = pie::context::create(fresh_pid().await)
            .await
            .unwrap();

        // Append tokens
        pie::context::append_working_page_tokens(
            id,
            vec![1, 2, 3, 4, 5],
            vec![0, 1, 2, 3, 4],
            vec![],
            None,
            None,
        );
        flush_mailbox(id).await;

        // working_page_token_count = 5
        let count = pie::context::working_page_token_count(id);
        assert_eq!(count, 5);

        // Drop the last 2 tokens, leaving 3.
        pie::context::truncate_working_page_tokens(id, 2)
            .await
            .unwrap();
        // The truncate dispatch publishes the working-token count to the cache
        // *after* sending its response, so flush the mailbox (FIFO) before
        // reading the sync getter — same pattern as the append above.
        flush_mailbox(id).await;
        assert_eq!(pie::context::working_page_token_count(id), 3);

        // Truncate out of range should fail
        let err = pie::context::truncate_working_page_tokens(id, 10).await;
        assert!(err.is_err(), "truncate beyond token count should error");
    });
}

#[test]
fn fork_context() {
    let s = state();
    s.rt.block_on(async {
        let parent_id = pie::context::create(fresh_pid().await)
            .await
            .unwrap();

        let child_id = pie::context::fork(parent_id, fresh_pid().await)
            .await
            .unwrap();

        assert_ne!(parent_id, child_id);
    });
}

/// Comprehensive test simulating a realistic multi-turn inference lifecycle.
///
/// Timeline:
///   1. Create anonymous context, fill 32 tokens
///   2. Commit first page → verify page counts
///   3. Commit second page → verify fully committed state
///   4. Append generation tokens via fill
///   5. Token truncation
///   6. Fork → verify child inherits state
#[test]
fn full_page_lifecycle() {
    let s = state();
    s.rt.block_on(async {
        const PAGE_SIZE: u32 = 16;

        // ── Phase 1: Create anonymous context and fill prompt tokens ──
        let prompt: Vec<u32> = (1000..1032).collect(); // 32 tokens
        let id = pie::context::create(fresh_pid().await)
            .await
            .unwrap();

        // Tokens per page should match the model config
        assert_eq!(
            pie::context::tokens_per_page(),
            PAGE_SIZE,
            "tokens_per_page should be 16"
        );

        assert_eq!(pie::context::committed_page_count(id), 0);
        assert_eq!(
            pie::context::working_page_token_count(id),
            0,
            "tokens start at 0"
        );

        // ── Phase 2: Mark all 32 tokens as forwarded ──
        let positions: Vec<u32> = (0..32).collect();
        pie::context::append_working_page_tokens(
            id,
            prompt.clone(),
            positions,
            vec![],
            None,
            None,
        );
        flush_mailbox(id).await;

        assert_eq!(pie::context::working_page_token_count(id), 32);

        // Reserve 2 pages (32 tokens / 16 per page) before committing
        pie::context::reserve_working_pages(id, 2)
            .await
            .unwrap();

        // Working page count should be 2 (actual allocated pages)
        assert_eq!(pie::context::working_page_count(id), 2);

        // ── Phase 3: Commit first page (positions 0..15) ──
        pie::context::commit_working_pages(id, 1)
            .await
            .unwrap();

        assert_eq!(pie::context::committed_page_count(id), 1);
        // 16 filled tokens remain (second page's worth)
        assert_eq!(
            pie::context::working_page_token_count(id),
            16,
            "16 tokens remain after first commit"
        );

        // ── Phase 4: Commit second page (positions 16..31) ──
        pie::context::commit_working_pages(id, 1)
            .await
            .unwrap();

        assert_eq!(pie::context::committed_page_count(id), 2);
        assert_eq!(
            pie::context::working_page_token_count(id),
            0,
            "0 tokens after full commit"
        );

        // ── Phase 5: Simulate generation — fill new tokens ──
        pie::context::append_working_page_tokens(
            id,
            vec![2000, 2001, 2002],
            vec![32, 33, 34],
            vec![],
            None,
            None,
        );
        flush_mailbox(id).await;
        assert_eq!(pie::context::working_page_token_count(id), 3);

        // ── Phase 6: Prepare state with filled tokens, then fork ──
        // Clear working page tokens from Phase 5
        pie::context::truncate_working_page_tokens(id, 3)
            .await
            .unwrap();
        // Fill 2 tokens with positions sequential from max_committed (31)
        pie::context::append_working_page_tokens(
            id,
            vec![3000, 3001],
            vec![32, 33],
            vec![],
            None,
            None,
        );
        flush_mailbox(id).await;
        assert_eq!(pie::context::working_page_token_count(id), 2);

        let child_id = pie::context::fork(id, fresh_pid().await)
            .await
            .unwrap();

        assert_ne!(id, child_id);

        // Verify child state
        assert_eq!(
            pie::context::committed_page_count(child_id),
            2,
            "child inherits committed pages"
        );

        // Fork preserves working_page_tokens — child inherits them
        assert_eq!(
            pie::context::working_page_token_count(child_id),
            2,
            "child inherits filled tokens"
        );
    });
}
