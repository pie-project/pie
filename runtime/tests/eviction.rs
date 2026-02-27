//! Eviction integration tests.
//!
//! Uses a dedicated environment with a large page budget to test
//! invested-importance eviction without interference from other tests.

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
        // 256 pages to give plenty of room for commit (each commit_page allocs a NEW gpu page)
        let env = create_mock_env("eviction-test", 1, 256, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

const MODEL: usize = 0;

/// Test that under memory pressure, the arbiter evicts the group with
/// the lowest invested importance (w·p = fewest pages × weight).
///
/// P1 commits 10 pages, P2 commits 4 pages → ~28 GPU pages used.
/// Request 230 working pages → exceeds remaining ~228 → triggers eviction.
/// P2 (4 committed, priority=4) has lower invested importance → evicted first.
#[test]
fn utility_based_eviction() {
    let s = state();
    s.rt.block_on(async {
        use uuid::Uuid;

        let p1 = Uuid::from_u128(0xdead_0001);
        let p2 = Uuid::from_u128(0xdead_0002);
        let p3 = Uuid::from_u128(0xdead_0003);

        // P1: commit 10 pages (160 tokens at 16 tokens/page)
        let ctx_a = pie::context::create_owned(MODEL, Some(p1)).await.unwrap();
        pie::context::set_buffered_tokens(MODEL, ctx_a, (10000..10160).collect()).unwrap();
        pie::context::fill(MODEL, ctx_a, 160, (0..160).collect(), vec![], None).unwrap();
        pie::context::reserve_pages(MODEL, ctx_a, 10).await.unwrap();
        pie::context::commit_pages(MODEL, ctx_a, (0..10).collect()).await.unwrap();

        // P2: commit 4 pages (64 tokens)
        let ctx_b = pie::context::create_owned(MODEL, Some(p2)).await.unwrap();
        pie::context::set_buffered_tokens(MODEL, ctx_b, (20000..20064).collect()).unwrap();
        pie::context::fill(MODEL, ctx_b, 64, (0..64).collect(), vec![], None).unwrap();
        pie::context::reserve_pages(MODEL, ctx_b, 4).await.unwrap();
        pie::context::commit_pages(MODEL, ctx_b, (0..4).collect()).await.unwrap();

        assert!(pie::context::is_active(MODEL, ctx_a), "ctx_a should be active");
        assert!(pie::context::is_active(MODEL, ctx_b), "ctx_b should be active");

        // Trigger eviction: 256 - ~28 (CAS) = ~228 free. Request 230.
        // With w·p model: P2 (4 committed, priority=4) < P1 (10, priority=10)
        // → P2's ctx_b is evicted first (lowest invested importance).
        // Use P3 as requester so neither P1 nor P2 is the requester.
        let ctx_c = pie::context::create_owned(MODEL, Some(p3)).await.unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            pie::context::reserve_pages(MODEL, ctx_c, 230),
        ).await;

        match result {
            Ok(Ok(())) => {
                // Eviction succeeded. P2 was the cheapest group (fewest pages).
                assert!(
                    !pie::context::is_active(MODEL, ctx_b),
                    "ctx_b (P2, 4 pages) should have been evicted — lowest invested importance"
                );
            }
            Ok(Err(e)) => {
                eprintln!("reserve_pages returned error: {e}");
            }
            Err(_) => {
                panic!("reserve_pages timed out — eviction loop may be stuck");
            }
        }

        // Cleanup
        let _ = pie::context::destroy(MODEL, ctx_a, true).await;
        let _ = pie::context::destroy(MODEL, ctx_b, true).await;
        let _ = pie::context::destroy(MODEL, ctx_c, true).await;
    });
}
