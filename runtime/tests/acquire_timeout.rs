//! #470/#485: bounded KV-page acquisition.
//!
//! When concurrent requests exceed the engine's KV slot count (or, as
//! exercised deterministically here, a single reservation asks for more
//! pages than the device can ever provide), the reservation defers on the
//! scheduler's restore queue. Before the fix that deferral had no timer of
//! its own: `context::reserve_working_pages` blocked on its response channel
//! forever, and because the single-threaded guest cannot make any other WIT
//! call while blocked, the request emitted no response until the *client's*
//! deadline. This test pins the fix — the reservation now fails fast with a
//! structured `server_busy` error instead of hanging.
//!
//! Own test binary so the runtime's `PIE_ACQUIRE_TIMEOUT_S` LazyLock is read
//! fresh (set before bootstrap, below) without racing other test files.

use std::sync::Arc;
use std::sync::OnceLock;

mod common;
use common::{create_mock_env, mock_device::EchoBehavior, MockEnv};

struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        // Bound the wait to a fraction of a second so an impossible
        // reservation fails fast within the test. Must be set before the
        // first `reserve_working_pages` so the LazyLock picks it up.
        // SAFETY: single-threaded at init time; no other thread reads the
        // environment concurrently here.
        unsafe { std::env::set_var("PIE_ACQUIRE_TIMEOUT_S", "0.5") };

        let rt = tokio::runtime::Runtime::new().unwrap();
        // 1 device × 8 pages × 16 tokens = 128 tokens of KV capacity.
        let env = create_mock_env("acquire-timeout", 1, 8, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

const MODEL: usize = 0;

/// A reservation that can never be satisfied (64 pages on an 8-page device,
/// with no other context to evict) must fail with the `server_busy`
/// sentinel within the bounded wait — not block forever.
#[test]
fn reserve_beyond_capacity_fails_fast_with_server_busy() {
    let s = state();
    s.rt.block_on(async {
        let pid = uuid::Uuid::new_v4();
        pie::context::register_process(pid, None).await.unwrap();
        let id = pie::context::create(MODEL, pid).await.unwrap();

        let start = std::time::Instant::now();
        let res = pie::context::reserve_working_pages(MODEL, id, 64).await;
        let elapsed = start.elapsed();

        let err = res.expect_err("reserve beyond capacity must fail, not hang");
        let msg = err.to_string();
        assert!(
            msg.contains("server_busy"),
            "expected the `server_busy` sentinel, got: {msg}"
        );
        // ~0.5s bound; allow generous slack for a loaded CI box. The point
        // is that it returns at all (the pre-fix behaviour was an unbounded
        // hang).
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "reserve should fail fast (~0.5s), took {elapsed:?}"
        );
    });
}
