//! Host-swap exhaustion policy (Project Rainer): with no swap room and no
//! fire in flight anywhere, the starvation predicate fails the YOUNGEST
//! parked ask loud — computed, never timed — and the fleet progresses.

use std::sync::Arc;
use std::time::Duration;

mod common;
use common::{
    create_mock_env, inferlets,
    mock_device::{DelayedBehavior, EchoBehavior},
};

use engine::inferlet::process;
use engine::inferlet::program::ProgramName;

#[test]
fn host_swap_exhaustion_kills_a_victim_without_wedging_the_fleet() {
    unsafe {
        std::env::set_var("PIE_KV_CACHE_ROOTS_MAX", "0");
    }
    inferlets::build_inferlets();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let env = create_mock_env(
        "contention-host-full",
        1,
        4,
        Arc::new(DelayedBehavior {
            inner: EchoBehavior(42),
            latency: Duration::from_millis(10),
        }),
    );
    let mut config = env.config();
    config.model.drivers[0].cpu_pages = 0;
    runtime.block_on(async {
        engine::bootstrap::bootstrap(config).await.unwrap();
        inferlets::add_and_install("generate").await;
    });

    let name = ProgramName::parse("generate@0.1.0").unwrap();
    const LANES: usize = 3;
    const TOKENS: usize = 40;
    let results = runtime.block_on(async {
        let receivers: Vec<_> = (0..LANES)
            .map(|_| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                process::spawn(
                    "contention-host-full-user".into(),
                    name.clone(),
                    TOKENS.to_string(),
                    None,
                    false,
                    Some(tx),
                )
                .unwrap();
                rx
            })
            .collect();
        let mut results = Vec::new();
        for receiver in receivers {
            results.push(
                tokio::time::timeout(Duration::from_secs(10), receiver)
                    .await
                    .expect("host-full fleet must not hang")
                    .expect("process result sender must remain live"),
            );
        }
        results
    });

    assert!(
        results.iter().any(Result::is_ok),
        "at least one process completes"
    );
    assert!(
        results.iter().any(|result| {
            result
                .as_ref()
                .is_err_and(|error| error.contains("no host swap room to evict into"))
        }),
        "host-full victim reports the kill reason: {results:?}"
    );
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !process::list().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("host-full fleet must tear down");
    });
    let diagnostics = engine::planner::planner()
        .expect("residency planner")
        .diagnostics();
    assert!(diagnostics.starvations_total > 0);
    assert_eq!(
        diagnostics.device_pages_free,
        diagnostics.device_pages_total
    );
}
