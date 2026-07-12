//! Deterministic host-only active KV preemption integration test.

use std::sync::Arc;
use std::time::Duration;

mod common;
use common::{create_mock_env, inferlets, mock_device::EchoBehavior};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

#[test]
fn active_preemption_swaps_and_restores_an_over_capacity_fleet() {
    unsafe {
        std::env::set_var("PIE_KV_CONTENTION", "preempt");
        std::env::set_var("PIE_KV_PREEMPT_ACTIVE", "1");
        std::env::set_var("PIE_KV_EXHAUSTION_MS", "5000");
        std::env::set_var("PIE_KV_CACHE_ROOTS_MAX", "0");
    }
    inferlets::build_inferlets();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let env = create_mock_env("contention-model", 1, 4, Arc::new(EchoBehavior(42)));
    runtime.block_on(async {
        pie_engine::bootstrap::bootstrap(env.config())
            .await
            .unwrap();
        inferlets::add_and_install("generate").await;
    });

    let name = ProgramName::parse("generate@0.1.0").unwrap();
    let results = runtime.block_on(async {
        let receivers: Vec<_> = (0..8)
            .map(|lane| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                process::spawn(
                    "contention-user".into(),
                    name.clone(),
                    format!(r#"{{"lane":{lane}}}"#),
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
                tokio::time::timeout(Duration::from_secs(20), receiver)
                    .await
                    .expect("contention fleet must not hang")
                    .expect("process result sender must remain live"),
            );
        }
        results
    });

    for result in results {
        assert_eq!(result.unwrap(), "generated 5 tokens: [42, 42, 42, 42, 42]");
    }
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let diagnostics = pie_engine::store::reclaim::contention()
                    .unwrap()
                    .diagnostics();
                if process::list().is_empty()
                    && diagnostics.host_slots_free == diagnostics.host_slots_total
                    && diagnostics.device_pages_free == diagnostics.device_pages_total
                    && diagnostics.waiters.is_empty()
                    && diagnostics.suspended.is_empty()
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("contention processes must reach teardown");
    });
    let orchestrator = pie_engine::store::reclaim::contention().unwrap();
    let diagnostics = orchestrator.diagnostics();
    assert!(diagnostics.suspends_total > 0);
    assert!(diagnostics.restores_total > 0);
    assert!(diagnostics.d2h_pages_total > 0);
    assert_eq!(diagnostics.d2h_pages_total, diagnostics.h2d_pages_total);
    assert_eq!(diagnostics.host_slots_free, diagnostics.host_slots_total);
    assert_eq!(
        diagnostics.device_pages_free,
        diagnostics.device_pages_total
    );
    assert!(diagnostics.waiters.is_empty());
    assert!(diagnostics.suspended.is_empty());
}
