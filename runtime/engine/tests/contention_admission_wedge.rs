//! Admission-gated wedge (Project Rainer): a fleet larger than the execution
//! admission cap must not deadlock the residency planner.
//!
//! The planner registers a process at spawn, because registration order is
//! the FCFS clock — but a process only reaches pooled KV after execution
//! admission. When the whole admitted cohort parks on an empty pool with no
//! swap room, the processes still queued for an execution permit hold
//! nothing and can never start; if the wedge predicate counts them as
//! "someone is still running", the starvation rung never fires and the
//! engine hangs forever with a head that wants a single page.

use std::sync::Arc;
use std::time::Duration;

mod common;
use common::{
    create_mock_env, inferlets,
    mock_device::{DelayedBehavior, EchoBehavior},
};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

#[test]
fn fleet_larger_than_the_admission_cap_does_not_wedge() {
    unsafe {
        std::env::set_var("PIE_KV_CACHE_ROOTS_MAX", "0");
    }
    inferlets::build_inferlets();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let env = create_mock_env(
        "contention-admission-wedge",
        1,
        4,
        Arc::new(DelayedBehavior {
            inner: EchoBehavior(42),
            latency: Duration::from_millis(10),
        }),
    );
    let mut config = env.config();
    // No swap room: eviction cannot fund the head, so the starvation
    // predicate is the only rung left.
    config.model.drivers[0].cpu_pages = 0;
    // Fewer execution permits than lanes: the remainder stay registered
    // (they own their FCFS seq) but page-less and unable to run.
    config.max_concurrent_processes = Some(2);
    runtime.block_on(async {
        pie_engine::bootstrap::bootstrap(config).await.unwrap();
        inferlets::add_and_install("generate").await;
    });

    let name = ProgramName::parse("generate@0.1.0").unwrap();
    const LANES: usize = 5;
    const TOKENS: usize = 40;
    let results = runtime.block_on(async {
        let receivers: Vec<_> = (0..LANES)
            .map(|_| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                process::spawn(
                    "contention-admission-wedge-user".into(),
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
        for (lane, receiver) in receivers.into_iter().enumerate() {
            let received = match tokio::time::timeout(Duration::from_secs(30), receiver).await {
                Ok(received) => received,
                Err(_) => {
                    let diagnostics = pie_engine::planner::planner().unwrap().diagnostics();
                    panic!(
                        "admission-capped fleet must not wedge: lane {lane} timed out; \
                         diagnostics: {diagnostics:#?}"
                    );
                }
            };
            results.push(received.expect("process result sender must remain live"));
        }
        results
    });

    assert!(
        results.iter().any(Result::is_ok),
        "the fleet makes progress: {results:?}"
    );
    let diagnostics = pie_engine::planner::planner()
        .expect("residency planner")
        .diagnostics();
    assert!(
        diagnostics.starvations_total > 0,
        "the wedge is broken by the computed starvation rung, not by a timer: {diagnostics:#?}"
    );
}
