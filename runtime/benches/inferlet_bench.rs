//! Benchmarks for inferlet spawn-to-completion throughput.
//!
//! Usage:
//!   cargo bench --bench inferlet_bench

#[path = "../tests/common/mod.rs"]
mod common;

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use common::{create_mock_env, MockEnv, mock_device::EchoBehavior, inferlets};
use pie::process;
use pie::program::ProgramName;

/// Shared state: MockEnv + tokio runtime.
struct BenchState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<BenchState> = OnceLock::new();

fn state() -> &'static BenchState {
    STATE.get_or_init(|| {
        inferlets::build_inferlets();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let env = create_mock_env("bench-model", 1, 16, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
            inferlets::add_and_install("echo").await;
        });
        BenchState { env, rt }
    })
}

fn program_name() -> ProgramName {
    ProgramName::parse("echo@0.1.0")
}

/// Spawn N concurrent echo inferlets and wait for all to complete.
fn spawn_and_wait_all(s: &BenchState, count: usize) {
    s.rt.block_on(async {
        let pids: Vec<_> = (0..count)
            .map(|i| {
                process::spawn(
                    "bench-user".into(),
                    program_name(),
                    vec![format!("{i}")],
                    None,
                    None,
                    false,
                )
                .expect("spawn")
            })
            .collect();

        let timeout = Duration::from_secs(60);
        for pid in pids {
            tokio::time::timeout(timeout, async {
                loop {
                    if !process::list().contains(&pid) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .expect("process did not complete in time");
        }
    });
}

fn bench_concurrent_inferlets(c: &mut Criterion) {
    let s = state();

    let mut group = c.benchmark_group("inferlet_spawn");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("1024_concurrent_echo", |b| {
        b.iter(|| spawn_and_wait_all(s, 1024));
    });

    group.finish();
}

criterion_group!(benches, bench_concurrent_inferlets);
criterion_main!(benches);
