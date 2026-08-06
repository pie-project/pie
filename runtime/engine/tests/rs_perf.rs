//! Recurrent-state scheduling benchmark (host-side, deterministic).
//!
//! This measures the ONE thing the RS publication seam changes: how many host
//! round trips a linear/SSM decode costs. The dummy driver's
//! `callback_delay_ms` stands in for device execution — each launch's
//! completion fires from its own worker thread after D ms, so launches
//! OVERLAP if and only if the engine actually lets them run ahead:
//!
//!   * strict host lockstep -> wall ~= tokens * D  (nothing overlaps)
//!   * run-ahead depth 2    -> wall ~= tokens * D / 2
//!   * k-wide frames        -> further, by folding seal boundaries
//!
//! Using a simulated device instead of a real one is deliberate: the change is
//! purely host scheduling, and a fixed D isolates it from kernel time, batch
//! composition and memory effects that would otherwise dominate a GPU run.
//!
//! Ignored by default (it sleeps). Drive it with env vars:
//!
//!   PIE_BENCH_PROGRAM   inferlet name (default `generate-gdn`)
//!   PIE_BENCH_TOKENS    tokens per lane (default 24)
//!   PIE_BENCH_DELAY_MS  simulated device latency (default 4)
//!   PIE_BENCH_LANES     concurrent processes (default 1)
//!   PIE_BENCH_DENSE     extra dense (no-RS) `generate` lanes (default 0)
//!   PIE_BENCH_FRAME_SIZE  k (default 2, the config default)
//!
//! It prints one `BENCH ...` line of `key=value` pairs on stdout.
//!
//! ```text
//! cargo test -p pie-engine --test rs_perf -- --ignored --nocapture
//! ```
//!
//! # Measured, 48 tokens, median of 9, vs the pre-publish-seam engine
//!
//! Single recurrent lane, production guest, k=2 — the barrier's cost is the
//! host time it refused to overlap, so it shows as a fixed ~31 ms saving over
//! the run's 47 decode fires (~0.66 ms/fire):
//!
//! | device D | before | after | |
//! |----------|--------|-------|-|
//! | 1 ms     | 105.5  | 74.6  | 1.41x |
//! | 2 ms     | 157.4  | 125.4 | 1.26x |
//!
//! Full-width recurrent frames (`generate-gdn-frame`, a shape the old engine
//! rejected outright): k=1 103.8 -> k=2 92.9 -> k=3 87.7 ms, i.e. -15.5%.
//!
//! The run-ahead depth lever was DEAD for recurrent workloads before: the old
//! engine measures flat across dispatch depth 1/2/3 (195-224 ms at
//! 8 lanes) because the barrier pinned every RS lane to depth 1. It is live
//! now (155/255/263 ms), which also means RS workloads inherit the wait-all
//! scheduler's known depth behaviour: in a FULLY BATCHED, zero-headroom fleet
//! (8 lanes, ~29 of 32 rows per launch) depth 2 costs ~1.7x versus depth 1.
//! That is pre-existing and RS-independent — the unmodified engine on a
//! PURE-ATTENTION model measures the same 610 -> 1034 ms going from depth 1 to
//! 2, and at matched depth 1 the two engines are identical (601-624 vs
//! 578-613 ms). `[model.scheduler] frame_dispatch_depth` is the deployment
//! lever; its default of 2 was tuned on real hardware, where run-ahead has
//! headroom to exploit.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

mod common;
use common::{
    MockEnv, create_mock_env, inferlets,
    mock_device::{DelayedBehavior, EchoBehavior},
};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

struct Bench {
    rt: tokio::runtime::Runtime,
    #[allow(dead_code)]
    env: MockEnv,
}

static BENCH: OnceLock<Bench> = OnceLock::new();

/// Programs installed up front. `add_and_install` shells out to `cargo build`
/// and mutates the shared program registry, so installing from several test
/// threads races; doing it once inside the `OnceLock` avoids that.
const PROGRAMS: [&str; 3] = ["generate-gdn", "generate-gdn-frame", "generate"];

/// Curated fixtures live in the repo-root `tests/inferlets` workspace, not the
/// engine's own. `text-completion-bench` is the one that matters here: it is
/// the production-shaped workload, the only benchmark guest that keeps a real
/// RUN-AHEAD WINDOW open (`submit_ahead`) instead of draining after every
/// submit, so it is the only one whose timing can show a host barrier at all.
const CURATED: [&str; 1] = ["text-completion-bench"];

fn curated_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets")
}

async fn install_curated(name: &str) {
    let workspace = curated_dir();
    let status = std::process::Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "--release",
            "-p",
            name,
        ])
        .current_dir(&workspace)
        .status()
        .unwrap_or_else(|e| panic!("build curated inferlet {name}: {e}"));
    assert!(status.success(), "build {name} failed");
    let wasm = std::fs::read(
        workspace
            .join("target/wasm32-wasip2/release")
            .join(format!("{}.wasm", name.replace('-', "_"))),
    )
    .expect("read curated wasm");
    let manifest = pie_engine::inferlet::program::Manifest::parse(
        &std::fs::read_to_string(workspace.join(name).join("Pie.toml")).unwrap(),
    )
    .unwrap();
    let program_name = ProgramName::parse(&format!("{name}@{}", manifest.package.version)).unwrap();
    pie_engine::inferlet::program::add(wasm, manifest, true)
        .await
        .unwrap();
    pie_engine::inferlet::program::install(&program_name)
        .await
        .unwrap();
}

fn bench() -> &'static Bench {
    BENCH.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        // A recurrent-state pool large enough that pool pressure never
        // confounds the timing, and a page pool sized for every lane.
        // Device time is modelled by `DelayedBehavior`, which sleeps on the
        // dummy driver's LAUNCH path — so a fire genuinely occupies the
        // device for D, and launches serialize there exactly as they do on a
        // GPU. `callback_delay_ms` would only delay the work-item completion
        // callback, leaving the launch path free.
        let delay = Duration::from_millis(env_usize("PIE_BENCH_DELAY_MS", 4) as u64);
        // `PIE_BENCH_LINEAR=0` bootstraps a PURE-ATTENTION model instead —
        // the control arm. On a linear model every forward must bind a
        // recurrent-state working set (`fire::rs::validate_count`), so a
        // dense tenant cannot coexist with an RS one on the same model; the
        // KV-only path has to be measured on its own deployment.
        let env = create_mock_env(
            "bench-model",
            1,
            4096,
            Arc::new(DelayedBehavior {
                inner: EchoBehavior(7),
                latency: delay,
            }),
        );
        let env = if env_usize("PIE_BENCH_LINEAR", 1) > 0 {
            env.with_recurrent_state(256, 4096)
        } else {
            env
        };
        // A harness parameter that selects an arm, not engine config: k
        // reaches the engine through `[model.scheduler] frame_size` below.
        let env = env.with_frame_size(env_usize("PIE_BENCH_FRAME_SIZE", 2) as u32);
        let config = env.config();
        rt.block_on(async {
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
            for name in PROGRAMS {
                inferlets::add_and_install(name).await;
            }
            for name in CURATED {
                install_curated(name).await;
            }
        });
        Bench { rt, env }
    })
}

/// Spawn arguments for one guest: curated fixtures take a JSON envelope and
/// report their token count differently from the small engine fixtures.
fn spawn_spec(name: &str, tokens: usize) -> (ProgramName, String, String) {
    if CURATED.contains(&name) {
        let manifest = pie_engine::inferlet::program::Manifest::parse(
            &std::fs::read_to_string(curated_dir().join(name).join("Pie.toml")).unwrap(),
        )
        .unwrap();
        return (
            ProgramName::parse(&format!("{name}@{}", manifest.package.version)).unwrap(),
            format!(
                r#"{{"prompt":"hi","max_tokens":{tokens},"temperature":0.0,"ignore_eos":true}}"#
            ),
            format!("\"num_output_tokens\":{tokens}"),
        );
    }
    (
        ProgramName::parse(&format!("{name}@0.1.0")).unwrap(),
        tokens.to_string(),
        format!("generated {tokens} tokens"),
    )
}

/// Wall time to run `lanes` copies of `program` plus `dense` copies of the
/// pure-attention `generate`, all concurrently, each for `tokens` tokens.
/// Returns `(total_wall, per_program_walls, dense_walls)`.
fn run_fleet(
    program: &str,
    lanes: usize,
    dense: usize,
    tokens: usize,
) -> (Duration, Vec<Duration>, Vec<Duration>) {
    let b = bench();
    b.rt.block_on(async move {
        let mut handles = Vec::new();
        let started = Instant::now();
        for (name, count, is_dense) in [(program, lanes, false), ("generate", dense, true)] {
            let (program_name, input, expect) = spawn_spec(name, tokens);
            for _ in 0..count {
                let (tx, rx) = tokio::sync::oneshot::channel();
                process::spawn(
                    "bench-user".into(),
                    program_name.clone(),
                    input.clone(),
                    None,
                    false,
                    Some(tx),
                )
                .expect("spawn");
                handles.push((is_dense, started, rx, expect.clone()));
            }
        }
        let mut program_walls = Vec::new();
        let mut dense_walls = Vec::new();
        for (is_dense, started, rx, expect) in handles {
            let result = tokio::time::timeout(Duration::from_secs(600), rx)
                .await
                .expect("lane finished")
                .expect("result channel");
            let elapsed = started.elapsed();
            match result {
                Ok(output) => assert!(
                    output.contains(&expect),
                    "lane produced {output}, expected {expect}"
                ),
                Err(error) => panic!("lane failed: {error}"),
            }
            if is_dense {
                dense_walls.push(elapsed);
            } else {
                program_walls.push(elapsed);
            }
        }
        (started.elapsed(), program_walls, dense_walls)
    })
}

fn mean_ms(samples: &[Duration]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / samples.len() as f64
}

#[test]
#[ignore = "performance measurement: sleeps for the simulated device latency"]
fn recurrent_decode_throughput() {
    let program = env_string("PIE_BENCH_PROGRAM", "generate-gdn");
    let tokens = env_usize("PIE_BENCH_TOKENS", 24);
    let delay_ms = env_usize("PIE_BENCH_DELAY_MS", 4);
    let lanes = env_usize("PIE_BENCH_LANES", 1);
    let dense = env_usize("PIE_BENCH_DENSE", 0);
    let linear = env_usize("PIE_BENCH_LINEAR", 1);
    let k = pie_engine::scheduler::configured_frame_size();

    // One untimed warm lane: first submit of a pass traces, hashes and binds
    // its PTIR program, and the wasm module/instance pool is cold.
    let _ = run_fleet(&program, 1, 0, tokens.min(4));

    // Structural counters, snapshotted around the timed run. `launches` is the
    // number of frame posts the driver saw: with a device that charges a fixed
    // cost per launch, FEWER and FATTER launches is strictly better, so this
    // separates a scheduling win from a batching regression.
    let ops_before = bench().env.operations().len();
    let launches_before = bench()
        .env
        .operations()
        .iter()
        .filter(|o| o.as_str() == "launch")
        .count();

    let (wall, program_walls, dense_walls) = run_fleet(&program, lanes, dense, tokens);

    let ops = bench().env.operations();
    let launches = ops.iter().filter(|o| o.as_str() == "launch").count() - launches_before;
    let rows: usize = ops[ops_before.min(ops.len())..]
        .iter()
        .filter_map(|o| o.strip_prefix("launch-shape tokens="))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter_map(|n| n.parse::<usize>().ok())
        .sum();

    // Tokens per lane include the prefill token, so the decode fires that the
    // scheduling change actually governs are `tokens - 1`.
    let decode_fires = tokens.saturating_sub(1);
    let lockstep_floor_ms = (decode_fires * delay_ms) as f64;
    let per_lane_ms = mean_ms(&program_walls);

    println!(
        "BENCH program={program} k={k} tokens={tokens} delay_ms={delay_ms} lanes={lanes} \
         dense={dense} linear={linear} wall_ms={:.1} lane_ms={:.1} dense_ms={:.1} \
         launches={launches} launch_tokens={rows} tokens_per_launch={:.2} \
         lockstep_floor_ms={lockstep_floor_ms:.1} speedup_vs_lockstep={:.2}",
        wall.as_secs_f64() * 1000.0,
        per_lane_ms,
        mean_ms(&dense_walls),
        if launches > 0 {
            rows as f64 / launches as f64
        } else {
            0.0
        },
        if per_lane_ms > 0.0 {
            lockstep_floor_ms / per_lane_ms
        } else {
            0.0
        }
    );
}
