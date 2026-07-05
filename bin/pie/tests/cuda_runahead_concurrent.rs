//! **Concurrent run-ahead-carrier device-correctness repro** (charlie).
//!
//! Companion to `cuda_concurrent` — that test drives the SYNCHRONOUS decode path
//! (`GenStep::execute`, no `next_inputs`); THIS one drives the RUN-AHEAD PIPELINED
//! path (`collect_tokens_pipelined`), which is the ONLY path that arms the
//! device-side next-input carrier (`set_pipeline_source_link` / `inject_next_input`
//! / `apply_next_input_carrier`). The `runahead` inferlet decodes "hello world"
//! greedily through the carrier and reports `pipelined=[...]`.
//!
//! Greedy (top-k=1, temp=0) is deterministic, so a request's `pipelined` stream
//! must be IDENTICAL whether it runs alone or concurrently with a co-batched
//! fleet. We run the fleet sequentially (reference) then concurrently and assert
//! each concurrent `pipelined` stream equals its sequential reference. A carrier
//! whose producer→consumer link doesn't survive the concurrent co-batch merge
//! (bravo's `e92d30f8` batch-level granularity) makes co-batched streams diverge.
//!
//! `#[ignore]` (needs 4090 + cuda + qwen3-0.6b). Run:
//!   PIE_COMPILER_LAUNCHER=env cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_runahead_concurrent -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

const FLEET: usize = 8;

/// Extract the `pipelined=[...]` token array from the `runahead` inferlet's
/// result string (`MATCH=.. ANCHOR_OK=.. CLEAR_OK=.. pipelined=[..] sync=[..]`).
fn parse_pipelined(json: &str) -> Option<Vec<i64>> {
    let key = "pipelined=[";
    let start = json.find(key)? + key.len();
    let end = json[start..].find(']')? + start;
    let toks: Vec<i64> = json[start..end]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if toks.is_empty() { None } else { Some(toks) }
}

async fn run_one(addr: &str) -> Result<Option<Vec<i64>>> {
    let c = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    c.authenticate("test-user", &None).await?;
    let mut proc = c.launch_process("runahead@0.1.0".into(), "8".into(), true).await?;
    Ok(parse_pipelined(&proc.wait_for_return().await?))
}

async fn run_fleet_concurrent(addr: &str) -> Vec<Option<Vec<i64>>> {
    let mut procs = Vec::new();
    for _ in 0..FLEET {
        let addr = addr.to_string();
        procs.push(tokio::spawn(async move { run_one(&addr).await.ok().flatten() }));
    }
    let mut out = Vec::new();
    for h in procs { out.push(h.await.unwrap_or(None)); }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "concurrent run-ahead carrier: needs the 4090 + cuda + qwen3-0.6b"]
async fn concurrent_runahead_matches_sequential() -> Result<()> {
    common::init_trace();

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    anyhow::ensure!(
        Command::new("cargo")
            .args(["build", "--target", "wasm32-wasip2", "-p", "runahead"])
            .current_dir(&ws).status()?.success(),
        "runahead wasm build failed"
    );
    let wasm = ws.join("target/wasm32-wasip2/debug/runahead.wasm");
    let manifest = ws.join("runahead/Pie.toml");

    let pie = common::boot_4090().await?;
    let addr = pie.listen_addr.to_string();
    eprintln!("[runahead-conc] booted, addr={addr}");

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    setup.authenticate("test-user", &None).await?;
    setup.add_program(&wasm, &manifest, true).await.context("add_program")?;

    // Reference: run ALONE, sequentially (no concurrency) — the carrier's RETAIN
    // strictly precedes its INJECT with no co-batch merge.
    let mut reference = Vec::new();
    for k in 0..FLEET {
        let r = run_one(&addr).await.ok().flatten();
        eprintln!("[runahead-conc] seq[{k}] = {r:?}");
        reference.push(r);
    }

    // Concurrent: launch the whole fleet at once → forces co-batched fires whose
    // producer→consumer carrier links must survive the concurrent merge.
    let concurrent = run_fleet_concurrent(&addr).await;

    let mut n_ok = 0usize;
    for k in 0..FLEET {
        let good = concurrent[k].is_some() && concurrent[k] == reference[k];
        if good { n_ok += 1; }
        eprintln!(
            "[runahead-conc] pipeline {k}: {} conc={:?} ref={:?}",
            if good { "OK" } else { "MISMATCH" }, concurrent[k], reference[k]
        );
    }
    eprintln!("[runahead-conc] {n_ok}/{FLEET} concurrent pipelined streams == their sequential reference");

    assert_eq!(
        n_ok, FLEET,
        "concurrent run-ahead-carrier corruption: {n_ok}/{FLEET} concurrent pipelined streams matched their own sequential reference"
    );
    Ok(())
}
