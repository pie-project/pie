//! **Concurrent-decode device-correctness repro** (charlie, #1 blocker).
//!
//! Greedy decode (top-k=1, temp=0) is deterministic: a request's token stream
//! must be IDENTICAL whether it runs alone or concurrently with others. This
//! test runs a fleet sequentially (the reference) then concurrently and asserts
//! each pipeline's concurrent stream equals its sequential stream.
//!
//! Modes:
//!   default        — 8 IDENTICAL prompts (triggers KV content-dedup/sharing).
//!   PIE_DISTINCT=1 — 8 DISTINCT prompts (no dedup) — isolates the sharing path.
//!
//! `#[ignore]` (needs 4090 + cuda + qwen3-0.6b). Run:
//!   PIE_COMPILER_LAUNCHER=env cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_concurrent -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

const FLEET: usize = 8;

fn parse_tokens(json: &str) -> Option<Vec<i64>> {
    let lb = json.rfind('[')?;
    let rb = json[lb..].find(']')? + lb;
    let toks: Vec<i64> = json[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if toks.is_empty() { None } else { Some(toks) }
}

async fn run_one(addr: &str, prompt: &str) -> Result<Option<Vec<i64>>> {
    let c = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    c.authenticate("test-user", &None).await?;
    let input = if prompt.is_empty() { "{}".to_string() } else { prompt.to_string() };
    let mut proc = c.launch_process("generate@0.1.0".into(), input, true).await?;
    Ok(parse_tokens(&proc.wait_for_return().await?))
}

async fn run_fleet_concurrent(addr: &str, prompts: &[String]) -> Vec<Option<Vec<i64>>> {
    let mut procs = Vec::new();
    for p in prompts {
        let addr = addr.to_string();
        let p = p.clone();
        procs.push(tokio::spawn(async move { run_one(&addr, &p).await.ok().flatten() }));
    }
    let mut out = Vec::new();
    for h in procs { out.push(h.await.unwrap_or(None)); }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "concurrent-decode device-correctness: needs the 4090 + cuda + qwen3-0.6b"]
async fn concurrent_decode_matches_sequential() -> Result<()> {
    common::init_trace();

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    anyhow::ensure!(
        Command::new("cargo")
            .args(["build", "--target", "wasm32-wasip2", "-p", "generate"])
            .current_dir(&ws).status()?.success(),
        "generate wasm build failed"
    );
    let wasm = ws.join("target/wasm32-wasip2/debug/generate.wasm");
    let manifest = ws.join("generate/Pie.toml");

    let pie = common::boot_4090().await?;
    let addr = pie.listen_addr.to_string();
    eprintln!("[concurrent] booted, addr={addr}");

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    setup.authenticate("test-user", &None).await?;
    setup.add_program(&wasm, &manifest, true).await.context("add_program")?;

    let distinct = std::env::var_os("PIE_DISTINCT").is_some();
    let prompts: Vec<String> = (0..FLEET)
        .map(|k| if distinct { format!("the quick brown fox number {k} jumps over the") } else { String::new() })
        .collect();
    eprintln!("[concurrent] mode={}", if distinct { "DISTINCT" } else { "IDENTICAL" });

    // Reference: run each prompt ALONE, sequentially (no concurrency).
    let mut reference = Vec::new();
    for (k, p) in prompts.iter().enumerate() {
        let r = run_one(&addr, p).await.ok().flatten();
        eprintln!("[concurrent] seq[{k}] = {r:?}");
        reference.push(r);
    }

    // Concurrent: launch the whole fleet at once.
    let concurrent = run_fleet_concurrent(&addr, &prompts).await;

    let mut n_ok = 0usize;
    for k in 0..FLEET {
        let good = concurrent[k].is_some() && concurrent[k] == reference[k];
        if good { n_ok += 1; }
        eprintln!(
            "[concurrent] pipeline {k}: {} conc={:?} ref={:?}",
            if good { "OK" } else { "MISMATCH" }, concurrent[k], reference[k]
        );
    }
    eprintln!("[concurrent] {n_ok}/{FLEET} concurrent streams == their sequential reference");

    assert_eq!(
        n_ok, FLEET,
        "concurrent-decode corruption: {n_ok}/{FLEET} concurrent streams matched their own sequential reference"
    );
    Ok(())
}
