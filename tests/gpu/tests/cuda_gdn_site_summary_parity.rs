//! Live check for the driver→engine site-summary handshake on the dense
//! qwen3.5-0.8B hybrid (the GDN model whose declared plan traces + validates
//! at boot — `declared_facts.cpp`).
//!
//! What the increment claims and this harness verifies on the real GPU:
//!
//! * With `PIE_DECLARED_FORWARD=1` the driver holds a validated plan and its
//!   capability payload carries `model_site_summary` — PRESENT-BUT-EMPTY on
//!   this dense model (no MoE checkpoint fits this GPU, so a populated
//!   summary is covered by unit/dummy tests instead). The one-line
//!   `[pie-driver-cuda] model_site_summary: declared_plan=yes expert_sites=0`
//!   statement is on this process's stderr (run with `--nocapture` and grep).
//! * With the gate OFF the summary is empty for the other reason
//!   (`declared_plan=no`), and either way the scheduler's fire planning
//!   reduces to exactly today's behavior: the greedy generate-gdn decode is
//!   BYTE-IDENTICAL off/on, compared through a cross-invocation record file
//!   (the `cuda_declared_forward_parity` idiom — that harness stays the
//!   qwen3 dense control).
//!
//! Uses the locally cached **Qwen3.5-0.8B-Base** snapshot (the variant on
//! this box); parity is within-model, so the Base/instruct distinction does
//! not matter here.
//!
//! `#[ignore]`, driver-cuda. Run OFF then ON, the second invocation gates:
//!   cargo test -p pie-gpu-tests --features driver-cuda \
//!     --test cuda_gdn_site_summary_parity -- --ignored --nocapture
//!   PIE_DECLARED_FORWARD=1 cargo test ... --test cuda_gdn_site_summary_parity ...

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

const MAX_TOKENS: usize = 32;

/// Resolve the locally cached Base snapshot (`resolve_qwen35_snapshot` in
/// common resolves the non-Base variant, which this box does not cache).
fn resolve_qwen35_base_snapshot() -> Result<String> {
    let hub = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join(".cache/huggingface")
        })
        .join("hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots");
    let snap = std::fs::read_dir(&hub)
        .with_context(|| format!("qwen3.5-0.8b-base not in HF cache at {}", hub.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.join("config.json").exists()
                && (p.join("model.safetensors").exists()
                    || p.join("model.safetensors.index.json").exists())
        })
        .with_context(|| format!("no complete snapshot under {}", hub.display()))?;
    Ok(snap.to_string_lossy().into_owned())
}

fn record_path(declared: bool) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pie_gdn_site_summary_parity_{}.txt",
        if declared { "on" } else { "off" }
    ))
}

/// The generate-gdn inferlet's `... [t0, t1, ...]` token list.
fn parse_generated_tokens(result: &str) -> Option<Vec<u32>> {
    let lb = result.find('[')?;
    let rb = result.find(']')?;
    let toks: Vec<u32> = result[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();
    (!toks.is_empty()).then_some(toks)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a CUDA GPU + qwen3.5-0.8b-base; run gate-OFF then gate-ON"]
async fn gdn_site_summary_parity() -> Result<()> {
    common::init_trace();
    let declared = std::env::var("PIE_DECLARED_FORWARD")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    let snapshot = resolve_qwen35_base_snapshot()?;
    let (controller, gateway, worker) =
        pie_bin::derive::derive_standalone(&common::cuda_standalone_toml(&snapshot))?;
    let pie = pie_bin::run_standalone(controller, gateway, worker).await?;
    eprintln!(
        "[gdn-site-summary] booted listen={} declared={declared}",
        pie.listen_addr
    );

    // generate-gdn: the greedy decode that binds BOTH working sets (KV +
    // recurrent state) — the hybrid's linear-attention layers need
    // runtime-assigned rs_cache slots (see cuda_mtp_stage1).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "generate-gdn"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "generate-gdn wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/generate_gdn.wasm");
    let manifest = ws.join("generate-gdn/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client
        .authenticate("test-user", &None)
        .await
        .context("auth")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program generate-gdn")?;

    let mut proc = client
        .launch_process(
            "generate-gdn@0.1.0".to_string(),
            MAX_TOKENS.to_string(),
            true,
        )
        .await
        .context("launch generate-gdn")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    drop(client);
    let tokens = parse_generated_tokens(&json)
        .with_context(|| format!("parse tokens from generate-gdn result: {json:?}"))?;
    eprintln!(
        "[gdn-site-summary] declared={declared}: {} tokens: {tokens:?}",
        tokens.len()
    );
    anyhow::ensure!(
        tokens.len() == MAX_TOKENS,
        "expected {MAX_TOKENS} decoded tokens, got {}",
        tokens.len()
    );

    let record = tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(record_path(declared), &record).context("write parity record")?;

    // Cross-invocation gate: when the counterpart record exists, the decode
    // must be byte-identical — the summary (present-but-empty vs absent) is
    // informational and changes nothing.
    if let Ok(other) = std::fs::read_to_string(record_path(!declared)) {
        anyhow::ensure!(
            other == record,
            "generate-gdn diverged across the PIE_DECLARED_FORWARD gate\n \
             {}: {other}\n {}: {record}",
            if declared { "off" } else { "on" },
            if declared { "on" } else { "off" },
        );
        eprintln!(
            "[gdn-site-summary] PARITY OK: {MAX_TOKENS} tokens byte-identical off/on"
        );
    } else {
        eprintln!(
            "[gdn-site-summary] no counterpart record yet; run the other gate to compare"
        );
    }

    pie.shutdown().await;
    Ok(())
}
