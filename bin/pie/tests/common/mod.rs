//! Shared boot helper for the `bin/pie` cuda_native integration tests
//! (Phase-2 WS7 — the programmable-sampler 4090 real-driver pass).
//!
//! `boot_4090()` owns ALL cuda_native boot details (TOML + Mode + addresses)
//! so the capability test bodies (golf's Client-submit + hotel's
//! `sampler_assert`) stay pure submit+assert and can't drift from the boot.
//! Imported via `mod common;` in each integration-test file.
//!
//! ONE boot per process (the runtime owns process-global singletons — `auth`
//! panics on a 2nd boot; the driver grabs a fixed POSIX shmem), so every test
//! that calls `boot_4090()` must live in its own `#[ignore]` test process.

// Not every integration-test file uses every helper (each `mod common;` is a
// separate compilation), so silence unused-helper warnings per test binary.
#![allow(dead_code)]

use anyhow::Result;
use pie_bin::derive::derive_standalone;
use pie_bin::{Mode, run_standalone};

/// Install a `tracing` subscriber driven by `RUST_LOG` so the inproc
/// forward-path debug probes (`pie::driver::inproc`) and any other `tracing`
/// events surface on the diagnostic runs. Idempotent + non-panicking: a 2nd
/// call (or a boot that already set a global) is a silent no-op.
pub fn init_trace() {
    use tracing_subscriber::{EnvFilter, fmt};
    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
}

/// Default model for the 4090 pass — HF-cached, resolved to a local snapshot
/// path (the cuda_native worker never downloads, per the R3 policy).
pub const QWEN3_0_6B_REPO: &str = "Qwen/Qwen3-0.6B";

/// Resolve `Qwen/Qwen3-0.6B` to its **local HF cache snapshot dir** (the dir
/// holding `config.json` + `model.safetensors` + `tokenizer.json`). The
/// cuda_native worker enforces R3 (never downloads), so `hf_repo` must be a
/// local path, and the snapshot hash is machine-specific — resolve it at
/// runtime from `$HF_HOME`/`~/.cache/huggingface/hub`.
pub fn resolve_qwen3_snapshot() -> Result<String> {
    let hub = std::env::var("HF_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::PathBuf::from(home).join(".cache/huggingface")
        })
        .join("hub/models--Qwen--Qwen3-0.6B/snapshots");
    let snap = std::fs::read_dir(&hub)
        .with_context(|| {
            format!(
                "qwen-3-0.6b not in HF cache at {} — run `huggingface-cli download Qwen/Qwen3-0.6B`",
                hub.display()
            )
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists() && p.join("model.safetensors").exists())
        .with_context(|| format!("no complete qwen-3-0.6b snapshot under {}", hub.display()))?;
    Ok(snap.to_string_lossy().into_owned())
}

/// The cuda_native standalone TOML (`[controller]/[gateway]/[worker]`).
///
/// `binary_path` is omitted (accepted-but-ignored for cuda_native — the
/// standalone embeds the driver as a static lib). The CUDA driver loads
/// `config.json` + `model.safetensors` + `tokenizer.json` from the snapshot
/// dir, so `hf_repo` is a **local snapshot path** (R3: the worker never
/// downloads). `device` is an array (`["cuda:0"]`); auth off; gateway on an
/// ephemeral loopback port.
pub fn cuda_standalone_toml(hf_repo: &str) -> String {
    format!(
        "[controller]\n\
         \n\
         [gateway]\n\
         listen = \"127.0.0.1:0\"\n\
         \n\
         [worker]\n\
         [worker.auth]\n\
         enabled = false\n\
         \n\
         [[worker.model]]\n\
         name = \"qwen3\"\n\
         hf_repo = \"{hf_repo}\"\n\
         \n\
         [worker.model.driver]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         \n\
         [worker.model.driver.options]\n\
         gpu_mem_utilization = 0.85\n"
    )
}

/// Boot the embedded standalone (controller + gateway + worker) with the real
/// CUDA driver + qwen-3-0.6b on the 4090. The client edge is at
/// `handle.listen_addr` (`ws://{listen_addr}` for the `pie-client`).
pub async fn boot_4090() -> Result<pie_bin::StandaloneHandle> {
    let snapshot = resolve_qwen3_snapshot()?;
    let (controller, gateway, worker) = derive_standalone(&cuda_standalone_toml(&snapshot))?;
    run_standalone(controller, gateway, worker, Mode::Local).await
}

/// Standalone TOML for the **dummy** driver: fabricates everything the portable
/// driver reads from weights (no GPU, no 20 GB load, ~instant boot), so it
/// exercises the *driver-agnostic* client edge (connect → add_program →
/// launch → forward round-trip) without CUDA. `vocab_size`/`arch_name` are
/// supplied so the driver doesn't read `config.json`; `hf_repo` still points at
/// the snapshot for the tokenizer the runtime registers.
pub fn dummy_standalone_toml(hf_repo: &str) -> String {
    format!(
        "[controller]\n\
         \n\
         [gateway]\n\
         listen = \"127.0.0.1:0\"\n\
         \n\
         [worker]\n\
         [worker.auth]\n\
         enabled = false\n\
         \n\
         [[worker.model]]\n\
         name = \"qwen3\"\n\
         hf_repo = \"{hf_repo}\"\n\
         \n\
         [worker.model.driver]\n\
         type = \"dummy\"\n\
         device = [\"cpu\"]\n\
         \n\
         [worker.model.driver.options]\n\
         vocab_size = 151936\n\
         arch_name = \"qwen3\"\n"
    )
}

/// Boot the embedded standalone with the **dummy** driver (no GPU). Fast repro
/// harness for the driver-agnostic client edge (e.g. the chunked-`add_program`
/// session-bridge path).
pub async fn boot_dummy() -> Result<pie_bin::StandaloneHandle> {
    let snapshot = resolve_qwen3_snapshot()?;
    let (controller, gateway, worker) = derive_standalone(&dummy_standalone_toml(&snapshot))?;
    run_standalone(controller, gateway, worker, Mode::Local).await
}

// ── Client submit (golf) ────────────────────────────────────────────────────
//
// The capability-test half: build a capability inferlet to wasm, submit it to
// the engine `boot_4090()` brought up, and return its structured-JSON result
// for hotel's `sampler_assert`. Pure client-side (no GPU), so it compiles +
// type-checks Rust-only (without `--features driver-cuda`).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use pie_client::client::Client;

/// Build the capability inferlets to `wasm32-wasip2` and return
/// `(wasm, manifest)` for `name` ∈ {"mirostat", "grammar"}. Paths resolve from
/// the `bin/pie` crate dir to the runtime test-inferlets workspace. Builds both
/// (one cargo invocation) so a multi-capability harness pays the build once.
pub fn build_inferlet(name: &str) -> (PathBuf, PathBuf) {
    let workspace =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "mirostat", "-p", "grammar"])
        .current_dir(&workspace)
        .status()
        .expect("spawn cargo build for capability inferlets")
        .success();
    assert!(ok, "capability inferlet wasm build failed");
    let wasm = workspace.join(format!("target/wasm32-wasip2/debug/{name}.wasm"));
    let manifest = workspace.join(format!("{name}/Pie.toml"));
    assert!(wasm.exists(), "missing inferlet wasm: {}", wasm.display());
    assert!(manifest.exists(), "missing manifest: {}", manifest.display());
    (wasm, manifest)
}

/// Submit a capability inferlet to the running engine at `listen_addr` and
/// return its structured-JSON result. Builds `name` (`mirostat`/`grammar`),
/// then runs the canonical submit flow (connect → authenticate → add_program →
/// launch_process → `wait_for_return`). `program_name` is `{name}@{version}`
/// (e.g. `mirostat@0.1.0`); `input` is the inferlet's JSON run-params (e.g.
/// `{"max_tokens":48}`, or `{}` for defaults).
pub async fn run_inferlet(
    listen_addr: &std::net::SocketAddr,
    name: &str,
    program_name: &str,
    input: &str,
) -> Result<String> {
    let (wasm, manifest) = build_inferlet(name);

    // The gateway serves the multi-turn client WebSocket at `/v1/ws`
    // (`gateway/src/ingress/mod.rs`), gated on the `x-pie-identity` trust-edge
    // header (else 401). Standalone has no edge proxy, so inject it here.
    let client = Client::connect_with_identity(&format!("ws://{listen_addr}/v1/ws"), "test-user")
        .await
        .with_context(|| format!("connect to engine at ws://{listen_addr}/v1/ws"))?;
    // The bench/test engine disables public-key auth, so this returns early.
    client
        .authenticate("test-user", &None)
        .await
        .context("authenticate")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .with_context(|| format!("add_program {program_name}"))?;

    let mut proc = client
        .launch_process(program_name.to_string(), input.to_string(), true, None)
        .await
        .with_context(|| format!("launch_process {program_name}"))?;

    proc.wait_for_return().await
}
