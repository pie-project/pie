//! Shared real-hardware (`cuda_native`) test harness: boots the worker's prod
//! embedded path in-proc and drives inferlets directly (`program::add` →
//! `process::spawn`), bypassing the gateway/client edge. Every cuda test is
//! `#[ignore]`d and boots once per process.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use ::runtime::inferlet::program::{Manifest, ProgramName};
use worker::WorkerHandle;

/// Default local HF snapshot (Qwen3-0.6B dense). Override with
/// `PIE_CUDA_TEST_SNAPSHOT`.
pub const DEFAULT_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca";

/// Local HF snapshot for the Qwen3.5-0.8B GDN model (RS-fold validation).
/// Override with `PIE_CUDA_TEST_GDN_SNAPSHOT`.
pub const DEFAULT_GDN_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

/// The dense model snapshot path (env-overridable).
pub fn snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_SNAPSHOT").unwrap_or_else(|_| DEFAULT_SNAPSHOT.to_string())
}

/// The GDN/hybrid-RS model snapshot path (env-overridable).
pub fn gdn_snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_GDN_SNAPSHOT").unwrap_or_else(|_| DEFAULT_GDN_SNAPSHOT.to_string())
}

/// Single-model worker config for `snapshot_path`: `cuda_native`, no cluster.
pub fn cuda_toml_for(snapshot_path: &str) -> String {
    let scratch = std::env::temp_dir().join("pie-cuda-test-scratch");
    let _ = std::fs::create_dir_all(&scratch);
    // Expert residency is `[model] device_weight_budget` / `host_weight_budget`;
    // KV pages are sized from remaining VRAM, so changing the expert slab
    // changes the attention plan's reduction order too.
    let kv = std::env::var("PIE_CUDA_TEST_KV_PAGES")
        .map(|v| format!("max_total_pages = {v}\n"))
        .unwrap_or_default();
    format!(
        "[server]\n\
         host = \"127.0.0.1\"\n\
         port = 0\n\
         \n\
         [sandbox]\n\
         allow_fs = true\n\
         fs_scratch_dir = \"{scratch}\"\n\
         \n\
         \n\
         [model]\n\
         name = \"default\"\n\
         model = \"{snapshot_path}\"\n\
         \n\
         [engine]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         gpu_mem_utilization = 0.90\n\
         {kv}",
        scratch = scratch.display(),
    )
}

/// Worker config for the default dense model.
pub fn cuda_toml() -> String {
    cuda_toml_for(&snapshot())
}

/// Route `tracing` to stderr, once per process, at whatever `RUST_LOG` says
/// (`error` if it says nothing). Needed because device compile/load/launch
/// failures are reported only through `tracing::error!`, not the returned error.
fn wire_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error"));
        // try_init: a second subscriber already wired is not a failure here.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init();
    });
}

/// Boot the embedded cuda engine in-proc with an explicit model snapshot.
/// Caller holds the handle and `shutdown()`s it.
pub async fn boot_cuda_model(snapshot_path: &str) -> WorkerHandle {
    wire_tracing();
    let cfg =
        worker::Config::parse(&cuda_toml_for(snapshot_path)).expect("parse cuda worker config");
    worker::run(cfg).await.expect("boot embedded cuda engine")
}

/// Boot the embedded cuda engine with the default dense model (Qwen3-0.6B).
pub async fn boot_cuda() -> WorkerHandle {
    boot_cuda_model(&snapshot()).await
}

/// Build a curated inferlet fixture → wasm + manifest + program id. Fixtures
/// live at the repository's `tests/inferlets`, two levels above this crate's
/// manifest.
pub fn load_curated_inferlet(name: &str) -> (Vec<u8>, Manifest, ProgramName) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/inferlets")
        .join(name);
    assert!(
        dir.is_dir(),
        "no inferlet fixture at {} -- this is the path, not `cargo`",
        dir.display()
    );
    let status = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "--release"])
        .current_dir(&dir)
        .status()
        .unwrap_or_else(|e| panic!("spawn cargo build for {name}: {e}"));
    assert!(status.success(), "build {name} failed");

    let wasm_path = dir
        .join("target/wasm32-wasip2/release")
        .join(format!("{}.wasm", name.replace('-', "_")));
    // Workspace members' artifacts land one level up; non-members build in place.
    let wasm_path = if wasm_path.exists() {
        wasm_path
    } else {
        dir.join("../target/wasm32-wasip2/release")
            .join(format!("{}.wasm", name.replace('-', "_")))
    };
    let wasm =
        std::fs::read(&wasm_path).unwrap_or_else(|e| panic!("read {}: {e}", wasm_path.display()));
    let manifest =
        Manifest::parse(&std::fs::read_to_string(dir.join("Pie.toml")).unwrap()).unwrap();
    let program_name = ProgramName::parse(&format!("{name}@{}", manifest.package.version)).unwrap();
    (wasm, manifest, program_name)
}

/// Build + add + install an inferlet once; returns its program id for repeated spawns.
pub async fn install_inferlet(name: &str) -> ProgramName {
    let (wasm, manifest, program_name) = load_curated_inferlet(name);
    ::runtime::inferlet::program::add(wasm, manifest, true)
        .await
        .expect("add program");
    ::runtime::inferlet::program::install(&program_name)
        .await
        .expect("install program");
    program_name
}

/// Spawn one inferlet run and capture its result (`Ok(text)` / `Err(msg)`).
/// Panics only on timeout.
pub async fn spawn_text(
    program: &ProgramName,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, String> {
    let input = format!(r#"{{"prompt":{prompt:?},"max_tokens":{max_tokens}}}"#);
    spawn_input(program, &input).await
}

/// Spawn an already-installed inferlet with a raw JSON input string.
pub async fn spawn_input(program: &ProgramName, input_json: &str) -> Result<String, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    ::runtime::inferlet::process::spawn(
        "cuda-test".into(),
        program.clone(),
        input_json.to_string(),
        None,
        false,
        Some(tx),
    )
    .expect("spawn process");
    // A timeout is an `Err`, not a panic: "did not answer" is a result about
    // the inferlet; callers that want it fatal use `.expect()`.
    match tokio::time::timeout(Duration::from_secs(180), rx).await {
        Err(_) => Err("no answer within 180s".to_string()),
        Ok(result) => result.expect("process result channel dropped"),
    }
}

/// Build + add + install + spawn an arbitrary curated inferlet fixture with a
/// raw JSON input. One-shot: installs then spawns.
pub async fn spawn_inferlet(name: &str, input_json: &str) -> Result<String, String> {
    let program = install_inferlet(name).await;
    spawn_input(&program, input_json).await
}

/// Refuse a completion that is not made of words: a non-empty check alone
/// passes garbage output from a broken reduction. Requires at least
/// `min_words` runs of 3+ letters and >=2/5 of non-space chars alphanumeric.
pub fn assert_coherent(text: &str, min_words: usize) {
    let words = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| w.chars().count() >= 3)
        .count();
    let dense: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    let alnum = dense.iter().filter(|c| c.is_alphanumeric()).count();
    assert!(
        words >= min_words,
        "completion has {words} word(s) of 3+ letters, wanted {min_words}: {text:?}"
    );
    assert!(
        !dense.is_empty() && alnum * 5 >= dense.len() * 2,
        "completion is {alnum}/{} alphanumeric, wanted two fifths: {text:?}",
        dense.len()
    );
}
