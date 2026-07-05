//! Shared real-hardware (`cuda_native`) test harness.
//!
//! Boots the worker's prod embedded path in-proc — `pie_worker::run` in
//! SingleNode mode loads the model onto the GPU via the embedded cuda driver and
//! co-resides `pie_engine::bootstrap::bootstrap` — then drives inferlets through the
//! same in-proc `program::add` → `process::spawn` flow the mock canary uses,
//! bypassing the gateway/client edge (no msgpack/JSON codec, no identity header,
//! no `pie-server-py`).
//!
//! Reused by the cuda validation tests (`cuda_forward` = dense forward; the
//! Lane-C CAS-dedup + Lane-D fold-parity tests compose on these helpers). Every
//! cuda test is `#[ignore]`d (real GPU + `--features driver-cuda`) and boots
//! ONCE per process (auth/shmem singletons forbid a 2nd boot).
//!
//! The model snapshot is overridable via `PIE_CUDA_TEST_SNAPSHOT` (a local HF
//! snapshot dir — R3: the worker never downloads); the default is the Qwen3-0.6B
//! dense model on the reference box. Use a GDN model (e.g. Qwen3.5-0.8B) for RS
//! fold validation.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use pie_engine::program::{Manifest, ProgramName};
use pie_worker::WorkerHandle;

/// Default local HF snapshot (Qwen3-0.6B dense) on the reference box. Override
/// with `PIE_CUDA_TEST_SNAPSHOT=/path/to/snapshot` for another model/host.
pub const DEFAULT_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca";

/// Local HF snapshot for the Qwen3.5-0.8B GDN (hybrid linear-attention) model —
/// the RS-fold validation model. Override with `PIE_CUDA_TEST_GDN_SNAPSHOT`.
pub const DEFAULT_GDN_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

/// The dense model snapshot path (env-overridable).
pub fn snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_SNAPSHOT").unwrap_or_else(|_| DEFAULT_SNAPSHOT.to_string())
}

/// The GDN/hybrid-RS model snapshot path (env-overridable) — for fold validation.
pub fn gdn_snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_GDN_SNAPSHOT").unwrap_or_else(|_| DEFAULT_GDN_SNAPSHOT.to_string())
}

/// Single-model worker config for `snapshot_path`: `cuda_native`, no cluster
/// (→ SingleNode), client edge on an ephemeral loopback port (unused — tests
/// drive `process::spawn` directly in-proc).
pub fn cuda_toml_for(snapshot_path: &str) -> String {
    // A writable scratch dir so snapshot-using canaries (demo-persistent-kv:
    // Context::save/open over `/scratch`) work; harmless for fs-free inferlets.
    let scratch = std::env::temp_dir().join("pie-cuda-test-scratch");
    let _ = std::fs::create_dir_all(&scratch);
    format!(
        "[server]\n\
         host = \"127.0.0.1\"\n\
         port = 0\n\
         \n\
         [runtime]\n\
         allow_fs = true\n\
         fs_scratch_dir = \"{scratch}\"\n\
         \n\
         [auth]\n\
         enabled = false\n\
         \n\
         [model]\n\
         name = \"default\"\n\
         hf_repo = \"{snapshot_path}\"\n\
         \n\
         [model.driver]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         \n\
         [model.driver.options]\n\
         gpu_mem_utilization = 0.90\n\
         memory_profile = \"latency\"\n",
        scratch = scratch.display(),
    )
}

/// Worker config for the default dense model.
pub fn cuda_toml() -> String {
    cuda_toml_for(&snapshot())
}

/// Boot the embedded cuda engine in-proc with an explicit model snapshot (loads
/// it onto the GPU + bootstraps the runtime). Use a GDN snapshot for RS fold
/// validation, the dense default otherwise. Caller holds the handle and
/// `shutdown()`s it.
pub async fn boot_cuda_model(snapshot_path: &str) -> WorkerHandle {
    let cfg = pie_worker::Config::parse(&cuda_toml_for(snapshot_path))
        .expect("parse cuda worker config");
    pie_worker::run(cfg).await.expect("boot embedded cuda engine")
}

/// Boot the embedded cuda engine with the default dense model (Qwen3-0.6B).
pub async fn boot_cuda() -> WorkerHandle {
    boot_cuda_model(&snapshot()).await
}

/// Build `../inferlets/<name>` → wasm + manifest + program id.
pub fn load_prod_inferlet(name: &str) -> (Vec<u8>, Manifest, ProgramName) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../inferlets")
        .join(name);
    let status = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "--release"])
        .current_dir(&dir)
        .status()
        .unwrap_or_else(|e| panic!("spawn cargo build for {name}: {e}"));
    assert!(status.success(), "build {name} failed");

    let wasm_path = dir
        .join("target/wasm32-wasip2/release")
        .join(format!("{}.wasm", name.replace('-', "_")));
    let wasm = std::fs::read(&wasm_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", wasm_path.display()));
    let manifest =
        Manifest::parse(&std::fs::read_to_string(dir.join("Pie.toml")).unwrap()).unwrap();
    let program_name =
        ProgramName::parse(&format!("{name}@{}", manifest.package.version)).unwrap();
    (wasm, manifest, program_name)
}

/// Build + add + install an inferlet once; returns its program id for repeated
/// spawns (one install per process; spawn many).
pub async fn install_inferlet(name: &str) -> ProgramName {
    let (wasm, manifest, program_name) = load_prod_inferlet(name);
    pie_engine::program::add(wasm, manifest, true)
        .await
        .expect("add program");
    pie_engine::program::install(&program_name)
        .await
        .expect("install program");
    program_name
}

/// Spawn one inferlet run and capture its result (`Ok(text)` / `Err(msg)`) — the
/// result-captured pattern that surfaces host/forward errors (e.g. the lost-KV
/// -commit bug) instead of a silent "completed". Panics only on timeout.
pub async fn spawn_text(
    program: &ProgramName,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, String> {
    let input = format!(r#"{{"prompt":{prompt:?},"max_tokens":{max_tokens}}}"#);
    spawn_input(program, &input).await
}

/// Spawn an already-installed inferlet with a raw JSON input string, capturing
/// its result.
pub async fn spawn_input(program: &ProgramName, input_json: &str) -> Result<String, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    pie_engine::process::spawn(
        "cuda-test".into(),
        program.clone(),
        input_json.to_string(),
        None,
        false,
        Some(tx),
    )
    .expect("spawn process");
    tokio::time::timeout(Duration::from_secs(180), rx)
        .await
        .expect("inferlet did not finish within 180s")
        .expect("process result channel dropped")
}

/// Build + add + install + spawn an arbitrary production inferlet with a raw
/// JSON input (for canaries — fork / spec / snapshot — that take
/// inferlet-specific inputs). One-shot: installs then spawns.
pub async fn spawn_inferlet(name: &str, input_json: &str) -> Result<String, String> {
    let program = install_inferlet(name).await;
    spawn_input(&program, input_json).await
}
