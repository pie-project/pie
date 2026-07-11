//! Test inferlet build helper.
//!
//! Provides functions to build and locate test inferlet WASM components.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use pie_engine::process::ProcessId;
use pie_engine::program::ProgramName;

/// Root directory of the test inferlets workspace.
fn inferlets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/inferlets")
}

fn target(name: &str) -> &'static str {
    // Guests using the `inferlet::ptir` bridge build as wasip2: the bridge's
    // channel registry pulls std `HashMap`, whose wasip3 std imports
    // `wasi:random/insecure-seed@0.3.0-rc-2026-03-15` — version-mismatched with
    // the engine wasmtime's `@0.3.0`, so the component fails to instantiate.
    if matches!(
        name,
        "direct-channel-e2e"
            | "direct-mixed-e2e"
            | "generate"
            | "grammar"
            | "runahead"
            | "lowlevel-chat"
            | "specverify"
            | "mtpverify"
    ) {
        "wasm32-wasip2"
    } else {
        "wasm32-wasip3"
    }
}

fn build_inferlet(name: &str) {
    let status = Command::new("cargo")
        .args(["build", "--target", target(name), "-p", name])
        .current_dir(inferlets_dir())
        .status()
        .unwrap_or_else(|error| panic!("failed to build test inferlet {name}: {error}"));
    assert!(status.success(), "test inferlet {name} build failed");
}

/// Build the current-SDK inferlets exercised by the executable e2e suite.
pub fn build_inferlets() {
    for name in [
        "echo",
        "context",
        "error",
        "direct-channel-e2e",
        "direct-mixed-e2e",
    ] {
        build_inferlet(name);
    }
}

/// Path to a compiled test inferlet WASM file.
pub fn inferlet_wasm_path(name: &str) -> PathBuf {
    // Cargo replaces hyphens with underscores in output filenames
    let filename = format!("{}.wasm", name.replace('-', "_"));
    inferlets_dir()
        .join(format!("target/{}/debug", target(name)))
        .join(filename)
}

/// Read the WASM binary for a test inferlet. Builds if needed.
pub fn read_inferlet_wasm(name: &str) -> Vec<u8> {
    let path = inferlet_wasm_path(name);
    if !path.exists() {
        build_inferlet(name);
    }
    std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e))
}

/// Read and parse the Pie.toml manifest for a test inferlet.
pub fn read_inferlet_manifest(name: &str) -> pie_engine::program::Manifest {
    let path = inferlets_dir().join(name).join("Pie.toml");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    pie_engine::program::Manifest::parse(&content)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path.display(), e))
}

/// Add and install a test inferlet in one step (async).
pub async fn add_and_install(name: &str) -> ProgramName {
    let wasm = read_inferlet_wasm(name);
    let manifest = read_inferlet_manifest(name);
    let program_name = ProgramName::parse(&format!("{name}@0.1.0")).unwrap();
    pie_engine::program::add(wasm, manifest, true)
        .await
        .unwrap();
    pie_engine::program::install(&program_name).await.unwrap();
    program_name
}

/// Wait for a process to complete (disappear from process::list()).
/// Returns true if the process exited within the timeout, false otherwise.
pub fn wait_for_process(id: ProcessId, timeout: Duration) -> bool {
    let start = Instant::now();
    loop {
        if !pie_engine::process::list().contains(&id) {
            return true;
        }
        if start.elapsed() > timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
